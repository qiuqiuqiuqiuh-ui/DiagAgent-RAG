import logging
from collections.abc import Callable
from contextlib import nullcontext
from time import perf_counter
from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from typing_extensions import TypedDict

from src.agents.citations import CitationError, bind_citations
from src.agents.policy import (
    DiagnosticPolicy,
    EvidenceContext,
    LLMDiagnosticPolicy,
    MockDiagnosticPolicy,
)
from src.agents.schemas import (
    Citation,
    CriticAssessment,
    DiagnosticRequest,
    DiagnosticResult,
    DiagnosticTrace,
    FaultCodeFinding,
    Intent,
    NodeName,
    NodeTrace,
    SensorFinding,
    StateTransition,
    ToolFinding,
)
from src.agents.tools import DiagnosticTools
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.llm import BaseLLMClient
from src.core.logging import log_event
from src.core.settings import Settings
from src.ingestion.schemas import Chunk
from src.retrieval.query import resolve_retrieval_query
from src.retrieval.schemas import SearchHit

_LOGGER = logging.getLogger("diagagent.agents")
_MAX_RETRIES = 2
_ESCALATION = "Escalate to qualified maintenance with the equipment ID, event log, and readings."
_MISSING_EVIDENCE = "Maintenance documentation that supports the requested diagnosis"


class EvidenceRetriever(Protocol):
    def search(self, query: str, top_k: int = 5, *, rerank: bool = True) -> list[SearchHit]:
        """Return the ranked evidence available for the query."""


class DiagnosticInput(TypedDict):
    request: DiagnosticRequest


class DiagnosticUpdate(TypedDict, total=False):
    intent: Intent
    chunks: tuple[Chunk, ...]
    tool_findings: tuple[ToolFinding, ...]
    context: EvidenceContext
    answer: str
    citations: tuple[Citation, ...]
    evidence_issue: str | None
    feedback: CriticAssessment
    retry_count: int
    top_k: int
    steps: tuple[NodeTrace, ...]
    result: DiagnosticResult


class DiagnosticState(DiagnosticInput, DiagnosticUpdate):
    """Input and channels shared by the diagnosis nodes."""


def _abstention(
    intent: Intent | None,
    missing: tuple[str, ...],
    findings: tuple[ToolFinding, ...] = (),
) -> DiagnosticResult:
    return DiagnosticResult(
        status="insufficient_evidence",
        intent=intent,
        answer="The available evidence is insufficient for a grounded diagnosis.",
        missing_aspects=missing,
        tool_findings=findings,
        escalation_recommendation=_ESCALATION,
    )


def _observe(
    name: NodeName, node: Callable[[DiagnosticState], DiagnosticUpdate]
) -> Callable[[DiagnosticState], DiagnosticUpdate]:
    def run(state: DiagnosticState) -> DiagnosticUpdate:
        started = perf_counter()
        updates = node(state)
        previous = state.get("steps", ())
        step = NodeTrace(
            node=name,
            latency_ms=(perf_counter() - started) * 1000,
            retry_count=state.get("retry_count", 0),
            top_k=updates.get("top_k"),
        )
        log_event(
            _LOGGER,
            "agent.transition",
            from_node=previous[-1].node if previous else "START",
            to_node=name,
            latency_ms=step.latency_ms,
            retry_count=step.retry_count,
            top_k=step.top_k,
        )
        updates["steps"] = (*previous, step)
        return updates

    return run


def _next_node(state: DiagnosticState) -> Literal["finish", "retrieve"]:
    return "finish" if "result" in state else "retrieve"


def _tool_issues(findings: tuple[ToolFinding, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for finding in findings:
        if finding.status in ("unknown", "ambiguous"):
            if isinstance(finding, FaultCodeFinding):
                missing.append(
                    f"Equipment and a documented mapping for fault code {finding.fault_code}"
                )
            else:
                missing.append(f"Equipment and a documented threshold for sensor {finding.sensor}")
    return tuple(missing)


def _recommend_escalation(findings: tuple[ToolFinding, ...]) -> str | None:
    recommendations: list[str] = []
    for finding in findings:
        if isinstance(finding, FaultCodeFinding) and finding.status == "found" and finding.rule:
            recommendations.append(finding.rule.escalation.rstrip(".") + ".")
        elif isinstance(finding, SensorFinding) and finding.status == "critical" and finding.rule:
            recommendations.append(
                f"Escalate the critical {finding.sensor} reading on {finding.rule.equipment_id} "
                "to qualified maintenance."
            )
    return " ".join(dict.fromkeys(recommendations)) or None


class DiagnosticWorkflow:
    """Run the typed diagnosis graph with injected retrieval, tools, and model policy."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        tools: DiagnosticTools,
        policy: DiagnosticPolicy,
        *,
        initial_top_k: int = 5,
    ) -> None:
        if initial_top_k < 1:
            raise ValueError("initial_top_k must be positive")
        self._retriever = retriever
        self._tools = tools
        self._policy = policy
        self._initial_top_k = initial_top_k

        graph = StateGraph(DiagnosticState, input_schema=DiagnosticInput)
        graph.add_node("intent_router", _observe("intent_router", self._route))
        graph.add_node("retriever", _observe("retriever", self._retrieve))
        graph.add_node("diagnostic_tool", _observe("diagnostic_tool", self._run_tools))
        graph.add_node("synthesizer", _observe("synthesizer", self._synthesize))
        graph.add_node("critic", _observe("critic", self._critique))
        graph.add_edge(START, "intent_router")
        graph.add_conditional_edges(
            "intent_router", _next_node, {"finish": END, "retrieve": "retriever"}
        )
        graph.add_edge("retriever", "diagnostic_tool")
        graph.add_edge("diagnostic_tool", "synthesizer")
        graph.add_edge("synthesizer", "critic")
        graph.add_conditional_edges("critic", _next_node, {"finish": END, "retrieve": "retriever"})
        self._graph = graph.compile()

    def invoke(self, request: DiagnosticRequest, *, include_trace: bool = True) -> DiagnosticResult:
        offline_policy = isinstance(self._policy, MockDiagnosticPolicy) or (
            isinstance(self._policy, LLMDiagnosticPolicy) and self._policy.uses_mock_client
        )
        trace_context = tracing_context(enabled=False) if offline_policy else nullcontext()
        with trace_context:
            state = self._graph.invoke({"request": request})
        result: DiagnosticResult = state["result"]
        steps: tuple[NodeTrace, ...] = state["steps"]
        retry_count = state.get("retry_count", 0)
        log_event(
            _LOGGER,
            "agent.transition",
            from_node=steps[-1].node,
            to_node="END",
            retry_count=retry_count,
            status=result.status,
        )
        if not include_trace:
            return result

        transitions = [
            StateTransition(from_node="START", to_node=steps[0].node),
            *(
                StateTransition(from_node=previous.node, to_node=current.node)
                for previous, current in zip(steps[:-1], steps[1:], strict=True)
            ),
            StateTransition(from_node=steps[-1].node, to_node="END"),
        ]
        return result.model_copy(
            update={
                "trace": DiagnosticTrace(
                    nodes=steps, transitions=tuple(transitions), retry_count=retry_count
                )
            }
        )

    def _route(self, state: DiagnosticState) -> DiagnosticUpdate:
        try:
            intent = self._policy.route(state["request"])
        except ModelClientError:
            return {"result": _abstention(None, ("A validated diagnostic intent",))}
        if intent == "out_of_scope":
            return {
                "intent": intent,
                "result": DiagnosticResult(
                    status="out_of_scope",
                    intent=intent,
                    answer="I can help with requests within the configured maintenance domain.",
                ),
            }
        return {"intent": intent}

    def _retrieve(self, state: DiagnosticState) -> DiagnosticUpdate:
        request = state["request"]
        top_k = self._initial_top_k * (1 + state.get("retry_count", 0))
        feedback = state.get("feedback")
        query = resolve_retrieval_query(request.query, feedback.refined_query if feedback else None)
        details = (request.equipment_id, request.fault_code, *request.sensor_readings)
        components = [
            detail for detail in details if detail and detail.casefold() not in query.casefold()
        ]
        query = " ".join([query, *components])
        try:
            hits = self._retriever.search(query, top_k=top_k, rerank=True)
        except ModelClientError:
            return {
                "chunks": (),
                "top_k": top_k,
                "evidence_issue": "A successful retrieval of relevant maintenance documentation",
            }
        return {
            "chunks": tuple(hit.chunk for hit in hits),
            "top_k": top_k,
            "evidence_issue": None if hits else _MISSING_EVIDENCE,
        }

    def _run_tools(self, state: DiagnosticState) -> DiagnosticUpdate:
        return {"tool_findings": self._tools.evaluate(state["request"])}

    def _synthesize(self, state: DiagnosticState) -> DiagnosticUpdate:
        context = EvidenceContext(
            request=state["request"],
            intent=state["intent"],
            equipment_id=self._tools.equipment_for(state["request"]),
            chunks=state["chunks"],
            tool_findings=state["tool_findings"],
        )
        issue = state.get("evidence_issue")
        answer = ""
        citations: tuple[Citation, ...] = ()
        if issue is None and not _tool_issues(context.tool_findings):
            try:
                answer, citations = bind_citations(self._policy.synthesize(context), context.chunks)
            except CitationError as exc:
                issue = str(exc)
            except ModelClientError:
                issue = "A validated diagnosis from the synthesis model"
        return {
            "context": context,
            "answer": answer,
            "citations": citations,
            "evidence_issue": issue,
        }

    def _critique(self, state: DiagnosticState) -> DiagnosticUpdate:
        context = state["context"]
        tool_issues = _tool_issues(context.tool_findings)
        if tool_issues:
            return {"result": _abstention(context.intent, tool_issues, context.tool_findings)}
        issue = state.get("evidence_issue")
        if issue:
            assessment = CriticAssessment(sufficient=False, missing_aspects=(issue,))
        else:
            try:
                assessment = self._policy.critique(context, state["answer"], state["citations"])
            except ModelClientError:
                assessment = CriticAssessment(
                    sufficient=False,
                    missing_aspects=("An independent review of the cited diagnosis",),
                )

        if assessment.sufficient and not assessment.missing_aspects and state["citations"]:
            return {
                "result": DiagnosticResult(
                    status="answered",
                    intent=context.intent,
                    answer=state["answer"],
                    citations=state["citations"],
                    tool_findings=context.tool_findings,
                    escalation_recommendation=_recommend_escalation(context.tool_findings),
                )
            }

        missing = assessment.missing_aspects or (_MISSING_EVIDENCE,)
        retry = state.get("retry_count", 0)
        if retry >= _MAX_RETRIES:
            return {"result": _abstention(context.intent, missing, context.tool_findings)}
        refinement = assessment.refined_query or f"{context.request.query} maintenance manual"
        return {
            "retry_count": retry + 1,
            "feedback": CriticAssessment(
                sufficient=False, missing_aspects=missing, refined_query=refinement
            ),
        }


def create_diagnostic_workflow(
    settings: Settings,
    retriever: EvidenceRetriever,
    *,
    llm_client: BaseLLMClient | None = None,
    tools: DiagnosticTools | None = None,
    initial_top_k: int = 5,
) -> DiagnosticWorkflow:
    policy: DiagnosticPolicy
    if llm_client is not None and (llm_client.provider, llm_client.model) != (
        settings.llm.provider,
        settings.llm.model,
    ):
        raise ConfigurationError("LLM client provider and model must match the diagnostic profile")
    if settings.llm.provider == "mock":
        policy = MockDiagnosticPolicy(settings.domain)
    elif llm_client is not None:
        policy = LLMDiagnosticPolicy(llm_client, settings.domain)
    else:
        raise ConfigurationError("The live diagnostic workflow requires an injected LLM client")
    return DiagnosticWorkflow(
        retriever,
        tools if tools is not None else DiagnosticTools.from_corpus(settings.paths.corpus_dir),
        policy,
        initial_top_k=initial_top_k,
    )
