import json
import logging
import socket
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from src.agents.citations import CitationError, bind_citations, validate_citations
from src.agents.policy import EvidenceContext, LLMDiagnosticPolicy, MockDiagnosticPolicy
from src.agents.schemas import DiagnosticRequest, DiagnosticResult, FaultCodeFinding, SensorFinding
from src.agents.tools import DiagnosticTools
from src.agents.workflow import DiagnosticWorkflow, create_diagnostic_workflow
from src.core.embedding import create_embedding_client
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.llm import ChatMessage, MockLLMClient
from src.core.reranker import create_reranker
from src.core.settings import DomainSettings, IngestionSettings, Settings, load_settings
from src.ingestion.chunking import chunk_sections
from src.ingestion.pipeline import build_index
from src.ingestion.schemas import Chunk, Section
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.index import load_index
from src.retrieval.schemas import SearchHit

ROOT = Path(__file__).resolve().parents[1]
NODES = ("intent_router", "retriever", "diagnostic_tool", "synthesizer", "critic")
DOMAIN = DomainSettings(name="Industrial Equipment Fault Diagnosis", language="en")


@pytest.fixture
def live_index(tmp_path: Path) -> Iterator[tuple[Settings, HybridRetriever]]:
    settings = load_settings(project_root=ROOT, env_file=None)
    settings.paths.index_dir = tmp_path / "index"
    with create_embedding_client(settings.embedding) as embedder:
        build_index(settings, embedder)
        stored = load_index(settings.paths.index_dir, settings.embedding, settings.ingestion)
        yield (
            settings,
            HybridRetriever(
                stored, embedder, create_reranker(settings.reranker), settings=settings.retrieval
            ),
        )


def test_mock_workflow_answers_and_traces_all_nodes_without_network(
    live_index: tuple[Settings, HybridRetriever],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("The offline workflow tried to connect to the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "mock-cannot-use-network")
    caplog.set_level(logging.INFO, logger="diagagent.agents")
    settings, retriever = live_index
    result = create_diagnostic_workflow(settings, retriever).invoke(
        DiagnosticRequest(
            query="What does VFD-OC3 mean and what should I check?", equipment_id="DRV-300"
        )
    )
    assert result.status == "answered"
    assert result.intent == "fault_diagnosis"
    assert "overcurrent" in result.answer.casefold()
    assert result.citations and all(
        f"[{item.chunk_id}]" in result.answer for item in result.citations
    )
    assert "data/raw" not in result.citations[0].source_path
    assert isinstance(result.tool_findings[0], FaultCodeFinding)
    assert result.tool_findings[0].status == "found"
    assert result.escalation_recommendation == "Electrical maintenance for repeat trips."
    assert result.trace is not None and result.trace.retry_count == 0
    assert tuple(step.node for step in result.trace.nodes) == NODES
    assert all(step.latency_ms >= 0 for step in result.trace.nodes)
    assert [(edge.from_node, edge.to_node) for edge in result.trace.transitions] == list(
        zip(("START", *NODES), (*NODES, "END"), strict=True)
    )
    logged = [
        record.event_fields for record in caplog.records if record.message == "agent.transition"
    ]
    assert [fields["to_node"] for fields in logged] == [*NODES, "END"]


def test_parameter_lookup_returns_asset_specific_sensor_finding(
    live_index: tuple[Settings, HybridRetriever],
) -> None:
    settings, retriever = live_index
    result = create_diagnostic_workflow(settings, retriever).invoke(
        DiagnosticRequest(
            query="What is the winding_temp warning threshold?",
            equipment_id="MX-200",
            sensor_readings={"winding_temp": 118},
        ),
        include_trace=False,
    )
    assert result.status == "answered"
    assert result.intent == "parameter_lookup"
    assert result.trace is None
    assert "105" in result.answer
    assert result.citations[0].section_title == "sensor_thresholds > MX-200 winding_temp"
    (finding,) = result.tool_findings
    assert isinstance(finding, SensorFinding) and finding.status == "warning"


def test_mock_abstains_after_two_retries_for_undocumented_specification(
    live_index: tuple[Settings, HybridRetriever],
) -> None:
    settings, retriever = live_index
    workflow = create_diagnostic_workflow(settings, retriever, initial_top_k=2)
    result = workflow.invoke(
        DiagnosticRequest(query="What is the mounting bolt torque for the MX-200 motor?")
    )
    assert result.status == "insufficient_evidence"
    assert result.intent == "parameter_lookup"
    assert any("torque" in item for item in result.missing_aspects)
    assert result.escalation_recommendation
    assert result.citations == ()
    assert result.trace is not None and result.trace.retry_count == 2
    assert [step.top_k for step in result.trace.nodes if step.node == "retriever"] == [2, 4, 6]
    assert len(result.trace.nodes) == 13


def test_ambiguous_sensor_reading_requires_equipment_identification(
    live_index: tuple[Settings, HybridRetriever],
) -> None:
    settings, retriever = live_index
    result = create_diagnostic_workflow(settings, retriever).invoke(
        DiagnosticRequest(
            query="Check the winding_temp threshold", sensor_readings={"winding_temp": 118}
        )
    )
    assert result.status == "insufficient_evidence"
    assert any(
        "documented threshold for sensor winding_temp" in item for item in result.missing_aspects
    )
    assert result.tool_findings[0].status == "ambiguous"
    assert result.trace is not None and result.trace.retry_count == 0


class RecordingRetriever:
    def __init__(self, chunks: Sequence[Chunk] = ()) -> None:
        self.hits = [SearchHit(chunk=chunk, score=1) for chunk in chunks]
        self.calls: list[tuple[str, int, bool]] = []

    def search(self, query: str, top_k: int = 5, *, rerank: bool = True) -> list[SearchHit]:
        self.calls.append((query, top_k, rerank))
        return self.hits[:top_k]


def test_mock_parameter_lookup_cannot_use_another_assets_threshold() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    chunks = chunk_sections(
        "manual.md",
        [
            Section(title="MX-200", text="The winding_temp measures the motor temperature."),
            Section(
                title="PM-14", text="winding_temp warning_threshold: 100, critical_threshold: 115."
            ),
        ],
        IngestionSettings(chunk_size=150, chunk_overlap=0),
    )
    result = create_diagnostic_workflow(settings, RecordingRetriever(chunks)).invoke(
        DiagnosticRequest(query="What is the winding_temp warning threshold for MX-200?")
    )
    assert result.status == "insufficient_evidence"
    assert result.citations == ()
    assert any("threshold" in missing for missing in result.missing_aspects)


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        chunk_id="vfd.pdf#s0#c0",
        doc_id="vfd.pdf",
        source_path="vfd.pdf",
        section_title="Acceleration Overcurrent",
        page_number=2,
        section_index=0,
        chunk_index=0,
        text="VFD-OC3 indicates an acceleration overcurrent trip. Review the ramp time.",
    )


def test_out_of_scope_short_circuits_before_retrieval() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    retriever = RecordingRetriever()
    result = create_diagnostic_workflow(settings, retriever).invoke(
        DiagnosticRequest(query="Give me the weather forecast for Tokyo.")
    )
    assert result.status == result.intent == "out_of_scope"
    assert retriever.calls == []
    assert result.tool_findings == result.citations == ()
    assert result.trace is not None and result.trace.retry_count == 0
    assert [step.node for step in result.trace.nodes] == ["intent_router"]


def test_critic_refines_query_and_expands_retrieval_before_accepting(chunk: Chunk) -> None:
    draft = {"answer": f"VFD-OC3 is an acceleration overcurrent trip. [{chunk.chunk_id}]"}
    model_outputs = [
        {"intent": "fault_diagnosis"},
        draft,
        {"sufficient": False, "missing_aspects": ["ramp"], "refined_query": "VFD-OC3 ramp current"},
        draft,
        {"sufficient": True, "missing_aspects": []},
    ]
    remaining = iter(model_outputs)
    prompts: list[list[ChatMessage]] = []

    def respond(messages: Sequence[ChatMessage]) -> str:
        prompts.append(list(messages))
        return json.dumps(next(remaining))

    retriever = RecordingRetriever([chunk])
    workflow = DiagnosticWorkflow(
        retriever,
        DiagnosticTools(),
        LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN),
        initial_top_k=3,
    )
    result = workflow.invoke(DiagnosticRequest(query="Diagnose VFD-OC3"))
    assert result.status == "answered" and result.citations[0].page_number == 2
    assert result.trace is not None and result.trace.retry_count == 1
    assert retriever.calls == [("Diagnose VFD-OC3", 3, True), ("VFD-OC3 ramp current", 6, True)]
    assert tuple(step.node for step in result.trace.nodes) == (*NODES, *NODES[1:])
    assert len(prompts) == 5
    assert json.loads(prompts[-1][-1].content)["citations"][0]["chunk_id"] == chunk.chunk_id


def test_untrusted_citations_are_rejected_even_with_valid_citations_present(chunk: Chunk) -> None:
    (citation,) = validate_citations(f"Review ramp [{chunk.chunk_id}] [{chunk.chunk_id}].", [chunk])
    assert citation.source_path == "vfd.pdf" and citation.page_number == 2
    with pytest.raises(CitationError, match="retrieved"):
        validate_citations(f"Review ramp [{chunk.chunk_id}] [invented#s0#c0].", [chunk])
    with pytest.raises(CitationError):
        validate_citations(f"Review ramp [{chunk.chunk_id}] [malformed", [chunk])
    with pytest.raises(CitationError):
        validate_citations("Review ramp without evidence.", [chunk])

    def respond(messages: Sequence[ChatMessage]) -> str:
        instruction = messages[0].content
        if "Classify the request" in instruction:
            return '{"intent":"fault_diagnosis"}'
        assert "Draft a concise" in instruction
        return json.dumps({"answer": f"Spurious claim [{chunk.chunk_id}] [invented#s0#c0]."})

    retriever = RecordingRetriever([chunk])
    result = DiagnosticWorkflow(
        retriever, DiagnosticTools(), LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN)
    ).invoke(DiagnosticRequest(query="Diagnose VFD-OC3"))
    assert result.status == "insufficient_evidence" and result.citations == ()
    assert result.trace is not None and result.trace.retry_count == 2
    assert "Spurious claim" not in result.answer
    assert "retrieved" in result.missing_aspects[0]
    assert len(retriever.calls) == 3


@pytest.mark.parametrize("invalid", ["{", '{"intent":"maintenance_record"}'])
def test_live_router_retries_bad_json_or_intent(chunk: Chunk, invalid: str) -> None:
    outputs = iter(
        (
            invalid,
            '{"intent":"fault_diagnosis"}',
            json.dumps({"answer": f"The trip indicates overcurrent. [{chunk.chunk_id}]"}),
            '{"sufficient":true,"missing_aspects":[]}',
        )
    )
    messages_by_call: list[list[ChatMessage]] = []

    def respond(messages: Sequence[ChatMessage]) -> str:
        messages_by_call.append(list(messages))
        return next(outputs)

    request = DiagnosticRequest(query="What readings should I record after VFD-OC3?")
    retriever = RecordingRetriever([chunk])
    result = DiagnosticWorkflow(
        retriever, DiagnosticTools(), LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN)
    ).invoke(request)
    assert result.status == "answered" and result.intent == "fault_diagnosis"
    assert len(messages_by_call) == 4 and messages_by_call[0] == messages_by_call[1]
    assert retriever.calls == [(request.query, 5, True)]
    assert result.trace is not None and result.trace.retry_count == 0
    assert tuple(step.node for step in result.trace.nodes) == NODES
    router_prompt, router_input = messages_by_call[0]
    assert router_prompt.role == "system" and router_input.role == "user"
    assert json.loads(router_input.content) == request.model_dump(mode="json")
    assert "observations or readings to record" in router_prompt.content
    assert "classifying one reading" in router_prompt.content
    assert "within the configured domain as in scope" in router_prompt.content
    assert "Use out_of_scope only for an unrelated" in router_prompt.content


@pytest.mark.parametrize(
    "draft",
    [
        "The cited manual does not specify the torque. Consult the approved site procedure.",
        "Use the authorized torque listed in the OEM specification.",
        "The authorized terminal screw torque is 12 N m.",
    ],
)
def test_live_critic_rejects_a_draft_without_evidence_of_requested_value(draft: str) -> None:
    (evidence,) = chunk_sections(
        "motor.md",
        [Section(title="MX-200", text="The nameplate and approved site procedure govern service.")],
        IngestionSettings(chunk_size=100, chunk_overlap=0),
    )
    missing = "The authorized MX-200 terminal screw torque"
    answer = f"{draft} [{evidence.chunk_id}]"
    critic_calls = 0

    def respond(messages: Sequence[ChatMessage]) -> str:
        nonlocal critic_calls
        prompt = messages[0].content
        if "Classify the request" in prompt:
            return '{"intent":"parameter_lookup"}'
        if "Draft a concise" in prompt:
            return json.dumps({"answer": answer})
        critic_calls += 1
        assert "must be supported for the requested asset and conditions" in prompt
        assert "only states that the requested information is missing or refers the user" in prompt
        assert "without providing the requested information, set sufficient to false" in prompt
        assert "list the required information in missing_aspects" in prompt
        assert json.loads(messages[1].content)["answer"] == answer
        return json.dumps({"sufficient": False, "missing_aspects": [missing]})

    workflow = DiagnosticWorkflow(
        RecordingRetriever([evidence]),
        DiagnosticTools(),
        LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN),
    )
    request = DiagnosticRequest(
        query="What is the authorized torque in N m for MX-200 terminal screws?",
        equipment_id="MX-200",
    )
    result = workflow.invoke(request)

    assert result.status == "insufficient_evidence" and result.intent == "parameter_lookup"
    assert result.missing_aspects == (missing,) and result.citations == ()
    assert draft not in result.answer
    assert result.trace is not None and result.trace.retry_count == 2
    assert critic_calls == 3


@pytest.mark.parametrize(("provider_error", "attempts"), [(False, 2), (True, 1)])
def test_invalid_model_routing_fails_closed_without_retrieval(
    provider_error: bool, attempts: int
) -> None:
    calls = 0

    def respond(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        assert "Classify the request" in messages[0].content
        calls += 1
        if provider_error:
            raise ModelClientError("provider rejected the credentials", status_code=401)
        return '{"intent":"unknown"}'

    retriever = RecordingRetriever()
    result = DiagnosticWorkflow(
        retriever, DiagnosticTools(), LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN)
    ).invoke(DiagnosticRequest(query="Diagnose VFD-OC3"))
    assert isinstance(result, DiagnosticResult)
    assert result.status == "insufficient_evidence" and result.intent is None
    assert retriever.calls == [] and calls == attempts
    assert result.trace is not None
    assert [step.node for step in result.trace.nodes] == ["intent_router"]


@pytest.mark.parametrize("equipment", [None, "CV-40"])
def test_tool_blocker_skips_model_synthesis_and_retrieval_retries(
    chunk: Chunk, equipment: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BlockNetwork:
        def __call__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Mock LLM must run without network calls")

    monkeypatch.setattr(socket.socket, "connect", BlockNetwork())
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "mock-cannot-use-network")
    calls = []

    def respond(messages: Sequence[ChatMessage]) -> str:
        calls.append(messages[0].content)
        assert "Classify the request" in calls[-1]
        return '```json {"intent":"fault_diagnosis"} ```'

    retriever = RecordingRetriever([chunk])
    workflow = DiagnosticWorkflow(
        retriever,
        DiagnosticTools.from_corpus(ROOT / "data" / "raw"),
        LLMDiagnosticPolicy(MockLLMClient(responder=respond), DOMAIN),
    )
    result = workflow.invoke(
        DiagnosticRequest(
            query="Check winding_temp",
            equipment_id=equipment,
            sensor_readings={"winding_temp": 118},
        )
    )
    assert result.status == "insufficient_evidence"
    assert result.trace is not None and result.trace.retry_count == 0
    assert len(retriever.calls) == len(calls) == 1


def test_workflow_factory_validates_injected_llm_provider_and_model() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    with pytest.raises(ConfigurationError, match="provider and model"):
        create_diagnostic_workflow(
            settings, RecordingRetriever(), llm_client=MockLLMClient("wrong")
        )
    settings.llm.provider = "openai_compatible"
    with pytest.raises(ConfigurationError, match="provider and model"):
        create_diagnostic_workflow(settings, RecordingRetriever(), llm_client=MockLLMClient())


def test_citation_binding_canonicalizes_spacing_groups_and_known_units(chunk: Chunk) -> None:
    second = chunk.model_copy(update={"chunk_id": "vfd.pdf#s0#c1", "chunk_index": 1})
    answer, citations = bind_citations(
        f"Pressure [bar] from [ {chunk.chunk_id} , {second.chunk_id} ].",
        [chunk, second],
    )
    assert answer == f"Pressure (bar) from [{chunk.chunk_id}] [{second.chunk_id}]."
    assert [citation.chunk_id for citation in citations] == [chunk.chunk_id, second.chunk_id]
    with pytest.raises(CitationError):
        bind_citations(f"Pressure [{chunk.chunk_id}, invented].", [chunk])


def test_mock_policy_matches_sensor_alias_and_numeric_asset_label() -> None:
    chunks = chunk_sections(
        "rules.md",
        [Section(title="1001 motor", text="The temperature warning threshold is 105 degC.")],
        IngestionSettings(chunk_size=100, chunk_overlap=0),
    )
    request = DiagnosticRequest(
        query="What is the temperature threshold?",
        equipment_id="1001",
        sensor_readings={"temp": 108},
    )
    context = EvidenceContext(
        request=request,
        intent="parameter_lookup",
        equipment_id="1001",
        chunks=chunks,
        tool_findings=(),
    )
    policy = MockDiagnosticPolicy(DOMAIN)
    answer = policy.synthesize(context)
    assert policy.critique(context, answer, validate_citations(answer, chunks)).sufficient
