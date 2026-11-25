from collections.abc import Iterable, Sequence
from math import log2
from statistics import fmean

import numpy as np
from pydantic import BaseModel, FiniteFloat

from src.agents.citations import CitationError, validate_citations
from src.agents.schemas import (
    DiagnosisStatus,
    DiagnosticResult,
    FaultCodeFinding,
    Intent,
    ToolFinding,
)
from src.evaluation.dataset import Category, EvalCase
from src.ingestion.schemas import Chunk


class RankingMetrics(BaseModel):
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


class CaseResult(BaseModel):
    id: str
    category: Category
    expected_status: DiagnosisStatus
    status: DiagnosisStatus
    intent: Intent | None
    retrieved_ids: tuple[str, ...]
    cited_ids: tuple[str, ...]
    ranking: RankingMetrics | None
    citation_coverage: float | None
    citation_valid: bool | None
    routing_correct: bool
    tools_correct: bool | None
    abstention_correct: bool
    latency_ms: FiniteFloat
    faithfulness: float | None = None


class MetricSummary(BaseModel):
    samples: int
    retrieval_samples: int
    citation_samples: int
    tool_samples: int
    judge_samples: int
    recall_at_k: float | None
    mrr_at_k: float | None
    ndcg_at_k: float | None
    citation_coverage: float | None
    citation_validity: float | None
    routing_accuracy: float
    tool_accuracy: float | None
    abstention_accuracy: float
    status_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    faithfulness: float | None


def score_ranking(ranked: Sequence[str], gold: set[str], k: int) -> RankingMetrics | None:
    if k < 1:
        raise ValueError("k must be positive")
    if not gold:
        return None
    positions = []
    seen: set[str] = set()
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold and chunk_id not in seen:
            positions.append(rank)
            seen.add(chunk_id)
    dcg = sum(1 / log2(rank + 1) for rank in positions)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return RankingMetrics(
        recall_at_k=len(positions) / len(gold),
        mrr_at_k=1 / positions[0] if positions else 0.0,
        ndcg_at_k=dcg / ideal_dcg,
    )


def _tool_record(finding: ToolFinding) -> tuple[str, str, str, str | None]:
    label = finding.fault_code if isinstance(finding, FaultCodeFinding) else finding.sensor
    asset = finding.rule.equipment_id.casefold() if finding.rule else None
    return finding.kind, label.casefold(), finding.status, asset


def score_case(
    case: EvalCase,
    result: DiagnosticResult,
    first_chunks: Sequence[Chunk],
    final_chunks: Sequence[Chunk],
    *,
    k: int,
    latency_ms: float,
    faithfulness: float | None = None,
) -> CaseResult:
    gold = set(case.relevant_chunks)
    retrieved = tuple(chunk.chunk_id for chunk in first_chunks[:k])
    cited = tuple(item.chunk_id for item in result.citations)
    citation_valid = None
    if result.status == "answered" or cited:
        try:
            citation_valid = validate_citations(result.answer, final_chunks) == result.citations
        except CitationError:
            citation_valid = False
    findings = [_tool_record(finding) for finding in result.tool_findings]
    tools_correct = None
    if case.expected_tools:
        tools_correct = all(
            any(
                (kind, label, status)
                == (expected.kind, expected.target.casefold(), expected.status)
                and (expected.equipment_id is None or asset == expected.equipment_id.casefold())
                for kind, label, status, asset in findings
            )
            for expected in case.expected_tools
        )
    return CaseResult(
        id=case.id,
        category=case.category,
        expected_status=case.expected_status,
        status=result.status,
        intent=result.intent,
        retrieved_ids=retrieved,
        cited_ids=cited,
        ranking=score_ranking(retrieved, gold, k),
        citation_coverage=(len(set(cited) & gold) / len(gold) if citation_valid else 0.0)
        if gold
        else None,
        citation_valid=citation_valid,
        routing_correct=result.intent == case.expected_intent,
        tools_correct=tools_correct,
        abstention_correct=(result.status == "answered") == (case.expected_status == "answered"),
        latency_ms=latency_ms,
        faithfulness=faithfulness,
    )


def _average(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return fmean(observed) if observed else None


def summarize(results: Sequence[CaseResult]) -> MetricSummary:
    if not results:
        raise ValueError("At least one case is required to summarize metrics")
    rankings = [case.ranking for case in results if case.ranking is not None]
    p50, p95 = np.percentile([case.latency_ms for case in results], [50, 95])
    return MetricSummary(
        samples=len(results),
        retrieval_samples=len(rankings),
        citation_samples=sum(case.citation_valid is not None for case in results),
        tool_samples=sum(case.tools_correct is not None for case in results),
        judge_samples=sum(case.faithfulness is not None for case in results),
        recall_at_k=_average(score.recall_at_k for score in rankings),
        mrr_at_k=_average(score.mrr_at_k for score in rankings),
        ndcg_at_k=_average(score.ndcg_at_k for score in rankings),
        citation_coverage=_average(case.citation_coverage for case in results),
        citation_validity=_average(case.citation_valid for case in results),
        routing_accuracy=fmean(case.routing_correct for case in results),
        tool_accuracy=_average(case.tools_correct for case in results),
        abstention_accuracy=fmean(case.abstention_correct for case in results),
        status_accuracy=fmean(case.status == case.expected_status for case in results),
        latency_p50_ms=float(p50),
        latency_p95_ms=float(p95),
        faithfulness=_average(case.faithfulness for case in results),
    )
