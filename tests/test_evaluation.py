import ast
import json
from math import log2
from pathlib import Path

import pytest

from src.agents.citations import validate_citations
from src.agents.schemas import DiagnosticResult
from src.agents.tools import DiagnosticTools
from src.core.exceptions import CorpusError
from src.core.llm import MockLLMClient
from src.core.settings import load_settings
from src.evaluation.dataset import EvalCase, load_dataset
from src.evaluation.metrics import score_case, score_ranking, summarize
from src.evaluation.runner import judge_faithfulness
from src.ingestion.corpus import read_corpus
from src.ingestion.schemas import Chunk

ROOT = Path(__file__).resolve().parents[1]


def test_golden_dataset_sizes_labels_and_tool_expectations() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    tools = DiagnosticTools.from_corpus(settings.paths.corpus_dir)
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    smoke = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    bench = load_dataset(ROOT / "data" / "eval" / "golden_bench.jsonl")
    assert len(smoke.cases) == 15
    assert sum(case.expected_status != "answered" for case in smoke.cases) == 3
    assert len(bench.cases) >= 50
    assert (
        0.13
        <= sum(case.expected_status != "answered" for case in bench.cases) / len(bench.cases)
        <= 0.17
    )
    assert {case.category for case in bench.cases} == {
        "single_hop",
        "multi_hop",
        "sensor_threshold",
        "unanswerable",
        "out_of_scope",
    }
    for case in (*smoke.cases, *bench.cases):
        assert set(case.relevant_chunks) <= chunk_ids
        verdicts = score_case(
            case,
            DiagnosticResult(
                status=case.expected_status,
                intent=case.expected_intent,
                answer="fixture",
                tool_findings=tools.evaluate(case.request),
            ),
            (),
            (),
            k=5,
            latency_ms=1.0,
        )
        assert verdicts.tools_correct in (None, True), case.id


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    original = (
        (ROOT / "data" / "eval" / "golden_smoke.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{original}\n{original}\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="duplicate case ID"):
        load_dataset(path)


def test_ranking_metrics_with_duplicate_ranks_and_missing_evidence() -> None:
    score = score_ranking(["x", "a", "a", "b"], {"a", "b", "c"}, 4)
    assert score is not None
    assert score.recall_at_k == pytest.approx(2 / 3)
    assert score.mrr_at_k == 0.5
    assert score.ndcg_at_k == pytest.approx((1 / log2(3) + 1 / log2(5)) / (1 + 1 / log2(3) + 0.5))
    assert score_ranking(["x"], set(), 5) is None
    assert score_ranking(["x"], {"a"}, 1).ndcg_at_k == 0


def test_case_citations_metric_aggregation_and_optional_judge() -> None:
    case = EvalCase(
        id="test-case",
        category="single_hop",
        request={"query": "What caused the alarm?"},
        expected_intent="fault_diagnosis",
        expected_status="answered",
        relevant_chunks=("f.md#s0#c0",),
    )
    chunk = Chunk(
        doc_id="f.md",
        source_path="f.md",
        section_title="Alarm",
        text="An overcurrent alarm",
        section_index=0,
        chunk_index=0,
        chunk_id="f.md#s0#c0",
    )
    answer = "An overcurrent alarm [f.md#s0#c0]."
    result = DiagnosticResult(
        status="answered",
        intent="fault_diagnosis",
        answer=answer,
        citations=validate_citations(answer, [chunk]),
    )
    judge = MockLLMClient(responder=lambda messages: '{"faithfulness":0.75}')
    faithfulness = judge_faithfulness(judge, case.request.query, result, [chunk])
    assert faithfulness == 0.75
    supported = score_case(
        case, result, [chunk], [chunk], k=5, latency_ms=2, faithfulness=faithfulness
    )
    mismatched = score_case(case, result, [chunk], [], k=5, latency_ms=8)
    report = summarize([supported, mismatched])
    assert report.recall_at_k == 1.0
    assert report.citation_coverage == report.citation_validity == 0.5
    assert report.latency_p50_ms == 5 and report.latency_p95_ms == pytest.approx(7.7)
    assert report.faithfulness == 0.75 and report.judge_samples == 1


def test_colab_notebook_matches_script_and_has_clean_cells() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "colab_gpu_pipeline.ipynb").read_text(encoding="utf-8")
    )
    script = (ROOT / "notebooks" / "colab_gpu_pipeline.py").read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU"
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 5 and all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    cells_code = "\n\n".join("".join(cell["source"]) for cell in cells)
    assert ast.dump(ast.parse(cells_code)) == ast.dump(ast.parse(script))
