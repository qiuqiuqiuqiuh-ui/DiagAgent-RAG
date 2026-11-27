from pathlib import Path

from src.evaluation.runner import EvaluationReport

_METRICS = (
    ("Recall@K", "recall_at_k"),
    ("MRR@K", "mrr_at_k"),
    ("NDCG@K", "ndcg_at_k"),
    ("Citation coverage", "citation_coverage"),
    ("Citation validity", "citation_validity"),
    ("Routing accuracy", "routing_accuracy"),
    ("Tool accuracy", "tool_accuracy"),
    ("Abstention accuracy", "abstention_accuracy"),
    ("Status accuracy", "status_accuracy"),
    ("P50 latency (ms)", "latency_p50_ms"),
    ("P95 latency (ms)", "latency_p95_ms"),
    ("Judge faithfulness", "faithfulness"),
)


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# {report.run_mode.title()} retrieval ablation",
        "",
        f"Dataset `{report.dataset}` · "
        f"{report.modes[0].metrics.samples} samples · K={report.top_k}",
        f"\nIndex `{report.index_version}` · {report.created_at.isoformat()}",
        "",
        *(f"- {name.title()} · `{provider}`" for name, provider in report.providers.items()),
        "",
        "| Metric | " + " | ".join(mode.mode for mode in report.modes) + " |",
        "|---|" + "---:|" * len(report.modes),
    ]
    attempts = [mode.metrics.retrieval_attempted_samples for mode in report.modes]
    if any(count is not None for count in attempts):
        lines.append(
            "| Retrieval attempts (gold cases) | "
            + " | ".join("—" if count is None else str(count) for count in attempts)
            + " |"
        )
    for label, name in _METRICS:
        values = (getattr(mode.metrics, name) for mode in report.modes)
        lines.append(
            f"| {label} | "
            + " | ".join("—" if value is None else f"{value:.4f}" for value in values)
            + " |"
        )
    judge_samples = [mode.metrics.judge_samples for mode in report.modes]
    if any(judge_samples) or "judge" in report.providers:
        lines.append("| Judge samples | " + " | ".join(map(str, judge_samples)) + " |")
    lines.extend(
        [
            "",
            "Ranking metrics use the first retrieval pass and cases with gold evidence.",
            "Citation coverage counts annotated chunks cited across answerable cases.",
            "Citation validity compares inline references and metadata to final retrieval.",
            "Tool accuracy uses annotated cases. Abstention accuracy checks the binary decision.",
            "Latency covers the workflow and excludes the optional judge.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: EvaluationReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"{stamp}_{report.run_mode}_{Path(report.dataset).stem}"
    json_path = directory / f"{prefix}.json"
    markdown_path = directory / f"{prefix}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
