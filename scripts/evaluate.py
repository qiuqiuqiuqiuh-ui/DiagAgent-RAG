import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.core.exceptions import DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import load_settings
from src.evaluation.dataset import load_dataset
from src.evaluation.report import write_reports
from src.evaluation.runner import ModeReport, evaluate


def _print_progress(report: ModeReport, elapsed: float) -> None:
    print(f"{report.mode}: {report.metrics.samples} cases in {elapsed:.1f}s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate four diagnostic retrieval variants")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/golden_smoke.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--judge", action="store_true", help="score answered cases with the configured live judge"
    )
    parser.add_argument(
        "--require-heterogeneous-judge",
        action="store_true",
        help="require --judge to use a model different from the answer model",
    )
    parser.add_argument(
        "--save-answers",
        action="store_true",
        help="save each mode's frozen answers, cited chunks and traces in the reports directory",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.require_heterogeneous_judge and not args.judge:
        parser.error("--require-heterogeneous-judge requires --judge")

    root = args.project_root or Path.cwd()
    try:
        settings = load_settings(args.config, project_root=root)
        configure_logging(settings.log_level)
        dataset = load_dataset(root / args.dataset)
        print(f"Evaluating {dataset.name}: {len(dataset.cases)} cases per mode", flush=True)
        report = evaluate(
            settings,
            dataset,
            top_k=args.top_k,
            judge=args.judge,
            require_heterogeneous_judge=args.require_heterogeneous_judge,
            save_answers_dir=settings.paths.reports_dir if args.save_answers else None,
            on_mode_complete=_print_progress,
        )
        json_path, markdown_path = write_reports(report, settings.paths.reports_dir)
    except (DiagAgentError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"{report.run_mode} evaluation completed: "
        f"{len(dataset.cases)} cases x {len(report.modes)} modes"
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if args.save_answers:
        print(f"Frozen answers: {settings.paths.reports_dir} ({len(report.modes)} mode files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
