import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from src.core.exceptions import DiagAgentError
from src.core.settings import load_settings
from src.ingestion.preflight import DEFAULT_MAX_FILE_BYTES, validate_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the corpus and estimate its index size")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus-dir", type=Path, help="override the configured corpus directory")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args(argv)
    if args.max_file_bytes < 1:
        parser.error("--max-file-bytes must be positive")

    try:
        settings = load_settings(args.config, project_root=args.project_root)
        corpus = (
            settings.paths.corpus_dir
            if args.corpus_dir is None
            else args.project_root / args.corpus_dir
        )
        report = validate_corpus(
            corpus, settings.ingestion, settings.rules, max_file_bytes=args.max_file_bytes
        )
    except (DiagAgentError, OSError, RuntimeError) as exc:
        print(f"Corpus preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS" if report.passed else "FAIL", **asdict(report)}, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
