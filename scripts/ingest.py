import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.core.embedding import create_embedding_client
from src.core.exceptions import DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import load_settings
from src.ingestion.pipeline import build_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a searchable index from the configured corpus"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config, project_root=args.project_root)
        configure_logging(settings.log_level)
        with create_embedding_client(settings.embedding) as embedder:
            manifest = build_index(settings, embedder)
    except DiagAgentError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
