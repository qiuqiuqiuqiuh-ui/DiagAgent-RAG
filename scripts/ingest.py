import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.core.embedding import create_embedding_client
from src.core.exceptions import DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import load_settings
from src.ingestion.pipeline import build_index
from src.retrieval.index import prune_generations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a searchable index from the configured corpus"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--prune-generations",
        type=int,
        metavar="N",
        help="after ingestion, keep the active index and N recent historical generations",
    )
    args = parser.parse_args(argv)
    if args.prune_generations is not None and args.prune_generations < 0:
        parser.error("--prune-generations must be nonnegative")

    try:
        settings = load_settings(args.config, project_root=args.project_root)
        configure_logging(settings.log_level)
        with create_embedding_client(settings.embedding) as embedder:
            manifest = build_index(settings, embedder)
    except DiagAgentError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(manifest.model_dump_json(indent=2))
    if args.prune_generations is not None:
        try:
            removed = prune_generations(
                settings.paths.index_dir, keep_latest=args.prune_generations
            )
        except DiagAgentError as exc:
            print(f"Index published, but pruning failed: {exc}", file=sys.stderr)
            return 1
        print(f"Pruned {len(removed)} historical index generation(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
