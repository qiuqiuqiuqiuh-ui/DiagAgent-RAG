import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from src.api.app import create_app
from src.core.exceptions import ConfigurationError, DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import load_settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the diagnostic FastAPI service")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--forwarded-allow-ips",
        help="comma-separated trusted proxy IPs for X-Forwarded-For (defaults to Uvicorn default)",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, project_root=args.project_root)
        for name, model in (("LLM", settings.llm), ("EMBEDDING", settings.embedding)):
            if model.provider == "openai_compatible" and (
                model.api_key is None or not model.api_key.get_secret_value().strip()
            ):
                raise ConfigurationError(
                    f"Set DIAGAGENT_{name}__API_KEY for the OpenAI-compatible provider."
                )
        configure_logging(settings.log_level)
        app = create_app(settings)
    except DiagAgentError as exc:
        print(f"API startup failed: {exc}", file=sys.stderr)
        return 1
    run_kwargs: dict[str, object] = {"host": args.host, "port": args.port, "workers": 1}
    if args.forwarded_allow_ips is not None:
        run_kwargs["forwarded_allow_ips"] = args.forwarded_allow_ips
    uvicorn.run(app, **run_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
