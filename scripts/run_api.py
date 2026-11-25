import argparse
import sys
from pathlib import Path

import uvicorn

from src.api.app import create_app
from src.core.exceptions import DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the diagnostic FastAPI service")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        settings = load_settings(args.config, project_root=args.project_root)
        configure_logging(settings.log_level)
        app = create_app(settings)
    except DiagAgentError as exc:
        print(f"API startup failed: {exc}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
