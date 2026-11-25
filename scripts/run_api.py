import argparse
from pathlib import Path

import uvicorn

from src.api.app import create_app
from src.core.logging import configure_logging
from src.core.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the diagnostic FastAPI service")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = load_settings(args.config, project_root=args.project_root)
    configure_logging(settings.log_level)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
