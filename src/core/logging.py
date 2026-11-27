import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TextIO

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "fields": getattr(record, "event_fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """Set up one JSON handler for loggers under the project's namespace."""
    logger = logging.getLogger("diagagent")
    logger.setLevel(level)
    logger.propagate = False
    for installed in list(logger.handlers):
        if installed.name == "diagagent-json":
            logger.removeHandler(installed)
            installed.close()

    handler = logging.StreamHandler(stream)
    handler.set_name("diagagent-json")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: object
) -> None:
    request_id = request_id_context.get()
    if request_id is not None:
        fields["request_id"] = request_id
    logger.log(level, event, extra={"event_fields": fields})
