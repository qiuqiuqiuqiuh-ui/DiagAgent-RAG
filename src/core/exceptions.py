class DiagAgentError(Exception):
    """Base error raised by the diagnostic system."""


class ConfigurationError(DiagAgentError):
    """A profile or provider configuration is unusable."""


class ModelClientError(DiagAgentError):
    """A model provider failed or returned an invalid result."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
