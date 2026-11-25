from contextlib import AbstractContextManager
from types import TracebackType


class ManagedClient(AbstractContextManager):
    """Allow callers to close resources consistently across providers."""

    def close(self) -> None:
        """Release resources owned by this client."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
