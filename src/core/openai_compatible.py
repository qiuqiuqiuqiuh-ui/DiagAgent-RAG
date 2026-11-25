from itertools import count
from time import sleep
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, SecretStr

from src.core.client import ManagedClient
from src.core.exceptions import ConfigurationError, ModelClientError

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAICompatibleTransport(ManagedClient):
    """Shared transport for OpenAI-compatible JSON endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None,
        timeout_seconds: float,
        *,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationError("The OpenAI-compatible provider requires an API key")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self._base_url = urlsplit(base_url)
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = httpx.Client() if client is None else client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def post(
        self,
        endpoint: str,
        payload: dict[str, object],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        path = f"{self._base_url.path.rstrip('/')}/{endpoint}"
        url = urlunsplit(self._base_url._replace(path=path, fragment=""))
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value().strip()}"}
        for attempt in count():
            try:
                response = self._client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                retryable = isinstance(exc, httpx.TransportError) or (
                    status is not None and (status == 429 or 500 <= status < 600)
                )
                if not retryable or attempt >= self._max_retries:
                    detail = f"HTTP {status}" if status is not None else type(exc).__name__
                    raise ModelClientError(
                        f"Request to {endpoint} failed ({detail})", status_code=status
                    ) from None
                sleep(0.25 * 2 ** min(attempt, 4))
            else:
                break

        try:
            return response_model.model_validate_json(response.content)
        except ValueError:
            raise ModelClientError(
                f"Invalid response from {endpoint}", status_code=response.status_code
            ) from None
