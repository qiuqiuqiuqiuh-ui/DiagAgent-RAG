from contextlib import nullcontext
from typing import TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from src.core.exceptions import ConfigurationError, ModelClientError

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAICompatibleTransport:
    """Shared transport for OpenAI-compatible JSON endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None,
        timeout_seconds: float,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationError("The OpenAI-compatible provider requires an API key")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    def post(
        self,
        endpoint: str,
        payload: dict[str, object],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        # The caller owns an injected client. An uninjected client is request-scoped.
        client_context = nullcontext(self._client) if self._client is not None else httpx.Client()
        try:
            with client_context as client:
                response = client.post(
                    f"{self._base_url}/{endpoint}",
                    headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            status = f" (HTTP {exc.response.status_code})" if isinstance(
                exc, httpx.HTTPStatusError
            ) else ""
            raise ModelClientError(f"Request to {endpoint} failed{status}") from None

        try:
            return response_model.model_validate_json(response.content)
        except ValueError:
            raise ModelClientError(f"Invalid response from {endpoint}") from None
