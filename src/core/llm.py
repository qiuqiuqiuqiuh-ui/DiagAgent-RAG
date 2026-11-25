import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from src.core.client import ManagedClient
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.logging import log_event
from src.core.openai_compatible import OpenAICompatibleTransport
from src.core.settings import LLMSettings

_LOGGER = logging.getLogger("diagagent.core.llm")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chat message content must not be blank")
        return value


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(BaseModel):
    text: str
    usage: TokenUsage | None = None


class BaseLLMClient(ManagedClient, ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identity used when validating a client's configuration."""

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        if not messages:
            raise ValueError("At least one chat message is required")
        started = perf_counter()
        result = self._generate(messages)
        if not result.text.strip():
            raise ModelClientError("The LLM returned blank text")
        log_event(
            _LOGGER,
            "llm.generate",
            model=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            token_usage=result.usage.model_dump() if result.usage is not None else None,
        )
        return result

    @abstractmethod
    def _generate(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Generate text from the supplied messages."""


class MockLLMClient(BaseLLMClient):
    provider = "mock"

    def __init__(
        self,
        model: str = "mock-llm-v1",
        *,
        responder: Callable[[Sequence[ChatMessage]], str] | None = None,
    ) -> None:
        super().__init__(model)
        self._responder = responder

    def _generate(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        if self._responder is not None:
            return LLMResponse(text=self._responder(messages))

        content = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            messages[-1].content,
        )
        return LLMResponse(text=f"Mock response for: {content.strip()}")


class _ChoiceMessage(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _ChoiceMessage


class _ChatCompletion(BaseModel):
    choices: list[_Choice] = Field(min_length=1)
    usage: TokenUsage | None = None


class OpenAICompatibleLLMClient(BaseLLMClient):
    provider = "openai_compatible"

    def __init__(self, settings: LLMSettings, *, http_client: httpx.Client | None = None) -> None:
        super().__init__(settings.model)
        self._generation_parameters = settings.model_dump(
            include={"temperature", "max_completion_tokens"}, exclude_none=True
        )
        self._transport = OpenAICompatibleTransport(
            str(settings.base_url),
            settings.api_key,
            settings.timeout_seconds,
            max_retries=settings.max_retries,
            client=http_client,
        )

    def close(self) -> None:
        self._transport.close()

    def _generate(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        result = self._transport.post(
            "chat/completions",
            {
                "model": self.model,
                "messages": [message.model_dump() for message in messages],
                **self._generation_parameters,
            },
            _ChatCompletion,
        )
        content = result.choices[0].message.content
        if content is None:
            raise ModelClientError("The LLM response contains no text")
        return LLMResponse(text=content, usage=result.usage)


def create_llm_client(
    settings: LLMSettings, *, http_client: httpx.Client | None = None
) -> BaseLLMClient:
    if settings.provider == "mock":
        return MockLLMClient(settings.model)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleLLMClient(settings, http_client=http_client)
    raise ConfigurationError("Unsupported LLM provider")
