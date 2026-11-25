import json
from collections.abc import Callable

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from src.core.embedding import create_embedding_client
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.llm import ChatMessage, create_llm_client
from src.core.settings import EmbeddingSettings, LLMSettings

KEY = SecretStr("test-key")
URL = "https://provider.invalid/compatible/v1"


def test_openai_compatible_chat() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compatible/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "chat-model",
            "messages": [{"role": "user", "content": "Explain ERR-402"}],
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Grounded text"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    settings = LLMSettings(
        provider="openai_compatible", model="chat-model", base_url=URL, api_key=KEY
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = create_llm_client(settings, http_client=client)
        result = llm.generate([ChatMessage(role="user", content="Explain ERR-402")])
        assert result.text == "Grounded text"
        assert result.usage is not None and result.usage.total_tokens == 7
        assert not client.is_closed


def embedding_client_settings(**overrides: object) -> EmbeddingSettings:
    values = dict(provider="openai_compatible", model="embedding-model", dimension=2, api_key=KEY)
    return EmbeddingSettings.model_validate({**values, "base_url": URL, **overrides})


def test_openai_compatible_embedding_restores_input_order() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compatible/v1/embeddings"
        assert json.loads(request.content) == {
            "model": "embedding-model",
            "input": ["first", "second"],
            "dimensions": 2,
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        settings = embedding_client_settings(request_dimensions=True)
        embedding = create_embedding_client(settings, http_client=client)
        result = embedding.embed(["first", "second"])
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.eye(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([{"index": 1, "embedding": [0, 1]}], "cover each input index exactly once"),
        ([{"index": 0, "embedding": [0, 1, 2]}], "Expected finite embeddings"),
        ([{"index": 0, "embedding": [1e40, 1]}], "Expected finite embeddings"),
    ],
)
def test_rejects_invalid_embeddings(data: list[dict[str, object]], error: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert "dimensions" not in json.loads(request.content)
        return httpx.Response(200, json={"data": data})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedding = create_embedding_client(embedding_client_settings(), http_client=client)
        with pytest.raises(ModelClientError, match=error):
            embedding.embed(["first"])


@pytest.mark.parametrize(
    ("respond", "error"),
    [
        (lambda request: httpx.Response(429, text="private details"), r"HTTP 429"),
        (lambda request: httpx.Response(200, json={"choices": []}), r"Invalid response"),
        (
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": " "}}]}),
            r"blank text",
        ),
    ],
)
def test_rejects_failed_chat_responses(
    respond: Callable[[httpx.Request], httpx.Response], error: str
) -> None:
    settings = LLMSettings(provider="openai_compatible", model="chat-model", api_key=KEY)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = create_llm_client(settings, http_client=client)
        with pytest.raises(ModelClientError, match=error) as raised:
            llm.generate([ChatMessage(role="user", content="Explain ERR-402")])
    assert "private details" not in str(raised.value)


def test_live_providers_require_credentials() -> None:
    with pytest.raises(ConfigurationError, match="requires an API key"):
        create_llm_client(LLMSettings(provider="openai_compatible", model="chat-model"))
    with pytest.raises(ConfigurationError, match="requires an API key"):
        create_embedding_client(embedding_client_settings(api_key=SecretStr(" ")))
