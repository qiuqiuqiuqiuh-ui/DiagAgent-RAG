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
        ([{"index": 0, "embedding": [0, 1, 2]}], r"shape \(1, 2\), got \(1, 3\)"),
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
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
            r"contains no text",
        ),
        (
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": " "}}]}),
            r"blank text",
        ),
    ],
)
def test_rejects_failed_chat_responses(
    respond: Callable[[httpx.Request], httpx.Response], error: str
) -> None:
    settings = LLMSettings(
        provider="openai_compatible", model="chat-model", api_key=KEY, max_retries=0
    )
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


def test_clean_bearer_query_and_generation_options() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.path == "/v1/chat/completions"
        assert request.url.query == b"api-version=1"
        assert json.loads(request.content)["temperature"] == 0.0
        assert json.loads(request.content)["max_completion_tokens"] == 16
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 3}},
        )

    settings = LLMSettings(
        provider="openai_compatible",
        model="chat-model",
        api_key=SecretStr(" \ttest-key\r\n"),
        base_url="https://provider.invalid/v1/?api-version=1",
        temperature=0,
        max_completion_tokens=16,
    )
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        create_llm_client(settings, http_client=client) as llm,
    ):
        result = llm.generate([ChatMessage(role="user", content="Query")])
    assert result.usage is not None and result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens is None


def test_manages_only_owned_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    )
    internal = httpx.Client(transport=transport)
    created: list[httpx.Client] = []

    def make_client() -> httpx.Client:
        created.append(internal)
        return internal

    settings = LLMSettings(provider="openai_compatible", model="chat-model", api_key=KEY)
    with httpx.Client(transport=transport) as external:
        with create_llm_client(settings, http_client=external) as injected:
            assert injected.generate([ChatMessage(role="user", content="Query")]).text == "OK"
        assert not external.is_closed

    monkeypatch.setattr("src.core.openai_compatible.httpx.Client", make_client)
    with create_llm_client(settings) as llm:
        for _ in range(2):
            assert llm.generate([ChatMessage(role="user", content="Query")]).text == "OK"
        assert created == [internal]
        assert not internal.is_closed
    assert internal.is_closed
    llm.close()


def test_opt_in_json_mode_and_legacy_token_limit() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["max_tokens"] == 32
        assert "max_completion_tokens" not in payload
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})

    settings = LLMSettings(
        provider="openai_compatible",
        model="gateway-chat",
        api_key=KEY,
        max_tokens=32,
        json_mode=True,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = create_llm_client(settings, http_client=client)
        assert llm.generate([ChatMessage(role="user", content="Return JSON")]).text == '{"ok":true}'


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("max_retries", [1, 2])
def test_bounded_retry_on_transient_status(
    monkeypatch: pytest.MonkeyPatch, status: int, max_retries: int
) -> None:
    backoffs: list[float] = []
    statuses = iter([status, status, 200])
    monkeypatch.setattr("src.core.openai_compatible.sleep", backoffs.append)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"choices": [{"message": {"content": "OK"}}]})

    settings = LLMSettings(
        provider="openai_compatible", model="chat-model", api_key=KEY, max_retries=max_retries
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = create_llm_client(settings, http_client=client)
        messages = [ChatMessage(role="user", content="Query")]
        if max_retries == 1:
            with pytest.raises(ModelClientError, match=f"HTTP {status}") as raised:
                llm.generate(messages)
            assert raised.value.status_code == status
        else:
            assert llm.generate(messages).text == "OK"
    assert backoffs == [0.25, 0.5][:max_retries]


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
def test_sanitized_transport_failure(
    monkeypatch: pytest.MonkeyPatch, failure: type[httpx.TransportError]
) -> None:
    backoffs: list[float] = []
    monkeypatch.setattr("src.core.openai_compatible.sleep", backoffs.append)

    def respond(request: httpx.Request) -> httpx.Response:
        raise failure("sensitive provider details", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        settings = embedding_client_settings(max_retries=1)
        embedding = create_embedding_client(settings, http_client=client)
        with pytest.raises(ModelClientError, match=failure.__name__) as raised:
            embedding.embed(["Query"])
        assert raised.value.status_code is None
        assert "sensitive provider details" not in str(raised.value)
    assert backoffs == [0.25]


def test_embeddings_batch_and_normalize() -> None:
    sent: list[list[str]] = []
    raw = {"first": [3, 4], "second": [0, 0], "third": [3e38, 3e38]}

    def respond(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.content)["input"]
        sent.append(batch)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": raw[text]}
                    for i, text in reversed(list(enumerate(batch)))
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedding = create_embedding_client(
            embedding_client_settings(batch_size=2), http_client=client
        )
        vectors = embedding.embed(["first", "second", "third"])
    assert sent == [["first", "second"], ["third"]]
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[0], [0.6, 0.8])
    np.testing.assert_array_equal(vectors[1], [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 0.0, 1.0], atol=1e-7)


def test_invalid_batch_stops_further_requests() -> None:
    sent: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        (text,) = json.loads(request.content)["input"]
        sent.append(text)
        return httpx.Response(
            200, json={"data": [{"index": int(text == "bad"), "embedding": [1, 0]}]}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedding = create_embedding_client(
            embedding_client_settings(batch_size=1), http_client=client
        )
        with pytest.raises(ModelClientError, match="each input index exactly once"):
            embedding.embed(["first", "bad", "last"])
    assert sent == ["first", "bad"]
