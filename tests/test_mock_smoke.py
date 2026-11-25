import json
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from src.core.embedding import MockEmbeddingClient, create_embedding_client
from src.core.llm import ChatMessage, MockLLMClient, create_llm_client
from src.core.logging import configure_logging
from src.core.reranker import MockReranker, create_reranker
from src.core.settings import load_settings
from src.core.tokenizer import tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mock_model_pipeline_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = logging.getLogger("diagagent")
    monkeypatch.setattr(namespace, "handlers", [])
    monkeypatch.setattr(namespace, "level", namespace.level)
    monkeypatch.setattr(namespace, "propagate", namespace.propagate)
    stream = StringIO()
    configure_logging(stream=stream)
    configure_logging(stream=stream)

    settings = load_settings(project_root=PROJECT_ROOT, env_file=None)
    llm = create_llm_client(settings.llm)
    embedding = create_embedding_client(settings.embedding)
    reranker = create_reranker(settings.reranker)
    assert isinstance(llm, MockLLMClient)
    assert isinstance(embedding, MockEmbeddingClient)
    assert isinstance(reranker, MockReranker)

    query = "VFD-OC3 overcurrent"
    passages = ["SKF-6205 bearing lubrication", "Inspect VFD-OC3 for motor overcurrent"]
    vectors = embedding.embed([query, *passages])
    assert vectors.shape == (3, settings.embedding.dimension)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(embedding.embed([query]), vectors[:1])
    assert vectors[0] @ vectors[2] > vectors[0] @ vectors[1]
    assert reranker.score(query, passages) == [0.0, 1.0]
    response = llm.generate([ChatMessage(role="user", content=query)])
    assert response.text == f"Mock response for: {query}"
    assert response.usage is None

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        "embedding.embed",
        "embedding.embed",
        "reranker.score",
        "llm.generate",
    ]
    assert all(event["fields"]["latency_ms"] >= 0 for event in events)
    assert events[2]["fields"]["scores"] == [0.0, 1.0]
    assert events[3]["fields"]["token_usage"] is None
    assert query not in stream.getvalue()


def test_mock_boundary_cases() -> None:
    assert tokenize("(ERR-402), SKF-6205; VFD-OC3!") == ["err-402", "skf-6205", "vfd-oc3"]
    embedding = MockEmbeddingClient(dimension=8)
    assert embedding.embed([]).shape == (0, 8)
    np.testing.assert_array_equal(embedding.embed(["..."]), np.zeros((1, 8), dtype=np.float32))
    reranker = MockReranker()
    assert reranker.score("ERR-402", []) == []
    assert reranker.score("", ["any text"]) == [0.0]
    with pytest.raises(ValueError, match="At least one"):
        MockLLMClient().generate([])
