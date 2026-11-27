import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.core.embedding import BaseEmbeddingClient, create_embedding_client
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.reranker import create_reranker
from src.core.settings import EmbeddingSettings, RerankerSettings

EMBEDDING = EmbeddingSettings(
    provider="sentence_transformers",
    model="test-embedding",
    dimension=2,
    batch_size=8,
    device="cpu",
)
RERANKER = RerankerSettings(provider="sentence_transformers", model="test-reranker", device="cpu")


def test_embeddings_normalize_to_contiguous_float32() -> None:
    class FortranEmbedding(BaseEmbeddingClient):
        provider = "test"

        def _embed(self, texts):
            return np.asfortranarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    result = FortranEmbedding("test", 2).embed(["one", "two"])
    assert result.flags.c_contiguous and result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 0.0]])


def test_optional_provider_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigurationError, match="Install the huggingface extra"):
        create_embedding_client(EMBEDDING).embed(["motor"])
    with pytest.raises(ConfigurationError, match="Install the huggingface extra"):
        create_reranker(RERANKER).score("motor", ["bearing"])


def test_adapts_huggingface_results_and_loads_each_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encode = Mock(return_value=np.array([[3.0, 4.0]], dtype=np.float32))
    predict = Mock(return_value=np.array([[0.2], [0.5]]))
    sentence_transformer = Mock(return_value=SimpleNamespace(encode=encode))
    cross_encoder = Mock(return_value=SimpleNamespace(predict=predict))
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = sentence_transformer
    module.CrossEncoder = cross_encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedding = create_embedding_client(EMBEDDING)
    reranker = create_reranker(RERANKER)
    sentence_transformer.assert_not_called()
    cross_encoder.assert_not_called()
    for _ in range(2):
        np.testing.assert_allclose(embedding.embed(["motor"]), [[0.6, 0.8]])
        assert reranker.score("motor", ["bearing", "motor bearing"]) == [0.2, 0.5]
    sentence_transformer.assert_called_once_with(
        "test-embedding", device="cpu", trust_remote_code=False
    )
    cross_encoder.assert_called_once_with("test-reranker", device="cpu", trust_remote_code=False)
    encode.assert_called_with(
        ["motor"], batch_size=8, convert_to_numpy=True, show_progress_bar=False
    )
    predict.assert_called_with(
        [("motor", "bearing"), ("motor", "motor bearing")],
        convert_to_numpy=True,
        show_progress_bar=False,
    )


@pytest.mark.parametrize("role", ["embedding", "reranker"])
def test_huggingface_inference_serializes_access_to_each_model(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    active = 0
    maximum = 0
    guard = Lock()
    workers = 6
    barrier = Barrier(workers)

    def infer(*args, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(active, maximum)
        sleep(0.01)
        with guard:
            active -= 1
        return [[3.0, 4.0]] if role == "embedding" else [0.5]

    factory = Mock(return_value=SimpleNamespace(encode=infer, predict=infer))
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embedding = create_embedding_client(EMBEDDING)
    reranker = create_reranker(RERANKER)

    def run(_: int):
        barrier.wait(timeout=5)
        if role == "embedding":
            np.testing.assert_allclose(embedding.embed(["motor"]), [[0.6, 0.8]])
        else:
            assert reranker.score("motor", ["bearing"]) == [0.5]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(workers)))
    assert maximum == 1
    factory.assert_called_once()


@pytest.mark.parametrize("failure", [OSError, RuntimeError, ValueError])
def test_huggingface_inference_errors(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    infer = Mock(side_effect=failure("failed"))
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = Mock(return_value=SimpleNamespace(encode=infer))
    module.CrossEncoder = Mock(return_value=SimpleNamespace(predict=infer))
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(ModelClientError, match="embedding failed"):
        create_embedding_client(EMBEDDING).embed(["motor"])
    with pytest.raises(ModelClientError, match="scoring failed"):
        create_reranker(RERANKER).score("motor", ["bearing"])
