from pathlib import Path

import pytest

from src.core.embedding import create_embedding_client
from src.core.exceptions import IndexValidationError
from src.core.reranker import create_reranker
from src.core.settings import Settings, load_settings
from src.ingestion.corpus import read_corpus
from src.ingestion.pipeline import build_index
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.index import IndexManifest, load_index
from src.retrieval.query import resolve_retrieval_query
from src.retrieval.tokenizer import tokenize_bm25

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def indexed_corpus(tmp_path: Path) -> tuple[Settings, IndexManifest]:
    settings = load_settings(project_root=PROJECT_ROOT, env_file=None)
    settings.paths.index_dir = tmp_path / "索引"
    with create_embedding_client(settings.embedding) as embedder:
        manifest = build_index(settings, embedder)
    return settings, manifest


def test_equipment_codes_and_lexical_components() -> None:
    text = "ERR-402, SKF-6205, VFD-OC3, E01.2, 6205-2RS/C3"
    assert tokenize_bm25(text) == ["err-402", "skf-6205", "vfd-oc3", "e01.2", "6205-2rs/c3"]
    assert tokenize_bm25(text, include_components=True) == [
        "err-402",
        "err",
        "402",
        "skf-6205",
        "skf",
        "6205",
        "vfd-oc3",
        "vfd",
        "oc3",
        "e01.2",
        "e01",
        "2",
        "6205-2rs/c3",
        "6205",
        "2rs",
        "c3",
    ]


def test_rrf_ranks_by_agreement_and_handles_duplicates() -> None:
    ranked = reciprocal_rank_fusion(
        [["dense", "dense", "both"], ["both", "sparse"]], rank_constant=10
    )
    assert [chunk_id for chunk_id, _ in ranked] == ["both", "dense", "sparse"]
    assert [score for _, score in ranked] == pytest.approx([1 / 12 + 1 / 11, 1 / 11, 1 / 12])
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([["first"], ["second"]])[0][0] == "first"
    assert resolve_retrieval_query(" motor ", " OC3 ramp ") == "OC3 ramp"
    assert resolve_retrieval_query(" motor ", "  ") == "motor"


def test_reload_manifest_and_all_retrieval_modes(
    indexed_corpus: tuple[Settings, IndexManifest],
) -> None:
    settings, manifest = indexed_corpus
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    assert corpus.document_count == 7
    index = load_index(
        settings.paths.index_dir,
        settings.embedding,
        settings.ingestion,
        expected_corpus_hash=corpus.corpus_hash,
    )
    assert index.manifest == manifest
    assert index.chunks == corpus.chunks
    assert index.vectors.d == manifest.embedding_dim
    assert index.vectors.ntotal == manifest.chunk_count
    assert manifest.created_at.tzinfo is not None
    with create_embedding_client(settings.embedding) as embedder:
        retriever = HybridRetriever(
            index, embedder, create_reranker(settings.reranker), settings=settings.retrieval
        )
        dense = retriever.dense.search("VFD-OC3 acceleration overcurrent", 3)
        assert len(dense) == 3
        assert dense[0].dense_score == dense[0].score
        assert "vfd-oc3" in dense[0].chunk.retrieval_text.lower()
        assert retriever.bm25.search("doesnotexist1234") == []
        assert retriever.bm25.search("OC3", 1)[0].bm25_score is not None
        assert "vfd-oc3" in retriever.bm25.search("OC3", 1)[0].chunk.retrieval_text.lower()
        assert "e01.2" in retriever.bm25.search("E01", 1)[0].chunk.retrieval_text.lower()
        assert "6205-2rs/c3" in retriever.bm25.search("6205-2RS/C3", 1)[0].chunk.text.lower()
        fusion = retriever.search("VFD-OC3 acceleration overcurrent", 3, rerank=False)
        reranked = retriever.search("VFD-OC3 acceleration overcurrent", 3)
        assert fusion[0].rrf_score == fusion[0].score
        assert reranked[0].reranker_score == reranked[0].score
        assert reranked[0].rrf_score is not None
        assert "vfd-oc3" in reranked[0].chunk.retrieval_text.lower()
        assert all(hit.reranker_score is None for hit in fusion)
        assert retriever.search("   ") == []


@pytest.mark.parametrize(
    ("section", "field", "value", "manifest_field"),
    [
        ("embedding", "model", "changed-model", "embedding_model"),
        ("embedding", "dimension", 32, "embedding_dim"),
        ("embedding", "provider", "sentence_transformers", "embedding_provider"),
        ("ingestion", "chunk_size", 512, "chunk_size"),
        ("ingestion", "chunk_overlap", 50, "chunk_overlap"),
    ],
)
def test_index_mismatch_rejection(
    indexed_corpus: tuple[Settings, IndexManifest],
    section: str,
    field: str,
    value: object,
    manifest_field: str,
) -> None:
    settings, _ = indexed_corpus
    setattr(getattr(settings, section), field, value)
    with pytest.raises(IndexValidationError, match=f"Manifest {manifest_field} mismatch"):
        load_index(settings.paths.index_dir, settings.embedding, settings.ingestion)


@pytest.mark.parametrize("name", ["vectors.faiss", "chunks.jsonl"])
def test_index_rejects_changed_artifacts(
    indexed_corpus: tuple[Settings, IndexManifest], name: str
) -> None:
    settings, manifest = indexed_corpus
    path = settings.paths.index_dir / "generations" / manifest.generation / name
    with path.open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(IndexValidationError, match="checksum mismatch"):
        load_index(settings.paths.index_dir, settings.embedding, settings.ingestion)


def test_rebuild_failure_keeps_previous_generation(
    indexed_corpus: tuple[Settings, IndexManifest], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, original = indexed_corpus
    root = settings.paths.index_dir
    entries = set(root.iterdir())
    generations = set((root / "generations").iterdir())

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with (
            create_embedding_client(settings.embedding) as embedder,
            pytest.raises(IndexValidationError, match="Could not publish index"),
        ):
            build_index(settings, embedder)
    assert set(root.iterdir()) == entries
    assert set((root / "generations").iterdir()) == generations
    assert load_index(root, settings.embedding, settings.ingestion).manifest == original


def test_index_version_is_stable_across_identical_generations(
    indexed_corpus: tuple[Settings, IndexManifest],
) -> None:
    settings, before = indexed_corpus
    with create_embedding_client(settings.embedding) as embedder:
        after = build_index(settings, embedder)
    assert after.generation != before.generation
    assert after.index_version == before.index_version
    assert (
        load_index(settings.paths.index_dir, settings.embedding, settings.ingestion).manifest
        == after
    )
    with pytest.raises(IndexValidationError, match="Manifest corpus_hash mismatch"):
        load_index(
            settings.paths.index_dir,
            settings.embedding,
            settings.ingestion,
            expected_corpus_hash="0" * 64,
        )
