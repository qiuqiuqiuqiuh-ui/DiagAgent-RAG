import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Annotated, BinaryIO, Literal
from uuid import uuid4

import faiss
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.exceptions import IndexValidationError
from src.core.settings import EmbeddingSettings, IngestionSettings
from src.ingestion.schemas import Chunk

_FORMAT_VERSION = 1
_CHUNKS_FILE = "chunks.jsonl"
_VECTORS_FILE = "vectors.faiss"
_MANIFEST_FILE = "manifest.json"
_IDENTITY_FIELDS = {
    "format_version",
    "embedding_provider",
    "embedding_model",
    "embedding_dim",
    "chunk_size",
    "chunk_overlap",
    "corpus_hash",
    "vectors_sha256",
    "chunks_sha256",
}
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _index_version(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"v{_FORMAT_VERSION}-{sha256(encoded).hexdigest()}"


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1]
    index_version: str
    embedding_provider: Literal["mock", "openai_compatible", "sentence_transformers"]
    embedding_model: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    corpus_hash: SHA256
    created_at: AwareDatetime
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    chunk_count: int = Field(gt=0)
    vectors_sha256: SHA256
    chunks_sha256: SHA256

    @model_validator(mode="after")
    def validate_identity(self) -> "IndexManifest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        identity = self.model_dump(include=_IDENTITY_FIELDS)
        if self.index_version != _index_version(identity):
            raise ValueError("index_version does not match the index identity")
        return self


@dataclass(frozen=True)
class StoredIndex:
    manifest: IndexManifest
    chunks: tuple[Chunk, ...]
    vectors: faiss.Index


def _profile_identity(
    embedding: EmbeddingSettings, ingestion: IngestionSettings
) -> dict[str, object]:
    return {
        "format_version": _FORMAT_VERSION,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "embedding_dim": embedding.dimension,
        "chunk_size": ingestion.chunk_size,
        "chunk_overlap": ingestion.chunk_overlap,
    }


def _digest(stream: BinaryIO) -> str:
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _artifact_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return _digest(stream)


def save_index(
    index_dir: Path,
    vectors: faiss.IndexFlatIP,
    chunks: tuple[Chunk, ...],
    corpus_hash: str,
    embedding: EmbeddingSettings,
    ingestion: IngestionSettings,
) -> IndexManifest:
    if vectors.ntotal != len(chunks) or vectors.d != embedding.dimension or not chunks:
        raise IndexValidationError("FAISS dimensions and rows must match the profile and chunks")

    generation = uuid4().hex
    artifacts = index_dir / "generations" / generation
    staged_manifest = index_dir / f".manifest-{generation}.tmp"
    vectors_path = artifacts / _VECTORS_FILE
    chunks_path = artifacts / _CHUNKS_FILE
    created_generation = False
    published = False
    try:
        artifacts.mkdir(parents=True)
        created_generation = True
        with vectors_path.open("wb") as stream:
            # Python opens paths so Windows can also use non-ASCII index directories.
            writer = faiss.PyCallbackIOWriter(stream.write)
            faiss.write_index(vectors, writer)
        with chunks_path.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(chunk.model_dump_json() + "\n")

        identity = {
            **_profile_identity(embedding, ingestion),
            "corpus_hash": corpus_hash,
            "vectors_sha256": _artifact_hash(vectors_path),
            "chunks_sha256": _artifact_hash(chunks_path),
        }
        manifest = IndexManifest.model_validate(
            {
                **identity,
                "index_version": _index_version(identity),
                "created_at": datetime.now(timezone.utc),
                "generation": generation,
                "chunk_count": len(chunks),
            }
        )
        staged_manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        staged_manifest.replace(index_dir / _MANIFEST_FILE)
        published = True
        return manifest
    except (OSError, RuntimeError, ValidationError) as exc:
        raise IndexValidationError(f"Could not publish index in {index_dir}") from exc
    finally:
        if created_generation and not published:
            for path in (staged_manifest, vectors_path, chunks_path):
                with suppress(OSError):
                    path.unlink(missing_ok=True)
            with suppress(OSError):
                artifacts.rmdir()


def load_index(
    index_dir: Path,
    embedding: EmbeddingSettings,
    ingestion: IngestionSettings,
    *,
    expected_corpus_hash: str | None = None,
) -> StoredIndex:
    try:
        manifest = IndexManifest.model_validate_json((index_dir / _MANIFEST_FILE).read_bytes())
    except (OSError, ValidationError) as exc:
        raise IndexValidationError(f"Invalid or missing manifest in {index_dir}: {exc}") from exc

    expected = _profile_identity(embedding, ingestion)
    if expected_corpus_hash is not None:
        expected["corpus_hash"] = expected_corpus_hash
    for field, value in expected.items():
        actual = getattr(manifest, field)
        if actual != value:
            raise IndexValidationError(
                f"Manifest {field} mismatch: expected {value!r}, got {actual!r}"
            )

    artifacts = index_dir / "generations" / manifest.generation
    try:
        with (artifacts / _CHUNKS_FILE).open("rb") as chunk_stream:
            if _digest(chunk_stream) != manifest.chunks_sha256:
                raise IndexValidationError("Chunk artifact checksum mismatch")
            chunk_stream.seek(0)
            chunks = tuple(Chunk.model_validate_json(line) for line in chunk_stream)
        if len(chunks) != manifest.chunk_count or len({chunk.chunk_id for chunk in chunks}) != len(
            chunks
        ):
            raise IndexValidationError("Chunk count or unique IDs do not match the manifest")
        with (artifacts / _VECTORS_FILE).open("rb") as vector_stream:
            if _digest(vector_stream) != manifest.vectors_sha256:
                raise IndexValidationError("FAISS artifact checksum mismatch")
            vector_stream.seek(0)
            reader = faiss.PyCallbackIOReader(vector_stream.read)
            vectors = faiss.read_index(reader)
    except (OSError, RuntimeError, ValidationError) as exc:
        raise IndexValidationError(f"Could not read index artifacts in {artifacts}") from exc

    if (
        not isinstance(vectors, faiss.IndexFlatIP)
        or vectors.d != manifest.embedding_dim
        or vectors.ntotal != manifest.chunk_count
    ):
        raise IndexValidationError(
            "FAISS index type, dimension, or count does not match the manifest"
        )
    return StoredIndex(manifest=manifest, chunks=chunks, vectors=vectors)
