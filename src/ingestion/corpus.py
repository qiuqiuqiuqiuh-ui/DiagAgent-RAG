import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from src.core.exceptions import CorpusError
from src.core.settings import IngestionSettings
from src.ingestion.chunking import chunk_sections
from src.ingestion.parsers import SUPPORTED_SUFFIXES, parse_document
from src.ingestion.schemas import Chunk


@dataclass(frozen=True)
class Corpus:
    chunks: tuple[Chunk, ...]
    corpus_hash: str
    document_count: int


def read_corpus(corpus_dir: Path, settings: IngestionSettings) -> Corpus:
    """Parse and hash the same byte snapshot of each supported file in path order."""
    root = corpus_dir.resolve()
    if not root.is_dir():
        raise CorpusError(f"Corpus directory does not exist: {root}")

    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    chunks: list[Chunk] = []
    fingerprints: list[tuple[str, str]] = []
    for path in paths:
        source_path = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise CorpusError(
                f"Corpus files must reside inside the corpus directory: {source_path}"
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CorpusError(f"Could not read corpus file {source_path}") from exc
        if path.suffix.lower() != ".pdf":
            content = content.replace(b"\r\n", b"\n")
        fingerprints.append((source_path, sha256(content).hexdigest()))
        chunks.extend(chunk_sections(source_path, parse_document(source_path, content), settings))
    if not chunks:
        raise CorpusError(f"No supported, searchable documents in {root}")
    corpus_hash = sha256(json.dumps(fingerprints, separators=(",", ":")).encode()).hexdigest()
    return Corpus(chunks=tuple(chunks), corpus_hash=corpus_hash, document_count=len(fingerprints))
