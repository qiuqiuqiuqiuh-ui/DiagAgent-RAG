import json
from collections.abc import Collection
from dataclasses import dataclass, field
from hashlib import sha256
from os import walk
from pathlib import Path
from typing import Literal

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
    captured_sources: dict[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusIssue:
    path: str
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    pages: tuple[int, ...] = ()


def scan_corpus_paths(root: Path) -> tuple[list[Path], list[CorpusIssue]]:
    """List supported sources without following links or traversing hidden directories."""
    paths = []
    issues = []

    def link_safe(path: Path) -> bool:
        name = path.relative_to(root).as_posix()
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            issues.append(CorpusIssue(name, "unreadable_path", "Could not resolve source path"))
            return False
        if not resolved.is_relative_to(root):
            issues.append(
                CorpusIssue(
                    name,
                    "symlink_outside",
                    f"Corpus files must reside inside the corpus directory: {name}",
                )
            )
            return False
        if resolved != path or path.is_symlink():
            issues.append(
                CorpusIssue(name, "symlink", f"Corpus sources must be regular paths: {name}")
            )
            return False
        return True

    def on_walk_error(error: OSError) -> None:
        location = Path(error.filename) if error.filename else root
        name = location.relative_to(root).as_posix() if location.is_relative_to(root) else "."
        issues.append(CorpusIssue(name, "unreadable_path", f"Could not read corpus path {name}"))

    for directory, children, files in walk(root, onerror=on_walk_error):
        parent = Path(directory)
        children[:] = [
            name
            for name in sorted(children)
            if not name.startswith(".") and link_safe(parent / name)
        ]
        for name in sorted(files):
            path = parent / name
            if name.startswith(".") or (
                path.suffix.lower() not in SUPPORTED_SUFFIXES and not path.is_symlink()
            ):
                continue
            if link_safe(path):
                if path.is_file():
                    paths.append(path)
                else:
                    issues.append(
                        CorpusIssue(
                            path.relative_to(root).as_posix(),
                            "unreadable_file",
                            "Source is not a regular file",
                        )
                    )
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix()), issues


def read_corpus(
    corpus_dir: Path, settings: IngestionSettings, *, capture: Collection[str] = ()
) -> Corpus:
    """Parse and hash the same byte snapshot of each supported file in path order."""
    root = corpus_dir.resolve()
    if not root.is_dir():
        raise CorpusError(f"Corpus directory does not exist: {root}")

    paths, issues = scan_corpus_paths(root)
    if issues:
        raise CorpusError(issues[0].message)
    chunks: list[Chunk] = []
    captured: dict[str, bytes] = {}
    fingerprints: list[tuple[str, str]] = []
    for path in paths:
        source_path = path.relative_to(root).as_posix()
        if path.is_symlink() or path.resolve() != path:
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
        if source_path in capture:
            captured[source_path] = content
        chunks.extend(chunk_sections(source_path, parse_document(source_path, content), settings))
    if not chunks:
        raise CorpusError(f"No supported, searchable documents in {root}")
    corpus_hash = sha256(json.dumps(fingerprints, separators=(",", ":")).encode()).hexdigest()
    return Corpus(
        chunks=tuple(chunks),
        corpus_hash=corpus_hash,
        document_count=len(fingerprints),
        captured_sources=captured,
    )
