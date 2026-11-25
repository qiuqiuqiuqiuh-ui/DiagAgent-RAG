import re
from collections.abc import Sequence

from src.agents.schemas import Citation
from src.core.exceptions import DiagAgentError
from src.ingestion.schemas import Chunk

_MARKER = re.compile(r"\[([^\[\]\r\n]+)\]")
_UNITS = frozenset({"degc", "°c", "°f", "bar", "psi", "mm/s rms", "l/min", "hz", "rpm", "a", "v"})


class CitationError(DiagAgentError):
    """The draft lacks valid inline citations to its retrieved evidence."""


def bind_citations(answer: str, context: Sequence[Chunk]) -> tuple[str, tuple[Citation, ...]]:
    """Validate each reference and render canonical inline chunk IDs."""
    by_id = {chunk.chunk_id: chunk for chunk in context}
    markers: dict[str, None] = {}

    def bind(match: re.Match[str]) -> str:
        label = match[1].strip()
        if label.casefold() in _UNITS:
            return f"({label})"
        ids = [item.strip() for item in label.split(",")]
        if any(chunk_id not in by_id for chunk_id in ids):
            raise CitationError(
                "Use only complete citations to passages retrieved for this diagnosis."
            )
        markers.update(dict.fromkeys(ids))
        return " ".join(f"[{chunk_id}]" for chunk_id in ids)

    remaining = _MARKER.sub("", answer)
    if "[" in remaining or "]" in remaining:
        raise CitationError("Use only complete citations to passages retrieved for this diagnosis.")
    bound = _MARKER.sub(bind, answer)
    if not markers:
        raise CitationError("Provide a diagnosis with at least one inline evidence citation.")
    if not any(char.isalnum() for char in remaining):
        raise CitationError("Provide a diagnosis supported by the cited passages.")
    return bound, tuple(
        Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source_path=chunk.source_path,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
        )
        for chunk in (by_id[marker] for marker in markers)
    )


def validate_citations(answer: str, context: Sequence[Chunk]) -> tuple[Citation, ...]:
    return bind_citations(answer, context)[1]
