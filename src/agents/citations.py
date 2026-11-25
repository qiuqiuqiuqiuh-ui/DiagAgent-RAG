import re
from collections.abc import Sequence

from src.agents.schemas import Citation
from src.core.exceptions import DiagAgentError
from src.ingestion.schemas import Chunk

_MARKER = re.compile(r"\[([^\[\]\r\n]+)\]")


class CitationError(DiagAgentError):
    """The draft lacks valid inline citations to its retrieved evidence."""


def validate_citations(answer: str, context: Sequence[Chunk]) -> tuple[Citation, ...]:
    by_id = {chunk.chunk_id: chunk for chunk in context}
    markers = tuple(dict.fromkeys(_MARKER.findall(answer)))
    if not markers:
        raise CitationError("Provide a diagnosis with at least one inline evidence citation.")
    remaining = _MARKER.sub("", answer)
    if "[" in remaining or "]" in remaining or any(marker not in by_id for marker in markers):
        raise CitationError("Use only complete citations to passages retrieved for this diagnosis.")
    if not any(char.isalnum() for char in remaining):
        raise CitationError("Provide a diagnosis supported by the cited passages.")
    return tuple(
        Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source_path=chunk.source_path,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
        )
        for chunk in (by_id[marker] for marker in markers)
    )
