from collections.abc import Iterator, Sequence

from pydantic import ValidationError

from src.core.exceptions import CorpusError
from src.core.settings import IngestionSettings
from src.ingestion.schemas import Chunk, Section, document_id

_SEPARATORS = ("\n\n", "\n", " ")


def _split_text(text: str, size: int, overlap: int) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            earliest = start + max(size // 2, overlap + 1)
            # Prefer a paragraph, line, then word boundary inside the current window.
            for separator in _SEPARATORS:
                position = text.rfind(separator, earliest - len(separator), end)
                if position != -1:
                    end = position + len(separator)
                    break
        segment = text[start:end].strip()
        if segment:
            yield segment
        if end == len(text):
            break
        start = end - overlap


def chunk_sections(
    source_path: str, sections: Sequence[Section], settings: IngestionSettings
) -> list[Chunk]:
    doc_id = document_id(source_path)
    chunks: list[Chunk] = []
    for section_index, section in enumerate(sections):
        for index, text in enumerate(
            _split_text(section.text, settings.chunk_size, settings.chunk_overlap)
        ):
            try:
                chunk = Chunk(
                    doc_id=doc_id,
                    source_path=source_path,
                    section_title=section.title,
                    page_number=section.page_number,
                    section_index=section_index,
                    chunk_index=index,
                    chunk_id=f"{doc_id}#s{section_index}#c{index}",
                    text=text,
                )
            except ValidationError as exc:
                raise CorpusError(f"{source_path}: invalid chunk metadata") from exc
            chunks.append(chunk)
    return chunks
