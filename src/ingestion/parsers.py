import csv
import json
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from src.core.exceptions import CorpusError
from src.ingestion.schemas import Section

SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".csv", ".json"}
_MARKDOWN = MarkdownIt("commonmark")
_PAGE_HEADING = re.compile(r"Page ([1-9][0-9]*)", re.IGNORECASE)


class DocumentParseError(CorpusError):
    def __init__(self, code: str, message: str, *, pages: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.pages = pages


@dataclass(frozen=True)
class ParsedDocument:
    sections: list[Section]
    pages_without_text: tuple[int, ...] = ()


def _markdown_section(headings: list[tuple[int, str]], title: str, body: str) -> Section:
    page = next(
        (
            int(match[1])
            for level, name in headings
            if level == 2 and (match := _PAGE_HEADING.fullmatch(name))
        ),
        None,
    )
    return Section(
        title=" > ".join(name for _, name in headings) or title, text=body, page_number=page
    )


def _markdown_sections(text: str, title: str) -> list[Section]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    tokens = _MARKDOWN.parse(text)
    sections: list[Section] = []
    headings: list[tuple[int, str]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0 or token.map is None:
            continue
        heading_start, heading_end = token.map
        body = "\n".join(lines[start:heading_start]).strip()
        if body:
            sections.append(_markdown_section(headings, title, body))
        level = int(token.tag[1:])
        headings = [(current, name) for current, name in headings if current < level]
        headings.append((level, tokens[index + 1].content.strip() or title))
        start = heading_end

    body = "\n".join(lines[start:]).strip()
    if body:
        sections.append(_markdown_section(headings, title, body))
    return sections


def _parse_pdf(content: bytes, source_path: str) -> ParsedDocument:
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise DocumentParseError("encrypted_pdf", f"{source_path}: Encrypted PDFs are unsupported")
    title = Path(source_path).stem
    sections: list[Section] = []
    empty = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(
                Section(title=f"{title} > Page {number}", page_number=number, text=text)
            )
        else:
            empty.append(number)
    return ParsedDocument(sections, tuple(empty))


def parse_csv_records(text: str) -> list[dict[str, str]]:
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    headers = [header.strip() for header in next(reader, [])]
    if not headers or any(not header for header in headers) or len(set(headers)) != len(headers):
        raise CorpusError("CSV requires unique, non-empty column names")
    records: list[dict[str, str]] = []
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(headers):
            raise CorpusError(
                f"CSV row ending on line {reader.line_num} has the wrong column count"
            )
        records.append(dict(zip(headers, (value.strip() for value in row), strict=True)))
    return records


def parse_json_records(text: str) -> list[dict[str, object]]:
    data = json.loads(text)
    if isinstance(data, dict):
        records = data.get("records")
        data = records if isinstance(records, list) or set(data) == {"records"} else [data]
    if not isinstance(data, list) or any(not isinstance(row, dict) or not row for row in data):
        raise CorpusError(
            "JSON tables require an object, an array of objects, or a records wrapper"
        )
    return data


def _record_title(record: dict[str, object], fallback: str) -> str:
    equipment = record.get("equipment_id")
    prefix = f"{equipment.strip()} " if isinstance(equipment, str) and equipment.strip() else ""
    for field in ("fault_code", "sensor", "id", "name"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return f"{prefix}{value.strip()}"
    return fallback


def _table_sections(
    records: list[dict[str, object]] | list[dict[str, str]], title: str
) -> list[Section]:
    sections = []
    for number, record in enumerate(records, start=1):
        fields = []
        for key, value in record.items():
            rendered = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, sort_keys=True)
            )
            fields.append(f"{key}: {rendered}")
        label = _record_title(dict(record), f"Record {number}")
        sections.append(Section(title=f"{title} > {label}", text="\n".join(fields)))
    return sections


def inspect_document(source_path: str, content: bytes) -> ParsedDocument:
    """Parse a source once and retain the original PDF pages without extractable text."""
    path = Path(source_path)
    suffix = path.suffix.lower()
    if not content:
        raise DocumentParseError("empty_file", f"Empty file {source_path}")
    pages: tuple[int, ...] = ()
    try:
        if suffix == ".pdf":
            parsed = _parse_pdf(content, source_path)
            sections, pages = parsed.sections, parsed.pages_without_text
        else:
            text = content.decode("utf-8-sig")
            if "\x00" in text:
                raise DocumentParseError(
                    "invalid_encoding", f"{source_path}: text contains NUL bytes"
                )
            if suffix in {".md", ".markdown"}:
                sections = _markdown_sections(text, path.stem)
            elif suffix == ".csv":
                sections = _table_sections(parse_csv_records(text), path.stem)
            elif suffix == ".json":
                sections = _table_sections(parse_json_records(text), path.stem)
            else:
                raise CorpusError(f"Unsupported extension {suffix}")
    except DocumentParseError:
        raise
    except UnicodeDecodeError:
        raise DocumentParseError(
            "invalid_encoding", f"{source_path}: expected UTF-8 text"
        ) from None
    except CorpusError as exc:
        raise DocumentParseError("malformed_document", f"{source_path}: {exc}") from exc
    except (ValueError, OSError, csv.Error, PyPdfError, DependencyError) as exc:
        raise DocumentParseError(
            "malformed_document", f"Could not parse {source_path} ({type(exc).__name__})"
        ) from exc
    if not sections:
        if suffix == ".pdf":
            raise DocumentParseError(
                "ocr_required",
                f"No searchable text in {source_path}. Run OCR upstream.",
                pages=pages,
            )
        raise DocumentParseError("empty_file", f"No searchable text in {source_path}")
    return ParsedDocument(sections, pages)


def parse_document(source_path: str, content: bytes) -> list[Section]:
    """Parse supported UTF-8 documents and extract text from PDF pages."""
    return inspect_document(source_path, content).sections
