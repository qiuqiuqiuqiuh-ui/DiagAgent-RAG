import csv
import json
from io import BytesIO, StringIO
from pathlib import Path

from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from src.core.exceptions import CorpusError
from src.ingestion.schemas import Section

SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".csv", ".json"}
_MARKDOWN = MarkdownIt("commonmark")


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
            sections.append(
                Section(title=" > ".join(name for _, name in headings) or title, text=body)
            )
        level = int(token.tag[1:])
        headings = [(current, name) for current, name in headings if current < level]
        headings.append((level, tokens[index + 1].content.strip() or title))
        start = heading_end

    body = "\n".join(lines[start:]).strip()
    if body:
        sections.append(Section(title=" > ".join(name for _, name in headings) or title, text=body))
    return sections


def _pdf_sections(content: bytes, title: str) -> list[Section]:
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise CorpusError("Encrypted PDFs are unsupported")
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(
                Section(title=f"{title} > Page {number}", page_number=number, text=text)
            )
    return sections


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


def parse_document(source_path: str, content: bytes) -> list[Section]:
    """Parse supported UTF-8 documents and extract text from PDF pages."""
    path = Path(source_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            sections = _pdf_sections(content, path.stem)
        else:
            text = content.decode("utf-8-sig")
            if suffix in {".md", ".markdown"}:
                sections = _markdown_sections(text, path.stem)
            elif suffix == ".csv":
                sections = _table_sections(parse_csv_records(text), path.stem)
            elif suffix == ".json":
                sections = _table_sections(parse_json_records(text), path.stem)
            else:
                raise CorpusError(f"Unsupported extension {suffix}")
    except CorpusError as exc:
        raise CorpusError(f"{source_path}: {exc}") from exc
    except (ValueError, csv.Error, PyPdfError, DependencyError) as exc:
        raise CorpusError(f"Could not parse {source_path} ({type(exc).__name__})") from exc
    if not sections:
        detail = " (scanned PDFs require OCR upstream)" if suffix == ".pdf" else ""
        raise CorpusError(f"No searchable text in {source_path}{detail}")
    return sections
