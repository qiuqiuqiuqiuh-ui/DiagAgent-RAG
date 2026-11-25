from hashlib import sha256
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.core.exceptions import CorpusError
from src.core.settings import IngestionSettings
from src.ingestion.chunking import chunk_sections
from src.ingestion.corpus import read_corpus
from src.ingestion.parsers import parse_document
from src.ingestion.schemas import Section


def test_markdown_structure_and_fenced_headings() -> None:
    sections = parse_document(
        "manuals/vfd.md",
        b"Before headings\n\nDrive\n=====\n\nIntro to the drive.\n\n"
        b"## OC3\n\nOvercurrent.\n\n```text\n# literal heading\n```\n\n"
        b"### Ramp time\n\nCheck the acceleration ramp.\n",
    )
    assert [section.title for section in sections] == [
        "vfd",
        "Drive",
        "Drive > OC3",
        "Drive > OC3 > Ramp time",
    ]
    assert "# literal heading" in sections[2].text
    assert all(section.page_number is None for section in sections)


def test_pdf_preserves_original_page_number(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    page = writer.add_blank_page(width=200, height=200)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 10 100 Td (VFD-OC3 acceleration) Tj ET")
    page.replace_contents(content)
    path = tmp_path / "motor.pdf"
    writer.write(path)
    sections = parse_document("manuals/motor.pdf", path.read_bytes())
    assert len(sections) == 1
    assert sections[0].page_number == 2
    assert "VFD-OC3 acceleration" in sections[0].text
    chunks = chunk_sections(
        "manuals/motor.pdf", sections, IngestionSettings(chunk_size=80, chunk_overlap=0)
    )
    assert chunks[0].page_number == 2


def test_structured_records_preserve_codes_and_values() -> None:
    csv_sections = parse_document(
        "fault_codes.csv",
        b'fault_code,meaning\r\nVFD-OC3,"Acceleration, overcurrent"\r\nERR-402,Feedback\r\n',
    )
    assert len(csv_sections) == 2
    assert csv_sections[0].title.endswith("VFD-OC3")
    assert "meaning: Acceleration, overcurrent" in csv_sections[0].text
    json_sections = parse_document(
        "sensor_thresholds.json",
        b'{"records":[{"sensor":"oil_pressure","warning_threshold":1.8,"direction":"low"}]}',
    )
    assert json_sections[0].title.endswith("oil_pressure")
    assert "warning_threshold: 1.8" in json_sections[0].text


@pytest.mark.parametrize(
    ("path", "content", "message"),
    [
        ("faults.csv", b"code,code\na,b", "unique, non-empty"),
        ("faults.csv", b"code,meaning\nERR-402", "wrong column count"),
        ("thresholds.json", b'["x"]', "array of objects"),
        ("blank.md", b"\n\t", "No searchable text"),
        ("invalid.pdf", b"not a PDF", "Could not parse"),
    ],
)
def test_invalid_documents_are_reported(path: str, content: bytes, message: str) -> None:
    with pytest.raises(CorpusError, match=message) as raised:
        parse_document(path, content)
    assert path in str(raised.value)


def test_chunk_bounds_deterministic_ids_and_overlap() -> None:
    settings = IngestionSettings(chunk_size=10, chunk_overlap=3)
    path = "manuals/drive notes].md"
    sections = [
        Section(title="Line 1", text="abcdefghijklmnopqrstuvwxyz"),
        Section(title="Other section", text="Separate"),
    ]
    chunks = chunk_sections(path, sections, settings)
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
        "Separate",
    ]
    assert chunks == chunk_sections(path, sections, settings)
    assert chunks[0].chunk_id == "manuals/drive%20notes%5D.md#s0#c0"
    assert chunks[-1].chunk_id == "manuals/drive%20notes%5D.md#s1#c0"
    assert all(chunk.source_path == path and len(chunk.text) <= 10 for chunk in chunks)


def test_chunking_prefers_paragraphs_and_handles_maximum_overlap() -> None:
    sections = [Section(title="Section", text="First paragraph.\n\nSecond paragraph.\n\nThird.")]
    chunks = chunk_sections(
        "manual.md", sections, IngestionSettings(chunk_size=23, chunk_overlap=0)
    )
    assert [chunk.text for chunk in chunks] == ["First paragraph.", "Second paragraph.", "Third."]
    tiny = [Section(title="Section", text="abcdefgh")]
    chunks = chunk_sections("manual.md", tiny, IngestionSettings(chunk_size=3, chunk_overlap=2))
    assert [chunk.text for chunk in chunks] == ["abc", "bcd", "cde", "def", "efg", "fgh"]


def test_corpus_hash_is_portable_and_sensitive_to_changes(tmp_path: Path) -> None:
    settings = IngestionSettings(chunk_size=100, chunk_overlap=0)
    doc = tmp_path / "drive.md"
    doc.write_bytes(b"# Drive\r\n\r\nDetails.\r\n")
    (tmp_path / "ignored.txt").write_bytes(b"irrelevant")
    first = read_corpus(tmp_path, settings)
    assert first.document_count == 1
    assert len(first.corpus_hash) == sha256().digest_size * 2
    doc.write_bytes(b"# Drive\n\nDetails.\n")
    assert read_corpus(tmp_path, settings) == first
    doc.write_bytes(b"# Drive\n\nUpdated details.\n")
    assert read_corpus(tmp_path, settings).corpus_hash != first.corpus_hash
