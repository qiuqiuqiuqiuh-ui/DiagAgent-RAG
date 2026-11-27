import json
from hashlib import sha256
from pathlib import Path
from runpy import run_path

import pytest
from pypdf import PdfWriter
from pypdf.errors import DependencyError, ParseError
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.core.exceptions import CorpusError
from src.core.settings import IngestionSettings, RulesSettings
from src.ingestion.chunking import chunk_sections
from src.ingestion.corpus import read_corpus
from src.ingestion.parsers import parse_csv_records, parse_document, parse_json_records
from src.ingestion.preflight import validate_corpus
from src.ingestion.schemas import Section

ROOT = Path(__file__).resolve().parents[1]


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


def test_markdown_uses_parser_line_boundaries() -> None:
    source = "# Title\r\n\r\nIntro\u2028Line2\r\r## Second\r\rBody of second\r"
    sections = parse_document("manual.md", source.encode("utf-8"))
    assert [(section.title, section.text) for section in sections] == [
        ("Title", "Intro\u2028Line2"),
        ("Title > Second", "Body of second"),
    ]


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
    report = validate_corpus(
        tmp_path,
        IngestionSettings(chunk_size=80, chunk_overlap=0),
        RulesSettings(allow_missing_rule_tables=True),
    )
    assert report.passed
    assert [(issue.code, issue.pages) for issue in report.issues] == [
        ("pdf_pages_without_text", (1,))
    ]


def test_ocr_markdown_page_heading_propagates_to_chunks() -> None:
    source = (
        "# 手册\n\n## Page 3\n\n启动检查。\n\n### 告警\n\n记录电流。\n\n## Page 8\n\n检查负载。\n"
    )
    sections = parse_document("manuals/ocr.md", source.encode())
    assert [section.page_number for section in sections] == [3, 3, 8]
    chunks = chunk_sections(
        "manuals/ocr.md", sections, IngestionSettings(chunk_size=100, chunk_overlap=0)
    )
    assert [(chunk.section_title, chunk.page_number) for chunk in chunks] == [
        ("手册 > Page 3", 3),
        ("手册 > Page 3 > 告警", 3),
        ("手册 > Page 8", 8),
    ]


def test_preflight_reports_failed_sources_independently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text("## Page 4\n\n核对设备铭牌。", encoding="utf-8")
    (corpus / "empty.md").touch()
    (corpus / "encoding.md").write_bytes(b"\xff\x00")
    (corpus / "large.md").write_bytes(b"x" * 8193)
    (corpus / "fault_codes.csv").write_text("fault_code,meaning\nVFD-OC3\n", encoding="utf-8")
    (corpus / "sensor_thresholds.json").write_text(
        json.dumps(
            [
                {
                    "equipment_id": "",
                    "sensor": "pressure",
                    "unit": "bar",
                    "warning_threshold": "bad",
                    "critical_threshold": 1.4,
                    "direction": "low",
                }
            ]
        ),
        encoding="utf-8",
    )
    scanned = PdfWriter()
    for _ in range(2):
        page = scanned.add_blank_page(width=100, height=100)
        image = DecodedStreamObject()
        image.set_data(b"BI /W 1 /H 1 /CS /RGB /BPC 8 ID \x00\x00\x00 EI")
        page.replace_contents(image)
    scanned.write(corpus / "scan.pdf")
    encrypted = PdfWriter()
    encrypted.add_blank_page(width=100, height=100)
    encrypted.encrypt("offline-test-password")
    encrypted.write(corpus / "encrypted.pdf")

    cli = run_path(str(ROOT / "scripts" / "validate_corpus.py"))["main"]
    assert (
        cli(
            [
                "--config",
                str(ROOT / "configs" / "default.yaml"),
                "--project-root",
                str(tmp_path),
                "--corpus-dir",
                str(corpus),
                "--max-file-bytes",
                "8192",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL" and result["document_count"] >= 1
    issues = {issue["path"]: issue for issue in result["issues"]}
    assert {path: issue["code"] for path, issue in issues.items()} == {
        "empty.md": "empty_file",
        "encoding.md": "invalid_encoding",
        "large.md": "oversized_file",
        "encrypted.pdf": "encrypted_pdf",
        "scan.pdf": "ocr_required",
        "fault_codes.csv": "invalid_rule_table",
        "sensor_thresholds.json": "invalid_rule_table",
    }
    assert issues["scan.pdf"]["pages"] == [1, 2] and "OCR" in issues["scan.pdf"]["message"]
    assert "wrong column count" in issues["fault_codes.csv"]["message"]
    assert "equipment_id" in issues["sensor_thresholds.json"]["message"]
    assert "warning_threshold" in issues["sensor_thresholds.json"]["message"]


def test_preflight_reports_duplicate_rules_and_inverted_thresholds(tmp_path: Path) -> None:
    rows = (ROOT / "data" / "raw" / "fault_codes.csv").read_text(encoding="utf-8").splitlines()
    (tmp_path / "fault_codes.csv").write_text("\n".join([*rows, rows[1]]), encoding="utf-8")
    (tmp_path / "sensor_thresholds.json").write_text(
        '[{"equipment_id":"MX-200","sensor":"temp","unit":"degC",'
        '"warning_threshold":105,"critical_threshold":90,"direction":"high"}]',
        encoding="utf-8",
    )
    report = validate_corpus(
        tmp_path, IngestionSettings(chunk_size=100, chunk_overlap=0), RulesSettings()
    )
    assert not report.passed
    issues = {issue.path: issue.message for issue in report.issues}
    assert "Duplicate rule" in issues["fault_codes.csv"]
    assert "Critical threshold must be beyond" in issues["sensor_thresholds.json"]


def test_preflight_ignores_link_targets_outside_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("Do not read", encoding="utf-8")
    try:
        (corpus / "external.md").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this host")
    report = validate_corpus(
        corpus,
        IngestionSettings(chunk_size=100, chunk_overlap=0),
        RulesSettings(allow_missing_rule_tables=True),
    )
    assert report.document_count == report.chunk_count == 0
    assert ("external.md", "symlink_outside") in [
        (issue.path, issue.code) for issue in report.issues
    ]


def test_preflight_accepts_shipped_corpus_without_touching_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = run_path(str(ROOT / "scripts" / "validate_corpus.py"))["main"]
    assert (
        cli(
            [
                "--config",
                str(ROOT / "configs" / "default.yaml"),
                "--project-root",
                str(tmp_path),
                "--corpus-dir",
                str(ROOT / "data" / "raw"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS" and result["issues"] == []
    assert (result["document_count"], result["chunk_count"]) == (7, 35)
    assert result["chunk_characters"]["maximum"] <= 1000
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [DependencyError, ParseError])
def test_pdf_parser_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def fail(*args: object) -> None:
        raise error("unreadable")

    monkeypatch.setattr("src.ingestion.parsers.PdfReader", fail)
    with pytest.raises(CorpusError, match=rf"manual\.pdf \({error.__name__}\)"):
        parse_document("manual.pdf", b"PDF fixture")


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


def test_record_parsers_handle_blank_rows_and_business_records_field() -> None:
    assert parse_csv_records("a,b\n1,2\n  \n,\n  , \n3,4\n") == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
    assert parse_json_records('{"equipment_id":"DRV-300","records":5}') == [
        {"equipment_id": "DRV-300", "records": 5}
    ]
    with pytest.raises(CorpusError, match="records wrapper"):
        parse_json_records('{"records":"bad"}')
    section = parse_document(
        "thresholds.json", b'{"equipment_id":"MX-200","sensor":"winding_temp"}'
    )[0]
    assert section.title == "thresholds > MX-200 winding_temp"


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


def test_ignores_hidden_files_and_directories(tmp_path: Path) -> None:
    settings = IngestionSettings(chunk_size=100, chunk_overlap=0)
    (tmp_path / "manual.md").write_text("Read me.", encoding="utf-8")
    expected = read_corpus(tmp_path, settings)
    (tmp_path / ".hidden.md").write_bytes(b"")
    checkpoint = tmp_path / "manuals" / ".ipynb_checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manual-checkpoint.md").write_bytes(b"")
    assert read_corpus(tmp_path, settings) == expected


def test_invalid_source_path_raises_corpus_error() -> None:
    with pytest.raises(CorpusError, match="bad:manual.md"):
        chunk_sections(
            "bad:manual.md",
            [Section(title="Manual", text="details")],
            IngestionSettings(chunk_size=100, chunk_overlap=0),
        )
