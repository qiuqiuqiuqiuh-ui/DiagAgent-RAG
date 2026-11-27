from dataclasses import dataclass
from pathlib import Path
from statistics import median

from src.agents.tools import DiagnosticTools, canonical_sensor_unit
from src.core.exceptions import CorpusError
from src.core.settings import IngestionSettings, RulesSettings
from src.ingestion.chunking import chunk_sections
from src.ingestion.corpus import CorpusIssue, scan_corpus_paths
from src.ingestion.parsers import DocumentParseError, inspect_document

DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class CharacterDistribution:
    minimum: int | None
    median: float | None
    maximum: int | None
    total: int

    @classmethod
    def from_counts(cls, counts: list[int]) -> "CharacterDistribution":
        return (
            cls(min(counts), median(counts), max(counts), sum(counts))
            if counts
            else cls(None, None, None, 0)
        )


@dataclass(frozen=True)
class CorpusPreflightReport:
    corpus_dir: str
    max_file_bytes: int
    scanned_files: int
    document_count: int
    chunk_count: int
    document_characters: CharacterDistribution
    chunk_characters: CharacterDistribution
    issues: tuple[CorpusIssue, ...]

    @property
    def passed(self) -> bool:
        return all(issue.severity != "error" for issue in self.issues)


def _rule_issues(
    captured: dict[str, bytes], rules: RulesSettings, invalid_paths: set[str]
) -> list[CorpusIssue]:
    result = []
    optional = rules.model_copy(update={"allow_missing_rule_tables": True})
    for path in rules.source_paths:
        if path in invalid_paths:
            continue
        if path not in captured:
            if not rules.allow_missing_rule_tables:
                result.append(
                    CorpusIssue(path, "missing_rule_table", "Required rule table is missing")
                )
            continue
        try:
            tools = DiagnosticTools.from_sources({path: captured[path]}, optional)
        except CorpusError as exc:
            result.append(CorpusIssue(path, "invalid_rule_table", str(exc)))
            continue
        for rule in tools.sensor_rules:
            if canonical_sensor_unit(rule.unit) is None:
                result.append(
                    CorpusIssue(
                        path,
                        "unknown_sensor_unit",
                        f"{rule.equipment_id}/{rule.sensor} uses {rule.unit!r}. "
                        "The original unit and thresholds were retained.",
                        severity="warning",
                    )
                )
    return result


def validate_corpus(
    corpus_dir: Path,
    ingestion: IngestionSettings,
    rules: RulesSettings,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> CorpusPreflightReport:
    """Inspect local sources using the ingestion parser without opening an index or a model."""
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    root = corpus_dir.resolve()
    paths, issues = (
        scan_corpus_paths(root)
        if root.is_dir()
        else ([], [CorpusIssue(".", "missing_corpus", "Corpus directory does not exist")])
    )
    captured: dict[str, bytes] = {}
    document_chars: list[int] = []
    chunk_chars: list[int] = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        try:
            if path.is_symlink() or path.resolve() != path:
                issues.append(
                    CorpusIssue(name, "unreadable_path", "Corpus path changed during preflight")
                )
                continue
            if path.stat().st_size > max_file_bytes:
                issues.append(
                    CorpusIssue(name, "oversized_file", f"File exceeds {max_file_bytes} bytes")
                )
                continue
            with path.open("rb") as source:
                data = source.read(max_file_bytes + 1)
        except (OSError, RuntimeError):
            issues.append(CorpusIssue(name, "unreadable_file", "Could not read source file"))
            continue
        if len(data) > max_file_bytes:
            issues.append(
                CorpusIssue(name, "oversized_file", f"File exceeds {max_file_bytes} bytes")
            )
            continue
        try:
            document = inspect_document(name, data)
            chunks = chunk_sections(name, document.sections, ingestion)
        except DocumentParseError as exc:
            code = (
                "invalid_rule_table"
                if name in rules.source_paths and exc.code == "malformed_document"
                else exc.code
            )
            issues.append(CorpusIssue(name, code, str(exc), pages=exc.pages))
            continue
        except CorpusError as exc:
            issues.append(CorpusIssue(name, "invalid_chunk", str(exc)))
            continue
        if document.pages_without_text:
            issues.append(
                CorpusIssue(
                    name,
                    "pdf_pages_without_text",
                    "Some PDF pages have no extractable text. Run OCR where needed.",
                    severity="warning",
                    pages=document.pages_without_text,
                )
            )
        document_chars.append(sum(len(section.text) for section in document.sections))
        chunk_chars.extend(len(chunk.text) for chunk in chunks)
        if name in rules.source_paths:
            captured[name] = data

    if root.is_dir():
        if not chunk_chars:
            issues.append(
                CorpusIssue(".", "no_searchable_documents", "No supported, searchable documents")
            )
        issues.extend(
            _rule_issues(
                captured, rules, {issue.path for issue in issues if issue.severity == "error"}
            )
        )
    return CorpusPreflightReport(
        corpus_dir=str(root),
        max_file_bytes=max_file_bytes,
        scanned_files=len(paths),
        document_count=len(document_chars),
        chunk_count=len(chunk_chars),
        document_characters=CharacterDistribution.from_counts(document_chars),
        chunk_characters=CharacterDistribution.from_counts(chunk_chars),
        issues=tuple(issues),
    )
