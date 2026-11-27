import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from src.agents.schemas import (
    DiagnosticRequest,
    FaultCodeFinding,
    FaultCodeRule,
    SensorFinding,
    SensorThresholdRule,
    ToolFinding,
)
from src.core.exceptions import CorpusError
from src.core.tokenizer import tokenize_codes
from src.ingestion.parsers import parse_csv_records, parse_json_records

Rule = TypeVar("Rule", FaultCodeRule, SensorThresholdRule)
FAULT_TABLE = "fault_codes.csv"
SENSOR_TABLE = "sensor_thresholds.json"
RULE_TABLES = (FAULT_TABLE, SENSOR_TABLE)


def _read_table(root: Path, filename: str) -> bytes | None:
    path = root / filename
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise CorpusError("Rule tables must reside inside the corpus directory")
        if not path.exists():
            return None
        return path.read_bytes()
    except (OSError, CorpusError) as exc:
        raise CorpusError(f"Invalid rule table {path.name}: {exc}") from exc


def _parse_rules(
    sources: Mapping[str, bytes],
    filename: str,
    parse: Callable[[str], Sequence[object]],
    model: type[Rule],
) -> list[Rule]:
    if filename not in sources:
        return []
    try:
        return [
            model.model_validate(record) for record in parse(sources[filename].decode("utf-8-sig"))
        ]
    except (ValueError, csv.Error, CorpusError) as exc:
        raise CorpusError(f"Invalid rule table {filename}: {exc}") from exc


def _index_rules(rules: Sequence[Rule], label: Callable[[Rule], str]) -> dict[str, dict[str, Rule]]:
    index: dict[str, dict[str, Rule]] = {}
    for rule in rules:
        name = label(rule).casefold()
        asset = rule.equipment_id.casefold()
        variants = index.setdefault(name, {})
        if asset in variants:
            raise CorpusError(f"Duplicate rule for {label(rule)} on {rule.equipment_id}")
        variants[asset] = rule
    return index


def _applicable_rules(variants: dict[str, Rule], equipment_id: str | None) -> tuple[Rule, ...]:
    if equipment_id is None:
        return tuple(variants.values())
    rule = variants.get(equipment_id.casefold())
    return () if rule is None else (rule,)


class DiagnosticTools:
    """Look up fault codes and evaluate sensor thresholds from curated tables."""

    def __init__(
        self,
        fault_codes: Sequence[FaultCodeRule] = (),
        thresholds: Sequence[SensorThresholdRule] = (),
    ) -> None:
        self._fault_codes = _index_rules(fault_codes, lambda rule: rule.fault_code)
        self._thresholds = _index_rules(thresholds, lambda rule: rule.sensor)
        self._equipment = {
            rule.equipment_id.casefold(): rule.equipment_id for rule in (*fault_codes, *thresholds)
        }

    @classmethod
    def from_corpus(cls, corpus_dir: Path) -> "DiagnosticTools":
        root = corpus_dir.resolve()
        if not root.is_dir():
            raise CorpusError(f"Corpus directory does not exist: {root}")
        sources = {
            filename: content
            for filename in RULE_TABLES
            if (content := _read_table(root, filename)) is not None
        }
        return cls.from_sources(sources)

    @classmethod
    def from_sources(cls, sources: Mapping[str, bytes]) -> "DiagnosticTools":
        return cls(
            _parse_rules(sources, FAULT_TABLE, parse_csv_records, FaultCodeRule),
            _parse_rules(sources, SENSOR_TABLE, parse_json_records, SensorThresholdRule),
        )

    @property
    def equipment_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._equipment.values(), key=str.casefold))

    @property
    def fault_codes_by_equipment(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for variants in self._fault_codes.values():
            for asset, rule in variants.items():
                groups.setdefault(self._equipment[asset], []).append(rule.fault_code)
        return {
            asset: tuple(sorted(groups[asset], key=str.casefold))
            for asset in self.equipment_ids
            if asset in groups
        }

    @property
    def sensor_rules(self) -> tuple[SensorThresholdRule, ...]:
        rules = (rule for variants in self._thresholds.values() for rule in variants.values())
        return tuple(
            sorted(rules, key=lambda rule: (rule.equipment_id.casefold(), rule.sensor.casefold()))
        )

    def equipment_for(self, request: DiagnosticRequest) -> str | None:
        if request.equipment_id is not None:
            return request.equipment_id
        matches = self._equipment.keys() & set(tokenize_codes(request.query))
        return self._equipment[next(iter(matches))] if len(matches) == 1 else None

    def evaluate(self, request: DiagnosticRequest) -> tuple[ToolFinding, ...]:
        findings: list[ToolFinding] = []
        terms = tokenize_codes(request.query)
        equipment_id = self.equipment_for(request)
        codes = (
            (request.fault_code,)
            if request.fault_code is not None
            else tuple(code for code in dict.fromkeys(terms) if code in self._fault_codes)
        )
        for code in codes:
            variants = self._fault_codes.get(code.casefold(), {})
            rules = _applicable_rules(variants, equipment_id)
            findings.append(
                FaultCodeFinding(
                    fault_code=rules[0].fault_code if len(rules) == 1 else code,
                    status="found" if len(rules) == 1 else "ambiguous" if rules else "unknown",
                    rule=rules[0] if len(rules) == 1 else None,
                )
            )

        for sensor, reading in request.sensor_readings.items():
            variants = self._thresholds.get(sensor.casefold(), {})
            rules = _applicable_rules(variants, equipment_id)
            if len(rules) != 1:
                findings.append(
                    SensorFinding(
                        sensor=sensor,
                        reading=reading,
                        status="ambiguous" if rules else "unknown",
                    )
                )
                continue

            (rule,) = rules
            polarity = 1 if rule.direction == "high" else -1
            measured = polarity * reading
            if measured >= polarity * rule.critical_threshold:
                status = "critical"
            elif measured >= polarity * rule.warning_threshold:
                status = "warning"
            else:
                status = "normal"
            findings.append(
                SensorFinding(sensor=rule.sensor, reading=reading, status=status, rule=rule)
            )
        return tuple(findings)
