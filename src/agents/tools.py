import csv
from collections.abc import Callable, Sequence
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


def _read_rules(
    path: Path, parse: Callable[[str], Sequence[object]], model: type[Rule]
) -> list[Rule]:
    try:
        if path.is_symlink():
            raise CorpusError("Rule tables must reside inside the corpus directory")
        if not path.exists():
            return []
        return [
            model.model_validate(record) for record in parse(path.read_text(encoding="utf-8-sig"))
        ]
    except (OSError, ValueError, csv.Error, CorpusError) as exc:
        raise CorpusError(f"Invalid rule table {path.name}: {exc}") from exc


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
        return cls(
            _read_rules(corpus_dir / "fault_codes.csv", parse_csv_records, FaultCodeRule),
            _read_rules(
                corpus_dir / "sensor_thresholds.json", parse_json_records, SensorThresholdRule
            ),
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
