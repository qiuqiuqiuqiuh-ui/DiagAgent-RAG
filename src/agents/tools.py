import csv
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from src.agents.schemas import (
    DiagnosticRequest,
    FaultCodeFinding,
    FaultCodeRule,
    SensorFinding,
    SensorThresholdRule,
    ToolFinding,
)
from src.core.exceptions import CorpusError
from src.core.logging import log_event
from src.core.settings import FAULT_TABLE, SENSOR_TABLE, RulesSettings
from src.core.tokenizer import tokenize_identifiers
from src.ingestion.parsers import parse_csv_records, parse_json_records

Rule = TypeVar("Rule", FaultCodeRule, SensorThresholdRule)
RULE_TABLES = (FAULT_TABLE, SENSOR_TABLE)
_LOGGER = logging.getLogger("diagagent.tools")
_SENSOR_UNITS = {
    unit.casefold(): unit
    for unit in (
        "degC",
        "degF",
        "K",
        "MPa",
        "kPa",
        "Pa",
        "bar",
        "psi",
        "mm/s RMS",
        "mm/s",
        "L/min",
        "Hz",
        "rpm",
        "A",
        "V",
        "%",
    )
} | {"°c": "degC", "℃": "degC", "celsius": "degC", "°f": "degF", "℉": "degF", "fahrenheit": "degF"}
KNOWN_SENSOR_UNITS: frozenset[str] = frozenset(_SENSOR_UNITS)


def canonical_sensor_unit(unit: str) -> str | None:
    return _SENSOR_UNITS.get(unit.strip().casefold())


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
    *,
    allow_missing: bool,
) -> list[Rule]:
    if filename not in sources:
        if allow_missing:
            return []
        raise CorpusError(f"Missing required rule table {filename}")
    try:
        records = parse(sources[filename].decode("utf-8-sig"))
    except (ValueError, csv.Error, CorpusError) as exc:
        raise CorpusError(f"Invalid rule table {filename}: {exc}") from exc
    if not records:
        raise CorpusError(f"Invalid rule table {filename}: no records")
    rules = []
    for number, record in enumerate(records, start=1):
        try:
            rules.append(model.model_validate(record))
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            details = "; ".join(
                ".".join(map(str, error["loc"])) + ": " + error["msg"]
                if error["loc"]
                else error["msg"]
                for error in errors
            )
            raise CorpusError(
                f"Invalid rule table {filename}, record {number}: {details}"
            ) from None
    return rules


def _normalize_threshold(rule: SensorThresholdRule) -> SensorThresholdRule:
    unit = canonical_sensor_unit(rule.unit)
    if unit is None:
        log_event(
            _LOGGER,
            "rules.unknown_sensor_unit",
            level=logging.WARNING,
            equipment_id=rule.equipment_id,
            sensor=rule.sensor,
            unit=rule.unit,
        )
        return rule
    return rule if unit == rule.unit else rule.model_copy(update={"unit": unit})


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
        self._thresholds = _index_rules(
            tuple(map(_normalize_threshold, thresholds)), lambda rule: rule.sensor
        )
        self._equipment = {
            rule.equipment_id.casefold(): rule.equipment_id for rule in (*fault_codes, *thresholds)
        }

    @classmethod
    def from_corpus(cls, corpus_dir: Path, rules: RulesSettings | None = None) -> "DiagnosticTools":
        root = corpus_dir.resolve()
        if not root.is_dir():
            raise CorpusError(f"Corpus directory does not exist: {root}")
        configured = rules if rules is not None else RulesSettings(allow_missing_rule_tables=True)
        sources = {
            filename: content
            for filename in configured.source_paths
            if (content := _read_table(root, filename)) is not None
        }
        return cls.from_sources(sources, configured)

    @classmethod
    def from_sources(
        cls, sources: Mapping[str, bytes], rules: RulesSettings | None = None
    ) -> "DiagnosticTools":
        configured = rules if rules is not None else RulesSettings(allow_missing_rule_tables=True)
        return cls(
            _parse_rules(
                sources,
                configured.fault_codes_path,
                parse_csv_records,
                FaultCodeRule,
                allow_missing=configured.allow_missing_rule_tables,
            ),
            _parse_rules(
                sources,
                configured.sensor_thresholds_path,
                parse_json_records,
                SensorThresholdRule,
                allow_missing=configured.allow_missing_rule_tables,
            ),
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
        matches = self._equipment.keys() & set(tokenize_identifiers(request.query))
        return self._equipment[next(iter(matches))] if len(matches) == 1 else None

    def evaluate(self, request: DiagnosticRequest) -> tuple[ToolFinding, ...]:
        findings: list[ToolFinding] = []
        terms = tokenize_identifiers(request.query)
        equipment_id = self.equipment_for(request)
        codes = (
            (request.fault_code,)
            if request.fault_code is not None
            else tuple(code for code in terms if code in self._fault_codes)
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
