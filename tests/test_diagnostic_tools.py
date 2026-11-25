import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.schemas import DiagnosticRequest, FaultCodeFinding, SensorFinding
from src.agents.tools import DiagnosticTools
from src.core.exceptions import CorpusError

CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_fault_lookup_uses_equipment_and_recognizes_full_code_in_query() -> None:
    tools = DiagnosticTools.from_corpus(CORPUS)
    (found,) = tools.evaluate(DiagnosticRequest(query="Diagnose vfd-oc3", equipment_id="drv-300"))
    assert isinstance(found, FaultCodeFinding)
    assert found.status == "found"
    assert found.rule is not None
    assert found.rule.meaning == "Acceleration overcurrent"

    (unknown,) = tools.evaluate(
        DiagnosticRequest(query="Diagnose this drive", fault_code="VFD-OC3", equipment_id="MX-200")
    )
    assert unknown.status == "unknown"
    assert unknown.rule is None


def test_uses_unambiguous_equipment_code_from_query_for_sensor_lookup() -> None:
    (finding,) = DiagnosticTools.from_corpus(CORPUS).evaluate(
        DiagnosticRequest(query="Inspect PM-14", sensor_readings={"winding_temp": 118})
    )
    assert isinstance(finding, SensorFinding)
    assert finding.status == "critical"
    assert finding.rule is not None and finding.rule.equipment_id == "PM-14"


@pytest.mark.parametrize(
    ("asset", "sensor", "reading", "expected"),
    [
        ("mx-200", "WINDING_TEMP", 104.9, "normal"),
        ("MX-200", "winding_temp", 105, "warning"),
        ("MX-200", "winding_temp", 120, "critical"),
        ("PM-14", "winding_temp", 118, "critical"),
        ("MX-200", "winding_temp", 118, "warning"),
        ("CV-40", "gearbox_oil_pressure", 1.81, "normal"),
        ("CV-40", "gearbox_oil_pressure", 1.8, "warning"),
        ("CV-40", "gearbox_oil_pressure", 1.4, "critical"),
        (None, "winding_temp", 118, "ambiguous"),
        ("CV-40", "winding_temp", 118, "unknown"),
    ],
)
def test_sensor_thresholds_use_correct_asset_direction_and_inclusive_boundaries(
    asset: str | None, sensor: str, reading: float, expected: str
) -> None:
    (finding,) = DiagnosticTools.from_corpus(CORPUS).evaluate(
        DiagnosticRequest(
            query="Check the sensor", equipment_id=asset, sensor_readings={sensor: reading}
        )
    )
    assert isinstance(finding, SensorFinding)
    assert finding.status == expected
    assert finding.reading == reading
    if expected not in ("unknown", "ambiguous"):
        assert finding.rule is not None
        assert finding.rule.equipment_id.casefold() == asset.casefold()
    else:
        assert finding.rule is None


def test_rule_loader_rejects_bad_threshold_order_and_duplicate_rules(tmp_path: Path) -> None:
    path = tmp_path / "sensor_thresholds.json"
    path.write_text(
        '[{"equipment_id":"MX-200","sensor":"temp","unit":"degC",'
        '"warning_threshold":100,"critical_threshold":90,"direction":"high"}]',
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="Critical threshold must be beyond"):
        DiagnosticTools.from_corpus(tmp_path)
    rows = (CORPUS / "fault_codes.csv").read_text(encoding="utf-8").splitlines()
    (tmp_path / "fault_codes.csv").write_text("\n".join([*rows, rows[1]]), encoding="utf-8")
    path.unlink()
    with pytest.raises(CorpusError, match="Duplicate rule for VFD-OC3"):
        DiagnosticTools.from_corpus(tmp_path)


def test_rule_loader_requires_existing_corpus_root(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="Corpus directory does not exist"):
        DiagnosticTools.from_corpus(tmp_path / "missing")
    assert DiagnosticTools.from_corpus(tmp_path).evaluate(DiagnosticRequest(query="Motor")) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"warning_threshold": False, "critical_threshold": True},
        {"warning_threshold": float("nan")},
        {"critical_threshold": float("inf")},
        {"warning_threshhold": 105},
    ],
)
def test_sensor_rules_reject_invalid_values_and_unknown_fields(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    record = {
        "equipment_id": "MX-200",
        "sensor": "temp",
        "unit": "degC",
        "warning_threshold": 105,
        "critical_threshold": 120,
        "direction": "high",
    }
    path = tmp_path / "sensor_thresholds.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    tools = DiagnosticTools.from_corpus(tmp_path)
    (finding,) = tools.evaluate(DiagnosticRequest(query="MX-200", sensor_readings={"temp": 105}))
    assert finding.status == "warning"
    path.write_text(json.dumps([{**record, **overrides}]), encoding="utf-8")
    with pytest.raises(CorpusError, match="Invalid rule table sensor_thresholds.json"):
        DiagnosticTools.from_corpus(tmp_path)


def test_fault_rule_rejects_extra_csv_column(tmp_path: Path) -> None:
    (tmp_path / "fault_codes.csv").write_text(
        "equipment_id,fault_code,severity,meaning,first_check,escalation,typo\n"
        "A1,F1,alarm,sample,inspect,notify,x\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="Invalid rule table fault_codes.csv"):
        DiagnosticTools.from_corpus(tmp_path)


@pytest.mark.parametrize(
    "data",
    [
        {"query": "  "},
        {"query": "Motor", "sensor_readings": {"temperature": float("nan")}},
        {"query": "Motor", "sensor_readings": {"temperature": True}},
        {"query": "Motor", "sensor_readings": {"Temp": 100, " temp ": 120}},
    ],
)
def test_request_rejects_invalid_user_inputs(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DiagnosticRequest.model_validate(data)
