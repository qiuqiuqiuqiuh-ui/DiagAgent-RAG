import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from src.core.exceptions import DiagAgentError
from src.evaluation.frozen import Mode
from src.evaluation.gate import GateReplayReport, check_eval_gate
from src.evaluation.runner import EvaluationReport


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check evaluation thresholds for a retrieval mode")
    parser.add_argument("--report", type=Path, required=True, help="JSON from scripts/evaluate.py")
    parser.add_argument("--mode", choices=get_args(Mode), default="hybrid_rerank")
    parser.add_argument("--replay", type=Path, help="matching JSON from scripts/replay_judge.py")
    args = parser.parse_args(argv)

    try:
        report = EvaluationReport.model_validate_json(args.report.read_bytes())
        replay = (
            GateReplayReport.model_validate_json(args.replay.read_bytes()) if args.replay else None
        )
        result = check_eval_gate(report, args.mode, replay=replay)
    except ValidationError:
        result = {"status": "FAIL", "error": "Invalid evaluation or replay report structure"}
    except (DiagAgentError, OSError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
