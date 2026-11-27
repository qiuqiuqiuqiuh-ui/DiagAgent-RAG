import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.core.exceptions import ConfigurationError, DiagAgentError
from src.core.settings import LLMSettings, load_settings
from src.evaluation.dataset import load_dataset
from src.evaluation.replay import replay_judge


def _judge_settings(args: argparse.Namespace) -> LLMSettings:
    if args.config is not None:
        settings = load_settings(args.config)
        if settings.judge_llm is None:
            raise ConfigurationError("The selected profile must configure judge_llm")
        return settings.judge_llm

    key = os.environ.get("DIAGAGENT_JUDGE_LLM__API_KEY", "").strip()
    if not key:
        raise ConfigurationError("Set DIAGAGENT_JUDGE_LLM__API_KEY in the process environment")
    try:
        return LLMSettings(
            provider="openai_compatible",
            model=args.model,
            base_url=args.base_url,
            api_key=key,
            temperature=0,
            json_mode=True,
            timeout_seconds=90,
        )
    except ValidationError:
        raise ConfigurationError("Invalid judge model or gateway URL") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score frozen diagnostic answers")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path, action="append", default=[])
    parser.add_argument("--case-id", action="append", help="repeat to select answered cases")
    parser.add_argument(
        "--paired", action="store_true", help="score with and without request context"
    )
    parser.add_argument("--allow-same-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="path to a new replay JSON file")
    parser.add_argument("--config", type=Path, help="YAML profile with a judge_llm section")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    if args.config is not None:
        if args.model is not None or args.base_url is not None:
            parser.error("--config cannot be combined with --model or --base-url")
    elif args.model is None or args.base_url is None:
        parser.error("specify --config or both --model and --base-url")

    try:
        report = replay_judge(
            _judge_settings(args),
            answers_path=args.answers,
            dataset=load_dataset(args.dataset),
            output_path=args.output,
            evidence_paths=args.evidence_report,
            case_ids=set(args.case_id) if args.case_id else None,
            paired=args.paired,
            allow_same_model=args.allow_same_model,
        )
    except (DiagAgentError, OSError) as exc:
        print(f"Judge replay failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(any(arm["failed"] for arm in report["summary"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
