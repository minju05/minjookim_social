"""
run_baseline_review.py
======================
baseline_review_2call 실험 CLI runner.

사용 예:
  python run_baseline_review.py --limit 900 --output outputs/cond_baseline_review_900.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_baseline_review import load_direct_cache, run_baseline_review_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline_review_2call experiment.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of questions to run. (default: 10)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Question offset. (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cond_baseline_review.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--load-direct",
        type=Path,
        default=None,
        help="기존 text_only JSONL 경로. 지정 시 1차 API 호출 스킵 (예: outputs/cond1_textonly_900.jsonl).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset and print a short summary.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()

    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)

    if args.download_only:
        print(
            json.dumps(
                {"dataset_dir": str(dataset_dir), "num_questions": len(questions)},
                indent=2,
            )
        )
        return

    selected = questions[args.offset : args.offset + args.limit]
    client = OpenAI(api_key=settings.openai_api_key)

    # 1차 캐시 로드 (--load-direct 지정 시)
    direct_cache = None
    if args.load_direct:
        direct_cache = load_direct_cache(args.load_direct)
        print(f"Loaded direct cache: {len(direct_cache)} entries from {args.load_direct}")

    summary = run_baseline_review_batch(
        client=client,
        settings=settings,
        records=selected,
        output_path=args.output,
        direct_cache=direct_cache,
    )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
