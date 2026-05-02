from __future__ import annotations

import argparse
import json
from pathlib import Path

from muma_eval import load_summary


DEFAULT_RUNS = [
    ("gpt4o", [Path("outputs/muma_gpt4o_predictions.summary.json"), Path("outputs/sanity5.summary.json"), Path("outputs/smoke_test.summary.json")]),
    (
        "gemini_2_5_pro_direct_video_text_qa",
        [Path("outputs/muma_gemini_predictions.summary.json"), Path("outputs/gemini_episode4.summary.json"), Path("outputs/gemini_smoke.summary.json")],
    ),
    ("limp", [Path("outputs/muma_limp_predictions.summary.json"), Path("outputs/limp_episode4.summary.json"), Path("outputs/limp_smoke.summary.json")]),
    ("llava_1_6_13b", [Path("outputs/muma_llava_13b.summary.json")]),
    ("llava_1_6_34b", [Path("outputs/muma_llava_34b.summary.json")]),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MuMA-ToM result summaries into a table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_results_table.md"),
        help="Output markdown path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = []
    for name, candidates in DEFAULT_RUNS:
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        summary = load_summary(path)
        rows.append(
            {
                "model": name,
                "source_file": str(path),
                "num_questions": summary["num_questions"],
                "overall_accuracy": summary["overall_accuracy"],
                "belief": summary["accuracy_by_type"].get("belief"),
                "social_goal": summary["accuracy_by_type"].get("social_goal"),
                "belief_of_goal": summary["accuracy_by_type"].get("belief_of_goal"),
            }
        )

    lines = [
        "| model | source_file | num_questions | overall_accuracy | belief | social_goal | belief_of_goal |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {source_file} | {num_questions} | {overall_accuracy:.4f} | {belief} | {social_goal} | {belief_of_goal} |".format(
                model=row["model"],
                source_file=row["source_file"],
                num_questions=row["num_questions"],
                overall_accuracy=row["overall_accuracy"],
                belief=f"{row['belief']:.4f}" if row["belief"] is not None else "-",
                social_goal=f"{row['social_goal']:.4f}" if row["social_goal"] is not None else "-",
                belief_of_goal=f"{row['belief_of_goal']:.4f}" if row["belief_of_goal"] is not None else "-",
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
