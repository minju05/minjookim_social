from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_gpt4o_frames import predict_question, prediction_to_dict
from muma_video import sample_video_frames


def select_questions(questions, limit, offset, per_type):
    if per_type:
        by_type = defaultdict(list)
        for q in questions:
            by_type[q.question_type].append(q)
        result = []
        for qs in by_type.values():
            result.extend(qs[:per_type])
        return result
    return questions[offset : offset + limit]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MuMA-ToM frame-based baseline (우리 버전 prompt + video frames).")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--per-type", type=int, default=None, help="N questions per question type (overrides --limit/--offset).")
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_gpt4o_frames_ours_predictions.jsonl"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()
    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)
    selected = select_questions(questions, args.limit, args.offset, args.per_type)

    frame_stride = args.frame_stride or settings.frame_stride
    max_frames = settings.max_frames if args.max_frames is None else (
        None if args.max_frames == 0 else args.max_frames
    )

    done_ids: set[str] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["question_id"])
        print(f"Resuming: {len(done_ids)} questions already done, skipping.")

    client = OpenAI(api_key=settings.openai_api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as f:
        for record in tqdm(selected, desc="frames (우리 버전)"):
            if record.question_id in done_ids:
                continue
            frames = sample_video_frames(
                record.video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
            prediction = predict_question(client, settings, record, frames)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            f.flush()

    rows = []
    with args.output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_type_total: dict[str, int] = defaultdict(int)
    by_type_correct: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type_total[row["question_type"]] += 1
        if row["correct"]:
            by_type_correct[row["question_type"]] += 1

    summary = {
        "num_questions": len(rows),
        "overall_accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "accuracy_by_type": {
            qt: by_type_correct[qt] / total
            for qt, total in sorted(by_type_total.items())
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()