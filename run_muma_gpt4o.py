from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_gpt4o import prediction_to_dict, predict_question
from muma_video import sample_video_frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GPT-4o MuMA-ToM baseline.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question offset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_gpt4o_predictions.jsonl"),
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset and print a short summary.",
    )
    parser.add_argument("--frame-stride", type=int, default=None, help="Override frame stride.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override max frames per question. Leave unset to use the configured default, or pass 0 for no cap.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v2_ours",
        help="Prompt version key (e.g. v2_ours, text_only).",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip frame sampling and send text only.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="콤마 구분 question_type 필터 (예: belief_of_goal,social_goal). 미지정 시 전체.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 output 파일에 이어서 쓰기 (덮어쓰기 대신 append).",
    )
    return parser


def summarize(records, predictions) -> dict[str, object]:
    by_type_total = defaultdict(int)
    by_type_correct = defaultdict(int)

    for record, prediction in zip(records, predictions):
        by_type_total[record.question_type] += 1
        if prediction.correct:
            by_type_correct[record.question_type] += 1

    summary = {
        "num_questions": len(predictions),
        "overall_accuracy": (sum(p.correct for p in predictions) / len(predictions)) if predictions else 0.0,
        "accuracy_by_type": {},
    }
    for question_type, total in sorted(by_type_total.items()):
        summary["accuracy_by_type"][question_type] = by_type_correct[question_type] / total
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()

    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)

    if args.download_only:
        print(json.dumps({"dataset_dir": str(dataset_dir), "num_questions": len(questions)}, indent=2))
        return

    # --question-types 필터 적용
    if args.question_types:
        allowed = {t.strip() for t in args.question_types.split(",")}
        questions = [q for q in questions if q.question_type in allowed]

    selected = questions[args.offset : args.offset + args.limit]
    client = OpenAI(api_key=settings.openai_api_key)
    frame_stride = args.frame_stride or settings.frame_stride
    max_frames = settings.max_frames if args.max_frames is None else (None if args.max_frames == 0 else args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    file_mode = "a" if args.append else "w"
    with args.output.open(file_mode, encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM GPT-4o"):
            if args.no_frames:
                frames = []
            else:
                frames = sample_video_frames(
                    record.video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                )
            prediction = predict_question(client, settings, record, frames, prompt_version=args.prompt_version)
            predictions.append(prediction)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            f.flush()

    summary = summarize(selected, predictions)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
