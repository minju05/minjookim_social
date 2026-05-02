from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions_from_paths
from muma_gemini import build_client, prediction_to_dict, predict_question, upload_video_and_wait


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Gemini MuMA-ToM baseline with direct video input.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question offset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_gemini_predictions.jsonl"),
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset and print a short summary.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help="Override questions.json path. Defaults to GitHub corrected file.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override Gemini model name.")
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
    if args.model:
        settings = settings.__class__(**{**settings.__dict__, "gemini_model": args.model})

    dataset_dir = ensure_dataset(settings)
    questions_path = args.questions_path or (settings.github_repo_dir / "Files" / "questions.json")
    texts_path = dataset_dir / "texts.json"
    questions = load_questions_from_paths(
        dataset_dir=dataset_dir,
        questions_path=questions_path,
        texts_path=texts_path,
    )

    if args.download_only:
        print(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "questions_path": str(questions_path),
                    "num_questions": len(questions),
                },
                indent=2,
            )
        )
        return

    selected = questions[args.offset : args.offset + args.limit]
    client = build_client(settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    uploaded_by_episode: dict[str, object] = {}
    with args.output.open("w", encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM Gemini"):
            uploaded = uploaded_by_episode.get(record.episode_id)
            if uploaded is None:
                uploaded = upload_video_and_wait(client, record.video_path)
                uploaded_by_episode[record.episode_id] = uploaded

            prediction = predict_question(client, settings, record, uploaded)
            predictions.append(prediction)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")

    summary = summarize(selected, predictions)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
