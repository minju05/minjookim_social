from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from muma_action_extraction import build_client as build_gemini_client
from muma_action_extraction import extract_action_text, upload_video_and_wait
from muma_config import load_settings
from muma_data import ensure_dataset, load_json, load_questions_from_paths
from muma_limp import answer_question, build_client as build_openai_client
from muma_limp import prediction_to_dict


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MuMA-ToM LIMP with Gemini visual extraction and GPT-4o reasoning.")
    parser.add_argument("--limit", type=int, default=4, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question offset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_limp_predictions.jsonl"),
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--actions-cache",
        type=Path,
        default=Path("outputs/actions_extracted_gemini.json"),
        help="Path to cached extracted actions. Missing episodes will be extracted and the cache will be updated.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional Gemini video FPS metadata override.",
    )
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model name.")
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
    if args.gemini_model:
        settings = settings.__class__(**{**settings.__dict__, "gemini_model": args.gemini_model})
    dataset_dir = ensure_dataset(settings)

    questions = load_questions_from_paths(
        dataset_dir=dataset_dir,
        questions_path=settings.github_repo_dir / "Files" / "questions.json",
        texts_path=dataset_dir / "texts.json",
    )
    selected = questions[args.offset : args.offset + args.limit]
    actions_prompt_data = load_json(settings.github_repo_dir / "Files" / "actions_extracted.json")

    actions_cache = {}
    if args.actions_cache.exists():
        actions_cache = load_json(args.actions_cache)

    openai_client = build_openai_client(settings.openai_api_key)
    gemini_client = build_gemini_client(settings)
    uploaded_by_episode: dict[str, object] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.actions_cache.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    with args.output.open("w", encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM LIMP"):
            cached = actions_cache.get(record.episode_id)
            raw_action_text = None
            if isinstance(cached, dict):
                raw_action_text = cached.get("action")

            if not raw_action_text:
                uploaded = uploaded_by_episode.get(record.episode_id)
                if uploaded is None:
                    uploaded = upload_video_and_wait(gemini_client, record.video_path)
                    uploaded_by_episode[record.episode_id] = uploaded

                prompt = actions_prompt_data[record.episode_id]["prompt"]
                raw_action_text = extract_action_text(
                    gemini_client,
                    settings,
                    uploaded,
                    prompt,
                    fps=args.fps,
                )
                actions_cache[record.episode_id] = {"prompt": prompt, "action": raw_action_text}
                args.actions_cache.write_text(json.dumps(actions_cache, ensure_ascii=False, indent=2), encoding="utf-8")

            prediction = answer_question(openai_client, record, raw_action_text)
            predictions.append(prediction)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")

    summary = summarize(selected, predictions)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
