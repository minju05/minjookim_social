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
from muma_limp import build_client as build_openai_client
from muma_limp_a import answer_question_a, prediction_to_dict


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
    parser = argparse.ArgumentParser(
        description="Run MuMA-ToM LIMP+A (Module A: Explicit Recursive Belief Estimation)."
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Select N questions per question type (overrides --limit/--offset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_limp_a_predictions.jsonl"),
    )
    parser.add_argument(
        "--actions-cache",
        type=Path,
        default=Path("outputs/actions_extracted_gemini.json"),
        help="Local cache of Gemini-extracted action text per episode.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gemini-model", type=str, default=None)
    return parser


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
    selected = select_questions(questions, args.limit, args.offset, args.per_type)
    actions_prompt_data = load_json(settings.github_repo_dir / "Files" / "actions_extracted.json")

    actions_cache: dict[str, dict[str, str]] = {}
    if args.actions_cache.exists():
        actions_cache = load_json(args.actions_cache)

    openai_client = build_openai_client(settings.openai_api_key)
    gemini_client = build_gemini_client(settings)
    uploaded_by_episode: dict[str, object] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.actions_cache.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["question_id"])
        print(f"Resuming: {len(done_ids)} questions already done, skipping.")

    with args.output.open("a", encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM LIMP+A"):
            if record.question_id in done_ids:
                continue

            cached = actions_cache.get(record.episode_id)
            raw_action_text = None
            if isinstance(cached, dict):
                raw_action_text = cached.get("action")

            if not raw_action_text:
                try:
                    uploaded = uploaded_by_episode.get(record.episode_id)
                    if uploaded is None:
                        uploaded = upload_video_and_wait(gemini_client, record.video_path)
                        uploaded_by_episode[record.episode_id] = uploaded

                    prompt = actions_prompt_data[record.episode_id]["prompt"]
                    raw_action_text = extract_action_text(
                        gemini_client, settings, uploaded, prompt, fps=args.fps
                    )
                    actions_cache[record.episode_id] = {"prompt": prompt, "action": raw_action_text}
                    args.actions_cache.write_text(
                        json.dumps(actions_cache, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception as e:
                    tqdm.write(f"[skip] episode {record.episode_id} Gemini failed, will retry next run: {e}")
                    continue

            prediction = answer_question_a(openai_client, record, raw_action_text)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            f.flush()

    final_predictions = []
    with args.output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                final_predictions.append(json.loads(line))

    by_type_total: dict[str, int] = defaultdict(int)
    by_type_correct: dict[str, int] = defaultdict(int)
    for row in final_predictions:
        by_type_total[row["question_type"]] += 1
        if row["correct"]:
            by_type_correct[row["question_type"]] += 1

    summary = {
        "num_questions": len(final_predictions),
        "overall_accuracy": (
            sum(r["correct"] for r in final_predictions) / len(final_predictions)
            if final_predictions
            else 0.0
        ),
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