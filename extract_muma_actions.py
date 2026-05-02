from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from muma_action_extraction import build_client, extract_action_text, upload_video_and_wait
from muma_config import load_settings
from muma_data import ensure_dataset, load_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MuMA-ToM visual actions with Gemini.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of episodes to process.")
    parser.add_argument("--offset", type=int, default=0, help="Episode offset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/actions_extracted_gemini.json"),
        help="Output json path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional Gemini video FPS metadata override.",
    )
    parser.add_argument(
        "--include-extra-prompts",
        action="store_true",
        help="Include prompt ids that are not part of the 225 benchmark episodes.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override Gemini model name.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()
    if args.model:
        settings = settings.__class__(**{**settings.__dict__, "gemini_model": args.model})
    dataset_dir = ensure_dataset(settings)
    client = build_client(settings)

    prompt_path = settings.github_repo_dir / "Files" / "actions_extracted.json"
    question_path = settings.github_repo_dir / "Files" / "questions.json"
    prompt_data = load_json(prompt_path)
    question_data = load_json(question_path)

    benchmark_ids = set(question_data.keys())
    all_ids = sorted(prompt_data.keys(), key=lambda x: int(x))
    selected_ids = [eid for eid in all_ids if args.include_extra_prompts or eid in benchmark_ids]
    selected_ids = selected_ids[args.offset : (args.offset + args.limit) if args.limit is not None else None]

    uploaded_by_episode: dict[str, object] = {}
    output_data: dict[str, dict[str, str]] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for episode_id in tqdm(selected_ids, desc="Extract actions"):
        video_path = dataset_dir / "videos" / f"video_{episode_id}.mp4"
        if not video_path.exists():
            continue

        uploaded = uploaded_by_episode.get(episode_id)
        if uploaded is None:
            uploaded = upload_video_and_wait(client, video_path)
            uploaded_by_episode[episode_id] = uploaded

        prompt = prompt_data[episode_id]["prompt"]
        action_text = extract_action_text(client, settings, uploaded, prompt, fps=args.fps)
        output_data[episode_id] = {"prompt": prompt, "action": action_text}

    args.output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": len(output_data), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
