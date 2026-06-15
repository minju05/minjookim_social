"""run_muma_internvl.py — MuMA-ToM InternVL3-14B local inference

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_muma_internvl.py --prompt-version text_only --no-frames --limit 900 --output outputs/internvl_cond1_textonly_900.jsonl
    CUDA_VISIBLE_DEVICES=0 python run_muma_internvl.py --prompt-version video_only --limit 900 --output outputs/internvl_cond4_videoonly_900.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_video import sample_video_frames

DEFAULT_MODEL_PATH = "/home/mindrium-admin3/models/InternVL3-14B"
HF_MODEL_ID = "OpenGVLab/InternVL3-14B"

# ── Image preprocessing (official InternVL3 recipe) ──────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    img_r = image.resize((tw, th))
    tiles = []
    for i in range(blocks):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        tiles.append(img_r.crop(box))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def pil_to_pixel_values(pil_img: Image.Image, max_num: int = 6, image_size: int = 448) -> torch.Tensor:
    transform = build_transform(image_size)
    tiles = dynamic_preprocess(pil_img, max_num=max_num, image_size=image_size, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles])


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ── Prompt builders (same logic as run_muma_qwen.py) ─────────────────────────

def build_payload(record, prompt_version: str) -> str:
    if prompt_version == "text_only":
        lines = [
            f"Question type: {record.question_type}",
            f"Text context: {record.text_context or 'N/A'}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        ]
    elif prompt_version in ("video_only", "qa_only"):
        lines = [
            f"Question type: {record.question_type}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        ]
    elif prompt_version == "generic_cot":
        lines = [
            f"Question type: {record.question_type}",
            f"Text context: {record.text_context or 'N/A'}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            "",
            "Think step by step and choose the best answer.",
            "",
            'Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        ]
    elif prompt_version == "qlogic":
        lines = [
            f"Question type: {record.question_type}",
            "",
            f"Text context: {record.text_context or 'N/A'}",
            "",
            f"Question: {record.question}",
            "",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            "",
            "Solve the question by explicitly analyzing its logic.",
            "",
            "Step 1. Identify whether the question asks for MOST likely or LEAST likely.",
            "",
            "Step 2. Identify the explicit condition in the question.",
            "Examples of conditions include:",
            "- whether a character is trying to help another character",
            "- whether a character is trying to hinder another character",
            "- whether a character knows what is inside a location",
            "- whether the answer should be inferred based on the agents' actions",
            "",
            "Step 3. Identify the target character whose belief, belief about another character's goal, or social goal is being evaluated.",
            "",
            "Step 4. Identify the key inference needed.",
            "Useful inference rules:",
            "- If a character knowingly gives false location information, this may indicate hindering rather than helping.",
            "- If a character gives true and useful location information, this may indicate helping rather than hindering.",
            "- If an agent placed an object somewhere and another agent moves it away, infer whether the mover believes the original location was the first agent's desired location.",
            "- For belief-of-goal questions, distinguish the other agent's actual goal from what the target character believes about that goal.",
            "- For LEAST likely questions, choose the option least consistent with the condition and inferred mental state.",
            "- For MOST likely questions, choose the option most consistent with the condition and inferred mental state.",
            "",
            "Additional targeted checks:",
            "- For belief questions with a LEAST likely polarity and a hindering condition, focus on the belief about the target object mentioned in the question, not on irrelevant objects. If the speaker gave a location for the target object and that location later appears false, then it is least likely that the speaker truly believed the target object was at the stated location.",
            "- For social-goal questions involving information about a target object's location, do not assume that giving information means helping. First compare the stated location with the observed outcome. If the speaker is assumed to know the relevant location and the target object is not found at the stated location, treat the statement as potentially misleading. Then decide whether the social goal is helping, obstructing, or indifference.",
            "",
            "Step 5. Evaluate each option independently:",
            "- A: Is this consistent with the question condition and the target character's mental state?",
            "- B: Is this consistent with the question condition and the target character's mental state?",
            "- C: Is this consistent with the question condition and the target character's mental state?",
            "",
            "Step 6. Choose the final answer according to the MOST/LEAST polarity.",
            "",
            "Respond in JSON:",
            "{",
            '  "question_polarity": "MOST likely or LEAST likely",',
            '  "condition": "...",',
            '  "target_character": "...",',
            '  "key_inference": "...",',
            '  "option_analysis": {',
            '    "A": "...",',
            '    "B": "...",',
            '    "C": "..."',
            "  },",
            '  "choice_letter": "A",',
            '  "reasoning": "..."',
            "}",
        ]
    else:
        lines = [
            f"Question type: {record.question_type}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        ]
    return "\n".join(lines)


# ── InternVL3 inference helpers ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the multiple-choice question carefully. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)


def build_internvl_input(record, frames, prompt_version: str):
    """Return (pixel_values | None, num_patches_list, question_str)."""
    text_payload = build_payload(record, prompt_version)

    if not frames:
        return None, [], text_payload

    pil_images = [b64_to_pil(f.image_b64) for f in frames]
    max_num_per_frame = max(1, 12 // len(pil_images)) if len(pil_images) <= 12 else 1
    max_num_per_frame = min(max_num_per_frame, 4)  # cap to save VRAM

    all_pv = []
    num_patches_list = []
    for pil in pil_images:
        pv = pil_to_pixel_values(pil, max_num=max_num_per_frame)
        all_pv.append(pv)
        num_patches_list.append(pv.size(0))

    pixel_values = torch.cat(all_pv, dim=0)

    # Build question with <image> tokens
    image_tokens = "\n".join(f"Frame {i+1}: <image>" for i in range(len(frames)))
    question_str = f"{image_tokens}\n\n{text_payload}"

    return pixel_values, num_patches_list, question_str


def parse_response(text: str) -> tuple[str, str]:
    text = text.strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            letter = d.get("choice_letter", "").strip().upper()
            if letter in ("A", "B", "C"):
                return letter, d.get("reasoning", "")
        except Exception:
            pass
    m2 = re.search(r'\b([ABC])\b', text)
    if m2:
        return m2.group(1), text
    return "", text


def letter_to_answer(record, letter: str) -> str:
    idx = {"A": 0, "B": 1, "C": 2}.get(letter)
    if idx is not None and idx < len(record.choices):
        return record.choices[idx]
    return ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MuMA-ToM with InternVL3-14B (local).")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/internvl_predictions.jsonl"))
    parser.add_argument(
        "--prompt-version", type=str, default="text_only",
        choices=["text_only", "video_only", "qa_only", "generic_cot", "qlogic"],
    )
    parser.add_argument("--no-frames", action="store_true", help="Skip frame sampling.")
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=f"모델 경로 또는 HF repo ID. 기본: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--append", action="store_true", help="기존 output에 이어서 쓰기.")
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()

    model_path = args.model_path or DEFAULT_MODEL_PATH
    # Fall back to HF download if local path does not exist
    if not Path(model_path).exists():
        print(f"[internvl] Local path {model_path} not found — loading from HF: {HF_MODEL_ID}")
        model_path = HF_MODEL_ID

    print(f"[internvl] Loading model from {model_path} ...")
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.system_message = SYSTEM_PROMPT
    print("[internvl] Model loaded.")

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)
    selected = questions[args.offset: args.offset + args.limit]

    frame_stride = args.frame_stride or settings.frame_stride
    max_frames = settings.max_frames if args.max_frames is None else (None if args.max_frames == 0 else args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.append else "w"

    results = []
    with args.output.open(file_mode, encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM InternVL3"):
            if args.no_frames:
                frames = []
            else:
                try:
                    frames = sample_video_frames(
                        record.video_path,
                        frame_stride=frame_stride,
                        max_frames=max_frames,
                    )
                except Exception as e:
                    print(f"\n[WARN] frame sampling failed for {record.question_id}: {e}")
                    frames = []

            try:
                pixel_values, num_patches_list, question_str = build_internvl_input(
                    record, frames, args.prompt_version
                )

                if pixel_values is not None:
                    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

                raw_text = model.chat(
                    tokenizer,
                    pixel_values,
                    question_str,
                    generation_config,
                    num_patches_list=num_patches_list if num_patches_list else None,
                    history=None,
                    return_history=False,
                )

                letter, reasoning = parse_response(raw_text)

            except torch.cuda.OutOfMemoryError as oom:
                print(f"\n[OOM] {record.question_id}: {oom} — ERR로 저장하고 계속")
                torch.cuda.empty_cache()
                letter, reasoning, raw_text = "ERR", f"OOM: {oom}", ""
            except Exception as e:
                print(f"\n[ERROR] {record.question_id}: {e}")
                letter, reasoning, raw_text = "ERR", f"ERROR: {e}", ""

            predicted_answer = letter_to_answer(record, letter) if letter not in ("ERR", "") else "ERR"
            correct = (predicted_answer == record.answer) if letter not in ("ERR", "") else False

            row = {
                "question_id": record.question_id,
                "episode_id": record.episode_id,
                "question_type": record.question_type,
                "gold_answer": record.answer,
                "predicted_answer": predicted_answer,
                "predicted_letter": letter,
                "correct": correct,
                "reasoning": reasoning,
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            torch.cuda.empty_cache()

    total = len(results)
    correct_n = sum(r["correct"] for r in results)
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["correct"])

    summary = {
        "num_questions": total,
        "overall_accuracy": correct_n / total if total else 0.0,
        "accuracy_by_type": {
            qt: sum(v) / len(v) for qt, v in sorted(by_type.items())
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
