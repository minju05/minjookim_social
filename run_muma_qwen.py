"""run_muma_qwen.py — MuMA-ToM Qwen2.5-VL local inference

Usage:
    CUDA_VISIBLE_DEVICES=2 python run_muma_qwen.py --prompt-version text_only --no-frames --limit 900 --output outputs/qwen_cond1_textonly_900.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from muma_data import ensure_dataset, load_questions
from muma_config import load_settings
from muma_video import sample_video_frames

DEFAULT_MODEL_PATH = "/home/mindrium-admin3/models/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT_TEXT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question based on the text context. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)

SYSTEM_PROMPT_VIDEO = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question about a video clip. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MuMA-ToM with Qwen2.5-VL (local).")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen_predictions.jsonl"))
    parser.add_argument("--prompt-version", type=str, default="text_only",
                        choices=["text_only", "video_only", "qa_only", "generic_cot", "qlogic"],
                        help="text_only: GT text context / video_only: frames only / qa_only: question only / generic_cot: plain CoT / qlogic: QLogic v2 targeted")
    parser.add_argument("--no-frames", action="store_true", help="Skip frame sampling.")
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="모델 경로 또는 HuggingFace repo ID. 미지정 시 기본 7B 경로 사용.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 output 파일에 이어서 쓰기.",
    )
    return parser


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
    elif prompt_version == "qa_only":
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
    else:  # video_only
        lines = [
            f"Question type: {record.question_type}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        ]
    return "\n".join(lines)


def build_messages(record, frames: list, prompt_version: str) -> list[dict]:
    text_versions = {"text_only", "generic_cot", "qlogic"}
    system_prompt = SYSTEM_PROMPT_TEXT if prompt_version in text_versions else SYSTEM_PROMPT_VIDEO
    text_payload = build_payload(record, prompt_version)

    content = []
    for frame in frames:
        # EncodedFrame(second, image_b64) — 이미 base64 인코딩된 상태
        content.append({"type": "image", "image": f"data:image/jpeg;base64,{frame.image_b64}"})
    content.append({"type": "text", "text": text_payload})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_response(text: str) -> tuple[str, str]:
    """Return (choice_letter, reasoning). Fallback to raw text if JSON fails."""
    import re
    text = text.strip()
    # Try to extract JSON
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            letter = d.get("choice_letter", "").strip().upper()
            if letter in ("A", "B", "C"):
                return letter, d.get("reasoning", "")
        except Exception:
            pass
    # Fallback: look for bare A/B/C
    m2 = re.search(r'\b([ABC])\b', text)
    if m2:
        return m2.group(1), text
    return "", text


def letter_to_answer(record, letter: str) -> str:
    mapping = {"A": 0, "B": 1, "C": 2}
    idx = mapping.get(letter)
    if idx is not None and idx < len(record.choices):
        return record.choices[idx]
    return ""


def summarize(records, results) -> dict:
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_correct: dict[str, int] = defaultdict(int)
    for rec, res in zip(records, results):
        by_type_total[rec.question_type] += 1
        if res["correct"]:
            by_type_correct[rec.question_type] += 1
    total = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "num_questions": total,
        "overall_accuracy": correct / total if total else 0.0,
        "accuracy_by_type": {
            qt: by_type_correct[qt] / by_type_total[qt]
            for qt in sorted(by_type_total)
        },
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()

    model_path = args.model_path or DEFAULT_MODEL_PATH
    print(f"[qwen] Loading model from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",  # AWQ/BnB 모두 호환
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print("[qwen] Model loaded.")

    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)
    selected = questions[args.offset: args.offset + args.limit]

    frame_stride = args.frame_stride or settings.frame_stride
    max_frames = settings.max_frames if args.max_frames is None else (None if args.max_frames == 0 else args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.append else "w"

    results = []
    with args.output.open(file_mode, encoding="utf-8") as f:
        for record in tqdm(selected, desc="MuMA-ToM Qwen"):
            if args.no_frames:
                frames = []
            else:
                frames = sample_video_frames(
                    record.video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                )

            try:
                messages = build_messages(record, frames, args.prompt_version)

                from qwen_vl_utils import process_vision_info
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                raw_text = processor.decode(generated, skip_special_tokens=True)

                letter, reasoning = parse_response(raw_text)
            except torch.cuda.OutOfMemoryError as oom:
                print(f"\n[OOM] {record.question_id}: {oom} — ERR로 저장하고 계속")
                torch.cuda.empty_cache()
                letter, reasoning, raw_text = "ERR", f"OOM: {oom}", ""
            except Exception as e:
                print(f"\n[ERROR] {record.question_id}: {e}")
                letter, reasoning, raw_text = "ERR", f"ERROR: {e}", ""
            predicted_answer = letter_to_answer(record, letter) if letter != "ERR" else "ERR"
            correct = predicted_answer == record.answer if letter != "ERR" else False

            row = {
                "question_id": record.question_id,
                "episode_id": record.episode_id,
                "question_type": record.question_type,
                "gold_answer": record.answer,
                "predicted_answer": predicted_answer,
                "predicted_letter": letter,
                "correct": correct,
                "reasoning": reasoning,
                "raw_output": raw_text,
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    summary = summarize(selected, results)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
