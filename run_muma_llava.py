from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions_from_paths
from muma_eval import summarize_predictions, write_jsonl, write_summary
from muma_image_frames import sample_video_frames_pil


LLAVA_MODEL_MAP = {
    "13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    "34b": "llava-hf/llava-v1.6-34b-hf",
}
LETTER_RE = re.compile(r"\b([A-C])\b")


@dataclass(frozen=True)
class LlavaPrediction:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    reasoning: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an LLaVA-1.6 baseline on MuMA-ToM.")
    parser.add_argument("--variant", choices=["13b", "34b"], required=True, help="LLaVA 1.6 checkpoint size.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question offset.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=20,
        help="Frame stride. Paper-faithful default is every 20 frames for non-video-native models.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on sampled frames. Leave unset for paper-faithful every-20-frames sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Defaults to outputs/muma_llava_<variant>.jsonl",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help="Override questions.json path. Defaults to GitHub corrected file.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", type=str, default="auto", help="torch_dtype argument.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load the model with 8-bit quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generation length.")
    return parser


def load_llava(variant: str, device: str, dtype: str, load_in_4bit: bool, load_in_8bit: bool):
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LLaVA baseline requires optional dependencies. Install torch, transformers, accelerate, and bitsandbytes first."
        ) from exc

    model_id = LLAVA_MODEL_MAP[variant]
    torch_dtype = dtype
    if dtype == "auto":
        torch_dtype = "auto"
    elif hasattr(torch, dtype):
        torch_dtype = getattr(torch, dtype)

    processor = AutoProcessor.from_pretrained(model_id)
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    return processor, model


def _build_prompt(record) -> str:
    return "\n".join(
        [
            f"Text: {record.text_context or 'N/A'}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            "Answer with only A, B, or C.",
        ]
    )


def predict_question(processor, model, record, frames, max_new_tokens: int) -> LlavaPrediction:
    prompt = _build_prompt(record)
    conversation = [
        {
            "role": "user",
            "content": ([{"type": "image"}] * len(frames)) + [{"type": "text", "text": prompt}],
        }
    ]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=frames, text=prompt_text, return_tensors="pt")

    if hasattr(model, "device") and str(model.device) != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer_text = decoded.split(prompt_text, 1)[-1].strip()

    match = LETTER_RE.search(answer_text.upper())
    if not match:
        raise ValueError(f"Could not parse A/B/C from LLaVA response: {answer_text}")
    letter = match.group(1)
    reasoning = answer_text[:500]

    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]
    return LlavaPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=reasoning,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()
    dataset_dir = ensure_dataset(settings)
    questions_path = args.questions_path or (settings.github_repo_dir / "Files" / "questions.json")
    questions = load_questions_from_paths(
        dataset_dir=dataset_dir,
        questions_path=questions_path,
        texts_path=dataset_dir / "texts.json",
    )
    selected = questions[args.offset : args.offset + args.limit]

    processor, model = load_llava(
        args.variant,
        args.device,
        args.dtype,
        args.load_in_4bit,
        args.load_in_8bit,
    )

    output_path = args.output or Path(f"outputs/muma_llava_{args.variant}.jsonl")
    predictions = []
    for record in tqdm(selected, desc=f"MuMA-ToM LLaVA-{args.variant}"):
        frames = sample_video_frames_pil(
            record.video_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        prediction = predict_question(processor, model, record, frames, args.max_new_tokens)
        predictions.append(prediction)

    write_jsonl(output_path, [asdict(p) for p in predictions])
    summary = summarize_predictions(selected, predictions)
    write_summary(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
