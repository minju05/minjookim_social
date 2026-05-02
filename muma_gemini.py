from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from google import genai
from google.genai import types

from muma_config import Settings
from muma_data import QuestionRecord


SYSTEM_PROMPT = """You are evaluating a multi-agent theory-of-mind question from MuMA-ToM.
Use the video and text context together.
Return strict JSON with keys:
{
  "choice_letter": "A" | "B" | "C",
  "choice_text": "...",
  "reasoning": "short explanation grounded in the video and text"
}
Only return valid JSON.
"""


@dataclass(frozen=True)
class GeminiPrediction:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    reasoning: str


def build_client(settings: Settings) -> genai.Client:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=settings.gemini_api_key)


def upload_video_and_wait(client: genai.Client, video_path: Path, poll_interval: float = 2.0) -> types.File:
    uploaded = client.files.upload(file=video_path)
    while uploaded.state == types.FileState.PROCESSING:
        time.sleep(poll_interval)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state == types.FileState.FAILED:
        raise RuntimeError(f"Gemini file processing failed for {video_path}: {uploaded.error}")
    if uploaded.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Unexpected Gemini file state for {video_path}: {uploaded.state}")
    return uploaded


def _question_payload(record: QuestionRecord) -> str:
    return "\n".join(
        [
            f"Question type: {record.question_type}",
            f"Text context: {record.text_context or 'N/A'}",
            f"Question: {record.question}",
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
            "Pick the single best answer.",
        ]
    )


def predict_question(
    client: genai.Client,
    settings: Settings,
    record: QuestionRecord,
    uploaded_video: types.File,
) -> GeminiPrediction:
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[uploaded_video, _question_payload(record)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema={
                "type": "object",
                "properties": {
                    "choice_letter": {"type": "string", "enum": ["A", "B", "C"]},
                    "choice_text": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["choice_letter", "choice_text", "reasoning"],
            },
        ),
    )

    text = (response.text or "").strip()
    parsed = json.loads(text)
    letter = parsed["choice_letter"].strip().upper()
    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    return GeminiPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=parsed.get("reasoning", "").strip(),
    )


def prediction_to_dict(prediction: GeminiPrediction) -> dict[str, object]:
    return asdict(prediction)
