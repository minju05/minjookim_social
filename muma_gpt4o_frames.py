from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from openai import OpenAI

from muma_config import Settings
from muma_data import QuestionRecord
from muma_video import EncodedFrame


SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question about a video clip. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)


@dataclass(frozen=True)
class Prediction:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    reasoning: str


def _question_payload(record: QuestionRecord) -> str:
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def predict_question(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
) -> Prediction:
    content: list = [{"type": "text", "text": _question_payload(record)}]
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {frame.second:.1f} seconds"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    resp_json = json.loads(response.choices[0].message.content)
    letter = resp_json["choice_letter"].strip().upper()
    reasoning = resp_json.get("reasoning", "")
    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    return Prediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=reasoning[:500],
    )


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    return asdict(prediction)