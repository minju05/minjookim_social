from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from openai import OpenAI

from muma_config import Settings
from muma_data import QuestionRecord


LETTER_RE = re.compile(r"\b([A-C])\b")


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
        f"Text: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "Answer with only A, B, or C.",
    ]
    return "\n".join(lines)


def predict_question(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
) -> Prediction:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "user", "content": _question_payload(record)},
        ],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    match = LETTER_RE.search(text.upper())
    if not match:
        raise ValueError(f"Could not parse A/B/C from response: {text}")
    letter = match.group(1)
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
        reasoning=text[:500],
    )


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    return asdict(prediction)
