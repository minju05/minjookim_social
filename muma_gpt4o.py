from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from openai import OpenAI

from muma_config import Settings
from muma_data import QuestionRecord
from muma_video import EncodedFrame


# ============================================================
# 민주 버전 (minju05/minjookim_social, 논문 제출 버전) — 57.2%
# detail=low 조건에서 실험한 결과
# ============================================================
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


# --- 민주 버전 프롬프트 ---
# 특징:
#   - system 메시지 없음 (user 메시지만)
#   - question_type 미포함
#   - "Answer with only A, B, or C." 단순 지시
#   - 출력: regex로 A/B/C 추출 (LETTER_RE)
#   - response_format 없음 (자유 텍스트)
#
# def _question_payload(record: QuestionRecord) -> str:
#     lines = [
#         f"Text: {record.text_context or 'N/A'}",
#         f"Question: {record.question}",
#         f"A) {record.choices[0]}",
#         f"B) {record.choices[1]}",
#         f"C) {record.choices[2]}",
#         "Answer with only A, B, or C.",
#     ]
#     return "\n".join(lines)
#
# messages=[{"role": "user", "content": content}]  # system 없음
# text = response.choices[0].message.content.strip()
# match = LETTER_RE.search(text.upper())
# letter = match.group(1)


# ============================================================
# 우리 버전 (수정 버전) — 70.1%  (+12.9%p)
# detail=low 조건, 900개 결과: outputs/muma_gpt4o_low_detail.jsonl
# ============================================================
# 특징:
#   - SYSTEM_PROMPT 추가 (theory of mind 전문가 역할)
#   - question_type을 프롬프트에 명시 → GPT가 문제 유형 인지
#   - JSON 출력 강제 (response_format={"type": "json_object"})
#   - 출력: {"choice_letter": "B", "reasoning": "..."} 파싱
#   - Pick the single best answer 지시
#
# SYSTEM_PROMPT = (
#     "You are an expert in theory of mind and social reasoning. "
#     "Answer the following multiple-choice question about a video clip. "
#     "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
# )
#
# def _question_payload(record: QuestionRecord) -> str:
#     lines = [
#         f"Question type: {record.question_type}",
#         f"Text context: {record.text_context or 'N/A'}",
#         f"Question: {record.question}",
#         f"A) {record.choices[0]}",
#         f"B) {record.choices[1]}",
#         f"C) {record.choices[2]}",
#         'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
#     ]
#     return "\n".join(lines)
#
# messages=[
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": content},
# ]
# response_format={"type": "json_object"}
# resp_json = json.loads(response.choices[0].message.content)
# letter = resp_json["choice_letter"].strip().upper()
# reasoning = resp_json.get("reasoning", "")


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
    frames: list[EncodedFrame],
) -> Prediction:
    content = [{"type": "text", "text": _question_payload(record)}]
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {frame.second:.1f} seconds"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}", "detail": "low"},
            }
        )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    match = LETTER_RE.search(text.upper())
    if not match:
        raise ValueError(f"Could not parse A/B/C from GPT-4o response: {text}")
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
