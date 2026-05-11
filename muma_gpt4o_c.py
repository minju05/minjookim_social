from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from openai import OpenAI

from muma_config import Settings
from muma_data import QuestionRecord
from muma_video import EncodedFrame


SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question about a video clip. "
    "Respond with only a single letter: A, B, or C."
)

CONSISTENCY_SYSTEM_PROMPT = (
    "You are analyzing whether an agent's utterance matches their observed behavior in a video. "
    "Respond with only A or B."
)


@dataclass(frozen=True)
class PredictionC:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    base_probs: dict
    adjusted_probs: dict
    consistency_score: float | None


def _question_payload(record: QuestionRecord) -> str:
    return "\n".join([
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
    ])


def _extract_abc_logprobs(response) -> dict[str, float]:
    result = {"A": 1e-9, "B": 1e-9, "C": 1e-9}
    try:
        for item in response.choices[0].logprobs.content[0].top_logprobs:
            tok = item.token.strip().upper()
            if tok in result and result[tok] == 1e-9:  # first match only
                result[tok] = math.exp(item.logprob)
    except (AttributeError, IndexError, TypeError):
        pass
    return result


def _get_base_probs(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
) -> dict[str, float]:
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
        logprobs=True,
        top_logprobs=5,
        max_tokens=1,
    )
    return _extract_abc_logprobs(response)


def _extract_utterance(client: OpenAI, model: str, text_context: str) -> str | None:
    prompt = (
        f"Read the following text describing a social interaction.\n"
        f"Text: {text_context}\n\n"
        "Extract any direct utterance (spoken words) made by one of the agents. "
        "If there is no utterance, reply with exactly 'NONE'. "
        "Otherwise reply with only the utterance text, nothing else."
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=150,
    )
    text = response.choices[0].message.content.strip()
    return None if text.upper() == "NONE" or not text else text


def _compute_visual_consistency(
    client: OpenAI,
    model: str,
    utterance: str,
    frames: list[EncodedFrame],
) -> float:
    """Returns P(consistent) between utterance and agent's video actions."""
    content: list = [{
        "type": "text",
        "text": (
            f"An agent made the following statement: \"{utterance}\"\n\n"
            "Watch the video frames and decide if the agent's actual behavior is consistent with this statement.\n"
            "A) Consistent — their actions match what they said\n"
            "B) Inconsistent — their actions contradict what they said\n"
            "Respond with only A or B."
        )
    }]
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=model,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
        max_tokens=1,
    )
    try:
        for item in response.choices[0].logprobs.content[0].top_logprobs:
            if item.token.strip().upper() == "A":
                return math.exp(item.logprob)
    except (AttributeError, IndexError, TypeError):
        pass
    return 0.5


def _classify_choice(choice_text: str) -> str:
    lower = choice_text.lower()
    if any(w in lower for w in ("hinder", "prevent", "obstruct", "mislead", "deceiv", "lie")):
        return "hinder"
    if any(w in lower for w in ("help", "assist", "cooperat", "support", "aid")):
        return "help"
    return "independent"


def _apply_consistency(
    base_probs: dict[str, float],
    choices: list[str],
    consistency_score: float,
    lambda_weight: float,
) -> dict[str, float]:
    letters = ["A", "B", "C"]
    adjusted = {}
    for letter, choice in zip(letters, choices):
        p = base_probs.get(letter, 1e-9)
        goal_type = _classify_choice(choice)
        if goal_type == "help":
            s_c = max(consistency_score, 1e-9)
        elif goal_type == "hinder":
            s_c = max(1.0 - consistency_score, 1e-9)
        else:
            adjusted[letter] = p
            continue
        adjusted[letter] = p * (s_c ** lambda_weight)

    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()} if total > 0 else adjusted


def predict_question_c(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
    lambda_weight: float = 1.0,
) -> PredictionC:
    # Step 1: base logprobs P(A), P(B), P(C) with 우리 버전 prompt
    base_probs = _get_base_probs(client, settings, record, frames)

    # Step 2: extract utterance from text_context, compute visual consistency
    consistency_score: float | None = None
    if record.text_context:
        utterance = _extract_utterance(client, settings.openai_model, record.text_context)
        if utterance:
            consistency_score = _compute_visual_consistency(
                client, settings.openai_model, utterance, frames
            )

    # Step 3: adjust probabilities using consistency score
    if consistency_score is not None:
        adjusted_probs = _apply_consistency(base_probs, record.choices, consistency_score, lambda_weight)
    else:
        total = sum(base_probs.values())
        adjusted_probs = {k: v / total for k, v in base_probs.items()} if total > 0 else base_probs

    letter = max(adjusted_probs, key=adjusted_probs.get)
    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    return PredictionC(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        base_probs=base_probs,
        adjusted_probs=adjusted_probs,
        consistency_score=consistency_score,
    )


def prediction_to_dict(prediction: PredictionC) -> dict[str, object]:
    return asdict(prediction)