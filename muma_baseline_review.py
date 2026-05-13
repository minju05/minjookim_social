"""
muma_baseline_review.py
=======================
baseline_review_2call 실험 전용 모듈.

구조:
  1차 호출: 기존 text_only baseline 그대로 (reasoning 포함 JSON)
  2차 호출: initial_choice + initial_reasoning 기반 ToM review

저장 필드:
  example_id, question_type, gold_answer,
  initial_choice, initial_reasoning,
  final_choice, initial_correct, final_correct,
  changed, review_reason,
  direct_response, review_response
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from muma_config import Settings
from muma_data import QuestionRecord


# ============================================================
# System Prompts
# ============================================================

# 1차 system prompt: text_only baseline 그대로
DIRECT_SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Answer the following multiple-choice question based on the text context. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)

# 2차 system prompt: reasoning 포함 review
REVIEW_SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "Your task is to review an initial answer and reasoning for a multiple-choice Theory-of-Mind question. "
    "Check whether the answer correctly reflects the relevant character's perspective, belief, goal, "
    "belief about another character's goal, or social intention. "
    "Only change the answer if there is a clear theory-of-mind error. "
    "Do not change the answer merely because another option also seems plausible. "
    "Respond in JSON with keys: final_choice, changed, review_reason."
)


# ============================================================
# Payload Builders
# ============================================================

def _payload_direct(record: QuestionRecord) -> str:
    """1차 호출: text_only baseline과 동일한 payload"""
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


def _payload_baseline_review_tom(
    record: QuestionRecord,
    initial_choice: str,
    initial_reasoning: str,
) -> str:
    """2차 호출: initial_choice + initial_reasoning 포함 ToM review"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        f"Initial answer: {initial_choice}",
        f"Initial reasoning: {initial_reasoning or 'N/A'}",
        "",
        "Review the initial answer and reasoning from a theory-of-mind perspective.",
        "",
        "Check:",
        "1. Whose mental state is the question asking about?",
        "2. Does the answer reflect that character's perspective, rather than the true world state?",
        "3. Does the reasoning confuse one character's belief, goal, or intention with another character's?",
        "4. If the question asks what one character believes about another character's goal, "
        "does the answer reflect the believer's view rather than the other character's actual goal?",
        "5. If the question asks about a social goal, does the answer capture the social purpose "
        "rather than just the physical action?",
        "",
        "Only change the answer if there is a clear theory-of-mind error.",
        "Do not change the answer merely because another option also seems plausible.",
        "If the initial answer is consistent with the relevant character's perspective, keep it.",
        "",
        'Respond in JSON: {"final_choice": "A", "changed": false, "review_reason": "..."}',
    ]
    return "\n".join(lines)


# ============================================================
# Parsers
# ============================================================

def _parse_direct(text: str) -> tuple[str | None, str]:
    """1차 응답 파서: choice_letter, reasoning 추출"""
    text = text.strip()
    data = None

    try:
        data = json.loads(text)
    except Exception:
        pass

    if data is None:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass

    if isinstance(data, dict):
        choice = data.get("choice_letter", "")
        if isinstance(choice, str):
            choice = choice.strip().upper()
        if choice in {"A", "B", "C"}:
            reasoning = data.get("reasoning", "")
            return choice, (reasoning if isinstance(reasoning, str) else "")

    # fallback: A/B/C 단독 등장
    m2 = re.search(r"\b([ABC])\b", text.upper())
    if m2:
        return m2.group(1), ""

    return None, ""


def _parse_review(text: str, initial_choice: str | None) -> dict:
    """
    2차 응답 파서: final_choice, changed, review_reason 추출.
    방어적 처리:
      - final_choice 파싱 실패 → initial_choice로 fallback
      - changed 없음 → final_choice != initial_choice 로 추론
    """
    result: dict = {
        "final_choice": initial_choice,  # fallback default
        "changed": None,
        "review_reason": "",
    }

    text = text.strip()
    data = None

    try:
        data = json.loads(text)
    except Exception:
        pass

    if data is None:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass

    if isinstance(data, dict):
        # final_choice
        choice = data.get("final_choice")
        if isinstance(choice, str):
            choice = choice.strip().upper()
            if choice in {"A", "B", "C"}:
                result["final_choice"] = choice

        # changed
        changed = data.get("changed")
        if isinstance(changed, bool):
            result["changed"] = changed
        elif isinstance(changed, str):
            result["changed"] = changed.strip().lower() in {"true", "yes", "1"}

        # review_reason
        reason = data.get("review_reason")
        if isinstance(reason, str):
            result["review_reason"] = reason.strip()

    # changed 추론 (없으면 final vs initial 비교)
    if result["changed"] is None and initial_choice is not None:
        result["changed"] = result["final_choice"] != initial_choice

    # final fallback: regex
    if result["final_choice"] is None:
        m2 = re.search(r"\b([ABC])\b", text.upper())
        if m2:
            result["final_choice"] = m2.group(1)
        else:
            result["final_choice"] = initial_choice  # 마지막 fallback

    return result


# ============================================================
# Result Dataclass
# ============================================================

@dataclass
class BaselineReviewResult:
    example_id: str
    question_type: str
    gold_answer: str
    initial_choice: str | None
    initial_reasoning: str
    final_choice: str | None
    initial_correct: bool
    final_correct: bool
    changed: bool | None
    review_reason: str
    direct_response: str
    review_response: str


def result_to_dict(r: BaselineReviewResult) -> dict:
    return asdict(r)


# ============================================================
# Direct cache loader (cond1_textonly 재사용)
# ============================================================

def load_direct_cache(jsonl_path: Path) -> dict[str, tuple[str | None, str]]:
    """
    기존 text_only JSONL에서 question_id → (predicted_letter, reasoning_text) 로드.
    reasoning 필드가 raw JSON string인 경우 내부 reasoning 텍스트를 추출.
    """
    cache: dict[str, tuple[str | None, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("question_id", "")
            letter = d.get("predicted_letter", "").strip().upper()
            if letter not in {"A", "B", "C"}:
                letter = None
            raw_reasoning = d.get("reasoning", "")
            # reasoning 필드가 raw JSON string인 경우 reasoning 텍스트만 추출
            reasoning_text = ""
            try:
                parsed = json.loads(raw_reasoning)
                if isinstance(parsed, dict):
                    reasoning_text = parsed.get("reasoning", "") or ""
            except Exception:
                reasoning_text = raw_reasoning  # 직접 텍스트인 경우 그대로
            cache[qid] = (letter, reasoning_text)
    return cache


# ============================================================
# Single-question 2-call function
# ============================================================

def _letter_to_text(letter: str | None, choices: list[str]) -> str | None:
    """choice letter (A/B/C) → 실제 텍스트 변환"""
    if letter is None:
        return None
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return None


def run_one(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    direct_cache: dict[str, tuple[str | None, str]] | None = None,
) -> BaselineReviewResult:
    """
    QuestionRecord 1개에 대해 2-call 수행 후 BaselineReviewResult 반환.
    direct_cache 제공 시 1차 API 호출 스킵 (기존 결과 재사용).
    """

    # ---------- 1차: 캐시 사용 or 실제 호출 ----------
    if direct_cache is not None and record.question_id in direct_cache:
        initial_choice, initial_reasoning = direct_cache[record.question_id]
        direct_text = f"[loaded from cache] letter={initial_choice}"
    else:
        direct_payload = _payload_direct(record)
        direct_messages = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": direct_payload}]},
        ]
        direct_resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=direct_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        direct_text = direct_resp.choices[0].message.content.strip()
        initial_choice, initial_reasoning = _parse_direct(direct_text)

    # 파싱 실패 시 review에 "?" 전달하되 기록은 None 유지
    choice_for_review = initial_choice if initial_choice else ""

    # ---------- 2차 호출: ToM Review ----------
    review_payload = _payload_baseline_review_tom(
        record, choice_for_review, initial_reasoning
    )
    review_messages = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": review_payload}]},
    ]
    review_resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=review_messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    review_text = review_resp.choices[0].message.content.strip()
    review_parsed = _parse_review(review_text, initial_choice)

    final_choice = review_parsed["final_choice"]
    changed = review_parsed["changed"]
    review_reason = review_parsed["review_reason"]

    gold = record.answer
    initial_text = _letter_to_text(initial_choice, record.choices)
    final_text = _letter_to_text(final_choice, record.choices)

    return BaselineReviewResult(
        example_id=record.question_id,
        question_type=record.question_type,
        gold_answer=gold,
        initial_choice=initial_choice,
        initial_reasoning=initial_reasoning,
        final_choice=final_choice,
        initial_correct=(initial_text == gold) if initial_text is not None else False,
        final_correct=(final_text == gold) if final_text is not None else False,
        changed=changed,
        review_reason=review_reason,
        direct_response=direct_text,
        review_response=review_text,
    )


# ============================================================
# Batch runner
# ============================================================

def run_baseline_review_batch(
    client: OpenAI,
    settings: Settings,
    records: list[QuestionRecord],
    output_path: Path,
    direct_cache: dict[str, tuple[str | None, str]] | None = None,
) -> dict:
    """
    records 전체에 대해 baseline_review 2-call 수행.
    direct_cache 제공 시 1차 API 호출 스킵 (기존 cond1_textonly 재사용).
    결과를 output_path에 JSONL로 저장하고 summary dict 반환.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_hits = sum(1 for r in records if direct_cache and r.question_id in direct_cache)
    desc = f"BaselineReview 2-call (cache {cache_hits}/{len(records)})"

    results: list[BaselineReviewResult] = []

    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc=desc):
            result = run_one(client, settings, record, direct_cache=direct_cache)
            results.append(result)
            f.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")
            f.flush()

    # ---- Summary ----
    total = len(results)
    initial_correct = sum(r.initial_correct for r in results)
    final_correct = sum(r.final_correct for r in results)
    changed_count = sum(1 for r in results if r.changed)

    # change_breakdown (transition table)
    cc = sum(1 for r in results if r.initial_correct and r.final_correct)
    cw = sum(1 for r in results if r.initial_correct and not r.final_correct)
    wc = sum(1 for r in results if not r.initial_correct and r.final_correct)
    ww = sum(1 for r in results if not r.initial_correct and not r.final_correct)

    # question_type별 accuracy (final_choice 기준)
    type_total: dict[str, int] = defaultdict(int)
    type_final_correct: dict[str, int] = defaultdict(int)
    type_initial_correct: dict[str, int] = defaultdict(int)
    for r in results:
        type_total[r.question_type] += 1
        if r.final_correct:
            type_final_correct[r.question_type] += 1
        if r.initial_correct:
            type_initial_correct[r.question_type] += 1

    summary = {
        "num_questions": total,
        "initial_accuracy": round(initial_correct / total, 4) if total else 0.0,
        "final_accuracy": round(final_correct / total, 4) if total else 0.0,
        "changed_count": changed_count,
        "change_breakdown": {
            "correct_to_correct": cc,
            "correct_to_wrong": cw,
            "wrong_to_correct": wc,
            "wrong_to_wrong": ww,
        },
        "initial_accuracy_by_type": {
            qt: round(type_initial_correct[qt] / type_total[qt], 4)
            for qt in sorted(type_total)
        },
        "final_accuracy_by_type": {
            qt: round(type_final_correct[qt] / type_total[qt], 4)
            for qt in sorted(type_total)
        },
    }
    return summary
