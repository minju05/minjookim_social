"""
gridtom_gpt4o.py

프롬프트 함수는 minjookim_social/muma_gpt4o.py (teahun0502 브랜치)에서 그대로 복사.
논문 공정한 비교를 위해 코드 동일성 유지.
변경 사항: QuestionRecord 어댑터 (GridToM 2지선다), 프레임 로딩 (PNG), Prediction import.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minjookim_social"))
from muma_config import Settings
from muma_data import QuestionRecord
from muma_gpt4o import Prediction

from gridtom_data import GridToMRecord

LETTER_RE = re.compile(r"\b([A-C])\b")

# ══════════════════════════════════════════════════════════════════════════════
# 아래 함수들은 minjookim_social/muma_gpt4o.py에서 그대로 복사 (teahun0502 브랜치)
# ══════════════════════════════════════════════════════════════════════════════

def _payload_text_only(record: QuestionRecord) -> str:
    """Text-only: 테이스트만 (프레임 없음)"""
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


def _parse_json_response(text: str) -> str:
    """JSON 응답에서 choice_letter 추출"""
    resp_json = json.loads(text)
    return resp_json["choice_letter"].strip().upper()


def _parse_json_choice_only(text: str) -> str:
    """choice_only_json 파서: choice_letter만 추출, 여분 키 무시, fallback 지원"""
    text = text.strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            letter = d.get("choice_letter", "").strip().upper()
            if letter in ("A", "B", "C"):
                return letter
        except Exception:
            pass
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return ""


def _parse_question_logic_rerank_response(text: str) -> str:
    """question_logic_rerank 응답 파서: choice_letter 추출, fallback 지원"""
    text = text.strip()
    try:
        data = json.loads(text)
        choice = data.get("choice_letter", "")
        if isinstance(choice, str):
            choice = choice.strip().upper()
            if choice in {"A", "B", "C"}:
                return choice
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            choice = data.get("choice_letter", "")
            if isinstance(choice, str):
                choice = choice.strip().upper()
                if choice in {"A", "B", "C"}:
                    return choice
        except Exception:
            pass
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return ""


def _payload_qlogic_v2_targeted(record: QuestionRecord) -> str:
    """qlogic_v2_targeted: qlogic with targeted fixes for belief LEAST-likely and social-goal errors."""
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
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# 이하 GridToM I/O 어댑터
# ══════════════════════════════════════════════════════════════════════════════

def to_question_record(rec: GridToMRecord) -> QuestionRecord:
    """
    GridToMRecord → QuestionRecord 변환.
    choices[2] = "" 로 C 옵션 빈칸 처리 (2지선다 대응).
    text_context = env_desc + caption.
    """
    return QuestionRecord(
        question_id  = f"{rec.index}_{rec.belief_type}",
        episode_id   = str(rec.index),
        question_type= rec.belief_type,
        question     = rec.question,
        choices      = [rec.options[0], rec.options[1], ""],
        answer       = rec.answer,
        text_context = f"{rec.env_desc}\n{rec.caption}",
        video_path   = Path("/dev/null"),
        raw          = {},
    )


def _build_generic_cot(record: QuestionRecord) -> str:
    """text_only + Think step by step (I/O 어댑터 범위)."""
    return _payload_text_only(record).replace(
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
        'Think step by step before answering.\nPick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    )


def _build_qa_only(record: QuestionRecord) -> str:
    """질문+선지만 (context 없음)."""
    lines = [
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


PROMPT_FNS = {
    "qa_only":     _build_qa_only,
    "text_only":   _payload_text_only,
    "text_frame":  _payload_text_only,
    "generic_cot": _build_generic_cot,
    "qlogic":      _payload_qlogic_v2_targeted,
}

PARSER_FNS = {
    "qa_only":     _parse_json_choice_only,
    "text_only":   _parse_json_choice_only,
    "text_frame":  _parse_json_choice_only,
    "generic_cot": _parse_json_choice_only,
    "qlogic":      _parse_question_logic_rerank_response,
}


def predict_gridtom(
    client: OpenAI,
    settings: Settings,
    record: GridToMRecord,
    frames_b64: list[str],
    condition: str = "text_only",
    max_retries: int = 3,
) -> Prediction:
    qr     = to_question_record(record)
    prompt = PROMPT_FNS[condition](qr)
    parser = PARSER_FNS[condition]

    content: list[dict] = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt})

    raw = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

    letter = parser(raw)

    # C는 GridToM에 없음 → 파싱 실패 처리
    if letter not in ("A", "B"):
        letter = ""
        predicted_answer = ""
    else:
        predicted_answer = qr.choices[0 if letter == "A" else 1].strip().lower()

    question_polarity = condition_field = target_character = ""
    key_inference = reasoning = ""
    option_analysis = None
    try:
        d = json.loads(raw)
        question_polarity = d.get("question_polarity", "")
        condition_field   = d.get("condition", "")
        target_character  = d.get("target_character", "")
        key_inference     = d.get("key_inference", "")
        option_analysis   = d.get("option_analysis")
        reasoning         = d.get("reasoning", "")
    except Exception:
        reasoning = raw

    return Prediction(
        question_id      = qr.question_id,
        episode_id       = qr.episode_id,
        question_type    = qr.question_type,
        gold_answer      = record.answer,
        predicted_answer = predicted_answer,
        predicted_letter = letter,
        correct          = predicted_answer == record.answer,
        reasoning        = reasoning,
        question_polarity= question_polarity,
        condition        = condition_field,
        target_character = target_character,
        key_inference    = key_inference,
        option_analysis  = option_analysis,
        raw_response     = raw,
    )
