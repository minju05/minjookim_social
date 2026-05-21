"""
MMToM-QA GPT-4o inference.
Prompt functions copied verbatim from minjookim_social/muma_gpt4o.py (commit 65fcdc3)
for fair comparison. Only I/O adapter and frame loading differ.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import sys
sys.path.insert(0, '/home/seohyeon/.00_project/26-01-SAI/minjookim_social')
from muma_data import QuestionRecord
from muma_config import Settings
from muma_gpt4o import Prediction

from mmtom_data import load_frames_b64

# ── copied from muma_gpt4o.py (commit 65fcdc3) ──────────────────────────────

LETTER_RE = re.compile(r'\b([A-B])\b')


def _payload_text_only(record: QuestionRecord) -> str:
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_generic_cot(record: QuestionRecord) -> str:
    lines = [
        f"Question type: {record.question_type}",
        "",
        f"Text context: {record.text_context or 'N/A'}",
        "",
        f"Question: {record.question}",
        "",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        "",
        "Think step by step and choose the best answer.",
        "",
        'Respond in JSON:',
        '{',
        '  "choice_letter": "A",',
        '  "reasoning": "..."',
        '}',
    ]
    return "\n".join(lines)


def _payload_qlogic_v2_targeted(record: QuestionRecord) -> str:
    lines = [
        f"Question type: {record.question_type}",
        "",
        f"Text context: {record.text_context or 'N/A'}",
        "",
        f"Question: {record.question}",
        "",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
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
        '    "B": "..."',
        "  },",
        '  "choice_letter": "A",',
        '  "reasoning": "..."',
        "}",
    ]
    return "\n".join(lines)


def _parse_json_choice_only(text: str) -> str:
    text = text.strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            letter = d.get("choice_letter", "").strip().upper()
            if letter in ("A", "B"):
                return letter
        except Exception:
            pass
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return ""


def _parse_question_logic_rerank_response(text: str) -> str:
    text = text.strip()
    try:
        data = json.loads(text)
        choice = data.get("choice_letter", "")
        if isinstance(choice, str):
            choice = choice.strip().upper()
            if choice in {"A", "B"}:
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
                if choice in {"A", "B"}:
                    return choice
        except Exception:
            pass
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return ""

# ── end copy ─────────────────────────────────────────────────────────────────

PROMPT_FNS = {
    "text_only":    _payload_text_only,
    "text_frame":   _payload_text_only,   # same prompt; frames injected separately
    "generic_cot":  _payload_generic_cot,
    "qlogic":       _payload_qlogic_v2_targeted,
}

PARSER_FNS = {
    "text_only":    _parse_json_choice_only,
    "text_frame":   _parse_json_choice_only,
    "generic_cot":  _parse_json_choice_only,
    "qlogic":       _parse_question_logic_rerank_response,
}

SYSTEM_PROMPTS = {
    "text_only": (
        "You are an expert in theory of mind and social reasoning. "
        "Answer the following multiple-choice question based on the text context. "
        "Respond in JSON with keys: choice_letter (A/B) and reasoning."
    ),
    "text_frame": (
        "You are an expert in theory of mind and social reasoning. "
        "Answer the following multiple-choice question based on the text context and video frames. "
        "Respond in JSON with keys: choice_letter (A/B) and reasoning."
    ),
    "generic_cot": (
        "You are an expert in theory of mind and social reasoning. "
        "Answer the following multiple-choice question based on the text context. "
        "Respond in JSON with keys: choice_letter (A/B) and reasoning."
    ),
    "qlogic": (
        "You are an expert in theory of mind and social reasoning. "
        "Your task is to answer a multiple-choice Theory-of-Mind question by explicitly analyzing the question logic. "
        "Do not answer immediately. "
        "First identify the question polarity, the explicit condition, the target character, "
        "and the key mental-state inference. "
        "Then evaluate each option independently and choose the best answer. "
        "Respond in JSON with keys: question_polarity, condition, target_character, "
        "key_inference, option_analysis, choice_letter, reasoning."
    ),
}


def predict_mmtom(
    record: QuestionRecord,
    condition: str,
    settings: Settings,
) -> Prediction:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

    prompt_text = PROMPT_FNS[condition](record)
    content: list[dict] = [{"type": "text", "text": prompt_text}]

    if condition == "text_frame":
        frames = load_frames_b64(record)
        for b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[condition]},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content.strip()
    letter = PARSER_FNS[condition](raw_text)

    if letter == "":
        predicted_answer = ""
        correct = False
    else:
        index = ord(letter) - ord("A")
        predicted_answer = record.choices[index] if index < len(record.choices) else ""
        correct = predicted_answer == record.answer

    return Prediction(
        question_id      = record.question_id,
        episode_id       = record.episode_id,
        question_type    = record.question_type,
        gold_answer      = record.answer,
        predicted_answer = predicted_answer,
        predicted_letter = letter,
        correct          = correct,
        reasoning        = raw_text[:500],
    )
