from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable

from openai import OpenAI

from muma_config import Settings
from muma_data import QuestionRecord
from muma_video import EncodedFrame


# ============================================================
# 정규표현식
# ============================================================
LETTER_RE = re.compile(r"\b([A-C])\b")


# ============================================================
# Prompt 설정 (하이브리드 방식)
# ============================================================
@dataclass
class PromptConfig:
    name: str
    description: str
    accuracy: float | str  # 실험 전이면 "TBD"
    system_prompt: str | None
    include_question_type: bool
    json_output: bool
    payload_fn: Callable[[QuestionRecord], str]
    response_parser_fn: Callable[[str], str]  # text → letter (A/B/C)


def _payload_v1_minju(record: QuestionRecord) -> str:
    """민주 버전: 단순 프롬프트"""
    lines = [
        f"Text: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "Answer with only A, B, or C.",
    ]
    return "\n".join(lines)


def _payload_v2_ours(record: QuestionRecord) -> str:
    """우리 버전: System Prompt + Question Type + JSON"""
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


def _payload_v3_perspective_state(record: QuestionRecord) -> str:
    """v3: Perspective-State Consistency (공통 오답 특화)"""
    lines = [
        f"Text: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Think through the scenario by keeping each character's mental state separate:",
        "- observations",
        "- beliefs",
        "- intentions",
        "- beliefs about others' intentions",
        "",
        "Avoid using facts that a character could not know.",
        "Distinguish a person's intention from the final outcome.",
        "For MOST/LEAST likely questions under an assumption, choose according to consistency with that assumption.",
        "",
        "Do not explain your reasoning.",
        "Answer with only A, B, or C.",
    ]
    return "\n".join(lines)


def _payload_v4_careful_observer(record: QuestionRecord) -> str:
    """v4: Careful Observer (공통 오답 특화 v2)"""
    lines = [
        f"Text: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before choosing, reason like a careful observer of people.",
        "Keep each character's perspective separate.",
        "Ask what each character saw, did not see, believed, and intended at that moment.",
        "Do not assume that a character knows facts they could not have observed.",
        "Judge intentions based on what the character likely knew or believed, not only on the final outcome.",
        "If a character is judging another character's goal, use the judging character's perspective.",
        "Read the wording carefully when it asks what is more or less likely.",
        "",
        "Do not explain your reasoning.",
        "Answer with only A, B, or C.",
    ]
    return "\n".join(lines)


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


def _payload_video_only(record: QuestionRecord) -> str:
    """Video-only: 프레임만 (GT 텍스트 없음)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_v5_few_shot_cot(record: QuestionRecord, examples: list[dict]) -> str:
    """v5: Few-Shot + Chain-of-Thought (v4 기반 + CoT 허용)"""
    # Few-shot 예제 포맷팅
    examples_text = "\n\n".join([
        f"Example {i+1}:\nQuestion type: {ex['question_type']}\n{ex['question']}\nA) {ex['choices'][0]}\nB) {ex['choices'][1]}\nC) {ex['choices'][2]}\nAnswer: {ex['answer']}"
        for i, ex in enumerate(examples)
    ])
    
    lines = [
        "You are an expert in theory of mind and social reasoning.",
        "Learn from these examples:",
        "",
        examples_text,
        "",
        "Now answer this question:",
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Reason through the scenario by keeping each character's mental state separate:",
        "- What did each character observe?",
        "- What did each character believe?",
        "- What was each character's intention?",
        "",
        "Then provide your answer in JSON format: {\"choice_letter\": \"A\", \"reasoning\": \"...\"}",
    ]
    return "\n".join(lines)


def _payload_cecr_wo_step1(record: QuestionRecord) -> str:
    """CECR ablation: w/o Step 1 (character state identification 제거)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before answering, reason about the consistency between what characters say, know, and do.",
        "",
        "Step 1. Check whether a character's statement is consistent with the events and evidence in the scenario.",
        "Step 2. Use that consistency to infer the character's likely belief or intention and answer the question.",
        "",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_cecr_wo_step2(record: QuestionRecord) -> str:
    """CECR ablation: w/o Step 2 (consistency check 제거, consistency 암시 문장도 제거)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before answering, reason about the characters' goals, statements, actions, and beliefs.",
        "",
        "Step 1. Identify the relevant characters' goals, statements, actions, and beliefs.",
        "Step 2. Use that to infer the character's likely belief or intention and answer the question.",
        "",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_v9_cecr_notype(record: QuestionRecord) -> str:
    """v9-notype: CECR without question_type hint"""
    lines = [
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before answering, reason about the consistency between what characters say, know, and do.",
        "",
        "Step 1. Identify the relevant characters' goals, statements, actions, and beliefs.",
        "Step 2. Check whether a character's statement is consistent with the events and evidence in the scenario.",
        "Step 3. Use that consistency to infer the character's likely belief or intention and answer the question.",
        "",
        'Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_v9_cecr(record: QuestionRecord) -> str:
    """v9: CECR (Consistency-Evidence-Coherence Reasoning)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before answering, reason about the consistency between what characters say, know, and do.",
        "",
        "Step 1. Identify the relevant characters' goals, statements, actions, and beliefs.",
        "Step 2. Check whether a character's statement is consistent with the events and evidence in the scenario.",
        "Step 3. Use that consistency to infer the character's likely belief or intention and answer the question.",
        "",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_v8_pgir(record: QuestionRecord) -> str:
    """v8: PGIR-Eval (Perspective-Grounded Intent Reasoning)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Use Perspective-Grounded Intent Reasoning before choosing.",
        "",
        "Step 1. Identify the relevant characters by name.",
        "Step 2. For each relevant character, track the facts from their perspective:",
        "  - what they saw or heard,",
        "  - what they did,",
        "  - what they likely believed,",
        "  - what they could not have known.",
        "Step 3. Read the question and determine whose perspective is required.",
        "Step 4. Reason from that perspective, not from the omniscient observer's perspective.",
        "Step 5. If the question involves intentions or goals, infer the character's likely intention from their actions, words, and available knowledge at the time.",
        "Step 6. Verify that your selected option is consistent with the character-level facts above.",
        "",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _payload_v7_action_check(record: QuestionRecord) -> str:
    """v7: Action-Check (말 vs 행동 불일치 감지)"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        "",
        "Before answering, follow these steps:",
        "Step 1. Identify what Person A SAID (the information they provided).",
        "Step 2. Identify where Person B actually found or grabbed the item.",
        "Step 3. Do the said location and the actual location match?",
        "  - YES (match) → Person A likely HELPED Person B",
        "  - NO  (mismatch) → Person A likely HINDERED or PREVENTED Person B",
        "  - No clear mismatch or no dialogue → consider indifferent or use other cues",
        "Step 4. Apply the question's assumption (e.g. 'if helping', 'if hindering', 'knows what is inside') to select the MOST or LEAST likely answer.",
        "",
        'Respond in JSON: {"choice_letter": "A", "reasoning": "..."}',
    ]
    return "\n".join(lines)


def _parse_text_response(text: str) -> str:
    """자유 텍스트 응답에서 A/B/C 추출"""
    match = LETTER_RE.search(text.upper())
    if not match:
        raise ValueError(f"Could not parse A/B/C from response: {text}")
    return match.group(1)


def _parse_json_response(text: str) -> str:
    """JSON 응답에서 choice_letter 추출"""
    resp_json = json.loads(text)
    return resp_json["choice_letter"].strip().upper()


def _parse_json_choice_only(text: str) -> str:
    """choice_only_json 파서: choice_letter만 추출, 여분 키 무시, fallback 지원"""
    text = text.strip()
    # JSON 파싱 시도
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            letter = d.get("choice_letter", "").strip().upper()
            if letter in ("A", "B", "C"):
                return letter
        except Exception:
            pass
    # fallback: A/B/C 추출
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return ""


def _payload_question_logic_rerank(record: QuestionRecord) -> str:
    """Question-logic-aware 6-step reranking payload (baseline 없이 새로 풀기)"""
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
        "Examples:",
        "- whether a character is trying to help another character",
        "- whether a character is trying to hinder another character",
        "- whether a character knows what is inside a location",
        "- whether the answer should be inferred based on the agents' actions",
        "",
        "Step 3. Identify the target character whose mental state is being evaluated.",
        "",
        "Step 4. Identify the key mental-state inference needed.",
        "Useful rules:",
        "- If a character knowingly gives false location information, this may indicate hindering rather than helping.",
        "- If a character gives true and useful location information, this may indicate helping rather than hindering.",
        "- If an agent placed an object somewhere and another agent moves it away, infer whether the mover believes the original location was the first agent's desired location.",
        "- For belief-of-goal questions, distinguish the other agent's actual goal from what the target character believes about that goal.",
        "- For LEAST likely questions, choose the option least consistent with the condition and inferred mental state.",
        "- For MOST likely questions, choose the option most consistent with the condition and inferred mental state.",
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
        '  },',
        '  "choice_letter": "A",',
        '  "reasoning": "..."',
        "}",
    ]
    return "\n".join(lines)


def _parse_question_logic_rerank_response(text: str) -> str:
    """question_logic_rerank 응답 파서: choice_letter 추출, fallback 지원"""
    text = text.strip()
    # 1. 직접 JSON 파싱
    try:
        data = json.loads(text)
        choice = data.get("choice_letter", "")
        if isinstance(choice, str):
            choice = choice.strip().upper()
            if choice in {"A", "B", "C"}:
                return choice
    except Exception:
        pass
    # 2. JSON 블록 추출 후 파싱
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
    # 3. fallback: A/B/C 단독 등장
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


def _payload_qlogic_v2_targeted_frames(record: QuestionRecord) -> str:
    """qlogic_v2_targeted_frames: qlogic_v2_targeted + video frames guidance."""
    lines = [
        f"Question type: {record.question_type}",
        "",
        "Video frames from the scene are provided above. Use them as additional evidence alongside the text context.",
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


def _payload_choice_only_json(record: QuestionRecord) -> str:
    """Choice-only JSON: text_only와 동일하되 reasoning 제거"""
    lines = [
        f"Question type: {record.question_type}",
        f"Text context: {record.text_context or 'N/A'}",
        f"Question: {record.question}",
        f"A) {record.choices[0]}",
        f"B) {record.choices[1]}",
        f"C) {record.choices[2]}",
        'Pick the single best answer. Respond in JSON: {"choice_letter": "A"}',
    ]
    return "\n".join(lines)


PROMPT_CONFIGS = {
    "v1_minju": PromptConfig(
        name="v1_minju",
        description="민주 버전 (논문 제출)",
        accuracy=0.572,
        system_prompt=None,
        include_question_type=False,
        json_output=False,
        payload_fn=_payload_v1_minju,
        response_parser_fn=_parse_text_response,
    ),
    "v2_ours": PromptConfig(
        name="v2_ours",
        description="우리 버전 (System + Type + JSON)",
        accuracy=0.701,
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_v2_ours,
        response_parser_fn=_parse_json_response,
    ),
    "v3_perspective_state": PromptConfig(
        name="v3_perspective_state",
        description="v3: Perspective-State Consistency (공통 오답 특화)",
        accuracy="TBD",
        system_prompt=None,
        include_question_type=False,
        json_output=False,
        payload_fn=_payload_v3_perspective_state,
        response_parser_fn=_parse_text_response,
    ),
    "v4_careful_observer": PromptConfig(
        name="v4_careful_observer",
        description="v4: Careful Observer (공통 오답 특화 v2)",
        accuracy="TBD",
        system_prompt=None,
        include_question_type=False,
        json_output=False,
        payload_fn=_payload_v4_careful_observer,
        response_parser_fn=_parse_text_response,
    ),
    "v9_cecr": PromptConfig(
        name="v9_cecr",
        description="v9: CECR (Consistency-Evidence-Coherence Reasoning)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_v9_cecr,
        response_parser_fn=_parse_json_response,
    ),
    "v9_cecr_notype": PromptConfig(
        name="v9_cecr_notype",
        description="v9 CECR without question_type hint",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=False,
        json_output=True,
        payload_fn=_payload_v9_cecr_notype,
        response_parser_fn=_parse_json_response,
    ),
    "cecr_wo_step1": PromptConfig(
        name="cecr_wo_step1",
        description="CECR ablation: w/o Step 1 (character state identification 제거)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_cecr_wo_step1,
        response_parser_fn=_parse_json_response,
    ),
    "cecr_wo_step2": PromptConfig(
        name="cecr_wo_step2",
        description="CECR ablation: w/o Step 2 (consistency check 제거)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_cecr_wo_step2,
        response_parser_fn=_parse_json_response,
    ),
    "v8_pgir": PromptConfig(
        name="v8_pgir",
        description="v8: PGIR-Eval (Perspective-Grounded Intent Reasoning)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_v8_pgir,
        response_parser_fn=_parse_json_response,
    ),
    "v7_action_check": PromptConfig(
        name="v7_action_check",
        description="v7: Action-Check (말 vs 행동 불일치 감지)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_v7_action_check,
        response_parser_fn=_parse_json_response,
    ),
    "text_only": PromptConfig(
        name="text_only",
        description="Text-Only (프레임 없음, v2 구조)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question based on the text context. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_text_only,
        response_parser_fn=_parse_json_response,
    ),
    "video_only": PromptConfig(
        name="video_only",
        description="Video-Only (프레임만, text context 없음)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question about a video clip. "
            "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_video_only,
        response_parser_fn=_parse_json_response,
    ),
    "choice_only_json": PromptConfig(
        name="choice_only_json",
        description="Choice-Only JSON (reasoning 제거, 변수 통제 실험)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Answer the following multiple-choice question based on the text context. "
            "Respond in JSON with only the key choice_letter. Do not include reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_choice_only_json,
        response_parser_fn=_parse_json_choice_only,
    ),
    "question_logic_rerank_without_initial": PromptConfig(
        name="question_logic_rerank_without_initial",
        description="Question-logic-aware 6-step reranking (baseline 없이 새로 풀기)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Your task is to answer a multiple-choice Theory-of-Mind question by explicitly analyzing the question logic. "
            "Do not answer immediately. "
            "First identify the question polarity, the explicit condition, the target character, "
            "and the key mental-state inference. "
            "Then evaluate each option independently and choose the best answer. "
            "Respond in JSON with keys: question_polarity, condition, target_character, "
            "key_inference, option_analysis, choice_letter, reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_question_logic_rerank,
        response_parser_fn=_parse_question_logic_rerank_response,
    ),
    "qlogic_v2_targeted": PromptConfig(
        name="qlogic_v2_targeted",
        description="QLogic with targeted checks for belief least-likely and social-goal truthfulness errors",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "Your task is to answer a multiple-choice Theory-of-Mind question by explicitly analyzing the question logic. "
            "Do not answer immediately. "
            "First identify the question polarity, the explicit condition, the target character, "
            "and the key mental-state inference. "
            "Then evaluate each option independently and choose the best answer. "
            "Respond in JSON with keys: question_polarity, condition, target_character, key_inference, option_analysis, choice_letter, reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_qlogic_v2_targeted,
        response_parser_fn=_parse_question_logic_rerank_response,
    ),
    "qlogic_v2_targeted_frames": PromptConfig(
        name="qlogic_v2_targeted_frames",
        description="qlogic_v2_targeted + video frames (text+visual fusion)",
        accuracy="TBD",
        system_prompt=(
            "You are an expert in theory of mind and social reasoning. "
            "You will be given video frames from the scene and a text transcript. "
            "Your task is to answer a multiple-choice Theory-of-Mind question by explicitly analyzing the question logic. "
            "Do not answer immediately. "
            "First identify the question polarity, the explicit condition, the target character, "
            "and the key mental-state inference. Use both the text context and the video frames as evidence. "
            "Then evaluate each option independently and choose the best answer. "
            "Respond in JSON with keys: question_polarity, condition, target_character, key_inference, option_analysis, choice_letter, reasoning."
        ),
        include_question_type=True,
        json_output=True,
        payload_fn=_payload_qlogic_v2_targeted_frames,
        response_parser_fn=_parse_question_logic_rerank_response,
    ),
}


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
    # structured fields (populated when model returns JSON with these keys)
    question_polarity: str = ""
    condition: str = ""
    target_character: str = ""
    key_inference: str = ""
    option_analysis: dict = None
    raw_response: str = ""


def predict_question(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
    prompt_version: str = "v2_ours",
) -> Prediction:
    """
    질문 예측 (프롬프트 버전 선택 가능)
    
    Args:
        client: OpenAI 클라이언트
        settings: 설정
        record: 질문 레코드
        frames: 인코딩된 프레임 목록
        prompt_version: 프롬프트 버전 (v1_minju, v2_ours, v3_perspective_state 등)
    """
    config = PROMPT_CONFIGS[prompt_version]
    
    # 1. 프롬프트 생성
    text_content = config.payload_fn(record)
    content = [{"type": "text", "text": text_content}]
    
    # 2. 프레임 추가
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {frame.second:.1f} seconds"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}", "detail": "low"},
            }
        )

    # 3. 메시지 구성
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": content})

    # 4. API 호출
    response_format = None
    if config.json_output:
        response_format = {"type": "json_object"}
    
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.0,
        response_format=response_format,
    )

    # 5. 응답 파싱
    text = response.choices[0].message.content.strip()
    letter = config.response_parser_fn(text)
    
    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    # 6. structured fields 추출 (JSON 응답인 경우)
    question_polarity = ""
    condition = ""
    target_character = ""
    key_inference = ""
    option_analysis = None
    reasoning_text = ""
    try:
        parsed = json.loads(text)
        question_polarity = parsed.get("question_polarity", "")
        condition = parsed.get("condition", "")
        target_character = parsed.get("target_character", "")
        key_inference = parsed.get("key_inference", "")
        option_analysis = parsed.get("option_analysis", None)
        reasoning_text = parsed.get("reasoning", "")
    except Exception:
        reasoning_text = text

    return Prediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=reasoning_text,
        question_polarity=question_polarity,
        condition=condition,
        target_character=target_character,
        key_inference=key_inference,
        option_analysis=option_analysis,
        raw_response=text,
    )


def predict_question_v5_few_shot(
    client: OpenAI,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
    examples: list[dict],
) -> Prediction:
    """
    v5 Few-Shot + CoT 예측
    
    Args:
        client: OpenAI 클라이언트
        settings: 설정
        record: 질문 레코드
        frames: 인코딩된 프레임 목록
        examples: Few-shot 예제 리스트 [{"question_type", "question", "choices", "answer"}, ...]
    """
    # 1. v5 프롬프트 생성
    text_content = _payload_v5_few_shot_cot(record, examples)
    content = [{"type": "text", "text": text_content}]
    
    # 2. 프레임 추가
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {frame.second:.1f} seconds"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}", "detail": "low"},
            }
        )

    # 3. 메시지 구성
    messages = [{"role": "user", "content": content}]

    # 4. API 호출 (JSON 응답 강제)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    # 5. 응답 파싱
    text = response.choices[0].message.content.strip()
    letter = _parse_json_response(text)
    
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
