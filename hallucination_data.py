from __future__ import annotations

import random
import re
from dataclasses import dataclass

from muma_data import QuestionRecord


NAME_RE = re.compile(r'\b([A-Z][a-z]{2,})\b')
_SKIP = {
    'Given', 'When', 'Meanwhile', 'The', 'Inside', 'After', 'Before',
    'While', 'Then', 'Also', 'Both', 'From', 'With', 'Into', 'Back',
    'She', 'Her', 'His', 'Him', 'They', 'Their', 'Has', 'Have', 'Had',
    'Was', 'Were', 'Did', 'Does', 'Just', 'That', 'This', 'What',
}

ANSWER_RE = re.compile(r'\b([A-C])\b')

QA_PROMPT = """\
Question type: {question_type}

Interaction:
{text}

Question:
{question}
A) {option_a}
B) {option_b}
C) {option_c}

Pick the single best answer. Respond in JSON: {{"choice_letter": "A", "reasoning": "..."}}"""

QA_SYSTEM_PROMPT = (
    "You are an expert in theory of mind and social reasoning. "
    "You are given a description of an interaction between two agents in a household environment. "
    "Answer the following multiple-choice question. "
    "Respond in JSON with keys: choice_letter (A/B/C) and reasoning."
)

VLM_PROMPT = """\
You are observing a video of two agents interacting in a household environment. \
Describe their actions and conversations as a concise narrative story, in chronological order.

RULES:
- Describe actions step by step as they happen
- Include ALL physical actions: walking, opening, closing, grabbing, placing objects
- Include exact dialogue when agents speak
- When an agent opens a container and closes it WITHOUT grabbing anything, explicitly state that
- Mention WHERE objects are grabbed FROM and placed TO
- If you cannot identify an exact object, write "an unknown object"
- Do NOT infer intentions or emotions
- End with whether they communicated further or not

[EXAMPLE OUTPUT]
Jessica walked into the kitchen while Michael stayed silent. Jessica then moved to the living room \
and asked, "Do you have any idea where the remote control might be?" Michael, who had just walked \
into the kitchen, replied, "I found the remote control inside the cabinet in the living room." \
Jessica walked to the cabinet, opened it, grabbed the remote control, and closed the cabinet. \
She then walked to the sofa and placed the remote control on it.

Meanwhile, Michael walked to the fridge, opened it, grabbed the milk, and closed the fridge. \
He then opened the microwave and placed the milk inside before closing it.

Jessica and Michael completed their tasks without further communication.

Now describe the following video in the same format.
Agent names: {name1}, {name2}"""


def extract_agent_names(text: str) -> tuple[str, str]:
    seen: list[str] = []
    for name in NAME_RE.findall(text):
        if name not in _SKIP and name not in seen:
            seen.append(name)
        if len(seen) == 2:
            break
    return (seen[0], seen[1]) if len(seen) >= 2 else ('Agent1', 'Agent2')


def build_episode_sg_type(questions: list[QuestionRecord]) -> dict[str, str]:
    """Extract help/hinder per episode from the MOST social_goal question answer."""
    result: dict[str, str] = {}
    for record in questions:
        if record.question_type != 'social_goal':
            continue
        if 'MOST' not in record.question:
            continue
        ans = record.answer.lower()
        if 'prevent' in ans or 'hinder' in ans:
            result[record.episode_id] = 'hinder'
        elif 'help' in ans or 'locate' in ans:
            result[record.episode_id] = 'help'
    return result


def select_pilot(
    questions: list[QuestionRecord],
    episode_sg_type: dict[str, str],
    seed: int = 42,
    n_per_type: int = 50,
) -> list[QuestionRecord]:
    """150 questions by default: belief=50, social_goal=50 (help=25, hinder=25), belief_of_goal=50."""
    rng = random.Random(seed)

    by_type: dict[str, list[QuestionRecord]] = {}
    for q in questions:
        by_type.setdefault(q.question_type, []).append(q)

    belief = rng.sample(by_type.get('belief', []), n_per_type)
    bog = rng.sample(by_type.get('belief_of_goal', []), n_per_type)

    sg_all = by_type.get('social_goal', [])
    sg_help = [q for q in sg_all if episode_sg_type.get(q.episode_id) == 'help']
    sg_hinder = [q for q in sg_all if episode_sg_type.get(q.episode_id) == 'hinder']
    half = n_per_type // 2
    sg = rng.sample(sg_help, half) + rng.sample(sg_hinder, n_per_type - half)

    return belief + sg + bog


def parse_letter(text: str) -> str | None:
    import json as _json
    try:
        parsed = _json.loads(text)
        letter = str(parsed.get('choice_letter', '')).strip().upper()
        if letter in {'A', 'B', 'C'}:
            return letter
    except Exception:
        pass
    m = ANSWER_RE.search(text.upper())
    return m.group(1) if m else None


@dataclass
class HalluResult:
    question_id: str
    episode_id: str
    condition: str
    question_type: str
    social_goal_type: str   # "help" | "hinder" | "N/A"
    gt_answer: str
    model_answer: str
    correct: bool
    vlm_output: str         # C3 LLaMA text; empty for C1/C2


def result_to_dict(r: HalluResult) -> dict:
    return {
        'question_id': r.question_id,
        'episode_id': r.episode_id,
        'condition': r.condition,
        'question_type': r.question_type,
        'social_goal_type': r.social_goal_type,
        'gt_answer': r.gt_answer,
        'model_answer': r.model_answer,
        'correct': r.correct,
        'vlm_output': r.vlm_output,
    }