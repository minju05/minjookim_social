from __future__ import annotations

import base64
import json
import pickle
import re
from pathlib import Path

import sys
sys.path.insert(0, '/home/seohyeon/.00_project/26-01-SAI/minjookim_social')
from muma_data import QuestionRecord

QUESTIONS_PATH = Path('/home/seohyeon/.00_project/26-01-SAI/MMToM-QA/Benchmark/questions.json')
VIDEOS_DIR     = Path('/home/seohyeon/.00_project/26-01-SAI/MMToM-QA/videos')

SELECTED_EPISODES: dict[float, list[int]] = {
    1.2: [663, 65, 124, 147, 187, 202, 252, 289, 482, 527, 811, 966, 1079, 1150, 1196, 43, 61],
    2.1: [475, 611, 464, 1025, 901, 14, 118, 833, 883, 230, 333, 340, 418],
}

TYPE_LABELS: dict[float, str] = {
    1.2: 'False Belief (short-term)',
    2.1: 'Goal Inference (true belief)',
}

_OPT_RE = re.compile(r'\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*Please respond', re.DOTALL)


def _parse_question_field(raw: str) -> tuple[str, str, str, str]:
    parts = raw.split('\nQuestion: ', 1)
    text_context = parts[0].strip()
    q_part = parts[1].strip() if len(parts) > 1 else raw

    m = _OPT_RE.search(q_part)
    if m:
        question_text = q_part[:m.start()].strip()
        option_a = m.group(1).strip()
        option_b = m.group(2).strip()
    else:
        question_text = q_part
        option_a = option_b = ''
    return text_context, question_text, option_a, option_b


def load_records(belief_type: str = 'both') -> list[QuestionRecord]:
    if belief_type == '1.2':
        target_types: list[float] = [1.2]
    elif belief_type == '2.1':
        target_types = [2.1]
    else:
        target_types = [1.2, 2.1]

    raw_qs = [json.loads(l) for l in QUESTIONS_PATH.open()]
    records: list[QuestionRecord] = []

    for t in target_types:
        allowed_eps = set(SELECTED_EPISODES[t])
        type_qs = [q for q in raw_qs if q['question_type'] == t and q['episode'] in allowed_eps]

        for idx, q in enumerate(type_qs):
            text_ctx, question_text, opt_a, opt_b = _parse_question_field(q['question'])
            answer_letter = q['answer'].strip().lower()
            answer_text = opt_a if answer_letter == 'a' else opt_b

            records.append(QuestionRecord(
                question_id  = f"mmtom_{q['episode']}_{t}_{idx}",
                episode_id   = str(q['episode']),
                question_type= TYPE_LABELS[t],
                question     = question_text,
                choices      = [opt_a, opt_b, ''],
                answer       = answer_text,
                text_context = text_ctx,
                video_path   = VIDEOS_DIR / f"task_{q['episode']}",
                raw          = q,
            ))

    return records


def load_frames_b64(record: QuestionRecord, num_frames: int = 8) -> list[str]:
    ep_dir = record.video_path
    frames_dir = ep_dir / 'script' / '0'
    end_time: int = record.raw.get('end_time', 999)

    pik_path = ep_dir / 'frame_intervals.pik'
    if pik_path.exists():
        with pik_path.open('rb') as f:
            intervals = pickle.load(f)
        times = [action[1] for action in intervals]
        end_frame = times[end_time] if end_time < len(times) else times[-1]
    else:
        end_frame = end_time * 40  # fallback heuristic

    all_pngs = sorted(
        frames_dir.glob('Action_*_0_normal.png'),
        key=lambda p: int(p.stem.split('_')[1])
    )
    valid_pngs = [p for p in all_pngs if int(p.stem.split('_')[1]) <= end_frame]
    if not valid_pngs:
        valid_pngs = all_pngs

    n = len(valid_pngs)
    if n <= num_frames:
        selected = valid_pngs
    else:
        step = (n - 1) / (num_frames - 1)
        selected = [valid_pngs[round(i * step)] for i in range(num_frames)]

    result = []
    for p in selected:
        with p.open('rb') as f:
            result.append(base64.b64encode(f.read()).decode('ascii'))
    return result
