from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import requests
from openai import OpenAI
from tqdm import tqdm

from hallucination_data import (
    HalluResult,
    QA_PROMPT,
    QA_SYSTEM_PROMPT,
    VLM_PROMPT,
    build_episode_sg_type,
    extract_agent_names,
    parse_letter,
    result_to_dict,
    select_pilot,
)
from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_video import sample_video_frames

OLLAMA_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'llama3.2-vision:11b'


# ---------------------------------------------------------------------------
# GPT-4o calls
# ---------------------------------------------------------------------------

def _format_qa_messages(prompt: str, content=None) -> list:
    user_content = content if content is not None else prompt
    return [
        {'role': 'system', 'content': QA_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content},
    ]


def _qa_response(client: OpenAI, model: str, text: str, record) -> str:
    prompt = QA_PROMPT.format(
        question_type=record.question_type,
        text=text,
        question=record.question,
        option_a=record.choices[0],
        option_b=record.choices[1],
        option_c=record.choices[2],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=_format_qa_messages(prompt),
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _qa_vision_response(client: OpenAI, model: str, text: str, record, frames) -> str:
    prompt = QA_PROMPT.format(
        question_type=record.question_type,
        text=text,
        question=record.question,
        option_a=record.choices[0],
        option_b=record.choices[1],
        option_c=record.choices[2],
    )
    content: list = [{'type': 'text', 'text': prompt}]
    for frame in frames:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{frame.image_b64}',
                'detail': 'low',
            },
        })
    resp = client.chat.completions.create(
        model=model,
        messages=_format_qa_messages(prompt, content),
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# LLaMA vision call (ollama)
# ---------------------------------------------------------------------------

def _tile_frames(frames) -> str:
    """Tile all frames into a single grid image (llama3.2-vision supports only 1 image)."""
    import base64
    import io
    from PIL import Image

    cols = 5
    images = [Image.open(io.BytesIO(base64.b64decode(f.image_b64))) for f in frames]
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    buf = io.BytesIO()
    grid.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _llama_describe(frames, name1: str, name2: str) -> str:
    prompt = VLM_PROMPT.format(name1=name1, name2=name2)
    # Use at most 10 frames for tiling — keeps grid small enough for reliable inference
    grid_b64 = _tile_frames(frames)
    payload = {
        'model': OLLAMA_MODEL,
        'messages': [{'role': 'user', 'content': prompt, 'images': [grid_b64]}],
        'stream': False,
        'options': {'temperature': 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    if 'error' in data:
        raise RuntimeError(f'ollama error: {data["error"]}')
    return data['message']['content']


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)['question_id'])
    if done:
        print(f'  Resuming: {len(done)} already done.')
    return done


def _log_error(path: Path, qid: str, condition: str, err: str) -> None:
    entries: list = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    entries.append({'question_id': qid, 'condition': condition, 'error': err})
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  [ERROR] {condition} {qid}: {err}')


def _write(f, result: HalluResult) -> None:
    f.write(json.dumps(result_to_dict(result), ensure_ascii=False) + '\n')
    f.flush()


def _make_result(record, condition, episode_sg_type, letter, model_answer, vlm_output=''):
    return HalluResult(
        question_id=record.question_id,
        episode_id=record.episode_id,
        condition=condition,
        question_type=record.question_type,
        social_goal_type=episode_sg_type.get(record.episode_id, 'N/A'),
        gt_answer=record.answer,
        model_answer=model_answer,
        correct=(model_answer == record.answer),
        vlm_output=vlm_output,
    )


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_c1(client, model, pilot, episode_sg_type, out_path, err_path):
    done = _load_done(out_path)
    with out_path.open('a', encoding='utf-8') as f:
        for record in tqdm(pilot, desc='C1'):
            if record.question_id in done:
                continue
            try:
                raw = _qa_response(client, model, record.text_context or '', record)
                letter = parse_letter(raw)
                if not letter:
                    raise ValueError(f'no A/B/C in: {raw[:100]}')
                model_answer = record.choices[ord(letter) - ord('A')]
                _write(f, _make_result(record, 'C1', episode_sg_type, letter, model_answer))
            except Exception as e:
                _log_error(err_path, record.question_id, 'C1', str(e))


def run_c2(client, model, settings, pilot, episode_sg_type, out_path, err_path):
    done = _load_done(out_path)
    with out_path.open('a', encoding='utf-8') as f:
        for record in tqdm(pilot, desc='C2'):
            if record.question_id in done:
                continue
            try:
                frames = sample_video_frames(
                    record.video_path,
                    frame_stride=settings.frame_stride,
                    max_frames=settings.max_frames,
                )
                raw = _qa_vision_response(client, model, record.text_context or '', record, frames)
                letter = parse_letter(raw)
                if not letter:
                    raise ValueError(f'no A/B/C in: {raw[:100]}')
                model_answer = record.choices[ord(letter) - ord('A')]
                _write(f, _make_result(record, 'C2', episode_sg_type, letter, model_answer))
            except Exception as e:
                _log_error(err_path, record.question_id, 'C2', str(e))


def run_c3(client, model, settings, pilot, episode_sg_type, out_path, vlm_dir, err_path):
    done = _load_done(out_path)
    vlm_dir.mkdir(exist_ok=True)
    with out_path.open('a', encoding='utf-8') as f:
        for record in tqdm(pilot, desc='C3'):
            if record.question_id in done:
                continue
            try:
                vlm_path = vlm_dir / f'{record.episode_id}.txt'
                if vlm_path.exists():
                    vlm_text = vlm_path.read_text(encoding='utf-8')
                else:
                    frames = sample_video_frames(
                        record.video_path,
                        frame_stride=settings.frame_stride,
                        max_frames=settings.max_frames,
                    )
                    n1, n2 = extract_agent_names(record.text_context or '')
                    vlm_text = _llama_describe(frames, n1, n2)
                    vlm_path.write_text(vlm_text, encoding='utf-8')

                raw = _qa_response(client, model, vlm_text, record)
                letter = parse_letter(raw)
                if not letter:
                    raise ValueError(f'no A/B/C in: {raw[:100]}')
                model_answer = record.choices[ord(letter) - ord('A')]
                _write(f, _make_result(record, 'C3', episode_sg_type, letter, model_answer, vlm_text))
            except Exception as e:
                _log_error(err_path, record.question_id, 'C3', str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(out_path: Path) -> dict:
    if not out_path.exists():
        return {}
    rows = [json.loads(l) for l in out_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not rows:
        return {}

    total = len(rows)
    correct = sum(r['correct'] for r in rows)
    print(f'\n=== {out_path.stem}: {correct}/{total} = {correct/total:.1%} ===')

    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_type[r['question_type']].append(r['correct'])

    acc_by_type = {}
    for qtype, bools in sorted(by_type.items()):
        acc = sum(bools) / len(bools)
        acc_by_type[qtype] = acc
        print(f'  {qtype}: {sum(bools)}/{len(bools)} = {acc:.1%}')

    sg = [r for r in rows if r['question_type'] == 'social_goal']
    for sgt in ('help', 'hinder'):
        sub = [r['correct'] for r in sg if r.get('social_goal_type') == sgt]
        if sub:
            a = sum(sub) / len(sub)
            acc_by_type[f'social_goal/{sgt}'] = a
            print(f'  social_goal/{sgt}: {sum(sub)}/{len(sub)} = {a:.1%}')

    return {'condition': out_path.stem.replace('results_', ''), 'total': total,
            'overall': correct / total, **acc_by_type}


def write_csv(summaries: list[dict], out_path: Path) -> None:
    if not summaries:
        return
    keys = ['condition', 'total', 'overall', 'belief', 'social_goal', 'belief_of_goal',
            'social_goal/help', 'social_goal/hinder']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    print(f'\nSaved: {out_path}')


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def check_gpu_for_c3() -> None:
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print('\n[C3] GPU free memory (MiB):')
            for line in result.stdout.strip().splitlines():
                idx, free = line.split(',')
                print(f'  GPU {idx.strip()}: {int(free.strip()):,} MiB free')
        # quick connectivity test
        resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        models = [m['name'] for m in resp.json().get('models', [])]
        if OLLAMA_MODEL in models:
            print(f'  ollama: {OLLAMA_MODEL} ready')
        else:
            print(f'  WARNING: {OLLAMA_MODEL} not in ollama ({models})')
    except Exception as e:
        print(f'  [GPU check failed] {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='MuMA-ToM Hallucination Pilot')
    parser.add_argument('--condition', choices=['C1', 'C2', 'C3', 'all'], default='C1')
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('hallucination_results'))
    parser.add_argument('--all-questions', action='store_true', help='전체 900개 사용 (샘플링 없음)')
    args = parser.parse_args()

    settings = load_settings()
    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)

    episode_sg_type = build_episode_sg_type(questions)
    if args.all_questions:
        pilot = questions
    else:
        pilot = select_pilot(questions, episode_sg_type, seed=args.seed)

    type_counts = Counter(q.question_type for q in pilot)
    sg_counts = Counter(
        episode_sg_type.get(q.episode_id, 'unknown')
        for q in pilot if q.question_type == 'social_goal'
    )
    print(f'Pilot: {len(pilot)} questions — {dict(type_counts)}')
    print(f'  social_goal breakdown: {dict(sg_counts)}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    err_path = args.output_dir / 'error_log.json'
    vlm_dir = args.output_dir / 'vlm_outputs'

    client = OpenAI(api_key=settings.openai_api_key)
    conditions = ['C1', 'C2', 'C3'] if args.condition == 'all' else [args.condition]

    summaries = []
    for cond in conditions:
        out = args.output_dir / f'results_{cond}.jsonl'
        if cond == 'C1':
            run_c1(client, args.model, pilot, episode_sg_type, out, err_path)
        elif cond == 'C2':
            run_c2(client, args.model, settings, pilot, episode_sg_type, out, err_path)
        elif cond == 'C3':
            check_gpu_for_c3()
            run_c3(client, args.model, settings, pilot, episode_sg_type, out, vlm_dir, err_path)
        s = print_summary(out)
        if s:
            summaries.append(s)

    if summaries:
        write_csv(summaries, args.output_dir / 'accuracy_summary.csv')


if __name__ == '__main__':
    main()