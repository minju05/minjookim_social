"""
Run MMToM-QA experiments for Table 5 (Dataset B).

Usage:
  python run_mmtom.py --condition text_only --limit 100
  python run_mmtom.py --condition text_frame --limit 100
  python run_mmtom.py --condition generic_cot --limit 100
  python run_mmtom.py --condition qlogic --limit 100
  python run_mmtom.py --profile
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, '/home/seohyeon/.00_project/26-01-SAI/minjookim_social')
from muma_config import load_settings
from muma_gpt4o import prediction_to_dict

from mmtom_data import load_records
from mmtom_gpt4o import predict_mmtom

OUTPUT_DIR = Path('/home/seohyeon/.00_project/26-01-SAI/minjookim_social_seohyeon/outputs/mmtom')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ['text_only', 'text_frame', 'generic_cot', 'qlogic']


def run(condition: str, limit: int | None, belief_type: str = 'both') -> None:
    settings = load_settings()
    records = load_records(belief_type)
    if limit:
        records = records[:limit]

    out_path = OUTPUT_DIR / f'mmtom_{condition}_{belief_type}.jsonl'
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            done_ids.add(json.loads(line)['question_id'])

    total = len(records)
    correct = 0
    with out_path.open('a') as f:
        for i, rec in enumerate(records):
            if rec.question_id in done_ids:
                continue
            try:
                pred = predict_mmtom(rec, condition, settings)
            except Exception as e:
                print(f"  ERROR {rec.question_id}: {e}")
                continue
            f.write(json.dumps(prediction_to_dict(pred)) + '\n')
            f.flush()
            if pred.correct:
                correct += 1
            if (i + 1) % 10 == 0:
                done_so_far = i + 1
                print(f"[{done_so_far}/{total}] acc so far: {correct}/{done_so_far} = {correct/done_so_far*100:.1f}%")

    all_preds = [json.loads(l) for l in out_path.open()]
    n = len(all_preds)
    c = sum(p['correct'] for p in all_preds)
    print(f"\n=== {condition} | {belief_type} ===")
    print(f"Total: {c}/{n} = {c/n*100:.1f}%")


def compute_profile() -> None:
    text_only_path  = OUTPUT_DIR / 'mmtom_text_only_both.jsonl'
    text_frame_path = OUTPUT_DIR / 'mmtom_text_frame_both.jsonl'

    if not text_only_path.exists() or not text_frame_path.exists():
        print("Need both text_only and text_frame results first.")
        return

    to_preds = {json.loads(l)['question_id']: json.loads(l)
                for l in text_only_path.open()}
    tf_preds = {json.loads(l)['question_id']: json.loads(l)
                for l in text_frame_path.open()}

    common = set(to_preds) & set(tf_preds)
    degradation_ids = [qid for qid in common
                       if to_preds[qid]['correct'] and not tf_preds[qid]['correct']]
    frame_only_ids  = [qid for qid in common
                       if not to_preds[qid]['correct'] and tf_preds[qid]['correct']]
    both_correct    = [qid for qid in common
                       if to_preds[qid]['correct'] and tf_preds[qid]['correct']]
    wrong_all       = [qid for qid in common
                       if not to_preds[qid]['correct'] and not tf_preds[qid]['correct']]

    profile = {
        'n_common': len(common),
        'text_only_acc':   sum(to_preds[q]['correct'] for q in common) / len(common),
        'text_frame_acc':  sum(tf_preds[q]['correct'] for q in common) / len(common),
        'degradation_n':   len(degradation_ids),
        'degradation_ids': degradation_ids,
        'frame_only_n':    len(frame_only_ids),
        'frame_only_ids':  frame_only_ids,
        'both_correct_n':  len(both_correct),
        'wrong_all_n':     len(wrong_all),
        'wrong_all_ids':   wrong_all,
    }

    # CoT and QLogic recovery on degradation set
    for cond in ['generic_cot', 'qlogic']:
        cond_path = OUTPUT_DIR / f'mmtom_{cond}_both.jsonl'
        if cond_path.exists():
            cond_preds = {json.loads(l)['question_id']: json.loads(l)
                          for l in cond_path.open()}
            recovered = sum(1 for qid in degradation_ids
                            if cond_preds.get(qid, {}).get('correct', False))
            profile[f'{cond}_recovery_n'] = recovered
            if degradation_ids:
                profile[f'{cond}_recovery_rate'] = recovered / len(degradation_ids)

    out = OUTPUT_DIR / 'mmtom_degradation_profile.json'
    out.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--belief-type', default='both', choices=['both', '1.2', '2.1'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    if args.profile:
        compute_profile()
    elif args.condition:
        run(args.condition, args.limit, args.belief_type)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
