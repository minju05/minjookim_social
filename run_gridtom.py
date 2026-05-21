"""
GridToM experiment runner.

Usage:
  python run_gridtom.py --condition text_only
  python run_gridtom.py --condition text_frame
  python run_gridtom.py --condition generic_cot
  python run_gridtom.py --condition qlogic
  python run_gridtom.py --profile   # degradation profile after all 4 runs
  python run_gridtom.py --condition text_only --limit 10  # quick test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minjookim_social"))
from muma_config import load_settings
from muma_gpt4o import prediction_to_dict

from gridtom_data import GridToMRecord, load_records, load_frames_b64
from gridtom_gpt4o import predict_gridtom

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "gridtom"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["qa_only", "text_only", "text_frame", "generic_cot", "qlogic"]


def run(condition: str, limit: int | None, belief_type: str) -> None:
    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    records = load_records(belief_type)
    if limit:
        records = records[:limit]

    out_path = OUTPUT_DIR / f"gridtom_{condition}_{belief_type}.jsonl"
    print(f"[{condition}] n={len(records)} → {out_path}")

    correct = 0
    with out_path.open("w") as f:
        for i, rec in enumerate(tqdm(records, desc=condition)):
            frames = load_frames_b64(rec) if condition == "text_frame" else []
            pred = predict_gridtom(client, settings, rec, frames, condition=condition)
            if pred.correct:
                correct += 1
            f.write(json.dumps(prediction_to_dict(pred), ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"  acc={correct/(i+1)*100:.1f}%  [{i+1}/{len(records)}]")

    acc = correct / len(records) * 100 if records else 0
    print(f"\nFinal: {acc:.1f}%  ({correct}/{len(records)})")

    # per-type breakdown
    by_type: dict[str, list[bool]] = defaultdict(list)
    with out_path.open() as f:
        for line in f:
            d = json.loads(line)
            by_type[d["question_type"]].append(d["correct"])
    for bt, vals in sorted(by_type.items()):
        print(f"  {bt}: {sum(vals)/len(vals)*100:.1f}%  ({sum(vals)}/{len(vals)})")


def compute_profile() -> None:
    def load(path: Path) -> dict[tuple, dict]:
        if not path.exists():
            return {}
        out = {}
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                key = (d["episode_id"], d["question_type"])
                out[key] = d
        return out

    text_only  = load(OUTPUT_DIR / "gridtom_text_only_both.jsonl")
    text_frame = load(OUTPUT_DIR / "gridtom_text_frame_both.jsonl")
    cot        = load(OUTPUT_DIR / "gridtom_generic_cot_both.jsonl")
    qlogic     = load(OUTPUT_DIR / "gridtom_qlogic_both.jsonl")

    if not text_only or not text_frame:
        print("Need text_only and text_frame results first.")
        return

    keys = set(text_only) & set(text_frame)
    degradation, frame_only, wrong_all, both_correct = [], [], [], []
    for k in keys:
        to_c = text_only[k]["correct"]
        tf_c = text_frame[k]["correct"]
        if to_c and not tf_c:
            degradation.append(k)
        elif not to_c and tf_c:
            frame_only.append(k)
        elif not to_c and not tf_c:
            wrong_all.append(k)
        else:
            both_correct.append(k)

    def recovery(subset, lookup):
        if not lookup or not subset:
            return None
        n = sum(1 for k in subset if lookup.get(k, {}).get("correct", False))
        return n, len(subset)

    n = len(keys)
    to_acc = sum(text_only[k]["correct"] for k in keys) / n * 100
    tf_acc = sum(text_frame[k]["correct"] for k in keys) / n * 100

    print("\n── GridToM Degradation Profile ──")
    print(f"n={n}  Text-only={to_acc:.1f}%  Text+Frame={tf_acc:.1f}%  Δ={tf_acc-to_acc:+.1f}pp")
    print(f"  Degradation (Text✓→Frame✗): {len(degradation)}")
    print(f"  Frame-only  (Text✗→Frame✓): {len(frame_only)}")
    print(f"  Wrong-all                 : {len(wrong_all)}")
    print(f"  Both-correct              : {len(both_correct)}")

    if cot:
        r = recovery(degradation, cot)
        print(f"\n  CoT    recovery on Degradation : {r[0]}/{r[1]}  ({r[0]/r[1]*100:.1f}%)")
    if qlogic:
        r = recovery(degradation, qlogic)
        print(f"  QLogic recovery on Degradation : {r[0]}/{r[1]}  ({r[0]/r[1]*100:.1f}%)")
        r2 = recovery(wrong_all, qlogic)
        if r2:
            print(f"  QLogic recovery on Wrong-all   : {r2[0]}/{r2[1]}  ({r2[0]/r2[1]*100:.1f}%)")

    profile = {
        "n": n,
        "text_only_acc": round(to_acc, 1),
        "text_frame_acc": round(tf_acc, 1),
        "degradation": [list(k) for k in degradation],
        "frame_only":  [list(k) for k in frame_only],
        "wrong_all":   [list(k) for k in wrong_all],
        "both_correct":[list(k) for k in both_correct],
    }
    out = OUTPUT_DIR / "degradation_profile.json"
    out.write_text(json.dumps(profile, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--belief-type", default="both",
                        choices=["TrueBelief", "FalseBelief", "both"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", action="store_true",
                        help="Compute degradation profile from saved results")
    args = parser.parse_args()

    if args.profile:
        compute_profile()
    elif args.condition:
        run(args.condition, args.limit, args.belief_type)
    else:
        parser.print_help()
