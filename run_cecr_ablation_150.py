"""
CECR Ablation — 150q (50×3 types), Video 버전
GitHub labels 기준으로 belief/bog/sg 각 50개 균등 샘플링 (seed=2025)

Variants:
  - v9_cecr       : 기존 cecr_gpt4o_video_900.jsonl에서 필터링 (API 호출 없음)
  - v1_minju      : 기존 muma_gpt4o_paper_minju.jsonl에서 필터링 (API 호출 없음)
  - cecr_wo_step1 : 새 API 호출 필요
  - cecr_wo_step2 : 새 API 호출 필요

출력:
  outputs/ablation150_v9_cecr.jsonl
  outputs/ablation150_baseline.jsonl
  outputs/ablation150_wo_step1.jsonl
  outputs/ablation150_wo_step2.jsonl
  sampled_150_questions.json
"""
import json
import random
import time
from pathlib import Path

import openai
from openai import OpenAI

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_gpt4o import prediction_to_dict, predict_question
from muma_video import sample_video_frames

# ── 설정 ─────────────────────────────────────────────────────────────────────
settings = load_settings()
dataset_dir = ensure_dataset(settings)
questions = load_questions(dataset_dir)

SEED = 2025
N_PER_TYPE = 50
SAMPLE_FILE = Path("sampled_150_questions.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0

# 기존 900q 결과에서 필터링할 variant (API 호출 없음)
PRECOMPUTED = [
    ("v9_cecr",  "outputs/cecr_gpt4o_video_900.jsonl",    "outputs/ablation150_v9_cecr.jsonl"),
    ("v1_minju", "outputs/muma_gpt4o_paper_minju.jsonl",  "outputs/ablation150_baseline.jsonl"),
]

# 새로 API 호출이 필요한 variant
NEW_VARIANTS = [
    ("cecr_wo_step1", "outputs/ablation150_wo_step1.jsonl"),
    ("cecr_wo_step2", "outputs/ablation150_wo_step2.jsonl"),
]

# 요약 출력용 (순서)
ALL_VARIANTS = [
    ("CECR Full",    "outputs/ablation150_v9_cecr.jsonl"),
    ("w/o Step1",    "outputs/ablation150_wo_step1.jsonl"),
    ("w/o Step2",    "outputs/ablation150_wo_step2.jsonl"),
    ("Baseline",     "outputs/ablation150_baseline.jsonl"),
]

# ── 150q 샘플링 (GitHub labels 기준, 50×3) ───────────────────────────────────
def sample_150(questions, seed=SEED, n=N_PER_TYPE):
    rng = random.Random(seed)
    by_type = {"belief": [], "belief_of_goal": [], "social_goal": []}
    for q in questions:
        if q.question_type in by_type:
            by_type[q.question_type].append(q)
    sampled = []
    for qt, qs in by_type.items():
        chosen = rng.sample(qs, min(n, len(qs)))
        sampled.extend(chosen)
        print(f"  {qt:<18}: {len(chosen)}개 샘플링")
    return sampled


if SAMPLE_FILE.exists():
    print(f"[샘플링] {SAMPLE_FILE} 이미 존재 — 로드합니다.")
    sampled_ids_ordered = [r["question_id"] for r in json.loads(SAMPLE_FILE.read_text())]
    id_to_q = {q.question_id: q for q in questions}
    test_records = [id_to_q[qid] for qid in sampled_ids_ordered if qid in id_to_q]
else:
    print("[샘플링] 150q 샘플링 중...")
    test_records = sample_150(questions)
    SAMPLE_FILE.write_text(
        json.dumps(
            [{"question_id": q.question_id, "question_type": q.question_type} for q in test_records],
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    print(f"  → {SAMPLE_FILE} 저장 완료 (총 {len(test_records)}개)")

print(f"\n총 {len(test_records)}개 문항 (video 포함)")
from collections import Counter
type_dist = Counter(q.question_type for q in test_records)
print(f"분포: {dict(type_dist)}\n")

client = OpenAI(api_key=settings.openai_api_key)

# ── 기존 900q 결과에서 150q 필터링 ───────────────────────────────────────────
sampled_ids = {q.question_id for q in test_records}

for variant_name, src_file, dst_file in PRECOMPUTED:
    dst = Path(dst_file)
    if dst.exists():
        print(f"[필터링] {variant_name}: {dst_file} 이미 존재 — 스킵")
        continue
    src = Path(src_file)
    if not src.exists():
        print(f"[필터링] {variant_name}: {src_file} 없음 — 스킵")
        continue
    filtered = []
    for line in src.read_text(encoding="utf-8").strip().splitlines():
        r = json.loads(line)
        if r["question_id"] in sampled_ids:
            filtered.append(line)
    dst.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    print(f"[필터링] {variant_name}: {len(filtered)}개 → {dst_file}")


def predict_with_retry(client, settings, record, frames, prompt_version, max_retries=MAX_RETRIES):
    delay = RETRY_BASE_DELAY
    for attempt in range(max_retries):
        try:
            return predict_question(client, settings, record, frames, prompt_version=prompt_version)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  RateLimitError — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2
        except openai.APIStatusError as e:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  APIStatusError {e.status_code} — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2
        except openai.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  APIConnectionError — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2


def run_variant(variant_name: str, output_file: str):
    output_path = Path(output_file)
    n_total = len(test_records)

    # Resume
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                done_ids.add(json.loads(line)["question_id"])
            except Exception:
                pass

    remaining = [q for q in test_records if q.question_id not in done_ids]
    if not remaining:
        print(f"  ✅ {variant_name}: 이미 완료 ({n_total}/{n_total})")
        return

    print(f"  진행: {len(done_ids)}/{n_total} 완료, {len(remaining)}개 남음")

    correct_this_run = 0
    total_this_run = 0
    start_time = time.time()

    with output_path.open("a", encoding="utf-8") as f:
        for record in remaining:
            frames = sample_video_frames(
                record.video_path,
                frame_stride=settings.frame_stride,
                max_frames=settings.max_frames,
            )
            pred = predict_with_retry(client, settings, record, frames, prompt_version=variant_name)
            f.write(json.dumps(prediction_to_dict(pred), ensure_ascii=False) + "\n")
            f.flush()

            total_this_run += 1
            if pred.correct:
                correct_this_run += 1

            done_total = len(done_ids) + total_this_run
            elapsed = time.time() - start_time
            avg_sec = elapsed / total_this_run
            eta_sec = avg_sec * (n_total - done_total)
            mark = "✅" if pred.correct else "❌"
            acc = correct_this_run / total_this_run * 100
            print(
                f"    [{done_total:3d}/{n_total}] {mark} {pred.question_id:<15} "
                f"({record.question_type:<18}) acc={acc:.1f}% ETA={eta_sec/60:.1f}m"
            )


# ── 새 API 호출 variant 실행 ──────────────────────────────────────────────────
for variant_name, output_file in NEW_VARIANTS:
    print("=" * 60)
    print(f"▶ {variant_name}")
    run_variant(variant_name, output_file)

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Method':<22} {'belief':^12} {'bog':^12} {'sg':^12} {'Total':^14}")
print("-" * 70)

QUESTION_TYPES = ["belief", "belief_of_goal", "social_goal"]

for label, output_file in ALL_VARIANTS:
    path = Path(output_file)
    if not path.exists():
        continue
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
    total_c = sum(r["correct"] for r in records)
    total_t = len(records)
    by_type = {}
    for qt in QUESTION_TYPES:
        subset = [r for r in records if r["question_type"] == qt]
        c = sum(r["correct"] for r in subset)
        by_type[qt] = (c, len(subset))

    b_c, b_t = by_type.get("belief", (0, 0))
    bog_c, bog_t = by_type.get("belief_of_goal", (0, 0))
    sg_c, sg_t = by_type.get("social_goal", (0, 0))

    print(
        f"{label:<22} "
        f"{b_c}/{b_t}({b_c/b_t*100:.1f}%)".center(12) + " " if b_t else f"{'—':^12} ",
        end=""
    )
    print(
        f"{bog_c}/{bog_t}({bog_c/bog_t*100:.1f}%)".center(12) + " " if bog_t else f"{'—':^12} ",
        end=""
    )
    print(
        f"{sg_c}/{sg_t}({sg_c/sg_t*100:.1f}%)".center(12) + " " if sg_t else f"{'—':^12} ",
        end=""
    )
    print(f"{total_c}/{total_t}({total_c/total_t*100:.1f}%)" if total_t else "")
