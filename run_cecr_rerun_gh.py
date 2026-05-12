"""
CECR GPT-4o 재실험 — GitHub labels 기준 (questions.json 교체 후)

Step 1: Text-Only 900q → outputs/cecr_gpt4o_textonly_900_gh.jsonl
Step 2: Video 900q     → outputs/cecr_gpt4o_video_900_gh.jsonl

Features:
  - Resume: 이미 완료된 question_id 자동 skip
  - Retry: 429 / 5xx 에러 시 exponential backoff (최대 5회)
  - tmux 친화적: 진행률 실시간 출력
  - text-only 완료 후 video 자동 시작
"""
import json
import time
import random
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
questions = load_questions(dataset_dir)  # GitHub labels 기준 900q

PROMPT_VERSION = "v9_cecr"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0

client = OpenAI(api_key=settings.openai_api_key)


# ── Retry wrapper ─────────────────────────────────────────────────────────────
def predict_with_retry(client, settings, record, frames, prompt_version, max_retries=MAX_RETRIES):
    delay = RETRY_BASE_DELAY
    for attempt in range(max_retries):
        try:
            return predict_question(client, settings, record, frames, prompt_version=prompt_version)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  ⚠️  RateLimitError — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2
        except openai.APIStatusError as e:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  ⚠️  APIStatusError {e.status_code} — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2
        except openai.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            wait = delay + random.uniform(0, 2)
            print(f"  ⚠️  APIConnectionError — {wait:.0f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            delay *= 2


# ── 공통 실행 함수 ────────────────────────────────────────────────────────────
def run_experiment(output_path: Path, use_video: bool, label: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                done_ids.add(json.loads(line)["question_id"])
            except Exception:
                pass

    remaining = [q for q in questions if q.question_id not in done_ids]
    n_total = len(questions)

    print(f"\n{'='*60}")
    print(f"▶  {label}")
    print(f"   Total   : {n_total}")
    print(f"   Done    : {len(done_ids)}")
    print(f"   Remaining: {len(remaining)}")
    print(f"   Output  : {output_path}")
    print(f"   Model   : {settings.openai_model} | prompt: {PROMPT_VERSION} | video: {use_video}")
    print(f"{'='*60}")

    if not remaining:
        print("✅ 이미 모두 완료되었습니다.")
        return

    total_done = len(done_ids)
    correct_this_run = 0
    total_this_run = 0
    start_time = time.time()

    with output_path.open("a", encoding="utf-8") as f:
        for record in remaining:
            if use_video:
                frames = sample_video_frames(
                    record.video_path,
                    frame_stride=settings.frame_stride,
                    max_frames=settings.max_frames,
                )
            else:
                frames = []

            pred = predict_with_retry(client, settings, record, frames, prompt_version=PROMPT_VERSION)

            f.write(json.dumps(prediction_to_dict(pred), ensure_ascii=False) + "\n")
            f.flush()

            total_done += 1
            total_this_run += 1
            if pred.correct:
                correct_this_run += 1

            elapsed = time.time() - start_time
            avg_sec = elapsed / total_this_run
            eta_sec = avg_sec * (n_total - total_done)

            mark = "✅" if pred.correct else "❌"
            acc = correct_this_run / total_this_run * 100
            print(
                f"[{total_done:3d}/{n_total}] {mark} {pred.question_id:<15} "
                f"({record.question_type:<15}) "
                f"acc={acc:.1f}% "
                f"ETA={eta_sec/60:.1f}m"
            )

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"✅ {label} 완료! 집계 중...")
    all_records = [json.loads(l) for l in output_path.read_text(encoding="utf-8").strip().splitlines()]
    total_correct = sum(r["correct"] for r in all_records)
    by_type = {}
    for qt in ["belief", "belief_of_goal", "social_goal"]:
        subset = [r for r in all_records if r["question_type"] == qt]
        c = sum(r["correct"] for r in subset)
        by_type[qt] = (c, len(subset))

    print(f"\n{label} — 900q 전체 결과 (GitHub labels)")
    print(f"  Total         : {total_correct}/{len(all_records)} ({total_correct/len(all_records)*100:.1f}%)")
    for qt, (c, t) in by_type.items():
        pct = c/t*100 if t > 0 else 0
        print(f"  {qt:<18}: {c}/{t} ({pct:.1f}%)")


# ── Step 1: Text-Only ─────────────────────────────────────────────────────────
run_experiment(
    output_path=Path("outputs/cecr_gpt4o_textonly_900_gh.jsonl"),
    use_video=False,
    label="CECR Text-Only (GitHub labels)",
)

# ── Step 2: Video ─────────────────────────────────────────────────────────────
run_experiment(
    output_path=Path("outputs/cecr_gpt4o_video_900_gh.jsonl"),
    use_video=True,
    label="CECR Video (GitHub labels)",
)

print("\n\n🎉 전체 재실험 완료!")
print("  - outputs/cecr_gpt4o_textonly_900_gh.jsonl")
print("  - outputs/cecr_gpt4o_video_900_gh.jsonl")
