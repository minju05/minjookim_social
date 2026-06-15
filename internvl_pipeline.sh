#!/usr/bin/env bash
# internvl_pipeline.sh — InternVL3-14B 6개 조건 병렬 실행
# GPU: 4x L40S → 4조건 동시 실행 후 나머지 2조건
# Usage: bash internvl_pipeline.sh

set -e
cd "$(dirname "$0")"
source /home/mindrium-admin3/.venv/bin/activate

MODEL_PATH="/home/mindrium-admin3/models/InternVL3-14B"
PY="python3 run_muma_internvl.py --limit 900 --model-path $MODEL_PATH"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo "[pipeline] Starting InternVL3-14B experiments (6 conditions, 4-GPU parallel)"
echo "[pipeline] Logs → $LOG_DIR"

# ── Batch 1: 4 conditions in parallel (GPU 0-3) ──────────────────────────────
echo "[Batch 1/2] QA-only(GPU0) | Video-only(GPU1) | Text-only(GPU2) | Text+Frame(GPU3) ..."

CUDA_VISIBLE_DEVICES=0 $PY \
  --prompt-version video_only --no-frames \
  --output outputs/internvl_cond5_qaonly_900.jsonl \
  > "$LOG_DIR/internvl_qaonly.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 $PY \
  --prompt-version video_only \
  --output outputs/internvl_cond4_videoonly_900.jsonl \
  > "$LOG_DIR/internvl_videoonly.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 $PY \
  --prompt-version text_only --no-frames \
  --output outputs/internvl_cond1_textonly_900.jsonl \
  > "$LOG_DIR/internvl_textonly.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 $PY \
  --prompt-version text_only \
  --output outputs/internvl_cond2_textframe_900.jsonl \
  > "$LOG_DIR/internvl_textframe.log" 2>&1 &
PID3=$!

wait $PID0 && echo "[1/6] QA-only done"    || echo "[1/6] QA-only FAILED"
wait $PID1 && echo "[2/6] Video-only done" || echo "[2/6] Video-only FAILED"
wait $PID2 && echo "[3/6] Text-only done"  || echo "[3/6] Text-only FAILED"
wait $PID3 && echo "[4/6] Text+Frame done" || echo "[4/6] Text+Frame FAILED"

echo "[Batch 1/2] Complete."

# ── Batch 2: CoT + QLogic in parallel (GPU 0, 1) ─────────────────────────────
echo "[Batch 2/2] Generic CoT(GPU0) | QLogic(GPU1) ..."

CUDA_VISIBLE_DEVICES=0 $PY \
  --prompt-version generic_cot --no-frames \
  --output outputs/internvl_cot_900.jsonl \
  > "$LOG_DIR/internvl_cot.log" 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES=1 $PY \
  --prompt-version qlogic --no-frames \
  --output outputs/internvl_qlogic_900.jsonl \
  > "$LOG_DIR/internvl_qlogic.log" 2>&1 &
PID5=$!

wait $PID4 && echo "[5/6] Generic CoT done" || echo "[5/6] Generic CoT FAILED"
wait $PID5 && echo "[6/6] QLogic done"      || echo "[6/6] QLogic FAILED"

echo "[Batch 2/2] Complete."

echo ""
echo "=== All 6 conditions complete ==="
python3 - << 'PYEOF'
import json, os
files = {
    'QA-only':    'outputs/internvl_cond5_qaonly_900.jsonl',
    'Video-only': 'outputs/internvl_cond4_videoonly_900.jsonl',
    'Text-only':  'outputs/internvl_cond1_textonly_900.jsonl',
    'Text+Frame': 'outputs/internvl_cond2_textframe_900.jsonl',
    'Generic CoT':'outputs/internvl_cot_900.jsonl',
    'QLogic':     'outputs/internvl_qlogic_900.jsonl',
}
print(f"{'Condition':<14} {'Correct':>8} {'Total':>7} {'Acc%':>7} {'ERR':>6}")
print('-'*48)
for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name:<14} {'(missing)':>8}")
        continue
    rows = [json.loads(l) for l in open(path)]
    total = len(rows)
    err   = sum(1 for r in rows if r.get('predicted_letter') == 'ERR')
    correct = sum(1 for r in rows if r.get('correct'))
    denom = total - err
    acc = correct / denom * 100 if denom else 0
    print(f"{name:<14} {correct:>8} {total:>7} {acc:>7.1f} {err:>6}")
PYEOF
