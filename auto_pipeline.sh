#!/bin/bash
set -e
cd /home/mindrium-admin3/minjookim_social
source /home/mindrium-admin3/.venv/bin/activate

COND2_FILE="outputs/gemini_cond2_text_frames_900.jsonl"
COND4_FILE="outputs/gemini_cond4_videoonly_900.jsonl"
LOG="outputs/auto_pipeline.log"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# ── 1. cond2 완료 대기 ──────────────────────────────────────────
log "=== STEP 1: cond2 완료 대기 (900줄) ==="
while true; do
    N=$(wc -l < "$COND2_FILE" 2>/dev/null || echo 0)
    log "cond2: ${N}/900"
    [ "$N" -ge 900 ] && break
    sleep 60
done
log "cond2 완료!"

# ── 2. cond4 시작 ───────────────────────────────────────────────
log "=== STEP 2: cond4 시작 (offset 137) ==="
python3 run_muma_gemini.py \
    --prompt-version video_only \
    --limit 900 --offset 137 --append \
    --output "$COND4_FILE" \
    2>&1 | tee -a outputs/gemini_cond4_videoonly_900.log
log "cond4 완료!"

# ── 3. cond2 ERR rerun ──────────────────────────────────────────
log "=== STEP 3: cond2 ERR rerun ==="
python3 run_muma_gemini.py \
    --rerun-err-from "$COND2_FILE" \
    --output "$COND2_FILE" \
    --prompt-version text_only \
    2>&1 | tee -a outputs/gemini_cond2_rerun.log
log "cond2 rerun 완료!"

# ── 4. cond4 ERR rerun ──────────────────────────────────────────
log "=== STEP 4: cond4 ERR rerun ==="
python3 run_muma_gemini.py \
    --rerun-err-from "$COND4_FILE" \
    --output "$COND4_FILE" \
    --prompt-version video_only \
    2>&1 | tee -a outputs/gemini_cond4_rerun.log
log "cond4 rerun 완료!"

# ── 5. 최종 결과 출력 ───────────────────────────────────────────
log "=== STEP 5: 최종 결과 ==="
python3 -c "
import json
from collections import defaultdict

def analyze(path, label):
    lines = open(path).readlines()
    by_type = defaultdict(lambda: {'total':0,'correct':0,'err':0})
    for l in lines:
        d = json.loads(l); qt = d['question_type']
        by_type[qt]['total'] += 1
        if d['predicted_letter']=='ERR': by_type[qt]['err'] += 1
        elif d['correct']: by_type[qt]['correct'] += 1
    total=len(lines); correct=sum(v['correct'] for v in by_type.values()); err=sum(v['err'] for v in by_type.values())
    print(f'\n=== {label} ===')
    print(f'n={total}, correct={correct}, ERR={err}, overall={correct/total:.1%}')
    for qt,v in sorted(by_type.items()): print(f'  {qt}: {v[\"correct\"]/v[\"total\"]:.1%} ({v[\"correct\"]}/{v[\"total\"]}) err={v[\"err\"]}')

analyze('$COND2_FILE', 'Gemini cond2 (text+frames)')
analyze('$COND4_FILE', 'Gemini cond4 (video_only)')
" 2>&1 | tee -a "$LOG"

log "=== ALL DONE ==="
