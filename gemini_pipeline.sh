#!/usr/bin/env bash
# gemini_pipeline.sh  (v2: 올바른 prompt_version 키 적용)
# qa_only=video_only+--no-frames / cond2=text_only+mp4api / ERR reruns
set -uo pipefail

cd /home/mindrium-admin3/minjookim_social
source /home/mindrium-admin3/.venv/bin/activate

COND4_FILE="outputs/gemini_cond4_videoonly_900.jsonl"

echo "=========================================="
echo "[1/6] cond4 완료 확인..."
echo "=========================================="
python3 - <<'EOF'
import json, sys
try:
    lines = open("outputs/gemini_cond4_videoonly_900.jsonl").readlines()
    n = len(lines)
    err = sum(1 for l in lines if json.loads(l).get("predicted_letter") == "ERR")
    print(f"  cond4: {n}/900줄, ERR={err}")
    if n < 900:
        print("  ⚠️  900줄 미만 — 계속 진행하지만 확인 필요")
    elif err > 0:
        print(f"  ⚠️  persistent ERR={err}개 있음 (무시하고 계속)")
    else:
        print("  ✅ cond4 완료 (ERR=0)")
except Exception as e:
    print(f"  ⚠️  파일 읽기 실패: {e}")
EOF
echo "  → cond2 rerun으로 진행"

echo ""
echo "=========================================="
echo "[2/6] preload_video_cache.py (나머지 에피소드 업로드)"
echo "=========================================="
python3 preload_video_cache.py 2>&1 | tee outputs/preload_video_cache.log
echo "  ✅ preload 완료"

echo ""
echo "=========================================="
echo "[3/6] Gemini cond2 (text+frame) ERR rerun"
echo "  ▶ 올바른 키: text_only + --video-mp4-api"
echo "=========================================="
python3 run_muma_gemini.py \
    --video-mp4-api \
    --video-uri-cache outputs/video_uri_cache.json \
    --rerun-err-from outputs/gemini_cond2_text_frames_900.jsonl \
    --output outputs/gemini_cond2_text_frames_900.jsonl \
    --prompt-version text_only \
    --limit 900 \
    2>&1 | tee outputs/gemini_cond2_mp4_rerun2.log
echo "  ✅ cond2 rerun 완료"

echo ""
echo "=========================================="
echo "[4/6] Gemini qa_only (질문+선택지만)"
echo "  ▶ 올바른 키: video_only + --no-frames"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version video_only \
    --no-frames \
    --limit 900 \
    --output outputs/gemini_cond5_qaonly_900.jsonl \
    2>&1 | tee outputs/gemini_qaonly_900.log
echo "  ✅ qa_only 완료"

echo ""
echo "=========================================="
echo "[5/7] Gemini generic_cot ERR rerun (ERR=11)"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version generic_cot \
    --no-frames \
    --rerun-err-from outputs/gemini_cot_900.jsonl \
    --output outputs/gemini_cot_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_cot_rerun.log
echo "  ✅ generic_cot ERR rerun 완료"

echo ""
echo "=========================================="
echo "[6/7] Gemini qlogic ERR rerun (ERR=4)"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version qlogic_v2_targeted \
    --no-frames \
    --rerun-err-from outputs/gemini_qlogic_900.jsonl \
    --output outputs/gemini_qlogic_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_qlogic_rerun.log
echo "  ✅ qlogic ERR rerun 완료"

echo ""
echo "=========================================="
echo "[7/7] Gemini cond4 video-only ERR rerun (ERR=8)"
echo "=========================================="
python3 run_muma_gemini.py \
    --video-mp4-api \
    --video-uri-cache outputs/video_uri_cache.json \
    --rerun-err-from outputs/gemini_cond4_videoonly_900.jsonl \
    --output outputs/gemini_cond4_videoonly_900.jsonl \
    --prompt-version video_only \
    --limit 900 \
    2>&1 | tee outputs/gemini_cond4_rerun.log
echo "  ✅ cond4 ERR rerun 완료"

echo ""
echo "=========================================="
echo "🎉 Gemini 전체 파이프라인 완료!"
echo "=========================================="
python3 - <<'EOF'
import json, os

files = {
    "qa_only (video_only+noframes)": "outputs/gemini_cond5_qaonly_900.jsonl",
    "cond2 text+frame (text_only+mp4)": "outputs/gemini_cond2_text_frames_900.jsonl",
    "cond4 video-only":  "outputs/gemini_cond4_videoonly_900.jsonl",
    "generic_cot":       "outputs/gemini_cot_900.jsonl",
    "qlogic":            "outputs/gemini_qlogic_900.jsonl",
}
for name, path in files.items():
    if not os.path.exists(path):
        print(f"  {name:20s}: ✗ 파일 없음")
        continue
    lines = open(path).readlines()
    if not lines:
        print(f"  {name:20s}: 0줄")
        continue
    n = len(lines)
    c = sum(1 for l in lines if json.loads(l)["correct"])
    e = sum(1 for l in lines if json.loads(l).get("predicted_letter") == "ERR")
    print(f"  {name:20s}: {c}/{n} ({c/n*100:.1f}%) ERR={e}")
EOF
