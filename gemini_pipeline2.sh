#!/usr/bin/env bash
# gemini_pipeline2.sh
# 이전 파이프라인에서 잘못된 prompt_version 키로 실패한 조건들 재실행
# - qa_only: video_only + --no-frames (PROMPT_CONFIGS에 'qa_only' 없음)
# - cond2 text+frame: text_only + mp4 api (PROMPT_CONFIGS에 'text_frames' 없음)
# - video-only ERR=8, generic_cot ERR=11, qlogic ERR=4 소규모 재시도
set -euo pipefail

cd /home/mindrium-admin3/minjookim_social
source /home/mindrium-admin3/.venv/bin/activate

export GEMINI_MODEL=gemini-2.5-flash
export GEMINI_API_KEY=AIzaSyCpe0JpA3WHFBN3bNBbiAugKyQFesSzm9s

echo "=========================================="
echo "[1/5] Gemini qa_only 완전 재실행"
echo "  prompt_version=video_only --no-frames"
echo "  (질문+선택지만, 텍스트/프레임 없음)"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version video_only \
    --no-frames \
    --limit 900 \
    --output outputs/gemini_cond5_qaonly_900.jsonl \
    2>&1 | tee outputs/gemini_qaonly_rerun2.log
echo "  ✅ qa_only 완료"

echo ""
echo "=========================================="
echo "[2/5] Gemini cond2 (text+frame) ERR=225 재실행"
echo "  prompt_version=text_only + mp4 api"
echo "  (text_context + 영상, ERR 항목만)"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version text_only \
    --video-mp4-api \
    --video-uri-cache outputs/video_uri_cache.json \
    --rerun-err-from outputs/gemini_cond2_text_frames_900.jsonl \
    --output outputs/gemini_cond2_text_frames_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_cond2_mp4_rerun2.log
echo "  ✅ cond2 rerun 완료"

echo ""
echo "=========================================="
echo "[3/5] Gemini video-only ERR=8 재시도"
echo "  video_only + mp4 api + rerun-err-from"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version video_only \
    --video-mp4-api \
    --video-uri-cache outputs/video_uri_cache.json \
    --rerun-err-from outputs/gemini_cond4_videoonly_900.jsonl \
    --output outputs/gemini_cond4_videoonly_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_videoonly_mp4_rerun2.log
echo "  ✅ video-only rerun 완료"

echo ""
echo "=========================================="
echo "[4/5] Gemini generic_cot ERR=11 재시도"
echo "  generic_cot + --no-frames + rerun-err-from"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version generic_cot \
    --no-frames \
    --rerun-err-from outputs/gemini_cot_900.jsonl \
    --output outputs/gemini_cot_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_cot_rerun2.log
echo "  ✅ generic_cot ERR rerun 완료"

echo ""
echo "=========================================="
echo "[5/5] Gemini qlogic ERR=4 재시도"
echo "  qlogic_v2_targeted + --no-frames + rerun-err-from"
echo "=========================================="
python3 run_muma_gemini.py \
    --prompt-version qlogic_v2_targeted \
    --no-frames \
    --rerun-err-from outputs/gemini_qlogic_900.jsonl \
    --output outputs/gemini_qlogic_900.jsonl \
    --limit 900 \
    2>&1 | tee outputs/gemini_qlogic_rerun2.log
echo "  ✅ qlogic ERR rerun 완료"

echo ""
echo "=========================================="
echo "🎉 Gemini 재실행 파이프라인 완료!"
echo "=========================================="
python3 - <<'EOF'
import json, os

files = {
    "qa_only":           "outputs/gemini_cond5_qaonly_900.jsonl",
    "video-only":        "outputs/gemini_cond4_videoonly_900.jsonl",
    "text-only":         "outputs/gemini_cond1_textonly_900.jsonl",
    "text+frame":        "outputs/gemini_cond2_text_frames_900.jsonl",
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
