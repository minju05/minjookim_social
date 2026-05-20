"""225개 에피소드 mp4를 Gemini File API에 미리 업로드해서 캐시에 저장.

캐시에 이미 있는 에피소드는 스킵.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai
from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_gemini import upload_video_file


def main():
    cache_path = Path("outputs/video_uri_cache.json")

    # 기존 캐시 로드
    if cache_path.exists():
        uri_cache: dict[str, str] = json.loads(cache_path.read_text())
        print(f"기존 캐시 로드: {len(uri_cache)}개 에피소드")
    else:
        uri_cache = {}
        print("캐시 파일 없음 → 새로 생성")

    # 225개 에피소드 수집
    settings = load_settings()
    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)
    all_eps: dict[str, str] = {str(q.episode_id): str(q.video_path) for q in questions}
    print(f"총 에피소드: {len(all_eps)}개")

    missing = {ep: vp for ep, vp in all_eps.items() if ep not in uri_cache}
    print(f"캐시 miss (업로드 필요): {len(missing)}개")

    if not missing:
        print("모든 에피소드 이미 캐시에 있음. 종료.")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
    client = genai.Client(api_key=api_key)

    failed = []
    for ep_id, vpath in tqdm(missing.items(), desc="Upload mp4"):
        if not Path(vpath).exists():
            print(f"  [SKIP] {ep_id}: 파일 없음 ({vpath})")
            continue
        try:
            uri = upload_video_file(client, vpath)
            uri_cache[ep_id] = uri
        except Exception as e:
            print(f"  [WARN] {ep_id} 업로드 실패: {e}")
            failed.append(ep_id)

    # 캐시 저장
    cache_path.write_text(json.dumps(uri_cache, indent=2, ensure_ascii=False))
    print(f"\n캐시 저장 완료: {cache_path} ({len(uri_cache)}개 에피소드)")
    if failed:
        print(f"실패: {len(failed)}개 → {failed}")


if __name__ == "__main__":
    main()
