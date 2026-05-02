from __future__ import annotations

import time
from pathlib import Path

from google import genai
from google.genai import types

from muma_config import Settings

_RETRY_WAIT = [10, 30, 60, 120]  # seconds between retries


def build_client(settings: Settings) -> genai.Client:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=settings.gemini_api_key)


def upload_video_and_wait(client: genai.Client, video_path: Path, poll_interval: float = 2.0) -> types.File:
    uploaded = client.files.upload(file=video_path)
    while uploaded.state == types.FileState.PROCESSING:
        time.sleep(poll_interval)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state == types.FileState.FAILED:
        raise RuntimeError(f"Gemini file processing failed for {video_path}: {uploaded.error}")
    if uploaded.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Unexpected Gemini file state for {video_path}: {uploaded.state}")
    return uploaded


def build_video_part(uploaded_video: types.File, fps: float | None = None) -> types.Part | types.File:
    if fps is None:
        return uploaded_video
    return types.Part(
        file_data=types.FileData(file_uri=uploaded_video.uri, mime_type=uploaded_video.mime_type),
        video_metadata=types.VideoMetadata(fps=fps),
    )


def extract_action_text(
    client: genai.Client,
    settings: Settings,
    uploaded_video: types.File,
    prompt: str,
    fps: float | None = None,
) -> str:
    video_part = build_video_part(uploaded_video, fps=fps)
    last_exc: Exception | None = None
    for attempt, wait in enumerate([0] + _RETRY_WAIT):
        if wait:
            print(f"  [retry {attempt}/{len(_RETRY_WAIT)}] waiting {wait}s...")
            time.sleep(wait)
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[video_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:
            last_exc = e
            print(f"  [error] {e}")
    raise RuntimeError(f"Gemini failed after {len(_RETRY_WAIT)} retries") from last_exc
