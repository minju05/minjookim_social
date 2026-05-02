from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def sample_video_frames_pil(
    video_path: Path,
    frame_stride: int = 20,
    max_frames: int | None = None,
) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames: list[Image.Image] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, frame_stride) != 0:
                frame_index += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if max_frames is not None and len(frames) >= max_frames:
                break
            frame_index += 1
    finally:
        capture.release()
    return frames
