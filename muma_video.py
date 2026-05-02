from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class EncodedFrame:
    second: float
    image_b64: str


def sample_video_frames(
    video_path: Path,
    frame_stride: int = 20,
    max_frames: int | None = None,
) -> list[EncodedFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    stride = max(1, frame_stride)

    frames: list[EncodedFrame] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                frame_index += 1
                continue

            frames.append(
                EncodedFrame(
                    second=frame_index / fps,
                    image_b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                )
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
            frame_index += 1
    finally:
        capture.release()

    return frames
