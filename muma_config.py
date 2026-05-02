from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    cache_dir: Path
    github_repo_dir: Path
    hf_repo_id: str
    hf_token: str | None
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str
    frame_stride: int
    max_frames: int | None


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")

    data_dir = ROOT / "data"
    cache_dir = ROOT / ".cache"

    max_frames_env = os.getenv("MAX_FRAMES")
    max_frames = None if max_frames_env in (None, "", "none", "None", "0") else int(max_frames_env)

    return Settings(
        root=ROOT,
        data_dir=data_dir,
        cache_dir=cache_dir,
        github_repo_dir=Path(os.getenv("GITHUB_REPO_DIR", ROOT / "muma-tom")),
        hf_repo_id=os.getenv("HF_REPO_ID", "SCAI-JHU/MUMA-TOM-BENCHMARK"),
        hf_token=os.getenv("HF_TOKEN"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        frame_stride=int(os.getenv("FRAME_STRIDE", "20")),
        max_frames=max_frames,
    )
