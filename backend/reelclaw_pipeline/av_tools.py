from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}... {stderr or 'unknown error'}")


def extract_audio(
    *,
    video_path: Path,
    output_audio_path: Path,
    timeout_s: float = 180.0,
) -> Path:
    """
    Extract an audio track from a video.

    Output is encoded as AAC-in-M4A for broad compatibility.
    """
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract audio. Please install ffmpeg and try again.")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_audio_path),
    ]
    _run(cmd, timeout_s=timeout_s)
    return output_audio_path

