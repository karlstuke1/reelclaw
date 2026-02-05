from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


# Default to a standard 9:16 canvas used by short-form content.
RESOLUTION: tuple[int, int] = (1080, 1920)


def escape_filter_value(text: str) -> str:
    # Escape a value used inside ffmpeg filter arguments (NOT shell escaping).
    escaped = (text or "").replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(" ", "\\ ")
    return escaped


def wrap_caption(text: str, max_chars: int = 22) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    if "\n" in raw:
        lines = [" ".join(line.split()) for line in raw.splitlines() if line.strip()]
        if len(lines) <= 2:
            return "\n".join(lines)
        return "\n".join([lines[0], " ".join(lines[1:])])

    normalized = " ".join(raw.split())
    if len(normalized) <= max_chars:
        return normalized

    words = normalized.split(" ")
    if len(words) <= 1:
        return normalized

    best: tuple[int, int, int] | None = None  # (max_len, imbalance, split_idx)
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        max_len = max(len(l1), len(l2))
        imbalance = abs(len(l1) - len(l2))
        cand = (max_len, imbalance, i)
        if best is None or cand < best:
            best = cand

    if best is None:
        return normalized

    i = best[2]
    return "\n".join([" ".join(words[:i]), " ".join(words[i:])])


def font_size_for_caption(caption: str, *, max_size: int = 96) -> int:
    longest = max((len(line) for line in (caption or "").split("\n")), default=0)
    if longest <= 0:
        return int(max_size)
    width, _height = RESOLUTION
    fit = int((width * 0.86) / (longest * 0.68))
    return max(32, min(int(max_size), fit))


def find_font_path() -> Path | None:
    candidates = [
        Path("/Library/Fonts/Impact.ttf"),
        Path("/System/Library/Fonts/Supplemental/Impact.ttf"),
        Path("/Library/Fonts/HelveticaNeue-CondensedBlack.ttf"),
        Path("/System/Library/Fonts/Supplemental/HelveticaNeue-CondensedBlack.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def merge_audio(*, video_path: Path, audio_path: Path, output_path: Path | None = None) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to merge audio. Please install ffmpeg and try again.")

    target = output_path or video_path
    temp_path = target.with_suffix(".tmp" + target.suffix)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-stream_loop",
        "-1",
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(temp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg audio merge failed: {stderr or 'unknown error'}")

    temp_path.replace(target)
    return target

