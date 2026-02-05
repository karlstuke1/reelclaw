from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
import typing as t


PTS_RE = re.compile(r"pts_time:([0-9]+(?:\\.[0-9]+)?)")


def _run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def ffprobe_duration_s(path: Path) -> float | None:
    ffprobe = os.getenv("FFPROBE", "ffprobe")
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def detect_scene_cuts(
    *,
    video_path: Path,
    min_scenes: int = 7,
    max_scenes: int = 9,
    target_scenes: int | None = None,
    min_spacing_s: float = 0.25,
    timeout_s: float = 180.0,
) -> tuple[list[float], float]:
    """
    Detect cut timestamps (seconds) using ffmpeg's scene detection.

    Returns (cut_times, duration_s) where cut_times excludes 0 and the end timestamp.
    """
    if min_scenes < 1 or max_scenes < 1 or min_scenes > max_scenes:
        raise ValueError("Invalid min/max scenes")

    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for cut detection. Please install ffmpeg and try again.")

    duration = ffprobe_duration_s(video_path) or 0.0
    if duration <= 0:
        raise RuntimeError("Could not determine video duration for cut detection")

    target = target_scenes or max_scenes
    thresholds = [0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.10]

    best: tuple[int, float, list[float]] | None = None  # (distance, threshold, cuts)

    for thr in thresholds:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{thr})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
        result = _run(cmd, timeout_s=timeout_s)
        if result.returncode != 0:
            continue

        times: list[float] = []
        for line in (result.stderr or "").splitlines():
            m = PTS_RE.search(line)
            if not m:
                continue
            try:
                t_s = float(m.group(1))
            except Exception:
                continue
            if t_s <= 0.0 or t_s >= duration:
                continue
            times.append(t_s)

        if not times:
            continue

        # Sort + de-dupe + enforce spacing.
        times = sorted(set(times))
        filtered: list[float] = []
        last = -1e9
        for t_s in times:
            if t_s - last < min_spacing_s:
                continue
            filtered.append(t_s)
            last = t_s

        scene_count = len(filtered) + 1
        dist = abs(scene_count - target)
        cand = (dist, thr, filtered)
        if best is None or cand < best:
            best = cand

        if min_scenes <= scene_count <= max_scenes:
            return filtered, duration

    if best is None:
        return [], duration

    return best[2], duration

