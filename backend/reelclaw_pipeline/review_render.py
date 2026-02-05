from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import typing as t


def _find_font_path() -> str | None:
    # Best-effort: use a stable system font when available.
    for p in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
    ]:
        try:
            if Path(p).exists():
                return p
        except Exception:
            continue
    return None


def _run(cmd: list[str], *, timeout_s: float) -> None:
    proc = subprocess.run(cmd, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def render_labeled_review_video(
    *,
    src_video: Path,
    segments: list[dict[str, t.Any]],
    out_path: Path,
    label_prefix: str = "S",
    keep_audio: bool,
    max_w: int = 360,
    max_h: int = 640,
    fps: int = 24,
    crf: int = 32,
    audio_bitrate: str = "64k",
    timeout_s: float = 240.0,
) -> None:
    """
    Render a small proxy MP4 with burned-in segment labels (S01..).
    segments: [{"id": int, "start_s": float, "end_s": float}, ...]
    """
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render review videos. Please install ffmpeg and try again.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = int(max(160, max_w))
    h = int(max(160, max_h))
    vf_parts: list[str] = []
    vf_parts.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
    vf_parts.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

    font_path = _find_font_path()
    font_arg = f"fontfile={font_path}:" if font_path else ""
    # Use a boxed label that stays readable over bright/dark shots.
    base = f"drawtext={font_arg}fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10:x=20:y=20"
    for seg in segments:
        try:
            seg_id = int(seg.get("id") or 0)
        except Exception:
            seg_id = 0
        if seg_id <= 0:
            continue
        try:
            start_s = float(seg.get("start_s") or 0.0)
            end_s = float(seg.get("end_s") or 0.0)
        except Exception:
            continue
        if end_s <= start_s + 1e-3:
            continue
        label = f"{label_prefix}{seg_id:02d}"
        vf_parts.append(f"{base}:text='{label}':enable='between(t,{start_s:.3f},{end_s:.3f})'")

    vf = ",".join(vf_parts)

    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-vf",
        vf,
        "-r",
        str(int(fps)),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if keep_audio:
        cmd.extend(["-c:a", "aac", "-b:a", str(audio_bitrate)])
    else:
        cmd.append("-an")
    cmd.append(str(out_path))
    _run(cmd, timeout_s=float(timeout_s))


def render_compare_video(
    *,
    ref_review: Path,
    out_review: Path,
    out_path: Path,
    max_w: int = 360,
    max_h: int = 640,
    fps: int = 24,
    crf: int = 32,
    audio_bitrate: str = "64k",
    timeout_s: float = 240.0,
) -> None:
    """
    Create a side-by-side (REF|OUT) proxy. Output audio only.
    """
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render review videos. Please install ffmpeg and try again.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = int(max(160, max_w))
    h = int(max(160, max_h))
    # Ensure both inputs are exactly w x h, then hstack.
    flt = "\n".join(
        [
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[rv];",
            f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[ov];",
            "[rv][ov]hstack=inputs=2[v]",
        ]
    )
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(ref_review),
        "-i",
        str(out_review),
        "-filter_complex",
        flt,
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-shortest",
        "-r",
        str(int(fps)),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        str(audio_bitrate),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run(cmd, timeout_s=float(timeout_s))

