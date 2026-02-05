from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import typing as t


def _mime_for_video(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "mp4": "video/mp4",
        "m4v": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
    }.get(ext, "video/mp4")


def _ffprobe_duration_s(path: Path) -> float | None:
    ffprobe = os.getenv("FFPROBE", "") or shutil.which("ffprobe") or "ffprobe"
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
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def _make_proxy_video(
    src: Path,
    *,
    dst: Path,
    max_w: int,
    max_h: int,
    fps: int,
    crf: int,
    audio_bitrate: str,
) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg") or "ffmpeg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='min({int(max_w)},iw)':'min({int(max_h)},ih)':force_original_aspect_ratio=decrease"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
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
        "-c:a",
        "aac",
        "-b:a",
        str(audio_bitrate),
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.check_call(cmd)


def ensure_inlineable_video(
    path: Path,
    *,
    max_mb: float,
    tmp_dir: Path,
    allow_proxy: bool = True,
) -> tuple[Path, dict[str, t.Any]]:
    """
    Return a path that is safe to inline as a data URL, creating a temporary proxy if needed.
    """
    meta: dict[str, t.Any] = {"source": str(path), "proxy": None}
    try:
        size_bytes = int(path.stat().st_size)
    except Exception:
        size_bytes = len(path.read_bytes())
    size_mb = float(size_bytes) / (1024.0 * 1024.0)
    meta["size_bytes"] = int(size_bytes)
    meta["size_mb"] = float(size_mb)
    if size_mb <= float(max_mb):
        return path, meta
    if not allow_proxy:
        raise ValueError(f"Video too large to inline without proxy: {size_mb:.2f} MB > {float(max_mb):.2f} MB ({path.name})")

    dur = _ffprobe_duration_s(path) or 0.0
    base = hashlib.sha256((str(path) + str(size_mb) + str(dur)).encode("utf-8")).hexdigest()[:12]
    proxy = (tmp_dir / f"proxy_{base}.mp4").resolve()
    try:
        if proxy.exists():
            psize = proxy.stat().st_size / (1024.0 * 1024.0)
            if float(psize) <= float(max_mb):
                meta["proxy"] = str(proxy)
                meta["proxy_size_mb"] = float(psize)
                meta["proxy_crf"] = None
                return proxy, meta
    except Exception:
        pass

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for crf in (32, 36, 40):
        try:
            _make_proxy_video(path, dst=proxy, max_w=360, max_h=640, fps=24, crf=crf, audio_bitrate="64k")
            psize = proxy.stat().st_size / (1024.0 * 1024.0)
            if float(psize) <= float(max_mb):
                meta["proxy"] = str(proxy)
                meta["proxy_size_mb"] = float(psize)
                meta["proxy_crf"] = int(crf)
                return proxy, meta
        except Exception:
            continue

    raise ValueError(f"Unable to create inlineable proxy under {max_mb} MB for {path.name}")


def encode_video_data_url(path: Path, *, max_mb: float) -> str:
    raw = path.read_bytes()
    size_mb = float(len(raw)) / (1024.0 * 1024.0)
    if size_mb > float(max_mb):
        raise ValueError(f"Video too large to inline: {size_mb:.2f} MB > {float(max_mb):.2f} MB ({path.name})")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_mime_for_video(path)};base64,{b64}"

