from __future__ import annotations

import json
import os
import shutil
import subprocess
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
import typing as t

from .av_tools import extract_audio
from .folder_edit_planner import (
    EditDecision,
    FolderEditPlan,
    ReferenceAnalysisPlan,
    analyze_reference_reel_segments,
    plan_folder_edit_edl,
    refine_inpoint_for_segment,
    tag_assets_from_thumbnails,
)
from .folder_edit_evaluator import evaluate_edit_similarity
from .media_encoding import encode_image_data_url as _encode_image_data_url
from .media_index import index_media_folder, load_index, load_or_build_cached_index, save_index
from .pipeline_output import OutputPaths, ensure_output_dirs, write_json
from .pipeline_types import CancelledError, ProgressUpdate
from .reel_cut_detect import detect_scene_cuts
from .reel_download import compress_for_analysis as _compress_for_analysis
from .reel_download import download_reel as _download_reel
from .video_tools import escape_filter_value as _escape_filter_value
from .video_tools import find_font_path as _find_font_path
from .video_tools import font_size_for_caption as _font_size_for_caption
from .video_tools import merge_audio
from .video_tools import wrap_caption as _wrap_caption

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None


@dataclass(frozen=True)
class FolderEditPipelineResult:
    output_paths: OutputPaths
    video_path: Path
    reference_analysis: ReferenceAnalysisPlan
    folder_plan: FolderEditPlan
    source_video_path: Path
    # Used by the GUI thumbnails preview (shows the *output* frames, not the reference reel frames).
    scene_images: list[Path] = field(default_factory=list)


def _truthy_env(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default).strip()
    try:
        return float(raw)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _candidate_times_for_refinement(
    *,
    asset_duration_s: float | None,
    segment_duration_s: float,
    initial_in_s: float,
) -> list[float]:
    """
    Pick a small set of timestamps for Gemini to choose from.
    Keep this generic (no story assumptions) and stable across assets.
    """
    try:
        init = max(0.0, float(initial_in_s))
    except Exception:
        init = 0.0

    dur = asset_duration_s if isinstance(asset_duration_s, (int, float)) and float(asset_duration_s) > 0 else None
    seg_dur = max(0.0, float(segment_duration_s or 0.0))
    if dur is None:
        # Unknown duration: sample around the initial in-point.
        max_start = init
    else:
        max_start = max(0.0, float(dur) - seg_dur)

    if max_start <= 0.0:
        return [0.0]

    anchors = [0.0, 0.15, 0.35, 0.55, 0.75, 0.9]
    times = [a * max_start for a in anchors]
    times.append(min(max_start, init))

    # De-dupe + keep deterministic ordering.
    out: list[float] = []
    for t_s in times:
        t3 = round(float(t_s), 3)
        if t3 < 0.0:
            continue
        if t3 not in out:
            out.append(t3)
    out.sort()
    return out[:7]


def _norm_tag(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _segment_energy_hint(duration_s: float) -> float:
    # 0..1 where 1 is highest energy. Use only timing so it's reel-agnostic.
    d = float(duration_s or 0.0)
    if d <= 0.8:
        return 1.0
    if d <= 1.2:
        return 0.75
    if d <= 1.7:
        return 0.45
    return 0.25


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _compute_eq_grade(
    *,
    ref_luma: float | None,
    ref_dark: float | None,
    out_luma: float | None,
    out_dark: float | None,
    ref_rgb: list[float] | None = None,
    out_rgb: list[float] | None = None,
) -> dict[str, float] | None:
    """Compute a mild per-segment eq() grade to reduce lighting mismatch.

    Keep this generic and conservative. We only nudge brightness/contrast when
    the reference and chosen footage are clearly mismatched.
    """
    if ref_luma is None or out_luma is None:
        return None

    grade: dict[str, float] = {}

    dl = float(ref_luma) - float(out_luma)
    dd = 0.0
    if isinstance(ref_dark, (int, float)) and isinstance(out_dark, (int, float)):
        dd = float(ref_dark) - float(out_dark)

    # If it's already close enough, don't grade brightness/contrast.
    luma_thresh = 0.06
    dark_thresh = 0.15
    try:
        if float(ref_luma) <= 0.03 or (ref_dark is not None and float(ref_dark) >= 0.93):
            # Dark references are sensitive to small absolute changes.
            luma_thresh = 0.03
            dark_thresh = 0.10
    except Exception:
        pass

    if abs(dl) >= luma_thresh or abs(dd) >= dark_thresh:
        # eq brightness is [-1..1] but practical, non-destructive moves are small.
        brightness = _clamp(dl * 0.85, -0.22, 0.22)

        # contrast default is 1.0; keep within a subtle range.
        contrast = 1.0
        if abs(dd) >= 0.05:
            contrast = _clamp(1.0 + (dd * 0.85), 0.85, 1.35)

        # If the reference is extremely dark, bias toward keeping things low-key.
        try:
            if ref_dark is not None and float(ref_dark) >= 0.93:
                brightness = _clamp(brightness - 0.03, -0.25, 0.18)
                contrast = _clamp(contrast + 0.06, 0.9, 1.45)
        except Exception:
            pass

        if abs(brightness) >= 0.01:
            grade["brightness"] = float(brightness)
        if abs(contrast - 1.0) >= 0.02:
            grade["contrast"] = float(contrast)

    # Optional: mild channel gain correction (acts like a simple white-balance / tint match).
    if (
        isinstance(ref_rgb, list)
        and isinstance(out_rgb, list)
        and len(ref_rgb) == 3
        and len(out_rgb) == 3
        and all(isinstance(x, (int, float)) for x in ref_rgb)
        and all(isinstance(x, (int, float)) for x in out_rgb)
    ):
        rg, gg, bg = 1.0, 1.0, 1.0
        # Avoid division blowups on near-black channels.
        denom = 0.06
        try:
            rg = float(ref_rgb[0]) / max(denom, float(out_rgb[0]))
            gg = float(ref_rgb[1]) / max(denom, float(out_rgb[1]))
            bg = float(ref_rgb[2]) / max(denom, float(out_rgb[2]))
        except Exception:
            rg, gg, bg = 1.0, 1.0, 1.0
        rg = _clamp(rg, 0.78, 1.32)
        gg = _clamp(gg, 0.78, 1.32)
        bg = _clamp(bg, 0.78, 1.32)
        # Only apply if there's a meaningful shift.
        if abs(rg - 1.0) >= 0.04 or abs(gg - 1.0) >= 0.04 or abs(bg - 1.0) >= 0.04:
            grade["r_gain"] = float(rg)
            grade["g_gain"] = float(gg)
            grade["b_gain"] = float(bg)

    return grade or None


def _shortlist_assets_for_segment(
    *,
    segment: t.Any,
    assets: list[dict[str, t.Any]],
    used_asset_ids: set[str],
    limit: int = 24,
) -> list[dict[str, t.Any]]:
    """
    Pre-filter assets for a segment using cheap metrics (lighting/negative space/motion + tag overlap).
    This reduces the chance the LLM picks something with the right 'idea' but wrong look/feel.
    """
    ref_luma = segment.ref_luma
    ref_dark = getattr(segment, "ref_dark_frac", None)
    desired_tags = getattr(segment, "desired_tags", []) or []
    desired_set = {_norm_tag(str(tg)) for tg in desired_tags if _norm_tag(str(tg))}

    # Optional hard gate for very dark reference segments.
    very_dark = False
    try:
        very_dark = (ref_dark is not None and float(ref_dark) >= 0.93) or (ref_luma is not None and float(ref_luma) <= 0.04)
    except Exception:
        very_dark = False

    def lighting_ok(a: dict[str, t.Any]) -> bool:
        if not very_dark:
            return True
        try:
            ad = a.get("dark_frac_max")
            if not isinstance(ad, (int, float)):
                ad = a.get("dark_frac")
            al = a.get("luma_min")
            if not isinstance(al, (int, float)):
                al = a.get("luma_mean")

            # Tighten luma thresholds when the reference is extremely dark.
            max_luma = 0.10
            if isinstance(ref_luma, (int, float)):
                rl = float(ref_luma)
                if rl <= 0.015:
                    max_luma = 0.055
                elif rl <= 0.03:
                    max_luma = 0.075
                else:
                    max_luma = 0.10

            if isinstance(ad, (int, float)) and float(ad) < 0.80:
                return False
            if isinstance(al, (int, float)) and float(al) > max_luma:
                return False
        except Exception:
            return False
        return True

    candidates = [a for a in assets if a.get("id")]
    gated = [a for a in candidates if lighting_ok(a)]
    if very_dark and len(gated) >= 10:
        candidates = gated

    energy = _segment_energy_hint(getattr(segment, "duration_s", 1.0) or 1.0)

    scored: list[tuple[float, dict[str, t.Any]]] = []
    for a in candidates:
        aid = str(a.get("id") or "")
        if not aid:
            continue

        # Penalize repeats when we have enough options.
        repeat_penalty = 0.0
        if aid in used_asset_ids and len(candidates) > 12:
            repeat_penalty = 0.35

        # Use the closest available luma/dark metrics (mean/min/max) to match the reference look.
        al: float | None = None
        ad: float | None = None
        if isinstance(ref_luma, (int, float)):
            rl = float(ref_luma)
            l_candidates = [a.get("luma_mean"), a.get("luma_min"), a.get("luma_max")]
            l_vals = [float(x) for x in l_candidates if isinstance(x, (int, float))]
            if l_vals:
                al = min(l_vals, key=lambda x: abs(x - rl))
        else:
            if isinstance(a.get("luma_mean"), (int, float)):
                al = float(a.get("luma_mean"))  # type: ignore[arg-type]

        if isinstance(ref_dark, (int, float)):
            rd = float(ref_dark)
            d_candidates = [a.get("dark_frac"), a.get("dark_frac_min"), a.get("dark_frac_max")]
            d_vals = [float(x) for x in d_candidates if isinstance(x, (int, float))]
            if d_vals:
                ad = min(d_vals, key=lambda x: abs(x - rd))
        else:
            if isinstance(a.get("dark_frac"), (int, float)):
                ad = float(a.get("dark_frac"))  # type: ignore[arg-type]
        am = a.get("motion_score")

        # Lighting distance (lower is better).
        dl = 0.18
        dd = 0.25
        if isinstance(ref_luma, (int, float)) and isinstance(al, (int, float)):
            dl = abs(float(al) - float(ref_luma))
        if isinstance(ref_dark, (int, float)) and isinstance(ad, (int, float)):
            dd = abs(float(ad) - float(ref_dark))
        lighting = (dl * 1.2) + (dd * 1.0)

        # Motion match.
        motion = 0.35
        if isinstance(am, (int, float)):
            motion = float(am)
        motion_dist = abs(motion - energy) * 0.45

        # Tag overlap bonus.
        tags = a.get("tags") or []
        tag_set: set[str] = set()
        if isinstance(tags, list):
            for tg in tags:
                nt = _norm_tag(str(tg))
                if nt:
                    tag_set.add(nt)
        for k in ("shot_type", "setting", "mood"):
            v = a.get(k)
            nv = _norm_tag(str(v)) if v else ""
            if nv:
                tag_set.add(nv)
        overlap = len(desired_set.intersection(tag_set))
        tag_bonus = -0.10 * min(overlap, 6)

        score = lighting + motion_dist + repeat_penalty + tag_bonus
        scored.append((float(score), a))

    scored.sort(key=lambda x: x[0])
    return [a for _s, a in scored[: max(6, int(limit))]]


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}... {stderr or 'unknown error'}")


def _maybe_analyze_music(
    *,
    audio_path: Path | None,
    output_dir: Path,
    timeout_s: float,
) -> dict[str, t.Any] | None:
    """
    Best-effort music analysis (beats/onsets) used for beat-snapping segment boundaries.
    No hard deps: uses our `scripts/music_analysis.py` no-deps baseline.
    """
    if not audio_path or not Path(audio_path).exists():
        return None
    try:
        from music_analysis import analyze_music  # local module
    except Exception:
        return None

    try:
        out = output_dir / "music_analysis.json"
        return analyze_music(audio_or_video_path=Path(audio_path), output_json_path=out, timeout_s=timeout_s)
    except Exception:
        return None


def _beat_snap_segments(
    *,
    segments: list[tuple[int, float, float]],
    duration_s: float,
    music_doc: dict[str, t.Any] | None,
) -> list[tuple[int, float, float]]:
    """
    Snap segment boundaries to nearby beats/onsets (within tolerance).
    Keeps segment count and ordering stable; does not hard-retime the whole reel.
    """
    if not segments or not music_doc:
        return segments
    beats = music_doc.get("beat_times") or []
    onsets = music_doc.get("onsets") or []
    if not isinstance(beats, list) or not isinstance(onsets, list):
        return segments

    try:
        from music_analysis import snap_times  # local module
    except Exception:
        return segments

    # Boundaries are [start0, end0, end1, ..., endN]. start0 is usually 0.
    boundaries: list[float] = []
    boundaries.append(float(segments[0][1]))
    for _sid, _s, e in segments:
        boundaries.append(float(e))

    snapped = snap_times(boundaries, beats=beats, onsets=onsets)
    # Clamp to [0, duration].
    snapped = [min(max(0.0, float(t)), float(duration_s)) for t in snapped]

    out: list[tuple[int, float, float]] = []
    for i, (seg_id, _s, _e) in enumerate(segments):
        s = float(snapped[i])
        e = float(snapped[i + 1]) if i + 1 < len(snapped) else float(duration_s)
        if e <= s + 1e-4:
            e = min(float(duration_s), s + 1e-3)
        out.append((int(seg_id), s, e))
    return out


# Cache stabilized video paths within a single process to avoid repeated work.
_STABILIZED_VIDEO_CACHE: dict[str, Path] = {}


def _file_fingerprint(path: Path) -> str:
    """
    Fingerprint to invalidate caches when a file changes.
    Keep it deterministic and cheap (no hashing file contents).
    """
    try:
        st = path.stat()
        return f"{path.resolve().as_posix()}::{int(st.st_size)}::{int(st.st_mtime_ns)}"
    except Exception:
        return path.resolve().as_posix()


def _ensure_stabilized_video(
    *,
    src: Path,
    cache_dir: Path,
    timeout_s: float,
) -> Path:
    """
    Create (and cache) a stabilized copy of a source video using ffmpeg vidstab.
    We stabilize the *entire* source once, then trim segments from the stabilized copy.

    Important: vidstab transform files are frame-indexed; trimming first would desync transforms.
    """
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to stabilize video. Please install ffmpeg and try again.")

    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _file_fingerprint(src)
    cached = _STABILIZED_VIDEO_CACHE.get(key)
    if cached and cached.exists():
        return cached

    hid = hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:12]
    trf = cache_dir / f"{hid}.trf"
    out = cache_dir / f"{hid}.mp4"
    if out.exists():
        _STABILIZED_VIDEO_CACHE[key] = out
        return out

    # Vidstab params: keep conservative defaults; tune via env if needed.
    shakiness = int(max(1, min(10, _float_env("FOLDER_EDIT_STABILIZE_SHAKINESS", "6"))))
    accuracy = int(max(1, min(15, _float_env("FOLDER_EDIT_STABILIZE_ACCURACY", "10"))))
    smoothing = int(max(0, min(60, _float_env("FOLDER_EDIT_STABILIZE_SMOOTHING", "12"))))

    # 1) Detect transforms.
    if not trf.exists():
        detect = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={_escape_filter_value(str(trf))}",
            "-f",
            "null",
            "-",
        ]
        _run(detect, timeout_s=timeout_s)

    # 2) Apply transforms (write to mp4 so we can seek cheaply later).
    tmp = out.with_suffix(".tmp.mp4")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    transform = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        f"vidstabtransform=input={_escape_filter_value(str(trf))}:smoothing={smoothing}:optzoom=1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    _run(transform, timeout_s=timeout_s)
    tmp.replace(out)

    _STABILIZED_VIDEO_CACHE[key] = out
    return out


def _extract_frame(
    *,
    video_path: Path,
    at_s: float,
    out_path: Path,
    timeout_s: float,
) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract frames. Please install ffmpeg and try again.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        max_w = int(max(160.0, float(os.getenv("FOLDER_EDIT_EXTRACT_FRAME_MAX_W", "768"))))
    except Exception:
        max_w = 768
    vf = f"scale={int(max_w)}:-2:flags=lanczos,format=yuvj420p"

    # ffmpeg can occasionally exit 0 yet produce no frame near the end of a clip.
    # Retry slightly earlier timestamps to ensure we get a usable frame.
    for attempt, dt in enumerate([0.0, -0.06, -0.12, -0.24], start=1):
        t_s = max(0.0, float(at_s) + float(dt))
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{t_s:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            # Ensure MJPEG gets a compatible full-range pixel format (some iPhone footage is tv-range).
            vf,
            "-q:v",
            "2",
            str(out_path),
        ]
        _run(cmd, timeout_s=timeout_s)
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                return
        except Exception:
            pass
        if attempt >= 4:
            break
    raise RuntimeError(f"ffmpeg produced no frame at ~{float(at_s):.3f}s for {video_path.name}")


def _luma_mean(path: Path) -> float | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L")
        img = img.resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return None
        return float(sum(pixels) / len(pixels)) / 255.0
    except Exception:
        return None


def _dark_frac(path: Path, *, threshold: int = 32) -> float | None:
    """
    Fraction of pixels considered "dark" (0..1). Helps detect "mostly black" frames.
    threshold is in 0..255 grayscale.
    """
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L")
        img = img.resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return None
        dark = sum(1 for p in pixels if int(p) < int(threshold))
        return float(dark) / float(len(pixels))
    except Exception:
        return None


def _rgb_mean(path: Path) -> list[float] | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("RGB").resize((64, 64))
        px = list(img.getdata())
        if not px:
            return None
        r = sum(int(p[0]) for p in px) / len(px)
        g = sum(int(p[1]) for p in px) / len(px)
        b = sum(int(p[2]) for p in px) / len(px)
        return [float(r) / 255.0, float(g) / 255.0, float(b) / 255.0]
    except Exception:
        return None


def _motion_score_from_thumbs(paths: list[Path]) -> float | None:
    """
    Cheap motion proxy: average pixel difference between multiple thumbnails from the same clip.
    Returns ~0 (static) .. ~1 (very dynamic / large changes).
    """
    if Image is None:
        return None
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("L").resize((64, 64))
        except Exception:
            continue
        imgs.append(list(img.getdata()))
    if len(imgs) < 2:
        return None
    diffs: list[float] = []
    for a, b in zip(imgs, imgs[1:], strict=False):
        if not a or not b:
            continue
        n = min(len(a), len(b))
        if n <= 0:
            continue
        diffs.append(sum(abs(int(a[i]) - int(b[i])) for i in range(n)) / (n * 255.0))
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))


def _beat_floor_index(t_s: float, beats: list[dict[str, t.Any]]) -> int | None:
    if not beats:
        return None
    t0 = float(t_s)
    best_i: int | None = None
    for b in beats:
        bt = b.get("t")
        bi = b.get("i")
        if not isinstance(bt, (int, float)) or not isinstance(bi, int):
            continue
        if float(bt) <= t0 + 1e-6:
            best_i = int(bi)
        else:
            break
    if best_i is not None:
        return best_i
    # If all beats are after t_s, return first.
    for b in beats:
        if isinstance(b.get("i"), int):
            return int(b["i"])
    return None


def _segment_music_energy(*, start_s: float, end_s: float, beats: list[dict[str, t.Any]]) -> float | None:
    if not beats:
        return None
    b0 = _beat_floor_index(float(start_s), beats)
    b1 = _beat_floor_index(float(end_s), beats)
    if b0 is None or b1 is None or b1 <= b0:
        return None
    strengths: list[float] = []
    for b in beats:
        bi = b.get("i")
        if not isinstance(bi, int):
            continue
        if bi < b0:
            continue
        if bi > b1:
            break
        s = b.get("strength")
        if isinstance(s, (int, float)):
            strengths.append(float(s))
    if not strengths:
        return None
    # Strengths are already roughly 0..1. Clamp defensively.
    e = sum(strengths) / len(strengths)
    return max(0.0, min(1.0, float(e)))


def _frame_motion_diff(a: Path, b: Path) -> float | None:
    if Image is None:
        return None
    try:
        ia = Image.open(a).convert("L").resize((64, 64))
        ib = Image.open(b).convert("L").resize((64, 64))
        pa = list(ia.getdata())
        pb = list(ib.getdata())
        n = min(len(pa), len(pb))
        if n <= 0:
            return None
        return sum(abs(int(pa[i]) - int(pb[i])) for i in range(n)) / (n * 255.0)
    except Exception:
        return None


def _estimate_shake_jitter_norm_p95(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    sample_fps: float = 8.0,
    max_samples: int = 48,
    max_w: int = 360,
) -> float | None:
    """
    Per-segment shake estimate (cheap, no vidstab) using phase correlation residuals.

    Returns a normalized jitter score (higher = shakier). Intended for stabilization decisions.
    """
    if cv2 is None or np is None:  # pragma: no cover
        return None
    if not video_path.exists():
        return None

    s0 = max(0.0, float(start_s))
    e0 = max(s0, float(end_s))
    if e0 <= s0 + 0.15:
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        margin = _float_env("FOLDER_EDIT_STABILIZE_SHAKE_MARGIN_S", "0.06")
        t0 = max(0.0, s0 + float(margin))
        t1 = max(t0, e0 - float(margin))
        if t1 <= t0 + 0.08:
            return None

        dt = 1.0 / max(0.5, float(sample_fps))
        times: list[float] = []
        t = float(t0)
        while t <= float(t1) + 1e-6 and len(times) < int(max_samples):
            times.append(float(t))
            t += float(dt)
        if len(times) < 4:
            return None

        prev_gray_f: t.Any | None = None
        hann: t.Any | None = None
        motions: list[tuple[float, float]] = []
        width: float | None = None
        min_corr = _float_env("FOLDER_EDIT_STABILIZE_SHAKE_MIN_CORR", "0.08")

        for tt in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(tt) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                continue
            if w > int(max_w):
                scale = float(max_w) / float(w)
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
            gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_f = cv2.GaussianBlur(gray_f, (0, 0), 1.2)
            hh, ww = gray_f.shape[:2]
            if ww <= 0 or hh <= 0:
                continue
            width = float(ww)
            if hann is None or hann.shape[0] != hh or hann.shape[1] != ww:
                hann = cv2.createHanningWindow((ww, hh), cv2.CV_32F)

            if prev_gray_f is None:
                prev_gray_f = gray_f
                continue

            (dx, dy), resp = cv2.phaseCorrelate(prev_gray_f, gray_f, hann)  # type: ignore[arg-type]
            if float(resp) >= float(min_corr):
                motions.append((float(dx), float(dy)))
            prev_gray_f = gray_f

        if width is None or width <= 1.0 or len(motions) < 4:
            return None

        # Convert to velocity (px/s) and compute jitter residual after smoothing.
        vel = np.array([(dx / max(1e-6, float(dt)), dy / max(1e-6, float(dt))) for dx, dy in motions], dtype=np.float32)
        n = int(vel.shape[0])
        win = int(max(3, min(9, float(_float_env("FOLDER_EDIT_STABILIZE_SHAKE_SMOOTH_WIN", "5")))))
        if win % 2 == 0:
            win += 1
        half = win // 2
        smooth = np.zeros_like(vel)
        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            smooth[i] = vel[a:b].mean(axis=0)
        resid = vel - smooth
        resid_mag = np.linalg.norm(resid, axis=1)
        resid_norm = (resid_mag / max(1.0, float(width))).astype(np.float32)
        return float(np.percentile(resid_norm, 95.0)) if resid_norm.size else None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _ensure_segment_count(
    *,
    cut_times: list[float],
    duration_s: float,
    min_scenes: int,
    max_scenes: int,
) -> list[tuple[int, float, float]]:
    # Convert cut times -> segments.
    times = [t for t in cut_times if 0.0 < t < duration_s]
    times = sorted(set(times))
    boundaries = [0.0] + times + [duration_s]
    segs = [(i + 1, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # If too few, split longest segments.
    while len(segs) < min_scenes:
        # Split the longest segment in half.
        longest = max(segs, key=lambda x: (x[2] - x[1]))
        segs.remove(longest)
        sid, s, e = longest
        mid = (s + e) / 2.0
        segs.append((sid, s, mid))
        segs.append((sid + 1, mid, e))
        segs = sorted(segs, key=lambda x: x[1])
        # Re-number
        segs = [(i + 1, s, e) for i, (_, s, e) in enumerate(segs)]

    # If too many, merge the shortest segments into a neighbor.
    while len(segs) > max_scenes:
        # Find shortest segment
        idx = min(range(len(segs)), key=lambda i: (segs[i][2] - segs[i][1]))
        sid, s, e = segs[idx]
        if idx == 0:
            # Merge into next
            nsid, ns, ne = segs[idx + 1]
            segs[idx + 1] = (nsid, s, ne)
            segs.pop(idx)
        else:
            psid, ps, pe = segs[idx - 1]
            segs[idx - 1] = (psid, ps, e)
            segs.pop(idx)
        segs = [(i + 1, s, e) for i, (_, s, e) in enumerate(segs)]

    return segs


def _render_segment(
    *,
    asset_path: Path,
    asset_kind: str,
    in_s: float,
    duration_s: float,
    speed: float = 1.0,
    crop_mode: str,
    reframe: dict[str, t.Any] | None = None,
    overlay_text: str,
    grade: dict[str, float] | None = None,
    stabilize: bool = False,
    stabilize_cache_dir: Path | None = None,
    zoom: float = 1.0,
    fade_in_s: float | None = None,
    fade_in_color: str | None = None,
    fade_out_s: float | None = None,
    fade_out_color: str | None = None,
    output_path: Path,
    burn_overlay: bool = True,
    fps: int = 30,
    timeout_s: float = 240.0,
) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render segments. Please install ffmpeg and try again.")

    # Allow fast draft renders for variant search by overriding output params via env.
    width, height = 1080, 1920
    try:
        w_env = int(float(os.getenv("FOLDER_EDIT_RENDER_WIDTH", str(width))))
        h_env = int(float(os.getenv("FOLDER_EDIT_RENDER_HEIGHT", str(height))))
        if w_env >= 160 and h_env >= 160:
            width, height = w_env, h_env
    except Exception:
        pass
    try:
        fps_env = int(float(os.getenv("FOLDER_EDIT_RENDER_FPS", str(int(fps)))))
        if fps_env >= 10 and fps_env <= 60:
            fps = int(fps_env)
    except Exception:
        pass

    preset = (os.getenv("FOLDER_EDIT_RENDER_PRESET", "") or "veryfast").strip() or "veryfast"
    crf = (os.getenv("FOLDER_EDIT_RENDER_CRF", "") or "20").strip() or "20"
    vcodec = (os.getenv("FOLDER_EDIT_RENDER_VCODEC", "") or os.getenv("FOLDER_EDIT_RENDER_ENCODER", "") or "").strip()
    if not vcodec and _truthy_env("FOLDER_EDIT_RENDER_HWENC", "0"):
        # Best-effort: pick a reasonable default encoder when explicitly requested.
        if sys.platform == "darwin":
            vcodec = "h264_videotoolbox"
    if not vcodec:
        vcodec = "libx264"
    vbitrate = (os.getenv("FOLDER_EDIT_RENDER_VBITRATE", "") or "2500k").strip() or "2500k"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional stabilization: stabilize the whole source once, then trim the segment.
    if stabilize and asset_kind == "video" and asset_path.exists():
        stab_dir = stabilize_cache_dir or (output_path.parent / "stabilized_cache")
        stab_timeout = max(timeout_s, _float_env("FOLDER_EDIT_STABILIZE_TIMEOUT", "600"))
        try:
            asset_path = _ensure_stabilized_video(src=asset_path, cache_dir=stab_dir, timeout_s=stab_timeout)
        except Exception:
            # If stabilization fails, fall back to the original source.
            pass

    draw_filter: str | None = None
    if burn_overlay and (overlay_text or "").strip():
        # Caption as a textfile to avoid quoting bugs.
        captions_dir = output_path.parent / "captions"
        captions_dir.mkdir(parents=True, exist_ok=True)
        caption_wrapped = _wrap_caption(overlay_text or "")
        caption_file = captions_dir / f"{output_path.stem}.txt"
        caption_file.write_text(caption_wrapped or "", encoding="utf-8")

        font_path = _find_font_path()
        font_arg = f"fontfile={font_path}" if font_path else ""
        # In real reels, overlay text is often smaller than the "year" caption style.
        # Keep it readable but avoid dominating the frame.
        font_size = _font_size_for_caption(caption_wrapped or " ", max_size=72)
        borderw = max(3, int(font_size * 0.07))
        line_spacing = max(10, int(font_size * 0.18))
        text_y = "(h*0.55-text_h/2)"

        textfile_arg = _escape_filter_value(str(caption_file))
        draw_parts = [
            "drawtext=",
            font_arg,
            f"fontsize={font_size}",
            "fontcolor=white",
            "bordercolor=black",
            f"borderw={borderw}",
            "expansion=none",
            f"textfile='{textfile_arg}'",
            "reload=0",
            f"line_spacing={line_spacing}",
            "x=(w-text_w)/2",
            f"y={text_y}",
        ]
        draw_filter = ":".join([p for p in draw_parts if p and p != "drawtext="])
        draw_filter = "drawtext=" + draw_filter

    # Crop positioning after aspect-fill scale.
    crop_x = "(iw-ow)/2"
    crop_y = "(ih-oh)/2"
    if crop_mode == "top":
        crop_y = "0"
    elif crop_mode == "bottom":
        crop_y = "(ih-oh)"
    elif crop_mode == "face":
        # Approximate face-safe framing by biasing the crop upward.
        crop_y = "(ih-oh)*0.25"
    elif crop_mode in {"smart", "auto"} and isinstance(reframe, dict):
        # Smart crop: linear reframe between (cx0,cy0)->(cx1,cy1) over the segment duration.
        try:
            cx0 = float(reframe.get("cx0"))
            cy0 = float(reframe.get("cy0"))
            cx1 = float(reframe.get("cx1"))
            cy1 = float(reframe.get("cy1"))
            d = max(0.001, float(duration_s))
            cx = f"({cx0:.5f}+({cx1:.5f}-{cx0:.5f})*t/{d:.5f})"
            cy = f"({cy0:.5f}+({cy1:.5f}-{cy0:.5f})*t/{d:.5f})"
            crop_x = f"max(0,min(iw-ow,(iw*{cx})-(ow/2)))"
            crop_y = f"max(0,min(ih-oh,(ih*{cy})-(oh/2)))"
            # ffmpeg filterchains use ',' as a separator, so commas inside expressions
            # must be escaped.
            crop_x = crop_x.replace(",", "\\,")
            crop_y = crop_y.replace(",", "\\,")
        except Exception:
            crop_x = "(iw-ow)/2"
            crop_y = "(ih-oh)/2"

    try:
        speed_f = float(speed)
    except Exception:
        speed_f = 1.0
    if speed_f <= 0.0:
        speed_f = 1.0
    speed_f = min(1.5, max(0.75, speed_f))

    # Zoom-in helps hide stabilization borders and can make shots feel punchier.
    try:
        zoom_f = float(zoom)
    except Exception:
        zoom_f = 1.0
    zoom_f = min(1.25, max(1.0, zoom_f))

    vf_parts: list[str] = []
    if asset_kind == "video" and abs(speed_f - 1.0) >= 0.01:
        vf_parts.append(f"setpts=PTS/{speed_f:.4f}")

    scale_w = width
    scale_h = height
    if zoom_f > 1.001:
        scale_w = int(round(width * zoom_f))
        scale_h = int(round(height * zoom_f))
    vf_parts.extend(
        [
            f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}:x={crop_x}:y={crop_y}",
            "setsar=1",
        ]
    )
    if grade:
        # Mild color gain correction before luminance/contrast correction.
        try:
            rg = float(grade.get("r_gain") or 1.0)
            gg = float(grade.get("g_gain") or 1.0)
            bg = float(grade.get("b_gain") or 1.0)
        except Exception:
            rg, gg, bg = 1.0, 1.0, 1.0
        if abs(rg - 1.0) >= 0.01 or abs(gg - 1.0) >= 0.01 or abs(bg - 1.0) >= 0.01:
            vf_parts.append(f"colorchannelmixer=rr={rg:.4f}:gg={gg:.4f}:bb={bg:.4f}")

        try:
            b = float(grade.get("brightness") or 0.0)
        except Exception:
            b = 0.0
        try:
            c = float(grade.get("contrast") or 1.0)
        except Exception:
            c = 1.0
        if abs(b) >= 0.005 or abs(c - 1.0) >= 0.01:
            vf_parts.append(f"eq=brightness={b:.4f}:contrast={c:.4f}")
    if draw_filter:
        vf_parts.append(draw_filter)

    # Optional start fade-in (used for length-preserving transitions in review/improve loops).
    if fade_in_s is not None:
        try:
            fi = float(fade_in_s)
        except Exception:
            fi = 0.0
        color_in = str(fade_in_color or "black").strip().lower()
        if color_in not in {"black", "white"}:
            color_in = "black"
        if fi > 0.01 and float(duration_s) > 0.12:
            d = min(float(fi), max(0.05, float(duration_s) * 0.5))
            vf_parts.append(f"fade=t=in:st=0:d={d:.3f}:color={color_in}")

    # Optional end fade to black (prevents abrupt endings on the last segment).
    if fade_out_s is not None:
        try:
            fo = float(fade_out_s)
        except Exception:
            fo = 0.0
        color_out = str(fade_out_color or "black").strip().lower()
        if color_out not in {"black", "white"}:
            color_out = "black"
        if fo > 0.01 and float(duration_s) > 0.12:
            # Clamp fade to the tail of the segment to avoid swallowing short clips.
            d = min(float(fo), max(0.05, float(duration_s) * 0.5))
            st = max(0.0, float(duration_s) - float(d))
            vf_parts.append(f"fade=t=out:st={st:.3f}:d={d:.3f}:color={color_out}")
    vf = ",".join(vf_parts)

    cmd: list[str] = [ffmpeg, "-y"]
    if asset_kind == "video":
        # If the requested window exceeds the clip, loop the source.
        src_window_s = float(duration_s) * float(speed_f)
        cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-ss", f"{in_s:.3f}", "-i", str(asset_path), "-t", f"{src_window_s:.3f}"])
        cmd.extend(["-an"])
    else:
        cmd.extend(["-loop", "1", "-t", f"{duration_s:.3f}", "-i", str(asset_path)])

    cmd.extend(
        [
            "-vf",
            vf,
            "-r",
            str(fps),
            "-c:v",
            vcodec,
            *(
                ["-preset", preset, "-crf", crf]
                if vcodec in {"libx264", "libx265"}
                else ["-b:v", vbitrate]
            ),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _run(cmd, timeout_s=timeout_s)


def _concat_segments(
    *,
    segment_paths: list[Path],
    output_path: Path,
    timeout_s: float = 180.0,
) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to concat segments. Please install ffmpeg and try again.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_path.parent / "concat_list.txt"
    # Use absolute paths to avoid concat demuxer interpreting paths relative to the list file.
    concat_list.write_text("".join([f"file '{p.resolve().as_posix()}'\n" for p in segment_paths]), encoding="utf-8")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]
    _run(cmd, timeout_s=timeout_s)


def run_folder_edit_pipeline(
    *,
    reel_url_or_path: str,
    media_folder: Path,
    output_base: Path,
    analysis_model: str,
    api_key: str,
    pro_mode: bool = False,
    reference_image_path: Path | None = None,
    min_scenes: int = 7,
    max_scenes: int = 9,
    use_reference_audio: bool = True,
    audio_path: Path | None = None,
    burn_overlays: bool = False,
    shared_media_index_path: Path | None = None,
    shared_shot_index_obj: t.Any = None,
    site_url: str | None = None,
    app_name: str | None = None,
    timeout_s: float = 240.0,
    iterations: int = 2,
    improve_iters: int = 0,
    critic_model: str | None = None,
    critic_max_mb: float = 8.0,
    critic_pro_mode: bool | None = None,
    cancel_event: t.Any = None,
    progress_cb: t.Callable[[ProgressUpdate], None] | None = None,
) -> FolderEditPipelineResult:
    if cancel_event is None:
        class _Dummy:
            def is_set(self) -> bool:
                return False

        cancel_event = _Dummy()

    def emit(stage: str, current: int, total: int, message: str) -> None:
        if progress_cb:
            progress_cb(ProgressUpdate(stage=stage, current=current, total=total, message=message))

    output_paths = ensure_output_dirs(output_base)
    if bool(burn_overlays):
        # Safety: never write burned captions to the canonical final_video.mov name.
        # If captions are requested, switch the canonical path instead of rendering twice.
        import dataclasses as _dc

        output_paths = _dc.replace(output_paths, video_path=(output_paths.root / "final_video_with_captions.mov"))
    improve_iters = max(0, int(improve_iters or 0))
    critic_model = str((critic_model or "").strip() or os.getenv("DIRECTOR_MODEL", "") or analysis_model)
    critic_max_mb = float(critic_max_mb or 8.0)
    critic_pro_mode = bool(critic_pro_mode) if critic_pro_mode is not None else bool(pro_mode)

    # Acquire reel video.
    emit("Reel", 0, 3, "Downloading reel")
    if cancel_event.is_set():
        raise CancelledError("Cancelled before downloading reel")

    source_dir = output_paths.root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = Path(reel_url_or_path).expanduser()
    if candidate_path.exists():
        src_path = candidate_path.resolve()
    else:
        src_path = _download_reel(reel_url_or_path, output_dir=source_dir, timeout_s=timeout_s)

    emit("Reel", 1, 3, "Preparing clip for analysis")
    analysis_clip = source_dir / "analysis_clip.mp4"
    _compress_for_analysis(src_path, dst=analysis_clip, timeout_s=timeout_s)

    emit("Reel", 2, 3, "Extracting audio + detecting cuts")
    if cancel_event.is_set():
        raise CancelledError("Cancelled before cut detection")

    extracted_audio = source_dir / "reel_audio.m4a"
    try:
        extract_audio(video_path=analysis_clip, output_audio_path=extracted_audio, timeout_s=timeout_s)
    except Exception:
        extracted_audio = None  # type: ignore[assignment]

    cut_times, duration_s = detect_scene_cuts(
        video_path=analysis_clip,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        target_scenes=max_scenes,
        timeout_s=timeout_s,
    )
    segments = _ensure_segment_count(cut_times=cut_times, duration_s=duration_s, min_scenes=min_scenes, max_scenes=max_scenes)

    emit("Reel", 3, 3, f"Found {len(segments)} segments")

    # Optional: beat/onset snap segment boundaries to match music timing more closely.
    beat_sync = _truthy_env("FOLDER_EDIT_BEAT_SYNC", "0")
    music_doc: dict[str, t.Any] | None = None
    if beat_sync:
        # Prefer the user's chosen audio for analysis; otherwise use extracted reel audio.
        music_src = audio_path if audio_path else (Path(extracted_audio) if extracted_audio else None)  # type: ignore[arg-type]
        music_dir = output_paths.root / "reference"
        music_doc = _maybe_analyze_music(audio_path=music_src, output_dir=music_dir, timeout_s=min(timeout_s, 240.0))
        if music_doc:
            segments = _beat_snap_segments(segments=segments, duration_s=duration_s, music_doc=music_doc)

    # Extract frames per segment for analysis.
    # We keep a midpoint frame for downstream metrics/evaluation, and (optionally) multiple
    # frames across the segment for better VLM understanding of motion/intent.
    emit("Analyze", 0, 2, "Extracting segment frames")
    if cancel_event.is_set():
        raise CancelledError("Cancelled before frame extraction")

    frames_dir = output_paths.root / "reel_segment_frames"
    segment_frames: list[tuple[int, float, float, Path]] = []
    segment_frames_multi: list[tuple[int, float, float, list[tuple[float, Path]]]] = []
    frame_count = int(max(1, min(5, float(os.getenv("REF_SEGMENT_FRAME_COUNT", "3")))))

    def _sample_times(start_s: float, end_s: float, n: int) -> list[float]:
        if n <= 1:
            return [float((start_s + end_s) / 2.0)]
        # Avoid extreme endpoints (often black/transition frames).
        eps = float(_float_env("REF_SEGMENT_FRAME_EPS_S", "0.03"))
        s0 = float(start_s) + eps
        e0 = float(end_s) - eps
        if e0 <= s0 + 1e-3:
            s0 = float(start_s)
            e0 = float(end_s)
        out: list[float] = []
        for i in range(int(n)):
            out.append(float(s0 + (e0 - s0) * (i / max(1, n - 1))))
        return out

    for seg_id, start_s, end_s in segments:
        mid = (start_s + end_s) / 2.0
        frame_path = frames_dir / f"seg_{seg_id:02d}_mid.jpg"
        _extract_frame(video_path=analysis_clip, at_s=mid, out_path=frame_path, timeout_s=timeout_s)
        segment_frames.append((seg_id, start_s, end_s, frame_path))
        # Extra frames for analysis prompt.
        sample_paths: list[tuple[float, Path]] = []
        for j, t_s in enumerate(_sample_times(float(start_s), float(end_s), frame_count), start=1):
            # Reuse midpoint extraction if we hit it.
            if abs(float(t_s) - float(mid)) <= 0.02:
                sample_paths.append((float(t_s), frame_path))
                continue
            p = frames_dir / f"seg_{seg_id:02d}_f{j:02d}.jpg"
            _extract_frame(video_path=analysis_clip, at_s=float(t_s), out_path=p, timeout_s=timeout_s)
            sample_paths.append((float(t_s), p))
        segment_frames_multi.append((seg_id, start_s, end_s, sample_paths))
    ref_frame_by_id = {sid: p for sid, _s, _e, p in segment_frames}

    emit("Analyze", 1, 2, "Analyzing reference reel structure")

    reference_data_url = _encode_image_data_url(reference_image_path) if reference_image_path else None
    ref_analysis = analyze_reference_reel_segments(
        api_key=api_key,
        model=analysis_model,
        segment_frames=t.cast(list[tuple[int, float, float, t.Any]], segment_frames_multi) if segment_frames_multi else segment_frames,
        reference_image_data_url=reference_data_url,
        timeout_s=timeout_s,
        site_url=site_url,
        app_name=app_name,
    )
    emit("Analyze", 2, 2, "Analysis complete")

    # Attach simple luminance + darkness metrics to each segment to help match low-key reels.
    seg_luma: dict[int, float | None] = {sid: _luma_mean(p) for sid, _s, _e, p in segment_frames}
    seg_dark: dict[int, float | None] = {sid: _dark_frac(p) for sid, _s, _e, p in segment_frames}
    seg_rgb: dict[int, list[float] | None] = {sid: _rgb_mean(p) for sid, _s, _e, p in segment_frames}
    try:
        from .folder_edit_planner import ReferenceAnalysisPlan, ReferenceSegmentPlan

        beats = music_doc.get("beat_times") if isinstance(music_doc, dict) else None
        if not isinstance(beats, list):
            beats = []

        ref_analysis = ReferenceAnalysisPlan(
            analysis=ref_analysis.analysis,
            raw=ref_analysis.raw,
            segments=[
                ReferenceSegmentPlan(
                    id=s.id,
                    start_s=s.start_s,
                    end_s=s.end_s,
                    duration_s=s.duration_s,
                    beat_goal=s.beat_goal,
                    overlay_text=s.overlay_text,
                    reference_visual=s.reference_visual,
                    desired_tags=s.desired_tags,
                    ref_luma=seg_luma.get(s.id),
                    ref_dark_frac=seg_dark.get(s.id),
                    ref_rgb_mean=seg_rgb.get(s.id),
                    music_energy=_segment_music_energy(start_s=float(s.start_s), end_s=float(s.end_s), beats=t.cast(list[dict[str, t.Any]], beats)),
                    start_beat=_beat_floor_index(float(s.start_s), t.cast(list[dict[str, t.Any]], beats)),
                    end_beat=_beat_floor_index(float(s.end_s), t.cast(list[dict[str, t.Any]], beats)),
                )
                for s in ref_analysis.segments
            ],
        )
    except Exception:
        pass

    # Index and tag the media folder.
    emit("Library", 0, 3, "Reusing shared media index" if shared_media_index_path else "Indexing media folder")
    if cancel_event.is_set():
        raise CancelledError("Cancelled before library indexing")

    index_path = output_paths.root / "media_index.json"
    tagged_path = output_paths.root / "media_index_tagged.json"
    # Allow bench runs (and other tooling) to share a stable tag cache across outputs/commits.
    tag_cache_override = os.getenv("FOLDER_EDIT_TAG_CACHE_PATH", "").strip()
    global_tagged_path = Path(tag_cache_override).expanduser().resolve() if tag_cache_override else (output_base / "_asset_tag_cache.json")

    use_cache = _truthy_env("FOLDER_EDIT_MEDIA_INDEX_CACHE", "0")
    refresh_cache = _truthy_env("FOLDER_EDIT_MEDIA_INDEX_REFRESH", "0")
    cache_root = Path(os.getenv("FOLDER_EDIT_MEDIA_INDEX_CACHE_ROOT", str(output_base / "_media_index_cache"))).expanduser().resolve()

    if shared_media_index_path:
        shared_path = Path(shared_media_index_path).expanduser().resolve()
        if not shared_path.exists():
            raise RuntimeError(f"shared_media_index_path not found: {shared_path}")
        try:
            index = load_index(shared_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load shared media index: {type(e).__name__}: {e}") from e

        # Safety: ensure the shared index matches the requested media_folder.
        try:
            if Path(str(index.source_folder or "")).expanduser().resolve() != media_folder.expanduser().resolve():
                raise RuntimeError(f"shared media index source_folder mismatch: {index.source_folder!r} != {str(media_folder)}")
        except Exception as e:
            raise RuntimeError(f"Invalid shared media index: {type(e).__name__}: {e}") from e

        # Symlink/copy the shared index into this output folder for auditability.
        try:
            if index_path.exists() or index_path.is_symlink():
                try:
                    index_path.unlink()
                except Exception:
                    pass
            index_path.symlink_to(shared_path)
        except Exception:
            try:
                shutil.copyfile(shared_path, index_path)
            except Exception:
                pass

        # Also link thumbnails into the output folder if available.
        shared_thumbs = shared_path.parent / "asset_thumbs"
        local_thumbs = output_paths.root / "asset_thumbs"
        if shared_thumbs.exists() and not local_thumbs.exists():
            try:
                local_thumbs.symlink_to(shared_thumbs)
            except Exception:
                pass
    elif index_path.exists():
        try:
            index = load_index(index_path)
        except Exception:
            index = index_media_folder(folder=media_folder, output_dir=output_paths.root, timeout_s=timeout_s)
            save_index(index, index_path)
    elif use_cache:
        index, cached_index_path = load_or_build_cached_index(folder=media_folder, cache_root=cache_root, timeout_s=timeout_s, refresh=refresh_cache)
        try:
            index_path.symlink_to(cached_index_path)
        except Exception:
            try:
                shutil.copyfile(cached_index_path, index_path)
            except Exception:
                pass
        cached_thumbs = cached_index_path.parent / "asset_thumbs"
        local_thumbs = output_paths.root / "asset_thumbs"
        if cached_thumbs.exists() and not local_thumbs.exists():
            try:
                local_thumbs.symlink_to(cached_thumbs)
            except Exception:
                pass
    else:
        index = index_media_folder(folder=media_folder, output_dir=output_paths.root, timeout_s=timeout_s)
        save_index(index, index_path)

    emit("Library", 1, 3, f"Found {len(index.assets)} assets")

    # Prepare assets for tagger.
    assets_for_tagger: list[dict[str, t.Any]] = []
    for a in index.assets:
        if not a.thumbnail_path:
            continue
        assets_for_tagger.append(
            {
                "id": a.id,
                "kind": a.kind,
                "path": a.path,
                "filename": Path(a.path).name,
                "duration_s": a.duration_s,
                "thumbnail_path": a.thumbnail_path,
                "thumbnail_paths": getattr(a, "thumbnail_paths", None),
            }
        )

    tags: dict[str, t.Any] = {}
    if not pro_mode:
        emit("Library", 1, 3, "Tagging thumbnails")
        TAG_CACHE_VERSION = 2  # bump when tagger inputs/outputs change materially (e.g. multi-thumbnail tagging)
        need_write_global_cache = False
        # Reuse tags across runs to avoid re-tagging the same local footage for every reel.
        if global_tagged_path.exists():
            try:
                cached = json.loads(global_tagged_path.read_text(encoding="utf-8"))
                cached_tags = cached.get("tags") or {}
                if isinstance(cached_tags, dict):
                    tags = cached_tags

                # Back-compat: older cache may not have a version key. Keep the tags to avoid re-tagging,
                # but upgrade the file on disk to the current versioned format.
                v = cached.get("version", None)
                if v is None:
                    need_write_global_cache = True
                else:
                    try:
                        if int(v) != TAG_CACHE_VERSION:
                            need_write_global_cache = True
                    except Exception:
                        need_write_global_cache = True
            except Exception:
                tags = {}
        if tagged_path.exists():
            try:
                local_doc = json.loads(tagged_path.read_text(encoding="utf-8"))
                local_tags = local_doc.get("tags") or {}
                if isinstance(local_tags, dict):
                    tags.update(local_tags)
            except Exception:
                pass

        # Seed/upgrade the global cache even if we didn't need to tag anything this run.
        if tags and (need_write_global_cache or not global_tagged_path.exists()):
            try:
                global_tagged_path.parent.mkdir(parents=True, exist_ok=True)
                global_tagged_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            except Exception:
                pass

        missing = [a for a in assets_for_tagger if str(a["id"]) not in tags]
        if missing:
            new_tags = tag_assets_from_thumbnails(
                api_key=api_key,
                model=analysis_model,
                assets=missing,
                timeout_s=timeout_s,
                site_url=site_url,
                app_name=app_name,
            )
            for aid, tinfo in new_tags.items():
                tags[aid] = {
                    "description": tinfo.description,
                    "tags": tinfo.tags,
                    "shot_type": tinfo.shot_type,
                    "setting": tinfo.setting,
                    "mood": tinfo.mood,
                }

            tagged_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            try:
                global_tagged_path.parent.mkdir(parents=True, exist_ok=True)
                global_tagged_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            except Exception:
                pass

    shot_index_obj = None
    if pro_mode:
        if shared_shot_index_obj is not None:
            emit("Library", 2, 3, "Reusing shared shot library")
            shot_index_obj = shared_shot_index_obj
        else:
            emit("Library", 2, 3, "Building shot library")
            try:
                from shot_index import build_or_load_shot_index

                def _shot_progress(cur: int, total: int, msg: str) -> None:
                    emit("ShotIndex", cur, total, msg)

                shot_index_obj = build_or_load_shot_index(
                    media_index_path=index_path,
                    # Use a global cache so the system is truly \"library-aware\" across runs and
                    # we don't re-tag/re-index the same folder for every new output.
                    cache_dir=(Path("Outputs") / "_shot_index_cache"),
                    api_key=api_key,
                    model=analysis_model,
                    timeout_s=timeout_s,
                    site_url=site_url,
                    app_name=app_name,
                    progress_cb=_shot_progress,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to build shot index: {e}")

    emit("Library", 2, 3, "Planning edit (EDL)")
    if cancel_event.is_set():
        raise CancelledError("Cancelled before EDL planning")

    assets_for_planner: list[dict[str, t.Any]] = []
    asset_path_by_id: dict[str, dict[str, t.Any]] = {}
    for a in index.assets:
        thumb_paths_raw = getattr(a, "thumbnail_paths", None)
        thumb_paths: list[Path] = []
        if isinstance(thumb_paths_raw, list):
            for tp in thumb_paths_raw[:3]:
                p = Path(str(tp))
                if p.exists():
                    thumb_paths.append(p)
        if not thumb_paths and a.thumbnail_path:
            p = Path(a.thumbnail_path)
            if p.exists():
                thumb_paths = [p]

        # Aggregate lighting metrics across start/mid/end thumbnails (when available).
        lumas: list[float] = []
        darks: list[float] = []
        for p in thumb_paths:
            l = _luma_mean(p)
            d = _dark_frac(p)
            if isinstance(l, (int, float)):
                lumas.append(float(l))
            if isinstance(d, (int, float)):
                darks.append(float(d))
        luma_mean = (sum(lumas) / len(lumas)) if lumas else None
        luma_min = (min(lumas) if lumas else None)
        luma_max = (max(lumas) if lumas else None)
        dark_mean = (sum(darks) / len(darks)) if darks else None
        dark_min = (min(darks) if darks else None)
        dark_max = (max(darks) if darks else None)

        motion = _motion_score_from_thumbs(thumb_paths) if thumb_paths else None
        meta = {
            "id": a.id,
            "kind": a.kind,
            "path": a.path,
            "duration_s": a.duration_s,
            "luma_mean": luma_mean,
            "luma_min": luma_min,
            "luma_max": luma_max,
            "dark_frac": dark_mean,
            "dark_frac_min": dark_min,
            "dark_frac_max": dark_max,
            "motion_score": motion,
        }
        if a.id in tags:
            meta.update(tags[a.id])
        assets_for_planner.append(meta)
        asset_path_by_id[a.id] = meta

    # Pro-mode (global optimizer) iteration overrides driven by the full-video judge.
    # These are updated between iterations (e.g., after iter1 judge, before iter2 render).
    pro_forced_shots_by_seg_id: dict[int, dict[str, t.Any]] = {}
    pro_force_stabilize_segs: set[int] = set()
    pro_env_overrides: dict[str, str] = {}
    # Render-time overrides (lane A fixes + transition fades) to apply during (re)renders.
    # Keys are segment_id -> patch dict (stabilize/crop/zoom/grade/overlay/fade).
    render_overrides_by_seg_id: dict[int, dict[str, t.Any]] = {}
    # Base stability gates for pro-mode (cheap insurance against extreme shake).
    if pro_mode and _truthy_env("FOLDER_EDIT_STABILITY_GATES", "1"):
        pro_env_overrides.setdefault("OPT_SHAKE_GATE", "1")
        pro_env_overrides.setdefault("OPT_SHAKE_MAX", str(_float_env("FOLDER_EDIT_STABILITY_SHAKE_MAX", "0.25")))
    pro_env_base_overrides: dict[str, str] = dict(pro_env_overrides)

    # Optional: load the latest learned grader to steer shot selection (pro-mode only).
    learned_grader = None
    learned_grader_info: dict[str, t.Any] | None = None
    if pro_mode and _truthy_env("FOLDER_EDIT_LEARNED_GRADER_STEER", "1"):
        try:
            from grader_steering import find_latest_grader_dir
            from learned_grader import GeminiGrader

            grader_dir_env = str(os.getenv("FOLDER_EDIT_LEARNED_GRADER_DIR", "") or "").strip()
            if grader_dir_env:
                grader_dir = Path(grader_dir_env).expanduser().resolve()
            else:
                grader_dir = find_latest_grader_dir(output_base) or None

            if grader_dir and grader_dir.exists():
                learned_grader = GeminiGrader.load(grader_dir)
                learned_grader_info = {"ok": True, "grader_dir": str(grader_dir)}
        except Exception as e:
            learned_grader = None
            learned_grader_info = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    from contextlib import contextmanager

    @contextmanager
    def _temp_env(overrides: dict[str, str]):
        old: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
        os.environ.update({k: str(v) for k, v in overrides.items()})
        try:
            yield
        finally:
            for k, prev in old.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev

    def plan_and_render_pro(*, iteration: int) -> tuple[FolderEditPlan, Path, list[dict[str, t.Any]]]:
        if not shot_index_obj:
            raise RuntimeError("pro_mode requested but shot_index_obj is missing")

        from edit_optimizer import optimize_shot_sequence
        from grader_steering import choose_beam_sequence_with_grader
        from micro_editor import micro_edit_sequence
        from .folder_edit_planner import ReferenceSegmentPlan

        segments_for_edit = list(ref_analysis.segments)
        story_diag: dict[str, t.Any] | None = None

        # Optional: story planner constraints to reduce \"randomness\" by selecting coherent
        # sequence groups across the whole library (agentic, but meta/reel-agnostic).
        if _truthy_env("FOLDER_EDIT_STORY_PLANNER", "0"):
            try:
                from story_planner import (
                    load_story_plans,
                    plan_story_plans,
                    save_story_plans,
                    summarize_sequence_groups,
                )

                story_dir = output_paths.root / "reference"
                story_dir.mkdir(parents=True, exist_ok=True)
                story_path = story_dir / "story_plans.json"

                refresh = _truthy_env("FOLDER_EDIT_STORY_REFRESH", "0")
                plans = [] if refresh else (load_story_plans(story_path) if story_path.exists() else [])
                if not plans:
                    group_summaries = summarize_sequence_groups(shots=list(shot_index_obj.shots or []))
                    plans = plan_story_plans(
                        api_key=api_key,
                        model=analysis_model,
                        segments=segments_for_edit,
                        group_summaries=group_summaries,
                        music_doc=music_doc,
                        timeout_s=min(timeout_s, 240.0),
                        site_url=site_url,
                        app_name=app_name,
                        num_plans=3,
                    )
                    if plans:
                        save_story_plans(plans, story_path)

                if plans:
                    chosen_plan = plans[0]
                    by_id = {int(s.id): s for s in chosen_plan.segments}
                    new_segments: list[ReferenceSegmentPlan] = []
                    for seg in segments_for_edit:
                        sp = by_id.get(int(seg.id))
                        if sp:
                            new_segments.append(
                                ReferenceSegmentPlan(
                                    id=seg.id,
                                    start_s=seg.start_s,
                                    end_s=seg.end_s,
                                    duration_s=seg.duration_s,
                                    beat_goal=seg.beat_goal,
                                    overlay_text=seg.overlay_text,
                                    reference_visual=seg.reference_visual,
                                    desired_tags=sp.desired_tags or seg.desired_tags,
                                    ref_luma=seg.ref_luma,
                                    ref_dark_frac=seg.ref_dark_frac,
                                    ref_rgb_mean=getattr(seg, "ref_rgb_mean", None),
                                    music_energy=getattr(seg, "music_energy", None),
                                    start_beat=getattr(seg, "start_beat", None),
                                    end_beat=getattr(seg, "end_beat", None),
                                    story_beat=sp.story_beat,
                                    preferred_sequence_group_ids=sp.preferred_sequence_group_ids,
                                    transition_hint=sp.transition_hint,
                                )
                            )
                        else:
                            new_segments.append(seg)
                    segments_for_edit = new_segments
                    story_diag = {"plan_id": chosen_plan.plan_id, "concept": chosen_plan.concept}
            except Exception:
                story_diag = None

        # Export a few optimizer candidates so the learned grader can steer selection.
        if learned_grader is not None:
            beam_k = int(max(1, _float_env("FOLDER_EDIT_LEARNED_GRADER_BEAM_TOPK", "8")))
            pro_env_overrides.setdefault("OPT_EXPORT_BEAM_TOPK", str(beam_k))

        with _temp_env(pro_env_overrides):
            # Macro: choose a globally coherent, diverse shot sequence.
            chosen_shots, opt_diag = optimize_shot_sequence(segments=segments_for_edit, shots=shot_index_obj.shots)
            if len(chosen_shots) != len(segments_for_edit):
                raise RuntimeError(f"Optimizer returned {len(chosen_shots)} shots for {len(segments_for_edit)} segments")

            # Learned grader steering: pick the best candidate sequence from the optimizer beam.
            if learned_grader is not None and isinstance(opt_diag, dict) and isinstance(opt_diag.get("beam_final"), list):
                try:
                    shots_by_id = {str(s.get("id") or ""): s for s in (shot_index_obj.shots or []) if str(s.get("id") or "")}
                    default_speed = _float_env("FOLDER_EDIT_DEFAULT_SPEED", "1.0")
                    default_zoom = _float_env("FOLDER_EDIT_ZOOM", "1.0")
                    stabilize_enabled = _truthy_env("FOLDER_EDIT_STABILIZE", "1")
                    stabilize_shake_th = _float_env("FOLDER_EDIT_STEER_STABILIZE_SHAKE_TH", str(pro_env_overrides.get("OPT_SHAKE_MAX", "0.25")))
                    best_seq, steer_diag = choose_beam_sequence_with_grader(
                        segments=segments_for_edit,
                        shots_by_id=shots_by_id,
                        beam_final=t.cast(list[dict[str, t.Any]], opt_diag.get("beam_final")),
                        optimizer_diag=opt_diag,
                        grader=learned_grader,
                        default_speed=default_speed,
                        default_zoom=default_zoom,
                        stabilize_enabled=stabilize_enabled,
                        stabilize_shake_th=stabilize_shake_th,
                    )
                    if best_seq is not None and len(best_seq) == len(segments_for_edit):
                        chosen_shots = best_seq
                        opt_diag = dict(opt_diag)
                        opt_diag["learned_grader_steer"] = steer_diag
                        if learned_grader_info is not None:
                            opt_diag["learned_grader_loaded"] = learned_grader_info
                except Exception as e:
                    try:
                        opt_diag = dict(opt_diag)
                        opt_diag["learned_grader_steer"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        if learned_grader_info is not None:
                            opt_diag["learned_grader_loaded"] = learned_grader_info
                    except Exception:
                        pass

            # Apply forced per-segment recasts (judge-driven) while keeping the rest of the
            # sequence stable to avoid thrash across iterations.
            if pro_forced_shots_by_seg_id:
                try:
                    opt_diag = dict(opt_diag)
                    opt_diag["forced_recasts"] = sorted([int(k) for k in pro_forced_shots_by_seg_id.keys()])
                except Exception:
                    pass
                for j, seg in enumerate(segments_for_edit):
                    forced = pro_forced_shots_by_seg_id.get(int(seg.id))
                    if isinstance(forced, dict):
                        chosen_shots[j] = forced

            # Micro: choose in-points within shots (code-only), attach beat indices if available.
            micro_dir = output_paths.root / f"micro_edits_iter{iteration}"
            micro_decisions, micro_diag = micro_edit_sequence(
                segments=segments_for_edit,
                chosen_shots=chosen_shots,
                music_doc=music_doc,
                work_dir=micro_dir,
                timeout_s=timeout_s,
                default_crop_mode=str(os.getenv("FOLDER_EDIT_DEFAULT_CROP", "center") or "center"),
                default_speed=_float_env("FOLDER_EDIT_DEFAULT_SPEED", "1.0"),
            )

        decisions: list[EditDecision] = []
        shot_by_id: dict[str, dict[str, t.Any]] = {str(s.get("id") or ""): s for s in chosen_shots if str(s.get("id") or "")}
        for md in micro_decisions:
            decisions.append(
                EditDecision(
                    segment_id=md.segment_id,
                    asset_id=md.asset_id,
                    in_s=md.in_s,
                    duration_s=md.duration_s,
                    speed=md.speed,
                    crop_mode=md.crop_mode,
                    notes=f"shot_id={md.shot_id}",
                )
            )

        plan = FolderEditPlan(
            analysis={
                "overall_edit_strategy": "pro_mode: shot-level library + global optimizer (beam search) + code micro-editor (in-point selection).",
                "optimizer": opt_diag,
                "micro_editor": micro_diag,
                "story_planner": story_diag,
            },
            decisions=decisions,
            raw={"optimizer": opt_diag, "micro_editor": micro_diag, "story_planner": story_diag},
        )

        # Render (reuse existing renderer, grading, stabilization). Apply judge-driven env overrides
        # here too (e.g., stricter shake gates / lower stabilize thresholds in iter2).
        with _temp_env(pro_env_overrides):
            emit("Rendering", 0, len(segments_for_edit) + 1, f"Rendering segments (pro, iter {iteration})")
            segments_dir = output_paths.root / f"segments_iter{iteration}"
            grade_samples_dir = output_paths.root / f"grade_samples_iter{iteration}"
            auto_grade = _truthy_env("FOLDER_EDIT_AUTO_GRADE", "1")

            decisions_by_seg = {d.segment_id: d for d in plan.decisions}
            segment_paths: list[Path] = []
            timeline_segments: list[dict[str, t.Any]] = []
            fade_tail = _float_env("FOLDER_EDIT_FADE_OUT_S", "0.18")
            for idx, seg in enumerate(segments_for_edit, start=1):
                if cancel_event.is_set():
                    raise CancelledError("Cancelled during rendering")

                dec = decisions_by_seg.get(seg.id)
                if not dec:
                    raise RuntimeError(f"Missing decision for segment {seg.id}")

                # Resolve chosen shot metadata (for better diagnostics and stabilization choice).
                chosen_shot = None
                md = next((m for m in micro_decisions if m.segment_id == seg.id), None)
                if md:
                    chosen_shot = shot_by_id.get(md.shot_id)

                asset_meta = asset_path_by_id.get(dec.asset_id) or {}
                asset_path = Path(str(asset_meta.get("path") or ""))
                asset_kind = str(asset_meta.get("kind") or "video")
                out_path = segments_dir / f"seg_{seg.id:02d}.mp4"

                # Grade by sampling at the chosen in-point.
                grade: dict[str, float] | None = None
                if auto_grade:
                    ref_luma = seg.ref_luma
                    ref_dark = seg.ref_dark_frac
                    ref_rgb = getattr(seg, "ref_rgb_mean", None)
                    out_luma: float | None = None
                    out_dark: float | None = None
                    out_rgb: list[float] | None = None
                    if asset_kind == "video" and asset_path.exists():
                        try:
                            grade_samples_dir.mkdir(parents=True, exist_ok=True)
                            safe = f"{float(dec.in_s):.3f}".replace(".", "p")
                            sample_path = grade_samples_dir / f"seg_{seg.id:02d}_t_{safe}.jpg"
                            if not sample_path.exists():
                                _extract_frame(video_path=asset_path, at_s=float(dec.in_s), out_path=sample_path, timeout_s=min(timeout_s, 120.0))
                            out_luma = _luma_mean(sample_path)
                            out_dark = _dark_frac(sample_path)
                            out_rgb = _rgb_mean(sample_path)
                        except Exception:
                            pass
                    grade = _compute_eq_grade(ref_luma=ref_luma, ref_dark=ref_dark, out_luma=out_luma, out_dark=out_dark, ref_rgb=ref_rgb, out_rgb=out_rgb)

                # Stabilization uses motion proxy; prefer shot-level if available.
                stabilize = False
                zoom = _float_env("FOLDER_EDIT_ZOOM", "1.0")
                seg_shake_p95 = None
                stabilize_reason: str | None = None
                if asset_kind == "video" and _truthy_env("FOLDER_EDIT_STABILIZE", "1"):
                    # Prefer a per-segment shake estimate (jitter residual) when cv2 is available.
                    # This is more reliable than shot thumb-based shake_score and reduces \"warpy\" stabilizations.
                    if _truthy_env("FOLDER_EDIT_STABILIZE_USE_CV2_SHAKE", "1"):
                        try:
                            seg_shake_p95 = _estimate_shake_jitter_norm_p95(
                                video_path=asset_path,
                                start_s=float(dec.in_s),
                                end_s=float(dec.in_s) + float(seg.duration_s),
                            )
                        except Exception:
                            seg_shake_p95 = None

                    shake_th = _float_env("FOLDER_EDIT_STABILIZE_SHAKE_P95_THRESHOLD", "0.06")
                    if isinstance(seg_shake_p95, (int, float)) and float(seg_shake_p95) >= float(shake_th) and float(seg.duration_s) >= 1.0:
                        stabilize = True
                        zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.08"))
                        stabilize_reason = "cv2_shake_p95"
                    else:
                        # Fallback: legacy motion proxy.
                        motion = None
                        if isinstance(chosen_shot, dict):
                            motion = chosen_shot.get("motion_score")
                        if motion is None:
                            motion = asset_meta.get("motion_score")
                        motion_th = _float_env("FOLDER_EDIT_STABILIZE_MOTION_THRESHOLD", "0.17")
                        if isinstance(motion, (int, float)) and float(motion) >= float(motion_th) and float(seg.duration_s) >= 1.0:
                            stabilize = True
                            zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.08"))
                            stabilize_reason = "motion_proxy"

                # Judge-driven stabilization override (e.g., when the critic flags low stability).
                if int(seg.id) in pro_force_stabilize_segs and asset_kind == "video":
                    stabilize = True
                    zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.10"))
                    stabilize_reason = "forced_by_judge"

                # Lane A + transition overrides (used by the Two-Lane improvement loop).
                crop_mode = dec.crop_mode
                overlay_text = seg.overlay_text
                fade_in_s: float | None = None
                fade_in_color: str | None = None
                fade_out_s: float | None = (fade_tail if idx == len(segments_for_edit) else None)
                fade_out_color: str | None = ("black" if fade_out_s is not None else None)
                overrides = render_overrides_by_seg_id.get(int(seg.id))
                if isinstance(overrides, dict):
                    if "overlay_text" in overrides and overrides.get("overlay_text") is not None:
                        overlay_text = str(overrides.get("overlay_text") or "")
                    if "crop_mode" in overrides and overrides.get("crop_mode") is not None:
                        crop_mode = str(overrides.get("crop_mode") or crop_mode)
                    if "stabilize" in overrides and overrides.get("stabilize") is not None:
                        try:
                            stabilize = bool(overrides.get("stabilize"))
                            stabilize_reason = stabilize_reason or "override"
                        except Exception:
                            pass
                    if "zoom" in overrides and overrides.get("zoom") is not None:
                        try:
                            zoom = float(overrides.get("zoom"))
                        except Exception:
                            pass
                    if "grade" in overrides:
                        og = overrides.get("grade")
                        if og is None:
                            grade = None
                        elif isinstance(og, dict):
                            grade = {str(k): float(v) for k, v in og.items() if isinstance(k, str) and isinstance(v, (int, float))}
                    if "fade_in_s" in overrides and overrides.get("fade_in_s") is not None:
                        try:
                            fade_in_s = float(overrides.get("fade_in_s"))
                        except Exception:
                            fade_in_s = None
                    if "fade_in_color" in overrides and overrides.get("fade_in_color") is not None:
                        fade_in_color = str(overrides.get("fade_in_color") or "").strip().lower() or None
                    if "fade_out_s" in overrides:
                        try:
                            fade_out_s = float(overrides.get("fade_out_s")) if overrides.get("fade_out_s") is not None else None
                        except Exception:
                            fade_out_s = None
                    if "fade_out_color" in overrides and overrides.get("fade_out_color") is not None:
                        fade_out_color = str(overrides.get("fade_out_color") or "").strip().lower() or None

                _render_segment(
                    asset_path=asset_path,
                    asset_kind=asset_kind,
                    in_s=dec.in_s,
                    duration_s=seg.duration_s,
                    speed=dec.speed,
                    crop_mode=crop_mode,
                    reframe=(md.reframe if md else None),
                    overlay_text=overlay_text,
                    grade=grade,
                    stabilize=stabilize,
                    stabilize_cache_dir=(output_paths.root / "stabilized") if stabilize else None,
                    zoom=zoom,
                    fade_in_s=fade_in_s,
                    fade_in_color=fade_in_color,
                    fade_out_s=fade_out_s,
                    fade_out_color=fade_out_color,
                    output_path=out_path,
                    burn_overlay=burn_overlays,
                    timeout_s=timeout_s,
                )
                segment_paths.append(out_path)
                emit("Rendering", idx, len(segments_for_edit) + 1, f"Rendered {idx}/{len(segments_for_edit)} (pro, iter {iteration})")

                shot_id = md.shot_id if md else None
                shot_shake = None
                try:
                    if isinstance(chosen_shot, dict):
                        shot_shake = chosen_shot.get("shake_score")
                except Exception:
                    shot_shake = None
                timeline_segments.append(
                    {
                        "id": seg.id,
                        "start_s": seg.start_s,
                        "end_s": seg.end_s,
                        "duration_s": seg.duration_s,
                        "ref_luma": seg.ref_luma,
                        "ref_dark_frac": seg.ref_dark_frac,
                        "ref_rgb_mean": getattr(seg, "ref_rgb_mean", None),
                        "music_energy": getattr(seg, "music_energy", None),
                        "start_beat": getattr(seg, "start_beat", None),
                        "end_beat": getattr(seg, "end_beat", None),
                        "beat_goal": seg.beat_goal,
                        "overlay_text": overlay_text,
                        "reference_visual": seg.reference_visual,
                        "desired_tags": seg.desired_tags,
                        "story_beat": getattr(seg, "story_beat", None),
                        "preferred_sequence_group_ids": getattr(seg, "preferred_sequence_group_ids", None),
                        "transition_hint": getattr(seg, "transition_hint", None),
                        "shot_id": shot_id,
                        "sequence_group_id": (chosen_shot.get("sequence_group_id") if isinstance(chosen_shot, dict) else None),
                        "shot_shake_score": shot_shake,
                        "seg_shake_p95": seg_shake_p95,
                        "asset_id": dec.asset_id,
                        "asset_path": str(asset_path),
                        "asset_kind": asset_kind,
                        "asset_in_s": dec.in_s,
                        "asset_out_s": float(dec.in_s) + (float(seg.duration_s) * float(dec.speed)),
                        "speed": dec.speed,
                        "crop_mode": crop_mode,
                        "reframe": (md.reframe if md else None),
                        "grade": grade,
                        "stabilize": stabilize,
                        "zoom": zoom,
                        "fade_in_s": fade_in_s,
                        "fade_in_color": fade_in_color,
                        "fade_out_s": fade_out_s,
                        "fade_out_color": fade_out_color,
                        "stabilize_reason": stabilize_reason,
                        "notes": dec.notes,
                        "micro_editor": (md.debug if md else None),
                    }
                )

            emit("Rendering", len(segments_for_edit), len(segments_for_edit) + 1, f"Concatenating segments (pro, iter {iteration})")
            silent_video = output_paths.root / f"final_video_silent_iter{iteration}.mp4"
            _concat_segments(segment_paths=segment_paths, output_path=silent_video, timeout_s=timeout_s)
            return plan, silent_video, timeline_segments

    def plan_and_render(*, extra_guidance: str | None, iteration: int) -> tuple[FolderEditPlan, Path, list[dict[str, t.Any]]]:
        if pro_mode:
            return plan_and_render_pro(iteration=iteration)

        segmented_planning = _truthy_env("FOLDER_EDIT_SEGMENTED_PLAN", "1")
        if segmented_planning:
            emit("Plan", 0, len(ref_analysis.segments), f"Planning segments (iter {iteration})")
            used: set[str] = set()
            decisions: list[EditDecision] = []
            raw_by_seg: dict[str, t.Any] = {}
            for idx, seg in enumerate(ref_analysis.segments, start=1):
                if cancel_event.is_set():
                    raise CancelledError("Cancelled during planning")

                shortlist = _shortlist_assets_for_segment(segment=seg, assets=assets_for_planner, used_asset_ids=used, limit=24)
                seg_guidance_parts: list[str] = []
                if used:
                    seg_guidance_parts.append(f"Used asset_ids already: {sorted(used)[:12]} (avoid repeats unless necessary).")
                # Keep any evaluator-derived guidance but keep it meta.
                if extra_guidance:
                    seg_guidance_parts.append(extra_guidance.strip())
                seg_guidance = "\n".join([p for p in seg_guidance_parts if p.strip()]) or None

                seg_plan = plan_folder_edit_edl(
                    api_key=api_key,
                    model=analysis_model,
                    segments=[seg],
                    assets=shortlist,
                    reference_image_data_url=reference_data_url,
                    timeout_s=timeout_s,
                    site_url=site_url,
                    app_name=app_name,
                    extra_guidance=seg_guidance,
                )
                raw_by_seg[str(seg.id)] = seg_plan.raw

                # Extract decision; be tolerant if the model returns segment_id=1 for single-seg calls.
                dec = next((d for d in seg_plan.decisions if d.segment_id == seg.id), None)
                if dec is None and seg_plan.decisions:
                    d0 = seg_plan.decisions[0]
                    dec = EditDecision(
                        segment_id=seg.id,
                        asset_id=d0.asset_id,
                        in_s=d0.in_s,
                        duration_s=seg.duration_s,
                        speed=d0.speed,
                        crop_mode=d0.crop_mode,
                        notes=d0.notes,
                    )
                if dec is None:
                    # Fallback to first shortlist item.
                    fallback = shortlist[0]
                    dec = EditDecision(segment_id=seg.id, asset_id=str(fallback.get("id") or ""), in_s=0.0, duration_s=seg.duration_s)

                # Always keep reference pacing (duration is determined by the reel cut).
                dec = EditDecision(
                    segment_id=seg.id,
                    asset_id=dec.asset_id,
                    in_s=dec.in_s,
                    duration_s=seg.duration_s,
                    speed=dec.speed,
                    crop_mode=dec.crop_mode,
                    notes=dec.notes,
                )

                decisions.append(dec)
                if dec.asset_id:
                    used.add(dec.asset_id)
                emit("Plan", idx, len(ref_analysis.segments), f"Planned {idx}/{len(ref_analysis.segments)} (iter {iteration})")

            plan = FolderEditPlan(
                analysis={
                    "overall_edit_strategy": "Segmented planning with per-segment shortlists (lighting/negative space/motion + tag overlap).",
                    "risk_notes": "If the footage library lacks matches for a reference segment's look, the planner may reuse assets or rely on grading.",
                    "segmented_planning": True,
                },
                decisions=decisions,
                raw={"segments": raw_by_seg},
            )
        else:
            plan = plan_folder_edit_edl(
                api_key=api_key,
                model=analysis_model,
                segments=ref_analysis.segments,
                assets=assets_for_planner,
                reference_image_data_url=reference_data_url,
                timeout_s=timeout_s,
                site_url=site_url,
                app_name=app_name,
                extra_guidance=extra_guidance,
            )

        # Optionally refine in-points by comparing reference frames against candidate frames sampled
        # from the selected asset. This makes the model decide *where* to cut within the user's footage.
        refine_inpoints = _truthy_env("FOLDER_EDIT_REFINE_INPOINTS", "1")
        decisions_by_seg = {d.segment_id: d for d in plan.decisions}
        refinement_by_seg_id: dict[int, dict[str, t.Any]] = {}

        if refine_inpoints:
            emit("Refine", 0, len(ref_analysis.segments), f"Analyzing local footage for cut points (iter {iteration})")
            refine_root = output_paths.root / f"refine_inpoints_iter{iteration}"
            for idx, seg in enumerate(ref_analysis.segments, start=1):
                if cancel_event.is_set():
                    raise CancelledError("Cancelled during in-point refinement")

                dec = decisions_by_seg.get(seg.id)
                if not dec:
                    continue

                asset_meta = asset_path_by_id.get(dec.asset_id) or {}
                asset_kind = str(asset_meta.get("kind") or "video")
                if asset_kind != "video":
                    continue

                asset_path = Path(str(asset_meta.get("path") or ""))
                if not asset_path.exists():
                    continue

                # Sample many in-points, then select the best candidates by lighting + negative-space match
                # (and a small energy/motion proxy) BEFORE asking the model to choose.
                seg_dir = refine_root / f"seg_{seg.id:02d}"
                sample_dir = seg_dir / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)

                asset_dur = asset_meta.get("duration_s")
                try:
                    asset_dur_f = float(asset_dur) if isinstance(asset_dur, (int, float, str)) else None
                except Exception:
                    asset_dur_f = None

                seg_dur = float(seg.duration_s)
                max_start = 0.0
                if isinstance(asset_dur_f, (int, float)) and asset_dur_f and asset_dur_f > 0.0:
                    max_start = max(0.0, float(asset_dur_f) - seg_dur)
                init_t = max(0.0, float(dec.in_s))
                if max_start > 0.0:
                    init_t = min(max_start, init_t)

                sample_count = 14 if max_start > 0.0 else 1
                base_times = [0.0] if sample_count == 1 else [i * max_start / (sample_count - 1) for i in range(sample_count)]
                base_times.append(init_t)

                # De-dupe / normalize times.
                times: list[float] = []
                for t_s in base_times:
                    t3 = round(float(t_s), 3)
                    if t3 < 0.0:
                        continue
                    if t3 not in times:
                        times.append(t3)
                times.sort()

                # Extract sample frames + metrics.
                samples: list[tuple[float, Path, float | None, float | None]] = []
                for t_s in times:
                    safe = f"{t_s:.3f}".replace(".", "p")
                    frame_path = sample_dir / f"t_{safe}.jpg"
                    if not frame_path.exists():
                        _extract_frame(video_path=asset_path, at_s=float(t_s), out_path=frame_path, timeout_s=timeout_s)
                    luma = _luma_mean(frame_path)
                    dark = _dark_frac(frame_path)
                    samples.append((float(t_s), frame_path, luma, dark))

                # Motion proxy from successive sampled frames.
                motion_by_time: dict[float, float | None] = {}
                for i in range(len(samples)):
                    t0, p0, _l0, _d0 = samples[i]
                    if i < len(samples) - 1:
                        _t1, p1, _l1, _d1 = samples[i + 1]
                        motion_by_time[t0] = _frame_motion_diff(p0, p1)
                    elif i > 0:
                        _tprev, pprev, *_ = samples[i - 1]
                        motion_by_time[t0] = _frame_motion_diff(pprev, p0)
                    else:
                        motion_by_time[t0] = None

                # Rank candidates by lighting/negative-space match, plus slight motion match.
                ref_luma = seg.ref_luma
                ref_dark = seg.ref_dark_frac
                energy = _segment_energy_hint(seg.duration_s)

                ranked: list[tuple[float, tuple[float, Path, float | None, float | None, float | None]]] = []
                for t_s, p, luma, dark in samples:
                    dl = 0.18
                    dd = 0.25
                    if isinstance(ref_luma, (int, float)) and isinstance(luma, (int, float)):
                        dl = abs(float(luma) - float(ref_luma))
                    if isinstance(ref_dark, (int, float)) and isinstance(dark, (int, float)):
                        dd = abs(float(dark) - float(ref_dark))
                    m = motion_by_time.get(t_s)
                    motion = float(m) if isinstance(m, (int, float)) else 0.35
                    score = (dl * 1.2) + (dd * 1.0) + (abs(motion - energy) * 0.25)
                    ranked.append((float(score), (t_s, p, luma, dark, motion)))
                ranked.sort(key=lambda x: x[0])

                # Build a small, high-quality candidate set to reduce LLM load and improve reliability.
                top_k = 6
                score_by: dict[tuple[float, str], float] = {}
                for s, (t_s, p, _l, _d, _m) in ranked:
                    score_by[(float(t_s), str(p))] = float(s)

                cand_full = [item for _s, item in ranked[:top_k]]
                # Ensure the initial in-point (or closest) is represented.
                if ranked and init_t not in [t for t, *_ in cand_full]:
                    closest = min(ranked, key=lambda x: abs(x[1][0] - init_t))[1]
                    cand_full.append(closest)

                cand_full.sort(key=lambda it: (score_by.get((float(it[0]), str(it[1])), 0.0), float(it[0])))

                seen: set[float] = set()
                candidate_frames_full: list[tuple[float, Path, float | None, float | None, float | None]] = []
                for t_s, p, luma, dark, motion in cand_full:
                    if float(t_s) in seen:
                        continue
                    seen.add(float(t_s))
                    candidate_frames_full.append((float(t_s), p, luma, dark, motion))

                max_candidates = 5
                candidate_frames = candidate_frames_full[:max_candidates]
                if candidate_frames_full and init_t not in [t for t, *_ in candidate_frames]:
                    init_candidate = min(candidate_frames_full, key=lambda cf: abs(float(cf[0]) - float(init_t)))
                    if init_candidate[0] not in [t for t, *_ in candidate_frames]:
                        if len(candidate_frames) >= max_candidates:
                            candidate_frames[-1] = init_candidate
                        else:
                            candidate_frames.append(init_candidate)

                candidate_times = [float(t_s) for t_s, _p, _l, _d, _m in candidate_frames]
                candidate_options = [
                    {
                        "time_s": float(t_s),
                        "luma": luma,
                        "dark_frac": dark,
                        "motion": motion,
                    }
                    for t_s, _p, luma, dark, motion in candidate_frames
                ]

                ref_frame = ref_frame_by_id.get(seg.id)
                if not ref_frame:
                    continue

                try:
                    refinement = refine_inpoint_for_segment(
                        api_key=api_key,
                        model=analysis_model,
                        segment=seg,
                        reference_frame_path=ref_frame,
                        asset_meta=asset_meta,
                        candidate_frames=candidate_frames,
                        timeout_s=min(timeout_s, 180.0),
                        site_url=site_url,
                        app_name=app_name,
                    )
                    decisions_by_seg[seg.id] = EditDecision(
                        segment_id=seg.id,
                        asset_id=dec.asset_id,
                        in_s=refinement.chosen_time_s,
                        duration_s=seg.duration_s,
                        speed=refinement.speed,
                        crop_mode=refinement.crop_mode,
                        notes=dec.notes,
                    )
                    chosen_tuple = next(
                        (cf for cf in candidate_frames if abs(float(cf[0]) - float(refinement.chosen_time_s)) < 0.001),
                        None,
                    )
                    refinement_by_seg_id[seg.id] = {
                        "candidate_time_s": candidate_times,
                        "candidate_options": candidate_options,
                        "chosen_time_s": refinement.chosen_time_s,
                        "chosen_luma": (chosen_tuple[2] if chosen_tuple else None),
                        "chosen_dark_frac": (chosen_tuple[3] if chosen_tuple else None),
                        "chosen_motion": (chosen_tuple[4] if chosen_tuple else None),
                        "crop_mode": refinement.crop_mode,
                        "speed": refinement.speed,
                        "reason": refinement.reason,
                    }
                except Exception as e:
                    msg = str(e)
                    # Fallback: choose the best-scoring candidate deterministically (keeps pipeline robust).
                    if candidate_frames:
                        fb_t, _fb_p, fb_l, fb_d, fb_m = candidate_frames[0]
                        decisions_by_seg[seg.id] = EditDecision(
                            segment_id=seg.id,
                            asset_id=dec.asset_id,
                            in_s=float(fb_t),
                            duration_s=seg.duration_s,
                            speed=dec.speed,
                            crop_mode=dec.crop_mode,
                            notes=dec.notes,
                        )
                        refinement_by_seg_id[seg.id] = {
                            "error": msg[:500],
                            "fallback_time_s": float(fb_t),
                            "candidate_time_s": candidate_times,
                            "candidate_options": candidate_options,
                            "chosen_time_s": float(fb_t),
                            "chosen_luma": fb_l,
                            "chosen_dark_frac": fb_d,
                            "chosen_motion": fb_m,
                            "crop_mode": dec.crop_mode,
                            "speed": dec.speed,
                            "reason": "fallback: best candidate by score (LLM refine failed)",
                        }
                    else:
                        refinement_by_seg_id[seg.id] = {"error": msg[:500]}

                emit("Refine", idx, len(ref_analysis.segments), f"Refined {idx}/{len(ref_analysis.segments)} (iter {iteration})")

        # Render.
        emit("Rendering", 0, len(ref_analysis.segments) + 1, f"Rendering segments (iter {iteration})")
        if cancel_event.is_set():
            raise CancelledError("Cancelled before rendering")

        segments_dir = output_paths.root / f"segments_iter{iteration}"
        grade_samples_dir = output_paths.root / f"grade_samples_iter{iteration}"
        auto_grade = _truthy_env("FOLDER_EDIT_AUTO_GRADE", "1")
        segment_paths: list[Path] = []
        timeline_segments: list[dict[str, t.Any]] = []
        fade_tail = _float_env("FOLDER_EDIT_FADE_OUT_S", "0.18")
        for idx, seg in enumerate(ref_analysis.segments, start=1):
            if cancel_event.is_set():
                raise CancelledError("Cancelled during rendering")
            dec = decisions_by_seg.get(seg.id)
            if not dec:
                # Fallback: use the first asset.
                fallback = index.assets[0]
                dec = EditDecision(segment_id=seg.id, asset_id=fallback.id, in_s=0.0, duration_s=seg.duration_s)

            asset_meta = asset_path_by_id.get(dec.asset_id) or {}
            asset_path = Path(str(asset_meta.get("path") or ""))
            asset_kind = str(asset_meta.get("kind") or "video")
            out_path = segments_dir / f"seg_{seg.id:02d}.mp4"

            # Optional: mild per-segment grading to reduce lighting mismatch vs reference.
            grade: dict[str, float] | None = None
            if auto_grade:
                ref_luma = seg.ref_luma
                ref_dark = seg.ref_dark_frac
                ref_rgb = getattr(seg, "ref_rgb_mean", None)
                out_luma: float | None = None
                out_dark: float | None = None
                out_rgb: list[float] | None = None
                refn = refinement_by_seg_id.get(seg.id)
                if isinstance(refn, dict):
                    ol = refn.get("chosen_luma")
                    od = refn.get("chosen_dark_frac")
                    orgb = refn.get("chosen_rgb_mean")
                    if isinstance(ol, (int, float)):
                        out_luma = float(ol)
                    if isinstance(od, (int, float)):
                        out_dark = float(od)
                    if isinstance(orgb, list) and len(orgb) == 3:
                        out_rgb = [float(x) for x in orgb if isinstance(x, (int, float))][:3]  # type: ignore[list-item]

                # If we don't have chosen metrics (e.g. refinement disabled/failed), sample at in_s.
                if out_luma is None or out_dark is None or out_rgb is None:
                    if asset_kind == "video" and asset_path.exists():
                        try:
                            grade_samples_dir.mkdir(parents=True, exist_ok=True)
                            safe = f"{float(dec.in_s):.3f}".replace(".", "p")
                            sample_path = grade_samples_dir / f"seg_{seg.id:02d}_t_{safe}.jpg"
                            if not sample_path.exists():
                                _extract_frame(
                                    video_path=asset_path,
                                    at_s=float(dec.in_s),
                                    out_path=sample_path,
                                    timeout_s=min(timeout_s, 120.0),
                                )
                            out_luma = out_luma if out_luma is not None else _luma_mean(sample_path)
                            out_dark = out_dark if out_dark is not None else _dark_frac(sample_path)
                            out_rgb = out_rgb if out_rgb is not None else _rgb_mean(sample_path)
                        except Exception:
                            pass
                    elif asset_path.exists():
                        out_luma = out_luma if out_luma is not None else _luma_mean(asset_path)
                        out_dark = out_dark if out_dark is not None else _dark_frac(asset_path)
                        out_rgb = out_rgb if out_rgb is not None else _rgb_mean(asset_path)

                grade = _compute_eq_grade(ref_luma=ref_luma, ref_dark=ref_dark, out_luma=out_luma, out_dark=out_dark, ref_rgb=ref_rgb, out_rgb=out_rgb)

            # Optional stabilization + punch-in.
            stabilize = False
            zoom = _float_env("FOLDER_EDIT_ZOOM", "1.0")
            if asset_kind == "video" and _truthy_env("FOLDER_EDIT_STABILIZE", "1"):
                try:
                    motion = asset_path_by_id.get(dec.asset_id, {}).get("motion_score")
                except Exception:
                    motion = None
                motion_th = _float_env("FOLDER_EDIT_STABILIZE_MOTION_THRESHOLD", "0.17")
                if isinstance(motion, (int, float)) and float(motion) >= float(motion_th) and float(seg.duration_s) >= 1.0:
                    stabilize = True
                    zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.08"))

            # Lane A + transition overrides (used by the Two-Lane improvement loop).
            crop_mode = dec.crop_mode
            overlay_text = seg.overlay_text
            fade_in_s: float | None = None
            fade_in_color: str | None = None
            fade_out_s: float | None = (fade_tail if idx == len(ref_analysis.segments) else None)
            fade_out_color: str | None = ("black" if fade_out_s is not None else None)
            overrides = render_overrides_by_seg_id.get(int(seg.id))
            if isinstance(overrides, dict):
                if "overlay_text" in overrides and overrides.get("overlay_text") is not None:
                    overlay_text = str(overrides.get("overlay_text") or "")
                if "crop_mode" in overrides and overrides.get("crop_mode") is not None:
                    crop_mode = str(overrides.get("crop_mode") or crop_mode)
                if "stabilize" in overrides and overrides.get("stabilize") is not None:
                    try:
                        stabilize = bool(overrides.get("stabilize"))
                    except Exception:
                        pass
                if "zoom" in overrides and overrides.get("zoom") is not None:
                    try:
                        zoom = float(overrides.get("zoom"))
                    except Exception:
                        pass
                if "grade" in overrides:
                    og = overrides.get("grade")
                    if og is None:
                        grade = None
                    elif isinstance(og, dict):
                        grade = {str(k): float(v) for k, v in og.items() if isinstance(k, str) and isinstance(v, (int, float))}
                if "fade_in_s" in overrides and overrides.get("fade_in_s") is not None:
                    try:
                        fade_in_s = float(overrides.get("fade_in_s"))
                    except Exception:
                        fade_in_s = None
                if "fade_in_color" in overrides and overrides.get("fade_in_color") is not None:
                    fade_in_color = str(overrides.get("fade_in_color") or "").strip().lower() or None
                if "fade_out_s" in overrides:
                    try:
                        fade_out_s = float(overrides.get("fade_out_s")) if overrides.get("fade_out_s") is not None else None
                    except Exception:
                        fade_out_s = None
                if "fade_out_color" in overrides and overrides.get("fade_out_color") is not None:
                    fade_out_color = str(overrides.get("fade_out_color") or "").strip().lower() or None

            _render_segment(
                asset_path=asset_path,
                asset_kind=asset_kind,
                in_s=dec.in_s,
                duration_s=seg.duration_s,
                speed=dec.speed,
                crop_mode=crop_mode,
                overlay_text=overlay_text,
                grade=grade,
                stabilize=stabilize,
                stabilize_cache_dir=(output_paths.root / "stabilized") if stabilize else None,
                zoom=zoom,
                fade_in_s=fade_in_s,
                fade_in_color=fade_in_color,
                fade_out_s=fade_out_s,
                fade_out_color=fade_out_color,
                output_path=out_path,
                burn_overlay=burn_overlays,
                timeout_s=timeout_s,
            )
            segment_paths.append(out_path)
            emit("Rendering", idx, len(ref_analysis.segments) + 1, f"Rendered {idx}/{len(ref_analysis.segments)} (iter {iteration})")
            timeline_segments.append(
                {
                    "id": seg.id,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "duration_s": seg.duration_s,
                    "ref_luma": seg.ref_luma,
                    "ref_dark_frac": seg.ref_dark_frac,
                    "ref_rgb_mean": getattr(seg, "ref_rgb_mean", None),
                    "music_energy": getattr(seg, "music_energy", None),
                    "start_beat": getattr(seg, "start_beat", None),
                    "end_beat": getattr(seg, "end_beat", None),
                    "beat_goal": seg.beat_goal,
                    "overlay_text": overlay_text,
                    "reference_visual": seg.reference_visual,
                    "desired_tags": seg.desired_tags,
                    "asset_id": dec.asset_id,
                    "asset_path": str(asset_path),
                    "asset_kind": asset_kind,
                    "asset_in_s": dec.in_s,
                    "asset_out_s": float(dec.in_s) + (float(seg.duration_s) * float(dec.speed)),
                    "speed": dec.speed,
                    "crop_mode": crop_mode,
                    "grade": grade,
                    "stabilize": stabilize,
                    "zoom": zoom,
                    "fade_in_s": fade_in_s,
                    "fade_in_color": fade_in_color,
                    "fade_out_s": fade_out_s,
                    "fade_out_color": fade_out_color,
                    "notes": dec.notes,
                    "inpoint_refinement": refinement_by_seg_id.get(seg.id),
                }
            )

        emit("Rendering", len(ref_analysis.segments), len(ref_analysis.segments) + 1, f"Concatenating segments (iter {iteration})")
        silent_video = output_paths.root / f"final_video_silent_iter{iteration}.mp4"
        _concat_segments(segment_paths=segment_paths, output_path=silent_video, timeout_s=timeout_s)
        return plan, silent_video, timeline_segments

    iterations = max(1, int(iterations))
    extra_guidance: str | None = None
    last_eval: dict[str, t.Any] | None = None

    folder_plan, silent_video, timeline_segments = plan_and_render(extra_guidance=None, iteration=1)
    emit("Library", 3, 3, "EDL ready (iter 1)")

    if iterations >= 2:
        # Pro-mode: use the full-video Gemini judge to drive stabilization/recast/retime in iter2.
        if pro_mode and _truthy_env("FOLDER_EDIT_JUDGE_LOOP", "1"):
            try:
                from .folder_edit_evaluator import evaluate_edit_full_video_compare
            except Exception:
                evaluate_edit_full_video_compare = None  # type: ignore[assignment]

            def _safe_float(x: t.Any) -> float | None:
                try:
                    return float(x)
                except Exception:
                    return None

            def _top_issue_contains(doc: dict[str, t.Any], needle: str) -> bool:
                issues = doc.get("top_issues") or []
                if not isinstance(issues, list):
                    return False
                n = str(needle).strip().lower()
                return any(n in str(x or "").lower() for x in issues)

            # Determine audio for judge (optional but helps rhythm scoring).
            final_audio_for_judge: Path | None = None
            if audio_path:
                final_audio_for_judge = audio_path
            elif use_reference_audio and extracted_audio:
                final_audio_for_judge = Path(extracted_audio)  # type: ignore[arg-type]

            judge_dir = output_paths.root / "judge"
            judge_dir.mkdir(parents=True, exist_ok=True)
            out_with_audio = judge_dir / "out_iter1_with_audio.mp4"
            try:
                shutil.copyfile(silent_video, out_with_audio)
                if final_audio_for_judge:
                    merge_audio(video_path=out_with_audio, audio_path=final_audio_for_judge, output_path=out_with_audio)
            except Exception:
                # Fallback: use silent output for judging.
                out_with_audio = silent_video

            out_proxy = judge_dir / "out_iter1_judge.mp4"
            ref_proxy = judge_dir / "ref_judge.mp4"
            # Build small judge proxies (fast to upload, stable to parse). Can be disabled via env.
            if not _truthy_env("FOLDER_EDIT_JUDGE_NO_PROXY", "0"):
                try:
                    if out_with_audio != silent_video:
                        _compress_for_analysis(out_with_audio, dst=out_proxy, timeout_s=min(timeout_s, 240.0))
                    else:
                        _compress_for_analysis(silent_video, dst=out_proxy, timeout_s=min(timeout_s, 240.0))
                except Exception:
                    out_proxy = out_with_audio
                try:
                    if not ref_proxy.exists():
                        _compress_for_analysis(analysis_clip, dst=ref_proxy, timeout_s=min(timeout_s, 240.0))
                except Exception:
                    ref_proxy = analysis_clip
            else:
                out_proxy = out_with_audio
                ref_proxy = analysis_clip

            critic_model = str(os.getenv("CRITIC_MODEL", analysis_model) or analysis_model)
            criteria = str(
                os.getenv(
                    "FOLDER_EDIT_JUDGE_CRITERIA",
                    "Be extremely strict about stability, transitions, and rhythm. If anything is shaky or messy, score stability low and cap overall accordingly.",
                )
            )
            prompt_variant = str(os.getenv("FOLDER_EDIT_JUDGE_PROMPT", "compare_general") or "compare_general")

            judge1: dict[str, t.Any] | None = None
            if callable(evaluate_edit_full_video_compare):
                emit("Evaluate", 0, 1, "Full-video judging iter 1 vs reference (Gemini Pro compare)")
                try:
                    ev1 = evaluate_edit_full_video_compare(
                        api_key=api_key,
                        model=critic_model,
                        output_video_path=out_proxy,
                        reference_video_path=ref_proxy,
                        criteria=criteria,
                        prompt_variant=prompt_variant,
                        timeout_s=min(timeout_s, 240.0),
                        site_url=site_url,
                        app_name=app_name,
                    )
                    judge1 = {
                        "ok": True,
                        "result": ev1.result,
                        "usage": ev1.usage,
                        "model_requested": ev1.model_requested,
                        "model_used": ev1.model_used,
                    }
                    (output_paths.root / "evaluation_fullvideo_iter1.json").write_text(json.dumps(judge1, indent=2), encoding="utf-8")
                except Exception as e:
                    judge1 = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    (output_paths.root / "evaluation_fullvideo_iter1.json").write_text(json.dumps(judge1, indent=2), encoding="utf-8")
                emit("Evaluate", 1, 1, "Full-video judge complete (iter 1)")

            # If judge failed, fall back to the legacy frame-based loop.
            if not (isinstance(judge1, dict) and judge1.get("ok") and isinstance(judge1.get("result"), dict)):
                # Evaluate similarity and derive reusable planner guidance.
                emit("Evaluate", 0, 1, "Evaluating edit vs reference")
                def eval_for_video(video_path: Path, *, tag: str) -> dict[str, t.Any]:
                    eval_dir = output_paths.root / f"evaluation_frames_{tag}"
                    out_frames: list[tuple[int, Path]] = []
                    ref_frames: list[tuple[int, Path]] = []
                    summaries: list[str] = []
                    for seg_id, start_s, end_s, ref_frame_path in segment_frames:
                        mid = (start_s + end_s) / 2.0
                        out_path = eval_dir / f"out_{seg_id:02d}.jpg"
                        _extract_frame(video_path=video_path, at_s=mid, out_path=out_path, timeout_s=timeout_s)
                        out_frames.append((seg_id, out_path))
                        ref_frames.append((seg_id, ref_frame_path))
                    for item in timeline_segments:
                        summaries.append(
                            f"beat={item.get('beat_goal')} overlay={item.get('overlay_text')!r} desired_tags={item.get('desired_tags')} asset_kind={item.get('asset_kind')} crop={item.get('crop_mode')}"
                        )

                    evaluation = evaluate_edit_similarity(
                        api_key=api_key,
                        model=analysis_model,
                        reference_frames=ref_frames,
                        output_frames=out_frames,
                        segment_summaries=summaries,
                        timeout_s=timeout_s,
                        site_url=site_url,
                        app_name=app_name,
                    )
                    (output_paths.root / f"evaluation_{tag}.json").write_text(json.dumps(evaluation.raw, indent=2), encoding="utf-8")
                    return evaluation.raw

                eval1 = eval_for_video(silent_video, tag="iter1")
                extra_guidance = str(eval1.get("planner_guidance") or "").strip() or None
                last_eval = {"iter1": eval1}

                try:
                    score1 = float(eval1.get("overall_score") or 0.0)
                except Exception:
                    score1 = 0.0
                emit("Evaluate", 1, 1, f"Evaluation score (iter 1): {score1:.1f}/10")

                if extra_guidance:
                    folder_plan, silent_video, timeline_segments = plan_and_render(extra_guidance=extra_guidance, iteration=2)
                    emit("Library", 3, 3, "EDL ready (iter 2)")

                    # Verify the improvement with a second evaluation pass.
                    emit("Evaluate", 0, 1, "Evaluating iter 2 vs reference")
                    eval2 = eval_for_video(silent_video, tag="iter2")
                    last_eval = {"iter1": eval1, "iter2": eval2}
                    try:
                        score2 = float(eval2.get("overall_score") or 0.0)
                    except Exception:
                        score2 = 0.0
                    emit("Evaluate", 1, 1, f"Evaluation score (iter 2): {score2:.1f}/10")
            else:
                # Judge succeeded: derive a conservative fix plan and re-render iter2.
                r1 = t.cast(dict[str, t.Any], judge1.get("result"))
                stability = _safe_float(r1.get("stability")) or 0.0
                rhythm = _safe_float(r1.get("rhythm")) or 0.0
                overall = _safe_float(r1.get("overall_score")) or 0.0

                # Tune policy knobs for iter2 (bounded, conservative).
                pro_env_overrides = dict(pro_env_base_overrides)
                if stability <= float(_float_env("FOLDER_EDIT_JUDGE_STABILITY_TUNE_TH", "3.5")) or _top_issue_contains(r1, "shak"):
                    # Make selection more stable and enable more stabilization.
                    pro_env_overrides["OPT_SHAKE_GATE"] = "1"
                    pro_env_overrides["OPT_SHAKE_MAX"] = str(_float_env("FOLDER_EDIT_JUDGE_SHAKE_MAX", "0.25"))
                    pro_env_overrides["FOLDER_EDIT_STABILIZE"] = "1"
                    pro_env_overrides["FOLDER_EDIT_STABILIZE_SHAKE_P95_THRESHOLD"] = str(_float_env("FOLDER_EDIT_JUDGE_SHAKE_P95_THRESHOLD", "0.05"))
                    pro_env_overrides["FOLDER_EDIT_STABILIZE_ZOOM"] = str(_float_env("FOLDER_EDIT_JUDGE_STABILIZE_ZOOM", "1.10"))
                if rhythm <= float(_float_env("FOLDER_EDIT_JUDGE_RHYTHM_TUNE_TH", "3.0")) or _top_issue_contains(r1, "rhythm") or _top_issue_contains(r1, "timing"):
                    # Search more inpoints for better micro-timing.
                    pro_env_overrides["MICRO_INPOINT_SAMPLES"] = str(int(_float_env("FOLDER_EDIT_JUDGE_MICRO_SAMPLES", "16")))
                    pro_env_overrides["MICRO_DP_KEEP"] = str(int(_float_env("FOLDER_EDIT_JUDGE_MICRO_KEEP", "12")))

                # Decide which segments to force-stabilize / recast based on measured shake.
                seg_rows = [s for s in timeline_segments if isinstance(s, dict)]
                # Prefer cv2 per-segment shake; fall back to shot_index shake_score.
                scored_segs: list[tuple[float, int]] = []
                for s in seg_rows:
                    sid = int(s.get("id") or 0)
                    if sid <= 0:
                        continue
                    sp = _safe_float(s.get("seg_shake_p95"))
                    if sp is None:
                        sp = _safe_float(s.get("shot_shake_score"))
                    if sp is None:
                        continue
                    scored_segs.append((float(sp), sid))
                scored_segs.sort(reverse=True)

                # Force stabilize on moderately shaky segments when judge stability is low.
                pro_force_stabilize_segs = set()
                force_thr = float(_float_env("FOLDER_EDIT_JUDGE_FORCE_STABILIZE_THR", "0.06"))
                if stability <= float(_float_env("FOLDER_EDIT_JUDGE_STABILITY_FORCE_TH", "3.5")):
                    for sp, sid in scored_segs:
                        if float(sp) >= float(force_thr):
                            pro_force_stabilize_segs.add(int(sid))

                # Recast the worst shaky segments (only a few) to avoid over-changing the edit.
                pro_forced_shots_by_seg_id = {}
                recast_topk = int(max(0, _float_env("FOLDER_EDIT_JUDGE_RECAST_TOPK", "2")))
                recast_thr = float(_float_env("FOLDER_EDIT_JUDGE_RECAST_THR", "0.10"))
                if stability <= float(_float_env("FOLDER_EDIT_JUDGE_STABILITY_RECAST_TH", "3.0")) or overall <= float(_float_env("FOLDER_EDIT_JUDGE_OVERALL_RECAST_TH", "5.5")):
                    try:
                        from edit_optimizer import shortlist_shots_for_segment
                    except Exception:
                        shortlist_shots_for_segment = None  # type: ignore[assignment]

                    if callable(shortlist_shots_for_segment):
                        used_assets: dict[str, int] = {}
                        for s in seg_rows:
                            aid = str(s.get("asset_id") or "").strip()
                            if aid:
                                used_assets[aid] = used_assets.get(aid, 0) + 1
                        # Prefer swapping only segments that are meaningfully shaky.
                        to_recast = [sid for sp, sid in scored_segs if float(sp) >= float(recast_thr)][: max(0, recast_topk)]
                        if to_recast:
                            # Build lookup for current shot ids.
                            shot_by_id_all = {str(x.get("id") or ""): x for x in (shot_index_obj.shots or []) if isinstance(x, dict) and str(x.get("id") or "")}
                            # Segment objects for cost-based candidate selection.
                            seg_obj_by_id = {int(s.id): s for s in list(ref_analysis.segments)}

                            shake_max = float(_float_env("FOLDER_EDIT_JUDGE_RECAST_SHAKE_MAX", "0.22"))
                            sharp_min = float(_float_env("FOLDER_EDIT_JUDGE_RECAST_SHARP_MIN", "80.0"))

                            def _f(x: t.Any) -> float | None:
                                try:
                                    return float(x)
                                except Exception:
                                    return None

                            for sid in to_recast:
                                seg_obj = seg_obj_by_id.get(int(sid))
                                if seg_obj is None:
                                    continue
                                row = next((r for r in seg_rows if int(r.get("id") or 0) == int(sid)), None)
                                cur_shot_id = str((row or {}).get("shot_id") or "").strip()
                                cur_shot = shot_by_id_all.get(cur_shot_id) if cur_shot_id else None
                                cands = shortlist_shots_for_segment(
                                    segment=seg_obj,
                                    shots=list(shot_index_obj.shots or []),
                                    limit=int(max(60, _float_env("FOLDER_EDIT_JUDGE_RECAST_CANDS", "80"))),
                                    max_per_asset=2,
                                )
                                picked = None
                                for cand in cands:
                                    if not isinstance(cand, dict):
                                        continue
                                    csid = str(cand.get("id") or "").strip()
                                    if csid and csid == cur_shot_id:
                                        continue
                                    aid = str(cand.get("asset_id") or "").strip()
                                    sh = _f(cand.get("shake_score"))
                                    sp0 = _f(cand.get("sharpness"))
                                    if sh is not None and float(sh) > float(shake_max):
                                        continue
                                    if sp0 is not None and float(sp0) < float(sharp_min):
                                        continue
                                    # Prefer new assets (but allow reuse if the library is small).
                                    if aid and used_assets.get(aid, 0) >= 2:
                                        continue
                                    picked = cand
                                    break
                                if picked is None:
                                    picked = next((c for c in cands if isinstance(c, dict) and str(c.get("id") or "").strip() != cur_shot_id), None)
                                if picked is not None:
                                    pro_forced_shots_by_seg_id[int(sid)] = t.cast(dict[str, t.Any], picked)

                folder_plan, silent_video, timeline_segments = plan_and_render(extra_guidance=None, iteration=2)
                last_eval = {"iter1_fullvideo": judge1}

                # Optionally judge iter2 (for logging).
                if callable(evaluate_edit_full_video_compare) and _truthy_env("FOLDER_EDIT_JUDGE_ITER2", "1"):
                    out2_with_audio = judge_dir / "out_iter2_with_audio.mp4"
                    try:
                        shutil.copyfile(silent_video, out2_with_audio)
                        if final_audio_for_judge:
                            merge_audio(video_path=out2_with_audio, audio_path=final_audio_for_judge, output_path=out2_with_audio)
                    except Exception:
                        out2_with_audio = silent_video
                    out2_proxy = judge_dir / "out_iter2_judge.mp4"
                    if not _truthy_env("FOLDER_EDIT_JUDGE_NO_PROXY", "0"):
                        try:
                            _compress_for_analysis(out2_with_audio, dst=out2_proxy, timeout_s=min(timeout_s, 240.0))
                        except Exception:
                            out2_proxy = out2_with_audio
                    else:
                        out2_proxy = out2_with_audio

                    emit("Evaluate", 0, 1, "Full-video judging iter 2 vs reference (Gemini Pro compare)")
                    try:
                        ev2 = evaluate_edit_full_video_compare(
                            api_key=api_key,
                            model=critic_model,
                            output_video_path=out2_proxy,
                            reference_video_path=ref_proxy,
                            criteria=criteria,
                            prompt_variant=prompt_variant,
                            timeout_s=min(timeout_s, 240.0),
                            site_url=site_url,
                            app_name=app_name,
                        )
                        judge2 = {
                            "ok": True,
                            "result": ev2.result,
                            "usage": ev2.usage,
                            "model_requested": ev2.model_requested,
                            "model_used": ev2.model_used,
                        }
                        (output_paths.root / "evaluation_fullvideo_iter2.json").write_text(json.dumps(judge2, indent=2), encoding="utf-8")
                        last_eval = {"iter1_fullvideo": judge1, "iter2_fullvideo": judge2}
                    except Exception as e:
                        judge2 = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        (output_paths.root / "evaluation_fullvideo_iter2.json").write_text(json.dumps(judge2, indent=2), encoding="utf-8")
                    emit("Evaluate", 1, 1, "Full-video judge complete (iter 2)")
        else:
            # Evaluate similarity and derive reusable planner guidance.
            emit("Evaluate", 0, 1, "Evaluating edit vs reference")

            def eval_for_video(video_path: Path, *, tag: str) -> dict[str, t.Any]:
                eval_dir = output_paths.root / f"evaluation_frames_{tag}"
                out_frames: list[tuple[int, Path]] = []
                ref_frames: list[tuple[int, Path]] = []
                summaries: list[str] = []
                for seg_id, start_s, end_s, ref_frame_path in segment_frames:
                    mid = (start_s + end_s) / 2.0
                    out_path = eval_dir / f"out_{seg_id:02d}.jpg"
                    _extract_frame(video_path=video_path, at_s=mid, out_path=out_path, timeout_s=timeout_s)
                    out_frames.append((seg_id, out_path))
                    ref_frames.append((seg_id, ref_frame_path))
                for item in timeline_segments:
                    summaries.append(
                        f"beat={item.get('beat_goal')} overlay={item.get('overlay_text')!r} desired_tags={item.get('desired_tags')} asset_kind={item.get('asset_kind')} crop={item.get('crop_mode')}"
                    )

                evaluation = evaluate_edit_similarity(
                    api_key=api_key,
                    model=analysis_model,
                    reference_frames=ref_frames,
                    output_frames=out_frames,
                    segment_summaries=summaries,
                    timeout_s=timeout_s,
                    site_url=site_url,
                    app_name=app_name,
                )
                (output_paths.root / f"evaluation_{tag}.json").write_text(json.dumps(evaluation.raw, indent=2), encoding="utf-8")
                return evaluation.raw

            eval1 = eval_for_video(silent_video, tag="iter1")
            extra_guidance = str(eval1.get("planner_guidance") or "").strip() or None
            last_eval = {"iter1": eval1}

            try:
                score1 = float(eval1.get("overall_score") or 0.0)
            except Exception:
                score1 = 0.0
            emit("Evaluate", 1, 1, f"Evaluation score (iter 1): {score1:.1f}/10")

            if extra_guidance:
                folder_plan, silent_video, timeline_segments = plan_and_render(extra_guidance=extra_guidance, iteration=2)
                emit("Library", 3, 3, "EDL ready (iter 2)")

                # Verify the improvement with a second evaluation pass.
                emit("Evaluate", 0, 1, "Evaluating iter 2 vs reference")
                eval2 = eval_for_video(silent_video, tag="iter2")
                last_eval = {"iter1": eval1, "iter2": eval2}
                try:
                    score2 = float(eval2.get("overall_score") or 0.0)
                except Exception:
                    score2 = 0.0
                emit("Evaluate", 1, 1, f"Evaluation score (iter 2): {score2:.1f}/10")

    # Choose audio.
    final_audio: Path | None = None
    if audio_path:
        final_audio = audio_path
    elif use_reference_audio and extracted_audio:
        final_audio = Path(extracted_audio)  # type: ignore[arg-type]

    # Final output.
    final_video = output_paths.video_path
    def _write_final_video_variant(*, silent_src: Path, tag: str) -> Path:
        out = output_paths.root / f"final_video_{tag}.mov"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(silent_src, out)
        if final_audio:
            merge_audio(video_path=out, audio_path=final_audio, output_path=out)
        # Keep the canonical output path pointing at the latest final.
        shutil.copyfile(out, final_video)
        return out

    current_variant_video = _write_final_video_variant(silent_src=silent_video, tag="iter00_baseline")

    # Review artifacts (always generated; used by the Two-Lane improvement loop).
    review_dir = output_paths.root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_ref = review_dir / "reference_review.mp4"
    review_out = review_dir / "output_review.mp4"
    review_compare = review_dir / "compare_review.mp4"
    try:
        from review_render import render_compare_video, render_labeled_review_video

        segs_for_review = [{"id": sid, "start_s": s, "end_s": e} for sid, s, e in segments]
        emit("Review", 0, 1, "Rendering review proxies (REF/OUT/COMPARE)")
        render_labeled_review_video(
            src_video=analysis_clip,
            segments=segs_for_review,
            out_path=review_ref,
            label_prefix="S",
            keep_audio=False,
            timeout_s=min(timeout_s, 240.0),
        )
        render_labeled_review_video(
            src_video=current_variant_video,
            segments=segs_for_review,
            out_path=review_out,
            label_prefix="S",
            keep_audio=True,
            timeout_s=min(timeout_s, 240.0),
        )
        render_compare_video(
            ref_review=review_ref,
            out_review=review_out,
            out_path=review_compare,
            timeout_s=min(timeout_s, 240.0),
        )
        emit("Review", 1, 1, "Review proxies ready")
    except Exception as e:
        # Review proxies are required for the compare-video critic; fail loudly.
        raise RuntimeError(f"Failed to render review proxies: {type(e).__name__}: {e}") from e

    # Two-Lane improvement loop (Single Edit).
    improve_dir = output_paths.root / "improve"
    timeline_doc: dict[str, t.Any] = {
        "schema_version": 2,
        "mode": "folder_edit",
        "reel_url_or_path": reel_url_or_path,
        "source_video": str(src_path),
        "analysis_clip": str(analysis_clip),
        "media_folder": str(media_folder),
        "reference_image": str(reference_image_path) if reference_image_path else None,
        "segments_detected": [{"id": sid, "start_s": s, "end_s": e} for sid, s, e in segments],
        "beat_sync": bool(beat_sync),
        "music_analysis": str((output_paths.root / "reference" / "music_analysis.json")) if (beat_sync and music_doc) else None,
        "analysis": ref_analysis.analysis,
        "edl_analysis": folder_plan.analysis,
        "evaluation": last_eval,
        "planner_guidance_used": extra_guidance,
        "timeline_segments": timeline_segments,
        "burn_overlays": bool(burn_overlays),
        "audio": str(final_audio) if final_audio else None,
        "models": {"analysis": analysis_model, "critic": critic_model},
    }

    def _timeline_summary_for_critic(doc: dict[str, t.Any]) -> dict[str, t.Any]:
        segs = doc.get("timeline_segments") if isinstance(doc.get("timeline_segments"), list) else []
        out: list[dict[str, t.Any]] = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            try:
                sid = int(s.get("id") or 0)
            except Exception:
                sid = 0
            if sid <= 0:
                continue
            out.append(
                {
                    "segment_id": sid,
                    "beat_goal": s.get("beat_goal"),
                    "overlay_text": s.get("overlay_text"),
                    "desired_tags": s.get("desired_tags"),
                    "story_beat": s.get("story_beat"),
                    "transition_hint": s.get("transition_hint"),
                }
            )
        return {"segments": out}

    def _symlink_latest_video(iter_root: Path) -> None:
        try:
            link = iter_root / "final_video.mov"
            if link.exists() or link.is_symlink():
                try:
                    link.unlink()
                except Exception:
                    pass
            link.symlink_to(current_variant_video)
        except Exception:
            # Best-effort only.
            pass

    scores: list[float] = []
    plateau_hits = 0
    improve_iters_total = int(improve_iters or 0)
    if improve_iters_total > 0:
        improve_dir.mkdir(parents=True, exist_ok=True)

    # Helper: deterministic rerender from timeline (Lane A only).
    def _rerender_from_timeline(*, doc: dict[str, t.Any], tag: str) -> Path:
        segs = doc.get("timeline_segments")
        if not isinstance(segs, list) or not segs:
            raise RuntimeError("timeline_segments missing for rerender")
        seg_dir = output_paths.root / f"segments_{tag}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_paths: list[Path] = []
        fade_tail = _float_env("FOLDER_EDIT_FADE_OUT_S", "0.18")
        for idx, seg in enumerate(segs, start=1):
            if not isinstance(seg, dict):
                continue
            sid = int(seg.get("id") or 0)
            asset_path = Path(str(seg.get("asset_path") or "")).expanduser()
            asset_kind = str(seg.get("asset_kind") or "video")
            in_s = float(seg.get("asset_in_s") or 0.0)
            duration_s = float(seg.get("duration_s") or 0.0)
            speed = float(seg.get("speed") or 1.0)
            crop_mode = str(seg.get("crop_mode") or "center")
            reframe = seg.get("reframe") if isinstance(seg.get("reframe"), dict) else None
            overlay_text = str(seg.get("overlay_text") or "")
            grade = seg.get("grade") if isinstance(seg.get("grade"), dict) else None
            stabilize = bool(seg.get("stabilize")) if seg.get("stabilize") is not None else False
            try:
                zoom = float(seg.get("zoom") or 1.0)
            except Exception:
                zoom = 1.0
            fade_in_s = None
            fade_in_color = None
            try:
                fade_in_s = float(seg.get("fade_in_s")) if seg.get("fade_in_s") is not None else None
            except Exception:
                fade_in_s = None
            if isinstance(seg.get("fade_in_color"), str):
                fade_in_color = str(seg.get("fade_in_color") or "").strip().lower() or None

            fade_out_s = None
            fade_out_color = None
            try:
                fade_out_s = float(seg.get("fade_out_s")) if seg.get("fade_out_s") is not None else None
            except Exception:
                fade_out_s = None
            if isinstance(seg.get("fade_out_color"), str):
                fade_out_color = str(seg.get("fade_out_color") or "").strip().lower() or None

            out_path = seg_dir / f"seg_{sid:02d}.mp4"
            _render_segment(
                asset_path=asset_path,
                asset_kind=asset_kind,
                in_s=in_s,
                duration_s=duration_s,
                speed=speed,
                crop_mode=crop_mode,
                reframe=reframe,
                overlay_text=overlay_text,
                grade=grade,
                stabilize=stabilize,
                stabilize_cache_dir=(output_paths.root / "stabilized") if stabilize else None,
                zoom=zoom,
                fade_in_s=fade_in_s,
                fade_in_color=fade_in_color,
                fade_out_s=(fade_out_s if fade_out_s is not None else (fade_tail if idx == len(segs) else None)),
                fade_out_color=fade_out_color,
                output_path=out_path,
                burn_overlay=burn_overlays,
                timeout_s=timeout_s,
            )
            segment_paths.append(out_path)

        silent_out = output_paths.root / f"final_video_silent_{tag}.mp4"
        _concat_segments(segment_paths=segment_paths, output_path=silent_out, timeout_s=timeout_s)
        return silent_out

    if improve_iters_total > 0:
        try:
            from compare_video_critic import critique_compare_video
            from critic_schema import severity_rank
            from fix_actions import apply_fix_actions, apply_segment_deltas_to_timeline, apply_transition_deltas
            import dataclasses as _dc
        except Exception as e:
            raise RuntimeError(f"Missing improve-loop dependencies: {type(e).__name__}: {e}") from e

        # Iteration counter for internal render artifacts (keeps segment folders distinct).
        render_iter = int(iterations or 1)
        if render_iter < 1:
            render_iter = 1

        # Keep ref_analysis in sync with overlay/tag deltas for future recasts.
        def _apply_segment_deltas_to_ref_analysis(deltas: list[dict[str, t.Any]]) -> None:
            nonlocal ref_analysis
            if not hasattr(ref_analysis, "segments"):
                return
            seg_map: dict[int, dict[str, t.Any]] = {}
            for d in deltas:
                try:
                    sid = int(d.get("segment_id") or 0)
                except Exception:
                    continue
                if sid > 0:
                    seg_map[sid] = d

            def _norm_tag(tag: str) -> str:
                return " ".join((tag or "").strip().lower().split())

            def _map_transition_hint(h: str) -> str | None:
                hh = (h or "").strip().lower()
                if not hh or hh == "neutral":
                    return None
                if hh == "continuity":
                    return "match cut"
                if hh == "contrast":
                    return "hard contrast"
                return None

            new_segments: list[t.Any] = []
            for seg in list(ref_analysis.segments):
                d = seg_map.get(int(getattr(seg, "id", 0)))
                if not d:
                    new_segments.append(seg)
                    continue

                desired = list(getattr(seg, "desired_tags", []) or [])
                desired_norm: list[str] = []
                for t0 in desired:
                    if not isinstance(t0, str):
                        continue
                    s0 = _norm_tag(t0)
                    if s0 and s0 not in desired_norm:
                        desired_norm.append(s0)
                add_raw = d.get("desired_tags_add") if isinstance(d.get("desired_tags_add"), list) else []
                rm_raw = d.get("desired_tags_remove") if isinstance(d.get("desired_tags_remove"), list) else []
                add = [_norm_tag(str(x)) for x in add_raw if _norm_tag(str(x))]
                rm = {_norm_tag(str(x)) for x in rm_raw if _norm_tag(str(x))}
                next_tags: list[str] = []
                for tg in desired_norm:
                    if tg in rm:
                        continue
                    next_tags.append(tg)
                for tg in add:
                    if tg in rm:
                        continue
                    if tg and tg not in next_tags:
                        next_tags.append(tg)
                next_tags = next_tags[:16]

                overlay = getattr(seg, "overlay_text", "")
                if d.get("overlay_text_rewrite") is not None:
                    overlay = str(d.get("overlay_text_rewrite") or "").strip()

                patch: dict[str, t.Any] = {"desired_tags": next_tags, "overlay_text": overlay}
                if critic_pro_mode:
                    if d.get("story_beat") is not None:
                        patch["story_beat"] = str(d.get("story_beat") or "").strip()[:80]
                    if d.get("transition_hint") is not None:
                        patch["transition_hint"] = _map_transition_hint(str(d.get("transition_hint") or ""))
                try:
                    new_segments.append(_dc.replace(seg, **patch))
                except Exception:
                    new_segments.append(seg)
            try:
                ref_analysis = _dc.replace(ref_analysis, segments=new_segments)
            except Exception:
                pass

        for iter_idx in range(0, improve_iters_total + 1):
            iter_name = "iter_00_baseline" if iter_idx == 0 else f"iter_{iter_idx:02d}"
            iter_root = improve_dir / iter_name
            iter_root.mkdir(parents=True, exist_ok=True)
            _symlink_latest_video(iter_root)

            emit("Improve", iter_idx, improve_iters_total + 1, f"Critiquing compare video ({iter_name})")

            # Refresh review proxies for the current output.
            try:
                segs_for_review = [{"id": sid, "start_s": s, "end_s": e} for sid, s, e in segments]
                render_labeled_review_video(
                    src_video=analysis_clip,
                    segments=segs_for_review,
                    out_path=review_ref,
                    label_prefix="S",
                    keep_audio=False,
                    timeout_s=min(timeout_s, 240.0),
                )
                render_labeled_review_video(
                    src_video=current_variant_video,
                    segments=segs_for_review,
                    out_path=review_out,
                    label_prefix="S",
                    keep_audio=True,
                    timeout_s=min(timeout_s, 240.0),
                )
                render_compare_video(
                    ref_review=review_ref,
                    out_review=review_out,
                    out_path=review_compare,
                    timeout_s=min(timeout_s, 240.0),
                )
            except Exception:
                # If proxies already exist, still attempt critique.
                pass

            critic_res = critique_compare_video(
                api_key=api_key,
                model=critic_model,
                compare_video_path=review_compare,
                timeline_summary=_timeline_summary_for_critic(timeline_doc),
                critic_pro_mode=bool(critic_pro_mode),
                max_mb=float(critic_max_mb),
                tmp_dir=(review_dir / "tmp"),
                timeout_s=min(timeout_s, 240.0),
                site_url=site_url,
                app_name=app_name,
            )
            report = critic_res.report
            (iter_root / "critique.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

            scores.append(float(report.overall_score))
            if len(scores) >= 2:
                delta = float(scores[-1] - scores[-2])
                if float(delta) < 0.2:
                    plateau_hits += 1
                else:
                    plateau_hits = 0

            actionable = bool(report.lane_a_actions or report.lane_b_deltas or report.transition_deltas)
            if iter_idx >= improve_iters_total:
                (iter_root / "applied_actions.json").write_text(
                    json.dumps(
                        {
                            "note": "final_iteration",
                            "actionable": actionable,
                            "plateau_hits": int(plateau_hits),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                break

            if not actionable:
                (iter_root / "applied_actions.json").write_text(json.dumps({"note": "no_actionable_deltas"}, indent=2) + "\n", encoding="utf-8")
                break

            if plateau_hits >= 2:
                (iter_root / "applied_actions.json").write_text(
                    json.dumps({"note": "plateau_stop", "plateau_hits": int(plateau_hits)}, indent=2) + "\n",
                    encoding="utf-8",
                )
                break

            # Convert dataclasses to dicts for executors.
            lane_a_actions = []
            for a in report.lane_a_actions:
                item: dict[str, t.Any] = {"type": a.type, "segment_id": int(a.segment_id)}
                if a.seconds is not None:
                    item["seconds"] = float(a.seconds)
                if a.value is not None:
                    item["value"] = a.value
                lane_a_actions.append(item)

            transition_deltas = [{"boundary_after_segment_id": int(td.boundary_after_segment_id), "type": td.type, "seconds": float(td.seconds)} for td in report.transition_deltas]

            # Enforce <=2 Lane B deltas (highest severity).
            severity_by_id = {int(s.segment_id): str(s.severity) for s in report.segments}
            lane_b_sorted = sorted(
                list(report.lane_b_deltas),
                key=lambda d: severity_rank(severity_by_id.get(int(d.segment_id), "low")),
                reverse=True,
            )
            lane_b_selected = lane_b_sorted[:2]
            lane_b_deltas: list[dict[str, t.Any]] = []
            for d in lane_b_selected:
                item2: dict[str, t.Any] = {
                    "segment_id": int(d.segment_id),
                    "desired_tags_add": list(d.desired_tags_add),
                    "desired_tags_remove": list(d.desired_tags_remove),
                }
                if d.story_beat is not None:
                    item2["story_beat"] = str(d.story_beat)
                if d.transition_hint is not None:
                    item2["transition_hint"] = str(d.transition_hint)
                if d.overlay_text_rewrite is not None:
                    item2["overlay_text_rewrite"] = str(d.overlay_text_rewrite)
                lane_b_deltas.append(item2)

            # Apply in a controlled way (reject/record invalid).
            patched_doc, lane_a_report = apply_fix_actions(timeline_doc, lane_a_actions)
            patched_doc, trans_report = apply_transition_deltas(patched_doc, transition_deltas)
            patched_doc, lane_b_report = apply_segment_deltas_to_timeline(patched_doc, lane_b_deltas, allow_pro_fields=bool(critic_pro_mode))

            applied_lane_b: list[dict[str, t.Any]] = []
            if isinstance(lane_b_report.get("applied"), list):
                applied_lane_b = t.cast(list[dict[str, t.Any]], lane_b_report.get("applied") or [])

            # Decide whether a reselection run is required (Lane B selection steering).
            reselect_ids: set[int] = set()
            for rec in applied_lane_b:
                try:
                    sid = int(rec.get("segment_id") or 0)
                except Exception:
                    sid = 0
                if sid <= 0:
                    continue
                add = rec.get("desired_tags_add") if isinstance(rec.get("desired_tags_add"), list) else []
                rm = rec.get("desired_tags_remove") if isinstance(rec.get("desired_tags_remove"), list) else []
                wants_reselect = bool(add or rm)
                if critic_pro_mode and (rec.get("story_beat") or rec.get("transition_hint")):
                    wants_reselect = True
                if wants_reselect:
                    reselect_ids.add(int(sid))
            need_reselect = bool(pro_mode and reselect_ids)

            # Build render overrides precisely (avoid overriding auto-grade on new clips unless requested).
            overrides: dict[int, dict[str, t.Any]] = {}

            def _ov(sid: int) -> dict[str, t.Any]:
                if sid not in overrides:
                    overrides[sid] = {}
                return overrides[sid]

            # Lane A render overrides.
            for a in (lane_a_report.get("applied") or []):
                if not isinstance(a, dict):
                    continue
                try:
                    sid = int(a.get("segment_id") or 0)
                except Exception:
                    continue
                if sid <= 0:
                    continue
                typ = str(a.get("type") or "").strip()
                if typ == "set_stabilize":
                    _ov(sid)["stabilize"] = bool(a.get("value"))
                elif typ == "set_crop_mode":
                    _ov(sid)["crop_mode"] = a.get("value")
                elif typ == "set_zoom":
                    _ov(sid)["zoom"] = a.get("value")
                elif typ == "set_grade":
                    _ov(sid)["grade"] = a.get("value")
                elif typ == "set_fade_out":
                    _ov(sid)["fade_out_s"] = a.get("seconds")
                    _ov(sid)["fade_out_color"] = "black"
                elif typ == "set_overlay_text":
                    _ov(sid)["overlay_text"] = a.get("value")

            # Lane B overlay rewrite (render-time).
            for b in applied_lane_b:
                try:
                    sid = int(b.get("segment_id") or 0)
                except Exception:
                    sid = 0
                if sid <= 0:
                    continue
                if b.get("overlay_text_rewrite") is not None:
                    _ov(sid)["overlay_text"] = b.get("overlay_text_rewrite")

            # Transition overrides: pull fade params from the patched timeline for the boundary segments only.
            patched_segs = patched_doc.get("timeline_segments") if isinstance(patched_doc.get("timeline_segments"), list) else []
            seg_by_id: dict[int, dict[str, t.Any]] = {}
            for s in patched_segs:
                if not isinstance(s, dict):
                    continue
                try:
                    sid = int(s.get("id") or 0)
                except Exception:
                    continue
                if sid > 0:
                    seg_by_id[int(sid)] = s

            for td in transition_deltas:
                try:
                    after = int(td.get("boundary_after_segment_id") or 0)
                except Exception:
                    after = 0
                if after <= 0:
                    continue
                prev = seg_by_id.get(int(after))
                nxt = seg_by_id.get(int(after + 1))
                if isinstance(prev, dict):
                    _ov(int(after))["fade_out_s"] = prev.get("fade_out_s")
                    _ov(int(after))["fade_out_color"] = prev.get("fade_out_color")
                if isinstance(nxt, dict):
                    _ov(int(after + 1))["fade_in_s"] = nxt.get("fade_in_s")
                    _ov(int(after + 1))["fade_in_color"] = nxt.get("fade_in_color")

            applied_actions_doc = {
                "critic_video_meta": critic_res.video_meta,
                "lane_a": lane_a_report,
                "transition": trans_report,
                "lane_b": lane_b_report,
                "need_reselect": bool(need_reselect),
                "reselect_segment_ids": sorted(list(reselect_ids)),
            }
            (iter_root / "applied_actions.json").write_text(json.dumps(applied_actions_doc, indent=2) + "\n", encoding="utf-8")
            (iter_root / "timeline_patched.json").write_text(json.dumps(patched_doc, indent=2) + "\n", encoding="utf-8")

            # Keep ref_analysis in sync for future iterations.
            ref_sync: list[dict[str, t.Any]] = list(applied_lane_b)
            for a in (lane_a_report.get("applied") or []):
                if not isinstance(a, dict):
                    continue
                if str(a.get("type") or "") != "set_overlay_text":
                    continue
                try:
                    sid = int(a.get("segment_id") or 0)
                except Exception:
                    continue
                if sid <= 0:
                    continue
                ref_sync.append({"segment_id": sid, "desired_tags_add": [], "desired_tags_remove": [], "overlay_text_rewrite": a.get("value")})
            _apply_segment_deltas_to_ref_analysis(ref_sync)

            if need_reselect:
                # Lock unchanged segments to avoid thrash across iterations.
                try:
                    shot_by_id_all = {str(x.get("id") or ""): x for x in (shot_index_obj.shots or []) if isinstance(x, dict) and str(x.get("id") or "")}
                except Exception:
                    shot_by_id_all = {}
                pro_forced_shots_by_seg_id = {}
                for row in (timeline_doc.get("timeline_segments") or []):
                    if not isinstance(row, dict):
                        continue
                    sid = int(row.get("id") or 0)
                    if sid <= 0 or sid in reselect_ids:
                        continue
                    shot_id = str(row.get("shot_id") or "").strip()
                    sh = shot_by_id_all.get(shot_id)
                    if isinstance(sh, dict):
                        pro_forced_shots_by_seg_id[int(sid)] = sh

                # Apply render overrides during the replan render.
                render_overrides_by_seg_id = overrides
                pro_force_stabilize_segs = set()
                pro_env_overrides = dict(pro_env_base_overrides)

                render_iter += 1
                folder_plan, silent_video, timeline_segments = plan_and_render(extra_guidance=None, iteration=render_iter)
                current_variant_video = _write_final_video_variant(silent_src=silent_video, tag=f"iter{iter_idx+1:02d}")

                # Clear one-shot overrides after this render.
                render_overrides_by_seg_id = {}
                pro_forced_shots_by_seg_id = {}
            else:
                silent_video = _rerender_from_timeline(doc=patched_doc, tag=f"improve_iter{iter_idx+1:02d}")
                timeline_segments = t.cast(list[dict[str, t.Any]], patched_doc.get("timeline_segments") or [])
                current_variant_video = _write_final_video_variant(silent_src=silent_video, tag=f"iter{iter_idx+1:02d}")

            # Update the rolling timeline_doc for the next iteration.
            timeline_doc = dict(timeline_doc)
            timeline_doc["timeline_segments"] = timeline_segments
            timeline_doc["edl_analysis"] = folder_plan.analysis

    # Always generate output preview frames for the GUI (so users don't confuse reference frames for output).
    preview_frames: list[Path] = []
    preview_dir = output_paths.root / "preview_frames"
    for seg_id, start_s, end_s, _ref_frame_path in segment_frames:
        mid = (start_s + end_s) / 2.0
        out_path = preview_dir / f"out_{seg_id:02d}.jpg"
        try:
            _extract_frame(video_path=silent_video, at_s=mid, out_path=out_path, timeout_s=min(timeout_s, 120.0))
        except Exception:
            continue
        if out_path.exists():
            preview_frames.append(out_path)

    emit("Done", 1, 1, "Done")

    # Persist the final (possibly improved) timeline.
    write_json(output_paths.timeline_path, timeline_doc)

    return FolderEditPipelineResult(
        output_paths=output_paths,
        video_path=final_video,
        reference_analysis=ref_analysis,
        folder_plan=folder_plan,
        source_video_path=src_path,
        scene_images=preview_frames,
    )
