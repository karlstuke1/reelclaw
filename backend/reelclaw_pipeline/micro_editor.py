from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import typing as t

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


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}... {stderr or 'unknown error'}")


def _extract_frame(*, video_path: Path, at_s: float, out_path: Path, timeout_s: float) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract frames. Please install ffmpeg and try again.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
            "scale=768:-2:flags=lanczos,format=yuvj420p",
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
        img = Image.open(path).convert("L").resize((64, 64))
        px = list(img.getdata())
        if not px:
            return None
        return float(sum(px) / len(px)) / 255.0
    except Exception:
        return None


def _dark_frac(path: Path, *, threshold: int = 32) -> float | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L").resize((64, 64))
        px = list(img.getdata())
        if not px:
            return None
        dark = sum(1 for p in px if int(p) < int(threshold))
        return float(dark) / float(len(px))
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


def _frame_motion_diff(a: Path, b: Path) -> float | None:
    if Image is None:
        return None
    try:
        ia = Image.open(a).convert("L").resize((64, 64))
        ib = Image.open(b).convert("L").resize((64, 64))
        pa = list(ia.getdata())
        pb = list(ib.getdata())
        if not pa or not pb:
            return None
        n = min(len(pa), len(pb))
        if n <= 0:
            return None
        return float(sum(abs(int(pa[i]) - int(pb[i])) for i in range(n)) / (n * 255.0))
    except Exception:
        return None


def _sharpness(path: Path) -> float | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L").resize((64, 64))
        px = list(img.getdata())
        if len(px) != 64 * 64:
            return None
        def p(x: int, y: int) -> int:
            return int(px[y * 64 + x])

        vals: list[float] = []
        for y in range(1, 63):
            for x in range(1, 63):
                c = p(x, y)
                lap = (p(x - 1, y) + p(x + 1, y) + p(x, y - 1) + p(x, y + 1) - 4 * c)
                vals.append(float(lap))
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        return float(var)
    except Exception:
        return None


def _segment_energy_hint(duration_s: float) -> float:
    d = float(duration_s or 0.0)
    if d <= 0.8:
        return 1.0
    if d <= 1.2:
        return 0.75
    if d <= 1.7:
        return 0.45
    return 0.25


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _motion_vec(a: Path, b: Path) -> tuple[float | None, float | None]:
    """
    Estimate dominant translation between two frames.
    Returns (mag_norm, angle_deg) where mag_norm is normalized by frame width.
    """
    if cv2 is None or np is None:  # pragma: no cover
        return None, None
    try:
        ia = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(b), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None:
            return None, None
        if ia.shape != ib.shape:
            # Resize to the smaller common size.
            h = min(int(ia.shape[0]), int(ib.shape[0]))
            w = min(int(ia.shape[1]), int(ib.shape[1]))
            if h <= 0 or w <= 0:
                return None, None
            ia = cv2.resize(ia, (w, h), interpolation=cv2.INTER_AREA)
            ib = cv2.resize(ib, (w, h), interpolation=cv2.INTER_AREA)

        gray0 = ia.astype(np.float32)
        gray1 = ib.astype(np.float32)
        gray0 = cv2.GaussianBlur(gray0, (0, 0), 1.2)
        gray1 = cv2.GaussianBlur(gray1, (0, 0), 1.2)

        hh, ww = gray0.shape[:2]
        if ww <= 1 or hh <= 1:
            return None, None
        hann = cv2.createHanningWindow((ww, hh), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(gray0, gray1, hann)  # type: ignore[arg-type]
        min_resp = float(os.getenv("MICRO_PHASECORR_MIN_RESP", "0.08") or 0.08)
        if float(resp) < float(min_resp):
            return None, None

        mag = float((float(dx) ** 2 + float(dy) ** 2) ** 0.5)
        mag_norm = float(mag / max(1.0, float(ww)))
        angle = 0.0
        if mag >= 1e-6:
            import math

            angle = float(math.degrees(math.atan2(float(dy), float(dx))))
        return mag_norm, angle
    except Exception:
        return None, None


_FACE_CASCADE: t.Any | None = None


def _face_center_norm(bgr: "np.ndarray") -> tuple[float, float] | None:
    if cv2 is None or np is None:  # pragma: no cover
        return None
    global _FACE_CASCADE
    try:
        if _FACE_CASCADE is None:
            try:
                cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]
            except Exception:
                cascade_path = ""
            if not cascade_path:
                return None
            _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        if boxes is None or len(boxes) <= 0:
            return None
        # Pick largest face.
        x, y, w, h = max(boxes, key=lambda bb: int(bb[2]) * int(bb[3]))
        hh, ww = gray.shape[:2]
        if ww <= 0 or hh <= 0:
            return None
        cx = (float(x) + (float(w) / 2.0)) / float(ww)
        cy = (float(y) + (float(h) / 2.0)) / float(hh)
        # Bias upward slightly for headroom (common reel framing).
        cy = max(0.0, min(1.0, float(cy) - 0.06))
        return float(cx), float(cy)
    except Exception:
        return None


def _saliency_center_norm(bgr: "np.ndarray") -> tuple[float, float] | None:
    if cv2 is None or np is None:  # pragma: no cover
        return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Edge density centroid (cheap, works for many non-face subjects).
        edges = cv2.Canny(gray, 80, 200)
        ys, xs = np.nonzero(edges)
        hh, ww = gray.shape[:2]
        if ww <= 0 or hh <= 0:
            return None
        if xs.size < 80:
            return None
        cx = float(xs.mean()) / float(ww)
        cy = float(ys.mean()) / float(hh)
        return float(max(0.0, min(1.0, cx))), float(max(0.0, min(1.0, cy)))
    except Exception:
        return None


def _compute_reframe(
    *,
    start_frame: Path,
    end_frame: Path,
    duration_s: float,
) -> dict[str, t.Any] | None:
    """
    Compute a \"smart\" crop path for the segment window using face detection + saliency.
    Returns a dict consumed by the renderer; None means \"fallback to center\".
    """
    if cv2 is None or np is None:  # pragma: no cover
        return None
    try:
        b0 = cv2.imread(str(start_frame), cv2.IMREAD_COLOR)
        b1 = cv2.imread(str(end_frame), cv2.IMREAD_COLOR)
        if b0 is None or b1 is None:
            return None

        allow_saliency = os.getenv("SMART_CROP_ALLOW_SALIENCY", "0").strip().lower() not in {"0", "false", "no", "off"}
        c0 = _face_center_norm(b0)
        c1 = _face_center_norm(b1)
        if c0 is None or c1 is None:
            if not allow_saliency:
                return None
        c0 = c0 or _saliency_center_norm(b0) or (0.5, 0.5)
        c1 = c1 or _saliency_center_norm(b1) or c0

        cx0, cy0 = float(c0[0]), float(c0[1])
        cx1, cy1 = float(c1[0]), float(c1[1])

        # Clamp pan speed so reframes feel intentional, not jittery.
        max_delta = float(os.getenv("SMART_CROP_MAX_DELTA", "0.12") or 0.12)
        if float(duration_s) > 0.2:
            max_delta = max_delta * max(1.0, float(duration_s) / 1.0)
        dx = cx1 - cx0
        dy = cy1 - cy0
        if abs(dx) > max_delta:
            cx1 = cx0 + (max_delta if dx > 0 else -max_delta)
        if abs(dy) > max_delta:
            cy1 = cy0 + (max_delta if dy > 0 else -max_delta)

        return {"mode": "linear", "cx0": cx0, "cy0": cy0, "cx1": cx1, "cy1": cy1}
    except Exception:
        return None

def _beat_floor_index(t_s: float, beats: list[dict[str, t.Any]]) -> int | None:
    if not beats:
        return None
    t0 = float(t_s)
    best_i: int | None = None
    for b in beats:
        bt = _safe_float(b.get("t"))
        bi = b.get("i")
        if bt is None or not isinstance(bi, int):
            continue
        if bt <= t0 + 1e-6:
            best_i = bi
        else:
            break
    if best_i is not None:
        return best_i
    # If all beats are after t_s, return first.
    for b in beats:
        if isinstance(b.get("i"), int):
            return int(b["i"])
    return None


@dataclass(frozen=True)
class MicroEditDecision:
    segment_id: int
    shot_id: str
    asset_id: str
    asset_path: str
    in_s: float
    duration_s: float
    speed: float
    crop_mode: str
    reframe: dict[str, t.Any] | None
    debug: dict[str, t.Any]


def pick_inpoint_for_shot(
    *,
    segment: t.Any,
    shot: dict[str, t.Any],
    work_dir: Path,
    timeout_s: float,
) -> tuple[float, dict[str, t.Any]]:
    """
    Code-only in-point picker inside a shot window. Samples candidate frames,
    ranks by lighting match + (small) motion/energy match + sharpness.
    """
    seg_dur = float(getattr(segment, "duration_s", 0.0) or 0.0)
    ref_luma = _safe_float(getattr(segment, "ref_luma", None))
    ref_dark = _safe_float(getattr(segment, "ref_dark_frac", None))
    # Target energy: duration-based hint blended with music-derived energy (when available).
    energy = _segment_energy_hint(seg_dur)
    me = _safe_float(getattr(segment, "music_energy", None))
    if me is not None:
        energy = float((0.65 * float(energy)) + (0.35 * float(me)))

    start_s = _safe_float(shot.get("start_s")) or 0.0
    end_s = _safe_float(shot.get("end_s")) or max(start_s, start_s + seg_dur)
    if end_s <= start_s + 1e-3:
        end_s = start_s + max(0.25, seg_dur)

    max_in = max(start_s, end_s - max(0.0, seg_dur))
    if max_in <= start_s + 1e-4:
        return float(start_s), {"candidates": [{"time_s": float(start_s)}], "chosen": {"time_s": float(start_s)}}

    # Candidate generation: uniform sampling + anchors (+ optional motion peaks).
    count = int(max(8, min(16, float(os.getenv("MICRO_INPOINT_SAMPLES", "12")))))
    times0 = [start_s + (max_in - start_s) * (i / (count - 1)) for i in range(count)]
    times0.extend([start_s, start_s + (max_in - start_s) * 0.33, start_s + (max_in - start_s) * 0.66, max_in])

    def _dedupe(times: list[float]) -> list[float]:
        uniq: list[float] = []
        for t_s in times:
            t3 = round(float(t_s), 3)
            if t3 < start_s - 1e-3 or t3 > max_in + 1e-3:
                continue
            if t3 not in uniq:
                uniq.append(t3)
        uniq.sort()
        return uniq

    uniq = _dedupe(times0)

    asset_path = Path(str(shot.get("asset_path") or "")).expanduser()
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing asset_path for shot: {asset_path}")

    sample_dir = work_dir / f"seg_{int(getattr(segment, 'id', 0)):02d}" / "inpoint_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    samples_by_time: dict[float, tuple[float, Path, float | None, float | None, float | None, list[float] | None]] = {}

    def _ensure_sample(t_s: float) -> None:
        tt = float(t_s)
        if tt in samples_by_time:
            return
        safe = f"{tt:.3f}".replace(".", "p")
        frame_path = sample_dir / f"t_{safe}.jpg"
        if not frame_path.exists():
            _extract_frame(video_path=asset_path, at_s=tt, out_path=frame_path, timeout_s=min(timeout_s, 120.0))
        luma = _luma_mean(frame_path)
        dark = _dark_frac(frame_path)
        sharp = _sharpness(frame_path)
        rgb = _rgb_mean(frame_path)
        samples_by_time[tt] = (tt, frame_path, luma, dark, sharp, rgb)

    for t_s in uniq:
        _ensure_sample(float(t_s))

    def _motion_map(times_sorted: list[float]) -> dict[float, float | None]:
        # Motion proxy from successive sampled frames (only a rough hint).
        ordered = [samples_by_time[t] for t in times_sorted if t in samples_by_time]
        motion_by_time: dict[float, float | None] = {}
        for i in range(len(ordered)):
            t0, p0, *_rest = ordered[i]
            if i < len(ordered) - 1:
                _t1, p1, *_rest2 = ordered[i + 1]
                motion_by_time[t0] = _frame_motion_diff(p0, p1)
            elif i > 0:
                _tprev, pprev, *_rest3 = ordered[i - 1]
                motion_by_time[t0] = _frame_motion_diff(pprev, p0)
            else:
                motion_by_time[t0] = None
        return motion_by_time

    motion_by_time = _motion_map(uniq)

    # Optional: add candidates around motion peaks/troughs to find more meaningful moments.
    if os.getenv("MICRO_USE_MOTION_PEAKS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        mot_vals = [(t, m) for t, m in motion_by_time.items() if isinstance(m, (int, float))]
        if mot_vals:
            mot_vals.sort(key=lambda x: float(x[1]))
            # For high energy segments, bias toward higher motion; for low energy, bias calmer.
            pick_high = energy >= 0.65
            pick_low = energy <= 0.35
            picks: list[float] = []
            if pick_high:
                picks = [float(t) for t, _m in mot_vals[-2:]]
            elif pick_low:
                picks = [float(t) for t, _m in mot_vals[:2]]
            else:
                # Mid energy: sample one high and one low.
                picks = [float(mot_vals[0][0]), float(mot_vals[-1][0])]
            extra_times: list[float] = []
            for t0 in picks:
                for dt in (-0.20, 0.0, 0.20):
                    tt = float(t0) + float(dt)
                    if tt < start_s - 1e-3 or tt > max_in + 1e-3:
                        continue
                    extra_times.append(tt)
            for t_s in _dedupe(extra_times):
                _ensure_sample(float(t_s))
            uniq = sorted(samples_by_time.keys())
            motion_by_time = _motion_map(uniq)

    samples = [samples_by_time[t] for t in uniq if t in samples_by_time]

    ranked: list[tuple[float, dict[str, t.Any]]] = []
    for t_s, p, luma, dark, sharp, rgb in samples:
        dl = 0.18
        dd = 0.25
        if ref_luma is not None and isinstance(luma, (int, float)):
            dl = abs(float(luma) - float(ref_luma))
        if ref_dark is not None and isinstance(dark, (int, float)):
            dd = abs(float(dark) - float(ref_dark))
        m = motion_by_time.get(t_s)
        motion = float(m) if isinstance(m, (int, float)) else 0.35
        sharp_pen = 0.0
        if isinstance(sharp, (int, float)) and float(sharp) < 60.0:
            sharp_pen = 0.08
        score = (dl * 1.2) + (dd * 1.0) + (abs(motion - energy) * 0.25) + sharp_pen
        ranked.append(
            (
                float(score),
                {
                    "time_s": float(t_s),
                    "luma": luma,
                    "dark_frac": dark,
                    "motion": motion,
                    "sharpness": sharp,
                    "rgb_mean": rgb,
                    "frame_path": str(p),
                },
            )
        )
    ranked.sort(key=lambda x: x[0])

    chosen = ranked[0][1]
    debug = {"candidates": [it for _s, it in ranked[: min(8, len(ranked))]], "chosen": chosen}
    return float(chosen["time_s"]), debug


def _build_inpoint_candidates(
    *,
    segment: t.Any,
    shot: dict[str, t.Any],
    work_dir: Path,
    timeout_s: float,
) -> list[dict[str, t.Any]]:
    """
    Build richer in-point candidates for DP micro-editing:
    - start/mid/end frames (within the segment window)
    - motion vectors (phase correlation) for start and end halves
    """
    seg_id = int(getattr(segment, "id", 0) or 0)
    seg_dur = float(getattr(segment, "duration_s", 0.0) or 0.0)
    ref_luma = _safe_float(getattr(segment, "ref_luma", None))
    ref_dark = _safe_float(getattr(segment, "ref_dark_frac", None))

    start_s = _safe_float(shot.get("start_s")) or 0.0
    end_s = _safe_float(shot.get("end_s")) or max(start_s, start_s + seg_dur)
    if end_s <= start_s + 1e-3:
        end_s = start_s + max(0.25, seg_dur)

    max_in = max(start_s, end_s - max(0.0, seg_dur))
    if max_in <= start_s + 1e-4:
        return [{"time_s": float(start_s)}]

    # Candidate generation: uniform sampling + anchors.
    count = int(max(8, min(16, float(os.getenv("MICRO_INPOINT_SAMPLES", "12")))))
    times0 = [start_s + (max_in - start_s) * (i / (count - 1)) for i in range(count)]
    times0.extend([start_s, start_s + (max_in - start_s) * 0.33, start_s + (max_in - start_s) * 0.66, max_in])

    def _dedupe(times: list[float]) -> list[float]:
        uniq: list[float] = []
        for t_s in times:
            t3 = round(float(t_s), 3)
            if t3 < start_s - 1e-3 or t3 > max_in + 1e-3:
                continue
            if t3 not in uniq:
                uniq.append(t3)
        uniq.sort()
        return uniq

    uniq = _dedupe(times0)
    asset_path = Path(str(shot.get("asset_path") or "")).expanduser()
    if not asset_path.exists():
        return [{"time_s": float(start_s)}]

    sample_dir = work_dir / f"seg_{seg_id:02d}" / "dp_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Cache frames by time to avoid repeated ffmpeg calls across candidates.
    frame_cache: dict[float, Path] = {}

    def _frame_at(t_s: float, *, label: str) -> Path:
        tt = round(float(t_s), 3)
        if tt in frame_cache:
            return frame_cache[tt]
        safe = f"{tt:.3f}".replace(".", "p")
        fp = sample_dir / f"{label}_{safe}.jpg"
        if not fp.exists():
            _extract_frame(video_path=asset_path, at_s=float(tt), out_path=fp, timeout_s=min(timeout_s, 120.0))
        frame_cache[tt] = fp
        return fp

    energy = _segment_energy_hint(seg_dur)
    me = _safe_float(getattr(segment, "music_energy", None))
    if me is not None:
        energy = float((0.65 * float(energy)) + (0.35 * float(me)))

    out: list[dict[str, t.Any]] = []
    for t_s in uniq:
        t0 = float(t_s)
        t2 = float(t0 + seg_dur)
        t1 = float(t0 + (seg_dur * 0.5))
        f0 = _frame_at(t0, label="s")
        f1 = _frame_at(t1, label="m")
        f2 = _frame_at(t2, label="e")

        l0 = _luma_mean(f0)
        d0 = _dark_frac(f0)
        sh0 = _sharpness(f0)
        rgb0 = _rgb_mean(f0)

        l2 = _luma_mean(f2)
        d2 = _dark_frac(f2)

        m0, a0 = _motion_vec(f0, f1)
        m1, a1 = _motion_vec(f1, f2)
        # Fallback motion magnitude if cv2 isn't available / fails.
        if m0 is None:
            md = _frame_motion_diff(f0, f1)
            m0 = float(md) if isinstance(md, (int, float)) else None
        if m1 is None:
            md = _frame_motion_diff(f1, f2)
            m1 = float(md) if isinstance(md, (int, float)) else None

        dl = 0.18
        dd = 0.25
        if ref_luma is not None and isinstance(l0, (int, float)):
            dl = abs(float(l0) - float(ref_luma))
        if ref_dark is not None and isinstance(d0, (int, float)):
            dd = abs(float(d0) - float(ref_dark))

        motion = float(m0) if isinstance(m0, (int, float)) else 0.35
        sharp_pen = 0.0
        if isinstance(sh0, (int, float)) and float(sh0) < 60.0:
            sharp_pen = 0.08
        score = (dl * 1.2) + (dd * 1.0) + (abs(motion - energy) * 0.25) + sharp_pen
        out.append(
            {
                "score": float(score),
                "time_s": float(t0),
                "frame_start": str(f0),
                "frame_mid": str(f1),
                "frame_end": str(f2),
                "luma": l0,
                "dark_frac": d0,
                "sharpness": sh0,
                "rgb_mean": rgb0,
                "luma_end": l2,
                "dark_frac_end": d2,
                "motion_start": m0,
                "motion_start_angle": a0,
                "motion_end": m1,
                "motion_end_angle": a1,
            }
        )

    out.sort(key=lambda x: float(x.get("score") or 0.0))
    keep = int(max(4, min(len(out), float(os.getenv("MICRO_DP_KEEP", "10")))))
    return out[:keep]


def micro_edit_sequence(
    *,
    segments: list[t.Any],
    chosen_shots: list[dict[str, t.Any]],
    music_doc: dict[str, t.Any] | None,
    work_dir: Path,
    timeout_s: float,
    default_crop_mode: str = "center",
    default_speed: float = 1.0,
) -> tuple[list[MicroEditDecision], dict[str, t.Any]]:
    """
    Convert a macro shot sequence into renderable decisions:
    - pick in-points within shots
    - attach beat indices (if music_doc available)
    """
    if len(segments) != len(chosen_shots):
        raise ValueError("segments and chosen_shots length mismatch")

    beats = music_doc.get("beat_times") if isinstance(music_doc, dict) else None
    if not isinstance(beats, list):
        beats = []

    use_dp = os.getenv("MICRO_DP", "1").strip().lower() not in {"0", "false", "no", "off"}
    smart_crop = os.getenv("FOLDER_EDIT_SMART_CROP", "1").strip().lower() not in {"0", "false", "no", "off"}

    # Fast path: old independent picker.
    if not use_dp:
        decisions: list[MicroEditDecision] = []
        for seg, shot in zip(segments, chosen_shots, strict=False):
            seg_id = int(getattr(seg, "id", 0) or 0)
            shot_id = str(shot.get("id") or "")
            asset_id = str(shot.get("asset_id") or "")
            asset_path = str(shot.get("asset_path") or "")
            if not shot_id or not asset_id or not asset_path:
                raise ValueError("Invalid chosen shot (missing id/asset_id/asset_path)")

            in_s, debug = pick_inpoint_for_shot(segment=seg, shot=shot, work_dir=work_dir, timeout_s=timeout_s)
            start_beat = _beat_floor_index(float(getattr(seg, "start_s", 0.0) or 0.0), t.cast(list[dict[str, t.Any]], beats))
            end_beat = _beat_floor_index(float(getattr(seg, "end_s", 0.0) or 0.0), t.cast(list[dict[str, t.Any]], beats))
            debug = dict(debug)
            debug["segment_start_beat"] = start_beat
            debug["segment_end_beat"] = end_beat

            reframe = None
            crop_mode = str(default_crop_mode)
            if smart_crop and str(default_crop_mode) in {"smart", "auto"}:
                chosen_fp = Path(str((debug.get("chosen") or {}).get("frame_path") or ""))
                if chosen_fp.exists():
                    # Best-effort: if we only have a start frame, compute a static center crop.
                    rf = _compute_reframe(start_frame=chosen_fp, end_frame=chosen_fp, duration_s=float(getattr(seg, "duration_s", 0.0) or 0.0))
                    if rf:
                        reframe = rf
                        crop_mode = "smart"

            decisions.append(
                MicroEditDecision(
                    segment_id=seg_id,
                    shot_id=shot_id,
                    asset_id=asset_id,
                    asset_path=asset_path,
                    in_s=float(in_s),
                    duration_s=float(getattr(seg, "duration_s", 0.0) or 0.0),
                    speed=float(default_speed),
                    crop_mode=crop_mode,
                    reframe=reframe,
                    debug=debug,
                )
            )
        diag = {"micro_editor": "inpoint_picker_v1", "count": len(decisions), "dp": False}
        return decisions, diag

    # DP path: build candidates for each segment and choose a globally coherent set of inpoints.
    candidates_by_seg: list[list[dict[str, t.Any]]] = []
    for seg, shot in zip(segments, chosen_shots, strict=False):
        candidates_by_seg.append(_build_inpoint_candidates(segment=seg, shot=shot, work_dir=work_dir, timeout_s=timeout_s))

    pair_w = float(os.getenv("MICRO_PAIR_W", "0.35") or 0.35)
    luma_w = float(os.getenv("MICRO_PAIR_LUMA_W", "0.10") or 0.10)

    def _mode(seg: t.Any) -> str:
        hint = str(getattr(seg, "transition_hint", "") or "").strip().lower()
        if not hint:
            return "neutral"
        if any(k in hint for k in ("match", "continue", "same", "smooth", "hold", "linger", "cut on action")):
            return "continuity"
        if any(k in hint for k in ("contrast", "hard", "jump", "smash", "whip")):
            return "contrast"
        return "neutral"

    def _pair_cost(prev_seg: t.Any, seg: t.Any, a: dict[str, t.Any], b: dict[str, t.Any]) -> float:
        if pair_w <= 1e-6:
            return 0.0
        mode = _mode(seg)
        a_mag = _safe_float(a.get("motion_end"))
        b_mag = _safe_float(b.get("motion_start"))
        a_ang = _safe_float(a.get("motion_end_angle"))
        b_ang = _safe_float(b.get("motion_start_angle"))
        strength = 0.0
        if a_mag is not None and b_mag is not None:
            strength = min(float(a_mag), float(b_mag))
        dir_cost = 0.0
        if a_ang is not None and b_ang is not None and strength > 1e-6:
            d = _angle_diff_deg(float(a_ang), float(b_ang)) / 180.0
            if mode == "contrast":
                dir_cost = float(d * 0.10 * strength)
            elif mode == "continuity":
                dir_cost = float(d * 0.55 * strength)
            else:
                dir_cost = float(d * 0.28 * strength)

        # Cut-on-action: prefer similar motion magnitude around the cut.
        act_cost = 0.0
        if a_mag is not None and b_mag is not None:
            act_cost = float(abs(float(a_mag) - float(b_mag)) * 0.35)

        # Avoid huge brightness jumps introduced by inpoint choice.
        lum_cost = 0.0
        al = _safe_float(a.get("luma_end"))
        bl = _safe_float(b.get("luma"))
        if al is not None and bl is not None:
            lum_cost = float(abs(float(al) - float(bl)) * float(luma_w))

        return float((dir_cost + act_cost + lum_cost) * float(pair_w))

    # DP tables.
    n = len(candidates_by_seg)
    dp: list[list[float]] = [[1e9 for _ in c] for c in candidates_by_seg]
    back: list[list[int]] = [[-1 for _ in c] for c in candidates_by_seg]

    for i in range(n):
        cands = candidates_by_seg[i]
        for k, cand in enumerate(cands):
            unary = float(cand.get("score") or 0.0)
            if i == 0:
                dp[i][k] = unary
                back[i][k] = -1
            else:
                best = 1e9
                best_j = -1
                prev_cands = candidates_by_seg[i - 1]
                prev_seg = segments[i - 1]
                seg = segments[i]
                for j, pc in enumerate(prev_cands):
                    v = float(dp[i - 1][j]) + float(unary) + _pair_cost(prev_seg, seg, pc, cand)
                    if v < best:
                        best = v
                        best_j = j
                dp[i][k] = best
                back[i][k] = best_j

    # Backtrack best path.
    last = n - 1
    best_k = min(range(len(dp[last])), key=lambda k: dp[last][k]) if dp[last] else 0
    chosen_idx: list[int] = [0 for _ in range(n)]
    cur = best_k
    for i in range(last, -1, -1):
        chosen_idx[i] = int(cur)
        cur = int(back[i][cur]) if i > 0 else -1

    decisions: list[MicroEditDecision] = []
    for seg, shot, cands, k in zip(segments, chosen_shots, candidates_by_seg, chosen_idx, strict=False):
        seg_id = int(getattr(seg, "id", 0) or 0)
        shot_id = str(shot.get("id") or "")
        asset_id = str(shot.get("asset_id") or "")
        asset_path = str(shot.get("asset_path") or "")
        if not shot_id or not asset_id or not asset_path:
            raise ValueError("Invalid chosen shot (missing id/asset_id/asset_path)")
        cand = cands[int(k)] if cands else {"time_s": float(_safe_float(shot.get("start_s")) or 0.0)}
        in_s = float(cand.get("time_s") or 0.0)

        start_beat = _beat_floor_index(float(getattr(seg, "start_s", 0.0) or 0.0), t.cast(list[dict[str, t.Any]], beats))
        end_beat = _beat_floor_index(float(getattr(seg, "end_s", 0.0) or 0.0), t.cast(list[dict[str, t.Any]], beats))

        reframe = None
        crop_mode = str(default_crop_mode)
        if smart_crop:
            fp0 = Path(str(cand.get("frame_start") or ""))
            fp2 = Path(str(cand.get("frame_end") or ""))
            if fp0.exists() and fp2.exists():
                rf = _compute_reframe(start_frame=fp0, end_frame=fp2, duration_s=float(getattr(seg, "duration_s", 0.0) or 0.0))
                if rf:
                    reframe = rf
                    crop_mode = "smart"

        debug = {
            "segment_start_beat": start_beat,
            "segment_end_beat": end_beat,
            "chosen_candidate": cand,
            "candidates": cands,
        }
        if reframe is not None:
            debug["reframe"] = reframe

        decisions.append(
            MicroEditDecision(
                segment_id=seg_id,
                shot_id=shot_id,
                asset_id=asset_id,
                asset_path=asset_path,
                in_s=float(in_s),
                duration_s=float(getattr(seg, "duration_s", 0.0) or 0.0),
                speed=float(default_speed),
                crop_mode=crop_mode,
                reframe=reframe,
                debug=debug,
            )
        )

    diag = {"micro_editor": "micro_dp_v2", "count": len(decisions), "dp": True, "pair_w": pair_w}
    return decisions, diag
