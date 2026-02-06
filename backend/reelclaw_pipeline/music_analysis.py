from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
import typing as t


SCHEMA_VERSION = 1


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}... {stderr or 'unknown error'}")


def extract_wav(
    *,
    input_path: Path,
    output_wav_path: Path,
    sr: int = 22050,
    timeout_s: float = 120.0,
) -> None:
    """
    Extract audio to a mono PCM WAV file for analysis.
    """
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for music analysis. Please install ffmpeg and try again.")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sr)),
        "-f",
        "wav",
        str(output_wav_path),
    ]
    _run(cmd, timeout_s=timeout_s)


def _read_wav_mono_f32(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n = int(wf.getnframes())
        sampwidth = int(wf.getsampwidth())
        nch = int(wf.getnchannels())
        raw = wf.readframes(n)

    # We always extract mono 16-bit PCM, but keep this a little tolerant.
    if nch != 1:
        raise RuntimeError("Expected mono wav for analysis; got channels=%d" % nch)
    if sampwidth == 2:
        # 16-bit signed little-endian.
        import struct

        vals = list(struct.unpack("<%dh" % (len(raw) // 2), raw))
        scale = 1.0 / 32768.0
        return [float(v) * scale for v in vals], sr
    if sampwidth == 4:
        import struct

        vals = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        scale = 1.0 / 2147483648.0
        return [float(v) * scale for v in vals], sr
    raise RuntimeError(f"Unsupported wav sample width: {sampwidth}")


def _rms_envelope(
    *,
    samples: list[float],
    sr: int,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> tuple[list[float], list[float]]:
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("Invalid frame/hop size")
    env: list[float] = []
    times: list[float] = []
    n = len(samples)
    i = 0
    while i + frame_size <= n:
        frame = samples[i : i + frame_size]
        s = 0.0
        for x in frame:
            s += x * x
        rms = math.sqrt(s / max(1, frame_size))
        env.append(float(rms))
        times.append(float(i) / float(sr))
        i += hop_size
    return env, times


def _normalize(x: list[float]) -> list[float]:
    if not x:
        return []
    mx = max(x)
    if mx <= 1e-9:
        return [0.0 for _ in x]
    return [float(v) / float(mx) for v in x]


def _diff_pos(x: list[float]) -> list[float]:
    if not x:
        return []
    out = [0.0]
    for i in range(1, len(x)):
        d = x[i] - x[i - 1]
        out.append(float(d) if d > 0 else 0.0)
    return out


def _peak_pick(
    *,
    values: list[float],
    times: list[float],
    min_spacing_s: float = 0.12,
    threshold: float = 0.20,
) -> list[dict[str, float]]:
    if len(values) != len(times):
        return []

    peaks: list[tuple[float, float]] = []  # (t, v)
    last_t = -1e9
    for i in range(1, len(values) - 1):
        v = float(values[i])
        if v < float(threshold):
            continue
        if v < float(values[i - 1]) or v < float(values[i + 1]):
            continue
        t_s = float(times[i])
        if t_s - last_t < float(min_spacing_s):
            continue
        peaks.append((t_s, v))
        last_t = t_s

    return [{"t": float(t_s), "strength": float(v)} for t_s, v in peaks]


def _estimate_bpm_from_onset_env(
    *,
    onset_env: list[float],
    sr: int,
    hop_size: int,
    bpm_min: float = 70.0,
    bpm_max: float = 190.0,
) -> float | None:
    if not onset_env:
        return None

    # Autocorrelation over plausible tempo lags.
    hop_s = float(hop_size) / float(sr)
    min_lag = int(round((60.0 / float(bpm_max)) / hop_s))
    max_lag = int(round((60.0 / float(bpm_min)) / hop_s))
    if max_lag <= min_lag + 1:
        return None

    # Normalize to reduce bias from long tracks.
    x = _normalize(onset_env)
    n = len(x)
    best_lag = None
    best_val = -1.0
    for lag in range(max(1, min_lag), min(max_lag, n - 1)):
        s = 0.0
        # Cheap autocorrelation. n is usually small (~1000), so this is fine.
        for i in range(0, n - lag):
            s += x[i] * x[i + lag]
        if s > best_val:
            best_val = s
            best_lag = lag
    if best_lag is None or best_val <= 1e-6:
        return None

    period_s = float(best_lag) * hop_s
    if period_s <= 1e-6:
        return None
    bpm = 60.0 / period_s
    if bpm < bpm_min or bpm > bpm_max:
        return None
    return float(bpm)


def _beats_from_bpm(
    *,
    bpm: float,
    duration_s: float,
    start_s: float = 0.0,
) -> list[float]:
    if bpm <= 1e-6:
        return []
    period = 60.0 / float(bpm)
    t_s = max(0.0, float(start_s))
    out: list[float] = []
    while t_s <= float(duration_s) + 1e-6:
        out.append(float(t_s))
        t_s += period
    return out


def _dedupe_times(values: list[float], *, tol_s: float = 1e-3) -> list[float]:
    if not values:
        return []
    values2 = sorted(float(x) for x in values)
    out = [float(values2[0])]
    for t_s in values2[1:]:
        if abs(float(t_s) - float(out[-1])) <= float(tol_s):
            continue
        out.append(float(t_s))
    return out


def _simple_snap_time(
    t_s: float,
    *,
    beats: list[dict[str, float]],
    onsets: list[dict[str, float]],
    onset_tol_s: float = 0.08,
    beat_tol_s: float = 0.12,
) -> float:
    t0 = float(t_s)
    # Prefer onset snap if very close.
    if onsets:
        best = min(onsets, key=lambda o: abs(float(o.get("t", 0.0)) - t0))
        dt = abs(float(best.get("t", 0.0)) - t0)
        if dt <= float(onset_tol_s):
            return float(best.get("t", t0))
    if beats:
        best = min(beats, key=lambda b: abs(float(b.get("t", 0.0)) - t0))
        dt = abs(float(best.get("t", 0.0)) - t0)
        if dt <= float(beat_tol_s):
            return float(best.get("t", t0))
    return t0


def snap_times(
    times_s: list[float],
    *,
    beats: list[dict[str, float]],
    onsets: list[dict[str, float]],
    onset_tol_s: float = 0.08,
    beat_tol_s: float = 0.12,
) -> list[float]:
    """
    Snap times to nearby beats/onsets.

    Modes:
    - dp (default): global dynamic programming that preserves per-segment durations
      while snapping boundaries to beats/onsets when it is safe.
    - simple: snap each boundary independently (legacy).
    """
    mode = str(os.getenv("MUSIC_SNAP_MODE", "dp") or "dp").strip().lower()
    if mode not in {"dp", "simple"}:
        mode = "dp"

    # Always keep start/end stable (prevents accidental global retiming).
    if not times_s:
        return []
    if len(times_s) <= 2:
        return [float(t) for t in times_s]
    if not beats and not onsets:
        return [float(t) for t in times_s]

    # Legacy mode (kept for debugging).
    if mode == "simple":
        out0 = [_simple_snap_time(t, beats=beats, onsets=onsets, onset_tol_s=onset_tol_s, beat_tol_s=beat_tol_s) for t in times_s]
        fixed: list[float] = []
        for t in out0:
            tt = float(t)
            if fixed and tt <= fixed[-1] + 1e-6:
                tt = fixed[-1] + 1e-3
            fixed.append(tt)
        return fixed

    # DP mode: choose a monotonic sequence of snapped boundaries that:
    # - stays close to original boundaries
    # - snaps to strong onsets/beats when close
    # - preserves each segment duration close to the original (prevents drift)
    orig = [float(t) for t in times_s]
    n = len(orig)
    seg_durs = [max(1e-3, float(orig[i + 1] - orig[i])) for i in range(n - 1)]

    # Tunables (kept conservative; deterministic).
    w_time = float(os.getenv("MUSIC_SNAP_W_TIME", "2.0") or 2.0)
    w_onset = float(os.getenv("MUSIC_SNAP_W_ONSET", "1.0") or 1.0)
    w_beat = float(os.getenv("MUSIC_SNAP_W_BEAT", "0.65") or 0.65)
    w_dur = float(os.getenv("MUSIC_SNAP_W_DUR", "4.0") or 4.0)
    min_gap = float(os.getenv("MUSIC_SNAP_MIN_GAP_S", "0.03") or 0.03)
    min_seg = float(os.getenv("MUSIC_SNAP_MIN_SEG_S", "0.25") or 0.25)

    # Candidate times per boundary.
    # First/last are fixed to original to avoid shifting the entire reel.
    cand_times: list[list[float]] = []
    for i, t0 in enumerate(orig):
        if i == 0 or i == n - 1:
            cand_times.append([float(t0)])
            continue
        cands: list[float] = [float(t0)]
        if onsets:
            for o in onsets:
                try:
                    tt = float(o.get("t", 0.0))
                except Exception:
                    continue
                if abs(tt - t0) <= float(onset_tol_s):
                    cands.append(float(tt))
        if beats:
            for b in beats:
                try:
                    tt = float(b.get("t", 0.0))
                except Exception:
                    continue
                if abs(tt - t0) <= float(beat_tol_s):
                    cands.append(float(tt))
        cand_times.append(_dedupe_times(cands, tol_s=1e-3))

    # Map exact candidate times to "alignment strength" (best of onset/beat).
    # These are bonuses (reduce cost), so they must be bounded.
    onset_strength_by_t: dict[float, float] = {}
    for o in onsets:
        try:
            tt = round(float(o.get("t", 0.0)), 3)
            st = float(o.get("strength", 0.0) or 0.0)
        except Exception:
            continue
        onset_strength_by_t[tt] = max(float(onset_strength_by_t.get(tt, 0.0)), float(st))
    beat_strength_by_t: dict[float, float] = {}
    for b in beats:
        try:
            tt = round(float(b.get("t", 0.0)), 3)
            st = float(b.get("strength", 0.0) or 0.0)
        except Exception:
            continue
        beat_strength_by_t[tt] = max(float(beat_strength_by_t.get(tt, 0.0)), float(st))

    def node_cost(i: int, t_s: float) -> float:
        dt = abs(float(t_s) - float(orig[i]))
        c = float(w_time) * float(dt)
        tt = round(float(t_s), 3)
        on = float(onset_strength_by_t.get(tt, 0.0))
        bt = float(beat_strength_by_t.get(tt, 0.0))
        if on > 0.0:
            c -= float(w_onset) * float(min(1.0, on))
        if bt > 0.0:
            c -= float(w_beat) * float(min(1.0, bt))
        return float(c)

    def trans_cost(i: int, t_prev: float, t_cur: float) -> float:
        # Segment i-1 duration preservation.
        dur = float(t_cur) - float(t_prev)
        if dur <= float(min_gap):
            return float("inf")
        if dur < float(min_seg):
            # Big penalty but still allow if no alternative.
            short_pen = (float(min_seg) - float(dur)) * 6.0
        else:
            short_pen = 0.0
        d0 = float(seg_durs[i - 1])
        return float(w_dur) * abs(float(dur) - float(d0)) + float(short_pen)

    # DP tables.
    dp: list[list[float]] = []
    back: list[list[int]] = []
    for i in range(n):
        m = len(cand_times[i])
        dp.append([float("inf")] * m)
        back.append([-1] * m)

    # Init boundary 0.
    dp[0][0] = node_cost(0, cand_times[0][0])
    back[0][0] = -1

    # DP.
    for i in range(1, n):
        for j, t_cur in enumerate(cand_times[i]):
            base = node_cost(i, t_cur)
            best_val = float("inf")
            best_k = -1
            for k, t_prev in enumerate(cand_times[i - 1]):
                v = dp[i - 1][k]
                if not math.isfinite(v):
                    continue
                tc = trans_cost(i, t_prev, t_cur)
                if not math.isfinite(tc):
                    continue
                val = float(v) + float(base) + float(tc)
                if val < best_val:
                    best_val = float(val)
                    best_k = int(k)
            dp[i][j] = float(best_val)
            back[i][j] = int(best_k)

    # Pick best end.
    end_j = min(range(len(dp[-1])), key=lambda j: dp[-1][j])
    if not math.isfinite(dp[-1][end_j]) or back[-1][end_j] < 0:
        # Fallback to simple snapping if DP fails.
        out0 = [_simple_snap_time(t, beats=beats, onsets=onsets, onset_tol_s=onset_tol_s, beat_tol_s=beat_tol_s) for t in times_s]
        fixed: list[float] = []
        for t in out0:
            tt = float(t)
            if fixed and tt <= fixed[-1] + 1e-6:
                tt = fixed[-1] + 1e-3
            fixed.append(tt)
        return fixed

    # Backtrack.
    snapped_rev: list[float] = []
    j = int(end_j)
    for i in reversed(range(n)):
        snapped_rev.append(float(cand_times[i][j]))
        j = int(back[i][j])
        if i > 0 and j < 0:
            # Safety fallback.
            snapped_rev = []
            break
    if not snapped_rev:
        return [float(t) for t in times_s]
    snapped = list(reversed(snapped_rev))

    # Final monotonic enforcement (guards float rounding).
    fixed2: list[float] = []
    for t in snapped:
        tt = float(t)
        if fixed2 and tt <= fixed2[-1] + 1e-6:
            tt = fixed2[-1] + 1e-3
        fixed2.append(tt)
    return fixed2


def analyze_music(
    *,
    audio_or_video_path: Path,
    output_json_path: Path,
    timeout_s: float = 240.0,
) -> dict[str, t.Any]:
    """
    Analyze an audio track into beats/onsets and a rough energy curve.

    This intentionally avoids hard dependencies. If librosa/aubio are available
    in the environment, they can be added later behind a feature flag. For now
    we ship a no-deps baseline that is good enough for beat snapping.
    """
    tmp_wav = output_json_path.with_suffix(".tmp.wav")
    try:
        extract_wav(input_path=audio_or_video_path, output_wav_path=tmp_wav, timeout_s=min(timeout_s, 180.0))
        samples, sr = _read_wav_mono_f32(tmp_wav)
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass

    # Envelope + onset strength.
    frame_size = int(os.getenv("MUSIC_FRAME_SIZE", "1024"))
    hop_size = int(os.getenv("MUSIC_HOP_SIZE", "512"))
    env, times = _rms_envelope(samples=samples, sr=sr, frame_size=frame_size, hop_size=hop_size)
    env_n = _normalize(env)
    onset_env = _normalize(_diff_pos(env_n))

    # Onsets.
    onset_thresh = float(os.getenv("MUSIC_ONSET_THRESHOLD", "0.22"))
    onsets = _peak_pick(values=onset_env, times=times, threshold=onset_thresh, min_spacing_s=0.10)

    # BPM + beats.
    bpm = _estimate_bpm_from_onset_env(onset_env=onset_env, sr=sr, hop_size=hop_size) or 120.0
    duration_s = float(times[-1] if times else 0.0)
    beat_times = _beats_from_bpm(bpm=bpm, duration_s=duration_s, start_s=0.0)
    beats: list[dict[str, float]] = []
    for i, bt in enumerate(beat_times):
        # Strength = best onset within a short window; otherwise energy.
        strength = 0.0
        if onsets:
            near = [o for o in onsets if abs(float(o["t"]) - float(bt)) <= 0.06]
            if near:
                strength = max(float(o["strength"]) for o in near)
        if strength <= 0.0:
            # Use envelope value near this time.
            if times:
                j = min(range(len(times)), key=lambda k: abs(float(times[k]) - float(bt)))
                strength = float(env_n[j]) if j < len(env_n) else 0.0
        beats.append({"i": int(i), "t": float(bt), "strength": float(strength)})

    # Rough sections: detect large changes in smoothed energy.
    sections: list[dict[str, t.Any]] = []
    if env_n and times:
        w = max(3, int(round(1.0 / (float(hop_size) / float(sr)))))  # ~1s window
        smooth: list[float] = []
        for i in range(len(env_n)):
            a = max(0, i - w)
            b = min(len(env_n), i + w)
            smooth.append(sum(env_n[a:b]) / max(1, b - a))
        deriv = _diff_pos(smooth)
        change_pts = _peak_pick(values=_normalize(deriv), times=times, threshold=0.35, min_spacing_s=2.0)
        cut_ts = [0.0] + [float(p["t"]) for p in change_pts] + [float(times[-1])]
        for si in range(len(cut_ts) - 1):
            s0 = cut_ts[si]
            s1 = cut_ts[si + 1]
            b0 = min(range(len(beats)), key=lambda j: abs(float(beats[j]["t"]) - s0)) if beats else 0
            b1 = min(range(len(beats)), key=lambda j: abs(float(beats[j]["t"]) - s1)) if beats else b0
            if b1 <= b0:
                continue
            a = min(range(len(times)), key=lambda j: abs(float(times[j]) - s0))
            b = min(range(len(times)), key=lambda j: abs(float(times[j]) - s1))
            if b <= a:
                continue
            e = sum(smooth[a:b]) / max(1, b - a)
            sections.append({"name": f"sec_{len(sections)+1}", "start_beat": int(b0), "end_beat": int(b1), "energy": float(e)})

    doc = {
        "version": SCHEMA_VERSION,
        "bpm": float(bpm),
        "beat_times": beats,
        "onsets": onsets,
        "sections": sections,
        "analysis": {
            "sr": int(sr),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "onset_threshold": float(onset_thresh),
            "snap_mode": str(os.getenv("MUSIC_SNAP_MODE", "dp") or "dp"),
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc
