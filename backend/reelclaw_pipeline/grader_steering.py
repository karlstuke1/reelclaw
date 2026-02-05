from __future__ import annotations

import math
import os
from pathlib import Path
import typing as t

from .learned_grader import GeminiGrader


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _norm_tag(tag: str) -> str:
    return " ".join((tag or "").strip().lower().split())


def _tokens(text: str) -> set[str]:
    import re

    s = (text or "").lower()
    toks = set(re.findall(r"[a-z0-9']{2,}", s))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "you",
        "your",
        "our",
        "are",
        "was",
        "were",
        "from",
        "into",
        "over",
        "under",
        "then",
        "than",
        "just",
        "like",
    }
    return {t for t in toks if t not in stop}


def _tag_set_for_shot(shot: dict[str, t.Any]) -> set[str]:
    out: set[str] = set()
    tags = shot.get("tags") or []
    if isinstance(tags, list):
        for tg in tags:
            nt = _norm_tag(str(tg))
            if nt:
                out.add(nt)
    for k in ("shot_type", "setting", "mood"):
        v = shot.get(k)
        nv = _norm_tag(str(v)) if v else ""
        if nv:
            out.add(nv)
    return out


def _sem_key_for_shot(shot: dict[str, t.Any]) -> str:
    gid = str(shot.get("sequence_group_id") or "").strip()
    st = _norm_tag(str(shot.get("shot_type") or ""))
    tags = shot.get("tags") or []
    t0 = ""
    if isinstance(tags, list) and tags:
        t0 = _norm_tag(str(tags[0]))
    parts = [p for p in (gid, st, t0) if p]
    return ("|".join(parts) or gid or st or t0 or "x")[:80]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    if mu is None:
        return None
    s = 0.0
    for v in values:
        dv = float(v) - float(mu)
        s += dv * dv
    return float((s / max(1, len(values))) ** 0.5)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(float(x) for x in values)
    if not v:
        return None
    qq = max(0.0, min(1.0, float(q)))
    idx = int(round(qq * float(len(v) - 1)))
    idx = max(0, min(len(v) - 1, idx))
    return float(v[idx])


def _srgb_to_linear(c: float) -> float:
    cc = float(max(0.0, min(1.0, c)))
    if cc <= 0.04045:
        return float(cc / 12.92)
    return float(((cc + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Convert sRGB (0..1) to CIE Lab (D65).
    DeltaE in this space roughly matches the scale used by palette continuity metrics.
    """
    r, g, b = (_srgb_to_linear(rgb[0]), _srgb_to_linear(rgb[1]), _srgb_to_linear(rgb[2]))
    # sRGB -> XYZ (D65)
    x = (0.4124 * r) + (0.3576 * g) + (0.1805 * b)
    y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    z = (0.0193 * r) + (0.1192 * g) + (0.9505 * b)
    # Normalize by reference white.
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = x / xn
    y = y / yn
    z = z / zn

    d = 6.0 / 29.0
    d3 = d**3
    a = 1.0 / (3.0 * d * d)
    b0 = 4.0 / 29.0

    def f(t0: float) -> float:
        if t0 > d3:
            return float(t0 ** (1.0 / 3.0))
        return float(a * t0 + b0)

    fx, fy, fz = f(x), f(y), f(z)
    L = (116.0 * fy) - 16.0
    aa = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return float(L), float(aa), float(bb)


def _delta_e_lab(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


def find_latest_grader_dir(outputs_dir: Path) -> Path | None:
    outs = outputs_dir.expanduser().resolve()
    if not outs.exists():
        return None
    cands: list[Path] = []
    for p in outs.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("gemini_graders_"):
            continue
        if not (p / "meta.json").exists():
            continue
        cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def fast_features_for_sequence(
    *,
    segments: list[t.Any],
    shots: list[dict[str, t.Any]],
    default_speed: float,
    default_zoom: float,
    stabilize_enabled: bool,
    stabilize_shake_th: float,
) -> dict[str, float | None]:
    """
    Fast (no-render) approximation of the deterministic variant features used by the learned grader.
    Uses only segment metadata + shot_index metrics (luma/dark/rgb/shake/sharpness).
    """
    seg_dur: list[float] = []
    luma_diff: list[float] = []
    dark_diff: list[float] = []
    asset_ids: list[str] = []
    shake_vals: list[float] = []
    sharp_vals: list[float] = []
    rgb_vals: list[tuple[float, float, float]] = []
    seg_rgb_diff: list[float] = []
    luma_vals: list[float] = []
    stabilize_flags: list[float] = []
    tag_hit_fracs: list[float] = []
    text_hit_fracs: list[float] = []
    sem_keys: list[str] = []
    group_ids: list[str] = []
    motion_vals: list[float] = []
    motion_seq: list[float | None] = []

    for seg, shot in zip(segments, shots):
        try:
            seg_dur.append(float(getattr(seg, "duration_s", 0.0) or 0.0))
        except Exception:
            pass

        rl = _safe_float(getattr(seg, "ref_luma", None))
        rd = _safe_float(getattr(seg, "ref_dark_frac", None))
        sl = _safe_float(shot.get("luma_mean"))
        sd = _safe_float(shot.get("dark_frac"))
        if rl is not None and sl is not None:
            luma_diff.append(abs(float(sl) - float(rl)))
        if rd is not None and sd is not None:
            dark_diff.append(abs(float(sd) - float(rd)))

        aid = str(shot.get("asset_id") or "").strip()
        if aid:
            asset_ids.append(aid)

        sh = _safe_float(shot.get("shake_score"))
        if sh is not None:
            shake_vals.append(float(sh))
            if stabilize_enabled and float(sh) >= float(stabilize_shake_th):
                stabilize_flags.append(1.0)
            else:
                stabilize_flags.append(0.0)
        else:
            stabilize_flags.append(0.0)

        sp = _safe_float(shot.get("sharpness"))
        if sp is not None:
            sharp_vals.append(float(sp))

        mv = _safe_float(shot.get("motion_score"))
        motion_seq.append(float(mv) if mv is not None else None)
        if mv is not None:
            motion_vals.append(float(mv))

        rgb = shot.get("rgb_mean")
        if isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(x, (int, float)) for x in rgb):
            rgb_vals.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))

        ref_rgb = getattr(seg, "ref_rgb_mean", None)
        if (
            isinstance(ref_rgb, list)
            and isinstance(rgb, list)
            and len(ref_rgb) == 3
            and len(rgb) == 3
            and all(isinstance(x, (int, float)) for x in ref_rgb)
            and all(isinstance(x, (int, float)) for x in rgb)
        ):
            try:
                seg_rgb_diff.append(sum(abs(float(ref_rgb[i]) - float(rgb[i])) for i in range(3)) / 3.0)
            except Exception:
                pass

        if sl is not None:
            luma_vals.append(float(sl))

        # Semantic/editorial alignment (cheap).
        desired = getattr(seg, "desired_tags", None)
        desired_set = {_norm_tag(str(tg)) for tg in (desired or []) if _norm_tag(str(tg))}
        if desired_set:
            tag_set = _tag_set_for_shot(shot)
            if tag_set:
                hits = len(desired_set.intersection(tag_set))
                tag_hit_fracs.append(float(hits) / float(max(1, len(desired_set))))

        seg_toks: set[str] = set()
        seg_toks |= _tokens(str(getattr(seg, "reference_visual", "") or ""))
        seg_toks |= _tokens(str(getattr(seg, "overlay_text", "") or ""))
        seg_toks |= _tokens(str(getattr(seg, "story_beat", "") or ""))
        if desired_set:
            seg_toks |= _tokens(" ".join(sorted(desired_set)))
        shot_toks = _tokens(str(shot.get("description") or ""))
        if seg_toks and shot_toks:
            text_hit_fracs.append(float(len(seg_toks.intersection(shot_toks))) / float(max(1, len(seg_toks))))

        sem_keys.append(_sem_key_for_shot(shot))
        gid = str(shot.get("sequence_group_id") or "").strip()
        if gid:
            group_ids.append(gid)

    avg_luma_diff = _mean(luma_diff)
    avg_dark_diff = _mean(dark_diff)
    cheap_score = None
    if avg_luma_diff is not None or avg_dark_diff is not None:
        dl = float(avg_luma_diff) if avg_luma_diff is not None else 9.0
        dd = float(avg_dark_diff) if avg_dark_diff is not None else 9.0
        cheap_score = float(dl + 0.8 * dd)

    segment_count = float(len(segments))
    unique_assets = float(len(set(asset_ids))) if asset_ids else None
    asset_unique_frac = None
    if segment_count > 0 and unique_assets is not None:
        asset_unique_frac = float(unique_assets / max(1.0, segment_count))

    # Palette continuity approximation from per-shot rgb_mean.
    delta_e: list[float] = []
    if len(rgb_vals) >= 2:
        labs = [_rgb_to_lab(rgb) for rgb in rgb_vals]
        for a, b in zip(labs[:-1], labs[1:]):
            delta_e.append(_delta_e_lab(a, b))
    palette_deltaE_p95 = _percentile(delta_e, 0.95) if delta_e else None

    luma_jump_max = None
    if len(luma_vals) >= 2:
        luma_jump_max = float(max(abs(float(b) - float(a)) for a, b in zip(luma_vals[:-1], luma_vals[1:])))

    sem_unique_frac = None
    sem_run_max = None
    if sem_keys:
        sem_unique_frac = float(len(set(sem_keys)) / max(1, len(sem_keys)))
        run = 0
        best = 0
        prev: str | None = None
        for k in sem_keys:
            if prev is not None and k == prev:
                run += 1
            else:
                run = 1
                prev = k
            best = max(best, run)
        sem_run_max = float(best)

    group_switch_frac = None
    group_dominant_frac = None
    if len(group_ids) >= 2:
        switches = sum(1 for a, b in zip(group_ids[:-1], group_ids[1:], strict=False) if a and b and a != b)
        group_switch_frac = float(switches) / float(max(1, len(group_ids) - 1))
        counts: dict[str, int] = {}
        for g in group_ids:
            counts[g] = counts.get(g, 0) + 1
        group_dominant_frac = float(max(counts.values())) / float(max(1, len(group_ids))) if counts else None

    arc_motion_delta = None
    if len(motion_seq) >= 4:
        first = [m for m in motion_seq[:2] if m is not None]
        last = [m for m in motion_seq[-2:] if m is not None]
        if first and last:
            arc_motion_delta = float(_mean(last) or 0.0) - float(_mean(first) or 0.0)

    feats: dict[str, float | None] = {
        # index.tsv-ish (reference match proxy)
        "cheap_score": cheap_score,
        "avg_luma_diff": avg_luma_diff,
        "avg_dark_diff": avg_dark_diff,
        "uniq_assets": unique_assets,
        "min_jaccard_dist": None,
        # segments
        "segment_count": segment_count,
        "segment_dur_mean": _mean(seg_dur),
        "segment_dur_std": _std(seg_dur),
        "speed_mean": float(default_speed),
        "speed_std": 0.0,
        "zoom_mean": float(default_zoom),
        "stabilize_frac": _mean(stabilize_flags),
        "asset_unique_frac": asset_unique_frac,
        # shake proxy
        "shake_score_mean": _mean(shake_vals),
        "shake_score_p95": _percentile(shake_vals, 0.95) if shake_vals else None,
        # semantic/editorial (cheap)
        "tag_hit_frac_mean": _mean(tag_hit_fracs),
        "tag_hit_frac_p05": _percentile(tag_hit_fracs, 0.05) if tag_hit_fracs else None,
        "text_hit_frac_mean": _mean(text_hit_fracs),
        "sem_unique_frac": sem_unique_frac,
        "sem_run_max": sem_run_max,
        "group_switch_frac": group_switch_frac,
        "group_dominant_frac": group_dominant_frac,
        "shot_motion_mean": _mean(motion_vals) if motion_vals else None,
        "shot_motion_p95": _percentile(motion_vals, 0.95) if motion_vals else None,
        "shot_sharpness_p05": _percentile(sharp_vals, 0.05) if sharp_vals else None,
        "arc_motion_delta": arc_motion_delta,
        # beat alignment (unknown without music_analysis)
        "beat_align_s_p95": None,
        "onset_align_s_p95": None,
        "beat_align_within_120ms_frac": None,
        "onset_align_within_80ms_frac": None,
        # palette continuity (approx)
        "palette_deltaE_p95": palette_deltaE_p95,
        "luma_jump_max": luma_jump_max,
        # frame quality / blur proxy (approx)
        "blur_lap_var_p05": _percentile(sharp_vals, 0.05) if sharp_vals else None,
        "clip_black_mean": _mean([float(x) for x in (shot.get("dark_frac") for shot in shots) if isinstance(x, (int, float))]),
        "clip_white_mean": None,
        # segment-local objective diffs (approx)
        "seg_luma_diff_mean": avg_luma_diff,
        "seg_luma_diff_p95": _percentile(luma_diff, 0.95) if luma_diff else None,
        "seg_dark_diff_mean": avg_dark_diff,
        "seg_dark_diff_p95": _percentile(dark_diff, 0.95) if dark_diff else None,
        "seg_rgb_diff_mean": _mean(seg_rgb_diff),
        "seg_rgb_diff_p95": _percentile(seg_rgb_diff, 0.95) if seg_rgb_diff else None,
        "seg_blur_lap_var_p05": _percentile(sharp_vals, 0.05) if sharp_vals else None,
        # embedding similarity (not available in fast/no-render mode)
        "clip_ref_out_cos_mean": None,
        "clip_ref_out_cos_p10": None,
        "clip_ref_out_cos_p50": None,
        "clip_ref_out_cos_p90": None,
        "clip_ref_out_cos_min": None,
        "clip_adj_cos_mean": None,
        "clip_adj_cos_p10": None,
    }
    return feats


def choose_beam_sequence_with_grader(
    *,
    segments: list[t.Any],
    shots_by_id: dict[str, dict[str, t.Any]],
    beam_final: list[dict[str, t.Any]],
    optimizer_diag: dict[str, t.Any],
    grader: GeminiGrader,
    default_speed: float,
    default_zoom: float,
    stabilize_enabled: bool,
    stabilize_shake_th: float,
) -> tuple[list[dict[str, t.Any]] | None, dict[str, t.Any]]:
    """
    Pick the best candidate sequence from optimizer beam using the learned grader (fast, no rendering).
    Returns (chosen_shots, steering_diagnostics).
    """
    rows: list[dict[str, t.Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_seq: list[dict[str, t.Any]] | None = None
    best_rank: int | None = None

    for item in beam_final:
        shot_ids = item.get("shot_ids")
        if not isinstance(shot_ids, list) or not shot_ids:
            continue
        seq: list[dict[str, t.Any]] = []
        ok = True
        for sid in shot_ids:
            s = shots_by_id.get(str(sid))
            if not isinstance(s, dict):
                ok = False
                break
            seq.append(s)
        if not ok:
            continue

        feats = fast_features_for_sequence(
            segments=segments,
            shots=seq,
            default_speed=default_speed,
            default_zoom=default_zoom,
            stabilize_enabled=stabilize_enabled,
            stabilize_shake_th=stabilize_shake_th,
        )
        pred = grader.predict(feats)

        overall = float(pred.get("overall_score") or pred.get("match_score") or 0.0)
        stability = float(pred.get("stability") or 0.0)
        cost = float(item.get("cost") or 0.0)

        key = (overall, stability, -cost)
        row = {
            "rank": int(item.get("rank") or 0),
            "optimizer_cost": cost,
            "pred": pred,
            "features": feats,
        }
        rows.append(row)

        if best_key is None or key > best_key:
            best_key = key
            best_seq = seq
            best_rank = int(item.get("rank") or 0)

    diag: dict[str, t.Any] = {
        "enabled": True,
        "grader_dir": str(grader.root),
        "beam_candidates": rows,
        "chosen_rank": best_rank,
        "optimizer": {"cost": optimizer_diag.get("cost"), "beam_size": optimizer_diag.get("beam_size")},
    }
    return best_seq, diag
