from __future__ import annotations

import os
import re
from dataclasses import dataclass
import math
import typing as t


def _truthy_env(name: str, default: str) -> bool:
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


def _norm_tag(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _tokens(text: str) -> set[str]:
    """
    Cheap lexical tokens for meta semantic matching (no embeddings).
    Keeps only short-ish alnum tokens and strips common stopwords.
    """
    s = (text or "").strip().lower()
    if not s:
        return set()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out: set[str] = set()
    for w in s.split():
        if len(w) < 3:
            continue
        if w in _STOPWORDS:
            continue
        out.add(w)
    return out


def _sem_key_for_shot(shot: dict[str, t.Any]) -> str:
    """
    Coarse semantic signature used to discourage repetitive shot "types" even across different assets.
    Keep it cheap, general, and stable (no reel-specific rules).
    """
    st = str(shot.get("shot_type") or "").strip().lower()
    setting = str(shot.get("setting") or "").strip().lower()
    tags = shot.get("tags")
    tag_bits: list[str] = []
    if isinstance(tags, list):
        stop = {
            "outdoor",
            "indoor",
            "daytime",
            "night",
            "low-key lighting",
            "high-key lighting",
            "high contrast",
            "wide shot",
            "close-up",
            "medium shot",
            "centered subject",
            "handheld",
            "static shot",
            "portrait",
            "landscape",
        }
        for raw in tags:
            s = str(raw or "").strip().lower()
            if not s or s in stop:
                continue
            tag_bits.append(s)
            if len(tag_bits) >= 2:
                break
    # Normalize empties so the key remains deterministic for sparse metadata.
    return "|".join([setting or "_", st or "_"] + (tag_bits if tag_bits else ["_"]))


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


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.0


def _is_very_dark(ref_luma: float | None, ref_dark: float | None) -> bool:
    try:
        if ref_dark is not None and float(ref_dark) >= 0.93:
            return True
        if ref_luma is not None and float(ref_luma) <= 0.04:
            return True
    except Exception:
        return False
    return False


def _tag_set(row: dict[str, t.Any]) -> set[str]:
    out: set[str] = set()
    tags = row.get("tags") or []
    if isinstance(tags, list):
        for tg in tags:
            nt = _norm_tag(str(tg))
            if nt:
                out.add(nt)
    for k in ("shot_type", "setting", "mood"):
        v = row.get(k)
        nv = _norm_tag(str(v)) if v else ""
        if nv:
            out.add(nv)
    return out


def _angle_diff_deg(a: float, b: float) -> float:
    """
    Smallest absolute difference between angles (degrees), in [0..180].
    """
    d = abs(float(a) - float(b)) % 360.0
    return float(360.0 - d) if d > 180.0 else float(d)


@dataclass(frozen=True)
class OptimizerConfig:
    candidates_per_slot: int = 40
    beam_size: int = 40
    max_per_asset_in_candidates: int = 2
    max_per_asset_total: int = 0  # 0 = auto
    max_per_group_total: int = 0  # 0 = auto
    usage_penalty: float = 0.25
    continuity_penalty: float = 0.35
    prefer_match_over_diversity: bool = False


def _slot_cost(
    *,
    segment: t.Any,
    shot: dict[str, t.Any],
) -> float:
    """
    Lower is better. Purely reel-agnostic heuristics:
    - lighting match (luma + dark fraction)
    - motion match (segment duration -> energy)
    - tag overlap (desired_tags)
    - light quality proxy (sharpness penalty)
    """
    ref_luma = _safe_float(getattr(segment, "ref_luma", None))
    ref_dark = _safe_float(getattr(segment, "ref_dark_frac", None))
    desired_tags = getattr(segment, "desired_tags", []) or []
    desired_set = {_norm_tag(str(tg)) for tg in desired_tags if _norm_tag(str(tg))}
    # Prefer music-aware energy when available (beat-synced projects).
    dur_energy = _segment_energy_hint(float(getattr(segment, "duration_s", 1.0) or 1.0))
    me = _safe_float(getattr(segment, "music_energy", None))
    if me is not None:
        # Blend instead of replacing: our no-deps music_energy is a conservative proxy and can
        # under-report "high-energy" tracks early on. Duration is still a strong cadence prior.
        energy = _clamp01((0.65 * float(dur_energy)) + (0.35 * _clamp01(float(me))))
    else:
        energy = float(dur_energy)

    # Emotional arc / intensity target (very lightweight).
    # We keep this conservative: it should nudge, not dominate lighting/tag matching.
    beat_goal = str(getattr(segment, "beat_goal", "") or "").strip().lower()
    target_map: dict[str, float] = {
        "hook": 0.82,
        "setup": 0.38,
        "escalation": 0.64,
        "payoff": 0.88,
        "cta": 0.48,
    }
    target_base = target_map.get(beat_goal)
    target_intensity = None
    if target_base is not None:
        # Blend reference-derived target with timing/music energy so this stays reel-agnostic.
        target_intensity = _clamp01((0.65 * float(target_base)) + (0.35 * _clamp01(float(energy))))

    # Hard gate: ensure the shot window is long enough for the segment.
    seg_dur = float(getattr(segment, "duration_s", 0.0) or 0.0)
    shot_dur = _safe_float(shot.get("duration_s")) or 0.0
    if seg_dur > 0.05 and shot_dur > 0.0 and shot_dur < seg_dur * 0.90:
        return 9.0  # effectively unusable

    # Lighting distance (lower is better).
    dl = 0.18
    dd = 0.25
    sl = _safe_float(shot.get("luma_mean"))
    sd = _safe_float(shot.get("dark_frac"))
    if ref_luma is not None and sl is not None:
        dl = abs(float(sl) - float(ref_luma))
    if ref_dark is not None and sd is not None:
        dd = abs(float(sd) - float(ref_dark))
    lighting = (dl * 1.2) + (dd * 1.0)

    # Color distance (mild). Helps avoid jarring hue/white-balance mismatches.
    ref_rgb = getattr(segment, "ref_rgb_mean", None)
    shot_rgb = shot.get("rgb_mean")
    color_dist = 0.0
    if (
        isinstance(ref_rgb, list)
        and isinstance(shot_rgb, list)
        and len(ref_rgb) == 3
        and len(shot_rgb) == 3
        and all(isinstance(x, (int, float)) for x in ref_rgb)
        and all(isinstance(x, (int, float)) for x in shot_rgb)
    ):
        color_dist = sum(abs(float(ref_rgb[i]) - float(shot_rgb[i])) for i in range(3)) / 3.0
        # Keep this weight modest so we don't overfit to a single frame's white balance.
        color_dist = color_dist * 0.55

    # Motion match.
    sm = _safe_float(shot.get("motion_score"))
    motion = float(sm) if sm is not None else 0.35
    motion_norm = _clamp01(float(motion))
    motion_dist = abs(motion - energy) * 0.45

    # Tag overlap bonus (negative cost).
    tag_set = _tag_set(shot)
    overlap = len(desired_set.intersection(tag_set)) if desired_set else 0
    tag_bonus = -0.10 * min(overlap, 6)
    # If a segment expresses intent (desired_tags) but a shot matches NONE of it, penalize slightly.
    # This reduces common failures like "random suitcase/dinner shot" sneaking in on lighting alone.
    tag_miss_pen = 0.0
    if desired_set and overlap <= 0:
        tag_miss_pen = float(_float_env("OPT_TAG_MISS_PEN", "0.10"))

    # Story-plan preference (optional): if a story planner provided preferred_sequence_group_ids,
    # treat them as a *soft* constraint. We still allow deviation if another shot matches the
    # reference much better, but we should prefer the planned groups when costs are similar.
    story_pref = getattr(segment, "preferred_sequence_group_ids", None)
    story_bonus = 0.0
    if isinstance(story_pref, list) and story_pref:
        gid = str(shot.get("sequence_group_id") or "").strip()
        prefs = [str(x).strip() for x in story_pref if str(x).strip()]
        if gid and prefs:
            if gid == prefs[0]:
                story_bonus = -0.10
            elif gid in prefs[:3]:
                story_bonus = -0.06
            elif gid in prefs:
                story_bonus = -0.03
            else:
                # Mild penalty for being outside the planned chapter.
                story_bonus = 0.06

    # Light semantic match between the segment's visual intent and the shot's description.
    # This stays meta: it operates on provided text (no video-specific rules).
    text_bonus = 0.0
    shot_desc = str(shot.get("description") or "").strip()
    if shot_desc:
        shot_toks = _tokens(shot_desc)
        seg_toks: set[str] = set()
        seg_toks |= _tokens(" ".join(desired_set)) if desired_set else set()
        seg_toks |= _tokens(str(getattr(segment, "reference_visual", "") or ""))
        seg_toks |= _tokens(str(getattr(segment, "story_beat", "") or ""))
        if seg_toks and shot_toks:
            overlap2 = len(seg_toks.intersection(shot_toks))
            # Keep weight modest; tags/lighting still dominate.
            text_bonus = -0.05 * min(overlap2, 8)

    # Quality: avoid extreme blur when we have options (soft penalty).
    sharp = _safe_float(shot.get("sharpness"))
    sharp_norm = 0.55
    if sharp is not None:
        # Heuristic: sharpness is a Laplacian-variance proxy on a tiny thumbnail.
        # Map into 0..1 so we can blend it into an "intensity" proxy.
        sharp_norm = _clamp01((float(sharp) - 60.0) / 180.0)
    sharp_pen = 0.0
    if sharp is not None:
        # Variance values vary wildly; clamp to a small penalty.
        if sharp < 60.0:
            sharp_pen = 0.12
        elif sharp < 120.0:
            sharp_pen = 0.06

    # Camera shake: prefer stable shots; this is separate from subject motion.
    #
    # NOTE: Our current shake_score is a cheap proxy computed from a few spaced thumbnails
    # (phase correlation deltas). It's useful as a debugging signal but can be noisy and
    # may correlate with deliberate motion (pans) or low-texture/dark shots. Therefore we
    # keep the default penalty effectively OFF unless explicitly enabled via env.
    shake = _safe_float(shot.get("shake_score"))
    shake_pen = 0.0
    if shake is not None:
        start = _float_env("OPT_SHAKE_PEN_START", "0.40")
        slope = _float_env("OPT_SHAKE_PEN_SLOPE", "8.0")
        if float(shake) > float(start):
            shake_pen = min(0.65, (float(shake) - float(start)) * float(slope))

    # Hook/payoff nudges: avoid "boring wide static" openings/closings.
    #
    # This is intentionally lightweight so it doesn't override lighting/tag matching.
    # It tends to fix the common critique: "good match but flat / no intentional hook".
    impact_pen = 0.0
    if beat_goal in {"hook", "payoff"}:
        st0 = str(shot.get("shot_type") or "").strip().lower()
        tags0 = shot.get("tags") or []
        tag_text = " ".join(str(x or "").strip().lower() for x in tags0) if isinstance(tags0, list) else ""
        is_close = ("close" in st0) or ("macro" in st0) or ("detail" in st0) or ("close-up" in tag_text) or ("macro" in tag_text) or ("detail" in tag_text)
        is_wide = ("wide" in st0) or ("establish" in st0) or ("wide shot" in tag_text) or ("establishing" in tag_text)
        is_static = ("static" in st0) or ("static shot" in tag_text) or (float(motion_norm) <= float(_float_env("OPT_STATIC_MOTION_MAX", "0.12")))

        if is_close:
            impact_pen -= float(_float_env("OPT_HOOK_CLOSE_BONUS", "0.06"))
        if is_wide:
            impact_pen += float(_float_env("OPT_HOOK_WIDE_PEN", "0.08"))
        if is_static:
            impact_pen += float(_float_env("OPT_HOOK_STATIC_PEN", "0.06"))
        if is_wide and is_static:
            impact_pen += float(_float_env("OPT_HOOK_WIDE_STATIC_PEN", "0.06"))

    arc_pen = 0.0
    arc_w = _float_env("OPT_ARC_W", "0.18")
    if target_intensity is not None and float(arc_w) > 1e-6:
        dark = _safe_float(shot.get("dark_frac"))
        dark_norm = _clamp01(float(dark)) if dark is not None else 0.5
        # Proxy for "visual intensity": more motion + crispness + (slightly) higher-key exposure.
        shot_intensity = (0.55 * motion_norm) + (0.30 * float(sharp_norm)) + (0.15 * (1.0 - dark_norm))
        arc_pen = abs(float(shot_intensity) - float(target_intensity)) * float(arc_w)

    return float(lighting + color_dist + motion_dist + tag_bonus + tag_miss_pen + text_bonus + sharp_pen + shake_pen + story_bonus + arc_pen + impact_pen)


def shortlist_shots_for_segment(
    *,
    segment: t.Any,
    shots: list[dict[str, t.Any]],
    limit: int,
    max_per_asset: int,
) -> list[dict[str, t.Any]]:
    ref_luma = _safe_float(getattr(segment, "ref_luma", None))
    ref_dark = _safe_float(getattr(segment, "ref_dark_frac", None))
    very_dark = _is_very_dark(ref_luma, ref_dark)

    desired_tags = getattr(segment, "desired_tags", []) or []
    desired_set = {_norm_tag(str(tg)) for tg in desired_tags if _norm_tag(str(tg))}

    seg_dur = float(getattr(segment, "duration_s", 0.0) or 0.0)
    # Motion gating: optionally filter out very static shots for high-energy segments.
    # This targets the common failure mode where pacing feels "too slow/static" even when
    # the cadence matches, because the chosen shots lack dynamic movement.
    #
    # IMPORTANT: gating can over-filter small libraries (or libraries where motion_score is noisy),
    # collapsing the candidate pool to a handful of shots and causing repeats. Therefore:
    # - Default is OFF (soft penalties in _slot_cost still prefer motion match).
    # - If enabled, we automatically relax it when it would shrink the pool too much.
    motion_gate = _truthy_env("OPT_MOTION_GATE", "0")
    motion_min_low = _float_env("OPT_MOTION_MIN_LOW", "0.06")
    # Default tuned to the observed distribution in our fast shot index (p95 ≈ 0.25).
    # Keep this conservative: motion gating is OFF by default and only applies when enabled.
    motion_min_high = _float_env("OPT_MOTION_MIN_HIGH", "0.18")
    energy_th = _float_env("OPT_MOTION_GATE_ENERGY_TH", "0.55")
    dur_energy = _segment_energy_hint(seg_dur if seg_dur > 0.0 else float(getattr(segment, "duration_s", 1.0) or 1.0))
    me = _safe_float(getattr(segment, "music_energy", None))
    if me is not None:
        energy_f = float(_clamp01((0.65 * float(dur_energy)) + (0.35 * _clamp01(float(me)))))
    else:
        energy_f = float(_clamp01(float(dur_energy)))
    motion_min = float(motion_min_high if energy_f >= float(energy_th) else motion_min_low)
    if very_dark:
        # Motion estimation is noisier on very dark/low-texture shots; relax slightly.
        motion_min = motion_min * float(_float_env("OPT_MOTION_MIN_DARK_SCALE", "0.85"))

    pref_groups = getattr(segment, "preferred_sequence_group_ids", None)
    pref_set: set[str] | None = None
    if isinstance(pref_groups, list):
        pref_set = {str(x).strip() for x in pref_groups if str(x).strip()}
        if not pref_set:
            pref_set = None

    # Keep both a motion-gated and a motion-relaxed pool so we can auto-relax when gating
    # would collapse diversity.
    preferred_gate: list[dict[str, t.Any]] = []
    others_gate: list[dict[str, t.Any]] = []
    preferred_all: list[dict[str, t.Any]] = []
    others_all: list[dict[str, t.Any]] = []
    strict_pref = _truthy_env("OPT_STRICT_STORY_PREF", "0")
    # Shake gating can easily over-filter when shake_score is noisy; default is OFF.
    shake_gate = _truthy_env("OPT_SHAKE_GATE", "0")
    # Default tuned to our fast shot index distribution (p90 ≈ 0.28).
    # When shake gating is enabled, the goal is to drop the *worst* shaky shots while still
    # allowing dynamic pans/whips. The prior default (0.08) was below the median and over-filtered.
    shake_max = _float_env("OPT_SHAKE_MAX", "0.30")
    # Blur gating: remove extremely blurry shots (uses shot_index sharpness proxy).
    sharp_gate = _truthy_env("OPT_SHARPNESS_GATE", "1")
    sharp_min = _float_env("OPT_SHARPNESS_MIN", "80.0")
    sharp_pct = _float_env("OPT_SHARPNESS_PCT_MIN", "0.0")
    if sharp_gate and sharp_pct > 0.0:
        # Allow percentile-based thresholds per library (e.g., 5 = bottom 5%).
        q = float(sharp_pct)
        if q > 1.0:
            q = q / 100.0
        q = max(0.0, min(0.5, q))
        sharp_vals = [float(_safe_float(s.get("sharpness"))) for s in shots if _safe_float(s.get("sharpness")) is not None]
        if sharp_vals:
            sharp_vals.sort()
            idx = int(round(q * float(len(sharp_vals) - 1)))
            idx = max(0, min(len(sharp_vals) - 1, idx))
            sharp_min = max(float(sharp_min), float(sharp_vals[idx]))
    for s in shots:
        sid = str(s.get("id") or "")
        if not sid:
            continue
        gid = str(s.get("sequence_group_id") or "").strip()
        # Only consider shots long enough for this segment.
        sd = _safe_float(s.get("duration_s"))
        if sd is not None and seg_dur > 0.05 and float(sd) < seg_dur * 0.90:
            continue
        if very_dark and _truthy_env("OPT_HARD_DARK_GATE", "0"):
            # Optional hard gate: for very dark reference reels, allow users to force the
            # candidate pool to be night/low-key only. Default is OFF (soft penalties handle it).
            dmax = _safe_float(s.get("dark_frac_max"))
            if dmax is None:
                dmax = _safe_float(s.get("dark_frac"))
            if dmax is not None and float(dmax) < float(_float_env("OPT_DARK_GATE_MIN", "0.78")):
                continue
        passed_motion = True
        if motion_gate:
            mv = _safe_float(s.get("motion_score"))
            # Only gate when motion is known; keep missing values.
            if mv is not None and float(mv) < float(motion_min):
                passed_motion = False
        if shake_gate:
            sh = _safe_float(s.get("shake_score"))
            # Only apply shake gating when we have an estimate.
            if sh is not None and float(sh) > float(shake_max):
                continue
        if sharp_gate:
            sh = _safe_float(s.get("sharpness"))
            # Only gate when sharpness is known; keep missing values.
            if sh is not None and float(sh) < float(sharp_min):
                continue
        is_pref = bool(pref_set is not None and gid and gid in t.cast(set[str], pref_set))
        if is_pref:
            preferred_all.append(s)
            if passed_motion:
                preferred_gate.append(s)
        else:
            others_all.append(s)
            if passed_motion:
                others_gate.append(s)

    # Auto-relax motion gating if it would collapse the pool (prevents repetition collapse).
    if motion_gate:
        # Aim for a minimally diverse pool; the ranker will still prefer higher-motion shots.
        min_pool = int(max(8, min(int(limit), 12)))
        if (len(preferred_gate) + len(others_gate)) >= int(min_pool):
            preferred = preferred_gate
            others = others_gate
        else:
            preferred = preferred_all
            others = others_all
    else:
        preferred = preferred_all
        others = others_all

    # Optional: tag gate. If the segment expresses desired_tags and we have enough tag-overlapping
    # shots, restrict the candidate pool to those to avoid irrelevant imagery.
    if desired_set and _truthy_env("OPT_TAG_GATE", "1"):
        def _has_overlap(row: dict[str, t.Any]) -> bool:
            try:
                return bool(desired_set.intersection(_tag_set(row)))
            except Exception:
                return False

        pref_hit = [s for s in preferred if _has_overlap(s)]
        other_hit = [s for s in others if _has_overlap(s)]
        hits = len(pref_hit) + len(other_hit)
        min_hits = int(max(8, min(int(limit), _float_env("OPT_TAG_GATE_MIN_HITS", "12"))))
        if hits >= int(min_hits):
            preferred = pref_hit
            others = other_hit

    # Rank preferred first, then fill from others if needed. This keeps story-planner influence
    # without becoming brittle (empty candidate sets cause randomness).
    def _take(scored: list[tuple[float, dict[str, t.Any]]], *, out: list[dict[str, t.Any]], per_asset: dict[str, int]) -> None:
        for _cost, s in scored:
            aid = str(s.get("asset_id") or "")
            if aid:
                c = per_asset.get(aid, 0)
                if c >= int(max_per_asset):
                    continue
                per_asset[aid] = c + 1
            out.append(s)
            if len(out) >= int(limit):
                return

    out: list[dict[str, t.Any]] = []
    per_asset: dict[str, int] = {}

    scored_pref: list[tuple[float, dict[str, t.Any]]] = [(_slot_cost(segment=segment, shot=s), s) for s in preferred]
    scored_pref.sort(key=lambda x: x[0])
    _take(scored_pref, out=out, per_asset=per_asset)

    # Story-planner preferences: by default we treat them as strong-but-soft. We take preferred
    # shots first, then fill remaining candidates from the rest of the library. This prevents the
    # system from getting stuck on a tiny subset if the story planner under-specifies the library.
    if pref_set is not None:
        if strict_pref and out:
            return out
        # Fill remaining slots from non-preferred shots (ranked by cost).
        if others and len(out) < int(limit):
            scored_other: list[tuple[float, dict[str, t.Any]]] = [(_slot_cost(segment=segment, shot=s), s) for s in others]
            scored_other.sort(key=lambda x: x[0])
            _take(scored_other, out=out, per_asset=per_asset)
        return out

    # No story preference: allow additional candidates from the whole library.
    if others:
        scored_other2: list[tuple[float, dict[str, t.Any]]] = [(_slot_cost(segment=segment, shot=s), s) for s in others]
        scored_other2.sort(key=lambda x: x[0])
        _take(scored_other2, out=out, per_asset=per_asset)

    # If still empty (should be rare), relax to best overall.
    if not out:
        scored_all = [(_slot_cost(segment=segment, shot=s), s) for s in shots]
        scored_all.sort(key=lambda x: x[0])
        _take(scored_all, out=out, per_asset=per_asset)

    return out


def _continuity_cost(
    *,
    prev_segment: t.Any,
    segment: t.Any,
    prev_shot: dict[str, t.Any],
    shot: dict[str, t.Any],
    w: float,
) -> float:
    """
    Penalize unnecessary adjacent jumps (lighting) when the reference doesn't jump.
    Keep conservative; we don't want to over-constrain stylistic contrasts.
    """
    if w <= 1e-6:
        return 0.0
    prl = _safe_float(getattr(prev_segment, "ref_luma", None))
    rrl = _safe_float(getattr(segment, "ref_luma", None))
    psl = _safe_float(prev_shot.get("luma_mean"))
    sl = _safe_float(shot.get("luma_mean"))

    prd = _safe_float(getattr(prev_segment, "ref_dark_frac", None))
    rrd = _safe_float(getattr(segment, "ref_dark_frac", None))
    psd = _safe_float(prev_shot.get("dark_frac"))
    sd = _safe_float(shot.get("dark_frac"))

    cost = 0.0
    # If the reference is stable but the output would jump, penalize.
    if prl is not None and rrl is not None and psl is not None and sl is not None:
        ref_jump = abs(float(rrl) - float(prl))
        out_jump = abs(float(sl) - float(psl))
        if ref_jump < 0.06 and out_jump > 0.14:
            cost += (out_jump - 0.14) * 1.2
    if prd is not None and rrd is not None and psd is not None and sd is not None:
        ref_jump = abs(float(rrd) - float(prd))
        out_jump = abs(float(sd) - float(psd))
        if ref_jump < 0.10 and out_jump > 0.22:
            cost += (out_jump - 0.22) * 1.0

    # Color continuity (very mild): avoid huge hue/white-balance jumps when the reference is stable.
    prgb = getattr(prev_segment, "ref_rgb_mean", None)
    rrgb = getattr(segment, "ref_rgb_mean", None)
    psgb = prev_shot.get("rgb_mean")
    sgb = shot.get("rgb_mean")
    if (
        isinstance(prgb, list)
        and isinstance(rrgb, list)
        and isinstance(psgb, list)
        and isinstance(sgb, list)
        and len(prgb) == 3
        and len(rrgb) == 3
        and len(psgb) == 3
        and len(sgb) == 3
    ):
        try:
            ref_jump = sum(abs(float(rrgb[i]) - float(prgb[i])) for i in range(3)) / 3.0
            out_jump = sum(abs(float(sgb[i]) - float(psgb[i])) for i in range(3)) / 3.0
            if ref_jump < 0.05 and out_jump > 0.14:
                cost += (out_jump - 0.14) * 0.8
        except Exception:
            pass

    return float(cost * w)


def _motion_continuity_cost(*, prev_shot: dict[str, t.Any], shot: dict[str, t.Any], mode: str) -> float:
    """
    Pairwise motion-direction continuity signal using dominant camera translation from thumbnails.
    Lower is better.
    """
    a0 = _safe_float(prev_shot.get("cam_motion_angle_deg"))
    a1 = _safe_float(shot.get("cam_motion_angle_deg"))
    m0 = _safe_float(prev_shot.get("cam_motion_mag"))
    m1 = _safe_float(shot.get("cam_motion_mag"))
    if a0 is None or a1 is None or m0 is None or m1 is None:
        return 0.0

    # Ignore near-static camera motion (direction is noisy).
    strength = min(float(m0), float(m1))
    if strength <= 0.008:
        return 0.0

    d = _angle_diff_deg(float(a0), float(a1)) / 180.0  # 0..1

    if mode == "continuity":
        return float(d * 0.42 * strength)
    if mode == "contrast":
        return float(d * 0.14 * strength)
    return float(d * 0.22 * strength)


def optimize_shot_sequence(
    *,
    segments: list[t.Any],
    shots: list[dict[str, t.Any]],
    config: OptimizerConfig | None = None,
) -> tuple[list[dict[str, t.Any]], dict[str, t.Any]]:
    """
    Global sequencing (beam search) to choose a coherent, diverse set of shots.
    Returns (chosen_shots_per_segment, diagnostics).
    """
    if config is None:
        config = OptimizerConfig(
            candidates_per_slot=int(max(12, _float_env("OPT_CANDIDATES_PER_SLOT", "40"))),
            beam_size=int(max(8, _float_env("OPT_BEAM_SIZE", "40"))),
            max_per_asset_in_candidates=int(max(1, _float_env("OPT_MAX_PER_ASSET_IN_CANDS", "2"))),
            max_per_asset_total=int(max(0, _float_env("OPT_MAX_ASSET_USAGE_TOTAL", "0"))),
            max_per_group_total=int(max(0, _float_env("OPT_MAX_GROUP_USAGE_TOTAL", "0"))),
            # Higher default: helps avoid the system \"getting stuck\" on a few similar clips.
            usage_penalty=_float_env("OPT_USAGE_PENALTY", "0.45"),
            continuity_penalty=_float_env("OPT_CONTINUITY_PENALTY", "0.35"),
            prefer_match_over_diversity=_truthy_env("OPT_PREFER_MATCH", "0"),
        )

    if not segments:
        return [], {"error": "no segments"}
    if not shots:
        return [], {"error": "no shots"}

    n_segments = len(segments)
    # Auto caps (only if not explicitly set). These prevent overusing the same source while
    # still allowing short continuity runs.
    max_asset_total = int(getattr(config, "max_per_asset_total", 0) or 0)
    if max_asset_total <= 0:
        max_asset_total = max(1, int(round(n_segments * 0.25)))
    max_group_total = int(getattr(config, "max_per_group_total", 0) or 0)
    if max_group_total <= 0:
        max_group_total = max(1, int(round(n_segments * 0.35)))

    def _transition_mode(seg: t.Any) -> str:
        hint = str(getattr(seg, "transition_hint", "") or "").strip().lower()
        if not hint:
            return "neutral"
        if any(k in hint for k in ("match", "continue", "same", "smooth", "hold", "linger", "cut on action")):
            return "continuity"
        if any(k in hint for k in ("contrast", "hard", "jump", "smash", "whip")):
            return "contrast"
        return "neutral"

    # Precompute candidates for each segment.
    candidates_by_seg: list[list[dict[str, t.Any]]] = []
    for seg in segments:
        cands = shortlist_shots_for_segment(
            segment=seg,
            shots=shots,
            limit=int(config.candidates_per_slot),
            max_per_asset=int(config.max_per_asset_in_candidates),
        )
        # Fallback: if hard gates filtered everything, relax gates and just pick the best matches.
        if not cands:
            scored = [(_slot_cost(segment=seg, shot=s), s) for s in shots]
            scored.sort(key=lambda x: x[0])
            relaxed: list[dict[str, t.Any]] = []
            per_asset: dict[str, int] = {}
            for _c, s in scored:
                aid = str(s.get("asset_id") or "")
                if aid:
                    c = per_asset.get(aid, 0)
                    if c >= int(config.max_per_asset_in_candidates):
                        continue
                    per_asset[aid] = c + 1
                relaxed.append(s)
                if len(relaxed) >= int(config.candidates_per_slot):
                    break
            cands = relaxed
        candidates_by_seg.append(cands)

    def _run_beam(*, asset_cap: int, group_cap: int) -> tuple[list[dict[str, t.Any]] | None, dict[str, t.Any] | None]:
        # Beam search state: (sequence, cost, asset_counts, group_counts)
        beam: list[tuple[list[dict[str, t.Any]], float, dict[str, int], dict[str, int]]] = [([], 0.0, {}, {})]
        repeat_window = int(max(0, _float_env("OPT_REPEAT_WINDOW", "3")))
        repeat_penalty = _float_env("OPT_REPEAT_PENALTY", "0.35")
        repeat_group_scale = _float_env("OPT_REPEAT_GROUP_SCALE", "0.7")
        for i, seg in enumerate(segments):
            new_beam: list[tuple[list[dict[str, t.Any]], float, dict[str, int], dict[str, int]]] = []
            cands = candidates_by_seg[i] or []
            if not cands:
                cands = shots[: int(max(1, min(len(shots), config.candidates_per_slot)))]
            for seq, cost, asset_counts, group_counts in beam:
                prev_shot = seq[-1] if seq else None
                prev_seg = segments[i - 1] if i > 0 else None
                prev_gid = str(prev_shot.get("sequence_group_id") or "") if isinstance(prev_shot, dict) else ""
                mode = _transition_mode(seg)
                for shot in cands:
                    sid = str(shot.get("id") or "")
                    if not sid:
                        continue
                    base = _slot_cost(segment=seg, shot=shot)
                    if base >= 8.0:
                        continue

                    aid = str(shot.get("asset_id") or "")
                    gid = str(shot.get("sequence_group_id") or "")

                    # Hard caps: prevent collapse onto a tiny subset of the library.
                    if aid and int(asset_counts.get(aid, 0)) + 1 > int(asset_cap):
                        continue
                    if gid and int(group_counts.get(gid, 0)) + 1 > int(group_cap):
                        continue

                    usage = 0.0
                    if aid:
                        prev = int(asset_counts.get(aid, 0))
                        usage = float(config.usage_penalty) * float(max(0, (prev + 1) - 1))

                    group_pen = 0.0
                    if gid:
                        prevg = int(group_counts.get(gid, 0))
                        afterg = prevg + 1
                        if afterg >= 2:
                            group_pen = 0.22 * float(afterg - 1)
                        if prevg == 0 and i <= 3:
                            group_pen -= 0.06

                        if prev_gid and gid == prev_gid:
                            if mode == "contrast":
                                group_pen += 0.35
                            elif mode == "continuity":
                                group_pen -= 0.10

                    cont = 0.0
                    if prev_shot is not None and prev_seg is not None:
                        cont = _continuity_cost(prev_segment=prev_seg, segment=seg, prev_shot=prev_shot, shot=shot, w=float(config.continuity_penalty))
                        # Disabled by default to preserve existing optimizer behavior; enable via env
                        # (e.g., OPT_MOTION_CONTINUITY_W=1.0) when you want motion-direction continuity.
                        motion_w = _float_env("OPT_MOTION_CONTINUITY_W", "0.0")
                        if motion_w >= 1e-6:
                            cont += _motion_continuity_cost(prev_shot=prev_shot, shot=shot, mode=mode) * float(motion_w)

                    if not config.prefer_match_over_diversity:
                        if aid and asset_counts.get(aid, 0) >= 1:
                            usage += 0.10

                    rep = 0.0
                    if repeat_window > 0 and seq:
                        recent = seq[-repeat_window:]
                        if aid and any(str(x.get("asset_id") or "") == aid for x in recent):
                            rep += float(repeat_penalty)
                        if gid and any(str(x.get("sequence_group_id") or "") == gid for x in recent):
                            rep += float(repeat_penalty) * float(repeat_group_scale)
                        # Semantic repetition (across different assets) makes edits feel \"looping\".
                        # Keep this conservative so we don't sacrifice reference matching.
                        sem_rep = _float_env("OPT_SEM_REPEAT_PENALTY", "0.18")
                        sem_consec = _float_env("OPT_SEM_CONSEC_PENALTY", "0.35")
                        if mode == "continuity":
                            sem_rep *= 0.20
                            sem_consec *= 0.20
                        elif mode == "contrast":
                            sem_rep *= 1.25
                            sem_consec *= 1.25
                        sem_key = _sem_key_for_shot(shot)
                        if sem_key and any(_sem_key_for_shot(x) == sem_key for x in recent):
                            rep += float(sem_rep)
                        # Allow 2-in-a-row for continuity, discourage 3+.
                        if sem_key and prev_shot is not None and _sem_key_for_shot(prev_shot) == sem_key:
                            run = 2  # prev + current
                            for x in reversed(recent[:-1]):  # exclude prev_shot which is already counted
                                if _sem_key_for_shot(x) == sem_key:
                                    run += 1
                                else:
                                    break
                            if run >= 3:
                                rep += float(sem_consec) * float(run - 2)

                    new_cost = float(cost + base + usage + group_pen + cont + rep)
                    new_seq = list(seq)
                    new_seq.append(shot)
                    new_asset_counts = dict(asset_counts)
                    if aid:
                        new_asset_counts[aid] = int(new_asset_counts.get(aid, 0)) + 1
                    new_group_counts = dict(group_counts)
                    if gid:
                        new_group_counts[gid] = int(new_group_counts.get(gid, 0)) + 1
                    new_beam.append((new_seq, new_cost, new_asset_counts, new_group_counts))

            new_beam.sort(key=lambda x: x[1])
            beam = new_beam[: int(config.beam_size)]

        if not beam:
            return None, None

        # Beam is sorted by cost (we sort + slice each step), but keep this robust.
        best_seq, best_cost, best_assets, best_groups = min(beam, key=lambda x: x[1])
        diag = {
            "optimizer": "beam_search_v2",
            "cost": float(best_cost),
            "beam_size": int(config.beam_size),
            "candidates_per_slot": int(config.candidates_per_slot),
            "usage_penalty": float(config.usage_penalty),
            "continuity_penalty": float(config.continuity_penalty),
            "sem_repeat_penalty": float(_float_env("OPT_SEM_REPEAT_PENALTY", "0.18")),
            "sem_consec_penalty": float(_float_env("OPT_SEM_CONSEC_PENALTY", "0.35")),
            "asset_usage": best_assets,
            "group_usage": best_groups,
            "asset_cap": int(asset_cap),
            "group_cap": int(group_cap),
        }

        # Optional: expose the final beam for downstream steering (e.g., learned grader).
        # Keep it lightweight and JSON-serializable (IDs and costs only).
        export_k = int(max(0, _float_env("OPT_EXPORT_BEAM_TOPK", "0")))
        if export_k > 0:
            # Ensure sorted for deterministic export.
            beam_sorted = sorted(beam, key=lambda x: x[1])
            out_items: list[dict[str, t.Any]] = []
            for rank, (seq0, cost0, assets0, groups0) in enumerate(beam_sorted[:export_k], start=1):
                try:
                    shot_ids = [str(s.get("id") or "") for s in seq0]
                    asset_ids = [str(s.get("asset_id") or "") for s in seq0]
                    group_ids = [str(s.get("sequence_group_id") or "") for s in seq0]
                    sem_keys = [_sem_key_for_shot(s) for s in seq0]
                    out_items.append(
                        {
                            "rank": int(rank),
                            "cost": float(cost0),
                            "shot_ids": shot_ids,
                            "asset_ids": asset_ids,
                            "group_ids": group_ids,
                            "sem_keys": sem_keys,
                            "asset_usage": dict(assets0),
                            "group_usage": dict(groups0),
                        }
                    )
                except Exception:
                    continue
            if out_items:
                diag["beam_final"] = out_items
        return best_seq, diag

    seq, diag = _run_beam(asset_cap=max_asset_total, group_cap=max_group_total)
    if seq is None:
        # Relax caps if the library is tiny or constraints are too strict.
        seq, diag = _run_beam(asset_cap=max(10, n_segments), group_cap=max(10, n_segments))
    if seq is None or diag is None:
        return [], {"error": "no feasible sequence"}
    return seq, diag
