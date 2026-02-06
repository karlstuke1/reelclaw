from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import typing as t

from .av_tools import extract_audio
from .folder_edit_planner import ReferenceAnalysisPlan, ReferenceSegmentPlan, analyze_reference_reel_segments, tag_assets_from_thumbnails
from .media_encoding import encode_image_data_url
from .media_index import index_media_folder, load_index, load_or_build_cached_index, save_index
from .pipeline_output import utc_timestamp, write_json
from .reel_cut_detect import detect_scene_cuts
from .reel_download import compress_for_analysis, download_reel
from .video_tools import merge_audio

# Reuse proven helpers from the main Folder Edit pipeline.
from .folder_edit_pipeline import (  # type: ignore
    _compute_eq_grade,
    _median,
    _median_rgb,
    _dark_frac,
    _rgb_mean,
    _robust_frame_stats,
    _ensure_segment_count,
    _extract_frame,
    _frame_motion_diff,
    _luma_mean,
    _maybe_analyze_music,
    _motion_score_from_thumbs,
    _beat_floor_index,
    _beat_snap_segments,
    _render_segment,
    _segment_music_energy,
    _segment_energy_hint,
    _shortlist_assets_for_segment,
    _smooth_grade_step,
)


def _load_api_key() -> str | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = Path("openrouter")
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


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


def _int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip()
    try:
        return int(float(raw))
    except Exception:
        try:
            return int(float(default))
        except Exception:
            return 0


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class VariantParams:
    tau: float  # selection randomness (higher = more exploration)
    top_n: int  # shortlist size for sampling
    inpoint_top_k: int  # choose among top-k ranked inpoints


@dataclass(frozen=True)
class GeminiDirectorPickResult:
    ok: bool
    picks: list[dict[str, t.Any]]
    error: str | None = None
    # A short snippet of the model output (useful for logs when parsing fails).
    text_snippet: str | None = None
    # Best-effort metadata extracted from the raw response (finish_reason, provider, etc).
    meta: dict[str, t.Any] | None = None


def _pick_weighted(
    candidates: list[dict[str, t.Any]],
    *,
    tau: float,
    rng: random.Random,
    usage: dict[str, int] | None = None,
    usage_penalty: float = 0.0,
) -> dict[str, t.Any]:
    if not candidates:
        raise ValueError("No candidates to pick from")
    # candidates are already sorted best->worst; sample by rank decay.
    weights: list[float] = []
    denom = max(0.25, float(tau))
    up = max(0.0, float(usage_penalty))
    for i, c in enumerate(candidates):
        aid = str(c.get("id") or "")
        u = int(usage.get(aid, 0)) if (usage and aid) else 0
        # Penalize assets that appear in many variants to encourage diversity.
        eff_rank = float(i) + (float(u) * up)
        weights.append(math.exp(-eff_rank / denom))
    return rng.choices(candidates, weights=weights, k=1)[0]


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    inter = a.intersection(b)
    return 1.0 - (len(inter) / len(union))


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


def _blur_lap_var(path: Path) -> float | None:
    """
    Cheap blur proxy: variance of a Laplacian-like response on a small grayscale image.
    Higher is sharper. Keep as a relative score only.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
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
        var = sum((v0 - mean) ** 2 for v0 in vals) / max(1, len(vals) - 1)
        return float(var)
    except Exception:
        return None


def _sem_key_for_shot(shot: dict[str, t.Any]) -> str:
    """
    A coarse semantic signature for within-variant diversity.
    Keep it cheap and general; avoid reel-specific rules.
    """
    st = str(shot.get("shot_type") or "").strip().lower()
    setting = str(shot.get("setting") or "").strip().lower()
    # Some libraries have sparse shot_type/setting; fall back to a couple tags.
    tags = shot.get("tags")
    tag_bits: list[str] = []
    if isinstance(tags, list):
        # Filter out very generic tags so we keep the key informative across niches.
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
            "slow motion",
            "portrait",
            "landscape",
        }
        for raw in tags:
            s = str(raw or "").strip().lower()
            if not s:
                continue
            if s in stop:
                continue
            tag_bits.append(s)
            if len(tag_bits) >= 2:
                break
    # Normalize empty to a stable token so the dict keys remain deterministic.
    key = "|".join([setting or "_", st or "_"] + (tag_bits if tag_bits else ["_"]))
    return key


def _compact_shot_candidate(shot: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Reduce a shot dict to stable, JSON-serializable fields for persistence/debugging.
    Keep this compact so per-project candidate dumps don't explode in size.
    """
    def _f(x: t.Any) -> float | None:
        try:
            return float(x)
        except Exception:
            return None

    def _s(x: t.Any, n: int) -> str | None:
        if not isinstance(x, str):
            return None
        s0 = x.strip()
        if not s0:
            return None
        return s0[:n]

    thumbs = shot.get("thumbnail_paths")
    thumb_list: list[str] = []
    if isinstance(thumbs, list):
        for p in thumbs[:3]:
            if isinstance(p, str) and p:
                thumb_list.append(p)

    tags = shot.get("tags")
    tag_list: list[str] = []
    if isinstance(tags, list):
        for tg in tags[:12]:
            if isinstance(tg, str) and tg.strip():
                tag_list.append(tg.strip().lower())

    rgb = shot.get("rgb_mean")
    rgb_mean = rgb if (isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(x, (int, float)) for x in rgb)) else None

    return {
        "id": str(shot.get("id") or ""),
        "asset_id": str(shot.get("asset_id") or ""),
        "asset_path": _s(shot.get("asset_path"), 320),
        "start_s": _f(shot.get("start_s")),
        "end_s": _f(shot.get("end_s")),
        "duration_s": _f(shot.get("duration_s")),
        "sequence_group_id": _s(shot.get("sequence_group_id"), 96),
        "luma_mean": _f(shot.get("luma_mean")),
        "dark_frac": _f(shot.get("dark_frac")),
        "rgb_mean": rgb_mean,
        "motion_score": _f(shot.get("motion_score")),
        "shake_score": _f(shot.get("shake_score")),
        "sharpness": _f(shot.get("sharpness")),
        "cam_motion_dx": _f(shot.get("cam_motion_dx")),
        "cam_motion_dy": _f(shot.get("cam_motion_dy")),
        "cam_motion_mag": _f(shot.get("cam_motion_mag")),
        "cam_motion_angle_deg": _f(shot.get("cam_motion_angle_deg")),
        "shot_type": _s(shot.get("shot_type"), 64),
        "setting": _s(shot.get("setting"), 64),
        "mood": _s(shot.get("mood"), 64),
        "tags": tag_list,
        "description": _s(shot.get("description"), 240),
        "thumbnail_paths": thumb_list or None,
    }


def _segment_mode(prev_seg: t.Any | None, seg: t.Any) -> str:
    """
    Decide whether the edit *expects* continuity or contrast between prev->seg.
    Used to avoid over-penalizing repetition when the reference itself repeats a setting/shot type.
    """
    try:
        hint = str(getattr(seg, "transition_hint", "") or "").strip().lower()
    except Exception:
        hint = ""
    if hint:
        if any(k in hint for k in ("match", "continue", "same", "smooth", "hold", "linger", "cut on action")):
            return "continuity"
        if any(k in hint for k in ("contrast", "hard", "jump", "smash", "whip")):
            return "contrast"

    if prev_seg is None:
        return "neutral"

    try:
        a = {str(x).strip().lower() for x in (getattr(prev_seg, "desired_tags", None) or []) if str(x).strip()}
        b = {str(x).strip().lower() for x in (getattr(seg, "desired_tags", None) or []) if str(x).strip()}
    except Exception:
        return "neutral"
    if not a or not b:
        return "neutral"
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    j = float(inter) / float(uni) if uni > 0 else 0.0
    cont_thr = _float_env("VARIANT_SEM_CONTINUITY_JACC", "0.35")
    contr_thr = _float_env("VARIANT_SEM_CONTRAST_JACC", "0.12")
    if j >= float(cont_thr):
        return "continuity"
    if j <= float(contr_thr):
        return "contrast"
    return "neutral"


def _load_variant_asset_set(variant_dir: Path) -> set[str]:
    """
    Read variants/v###/timeline.json and return the set of asset_ids used.
    Best-effort (empty set on failure).
    """
    try:
        p = variant_dir / "timeline.json"
        if not p.exists():
            return set()
        doc = json.loads(p.read_text(encoding="utf-8"))
        segs = doc.get("timeline_segments") or []
        if not isinstance(segs, list):
            return set()
        out: set[str] = set()
        for s in segs:
            if not isinstance(s, dict):
                continue
            aid = str(s.get("asset_id") or "").strip()
            if aid:
                out.add(aid)
        return out
    except Exception:
        return set()


def _auto_pick_inpoint(
    *,
    asset_path: Path,
    asset_duration_s: float | None,
    segment_duration_s: float,
    initial_in_s: float,
    ref_luma: float | None,
    ref_dark: float | None,
    energy_hint: float,
    sample_dir: Path,
    timeout_s: float,
    rng: random.Random,
    choose_top_k: int = 1,
) -> tuple[float, dict[str, t.Any]]:
    """
    Code-only in-point selection. Samples several timestamps across the clip,
    ranks by (luma + dark + slight motion/energy), and returns a chosen time.
    """
    seg_dur = float(segment_duration_s)
    dur = float(asset_duration_s) if isinstance(asset_duration_s, (int, float)) and float(asset_duration_s) > 0 else None
    max_start = 0.0
    if dur is not None:
        max_start = max(0.0, dur - seg_dur)

    init_t = max(0.0, float(initial_in_s))
    if max_start > 0.0:
        init_t = min(max_start, init_t)

    sample_count = 14 if max_start > 0.0 else 1
    base_times = [0.0] if sample_count == 1 else [i * max_start / (sample_count - 1) for i in range(sample_count)]
    base_times.append(init_t)

    # De-dupe/normalize.
    times: list[float] = []
    for t_s in base_times:
        t3 = round(float(t_s), 3)
        if t3 < 0.0:
            continue
        if t3 not in times:
            times.append(t3)
    times.sort()

    sample_dir.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[float, Path, float | None, float | None]] = []
    for t_s in times:
        safe = f"{t_s:.3f}".replace(".", "p")
        frame_path = sample_dir / f"t_{safe}.jpg"
        if not frame_path.exists():
            _extract_frame(video_path=asset_path, at_s=float(t_s), out_path=frame_path, timeout_s=timeout_s)
        luma = _luma_mean(frame_path)
        dark = _dark_frac(frame_path)
        samples.append((float(t_s), frame_path, luma, dark))

    # Motion proxy from successive frames.
    motion_by_time: dict[float, float | None] = {}
    for i in range(len(samples)):
        t0, p0, _l0, _d0 = samples[i]
        if i < len(samples) - 1:
            _t1, p1, *_ = samples[i + 1]
            motion_by_time[t0] = _frame_motion_diff(p0, p1)
        elif i > 0:
            _tprev, pprev, *_ = samples[i - 1]
            motion_by_time[t0] = _frame_motion_diff(pprev, p0)
        else:
            motion_by_time[t0] = None

    ranked: list[tuple[float, dict[str, t.Any]]] = []
    for t_s, p, luma, dark in samples:
        dl = 0.18
        dd = 0.25
        if isinstance(ref_luma, (int, float)) and isinstance(luma, (int, float)):
            dl = abs(float(luma) - float(ref_luma))
        if isinstance(ref_dark, (int, float)) and isinstance(dark, (int, float)):
            dd = abs(float(dark) - float(ref_dark))
        m = motion_by_time.get(t_s)
        motion = float(m) if isinstance(m, (int, float)) else 0.35
        score = (dl * 1.2) + (dd * 1.0) + (abs(motion - float(energy_hint)) * 0.25)
        ranked.append(
            (
                float(score),
                {
                    "time_s": float(t_s),
                    "luma": luma,
                    "dark_frac": dark,
                    "motion": motion,
                    "frame_path": str(p),
                },
            )
        )
    ranked.sort(key=lambda x: x[0])

    # Pick deterministically or randomly among the best few.
    k = max(1, min(int(choose_top_k), len(ranked)))
    chosen = ranked[0][1] if k == 1 else rng.choice([it for _s, it in ranked[:k]])
    return float(chosen["time_s"]), {"candidates": [it for _s, it in ranked[: min(7, len(ranked))]], "chosen": chosen}


def _strip_code_fences(text: str) -> str:
    import re

    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\s*```$", "", s)
    return s.strip()


def _extract_json_object(text: str) -> dict[str, t.Any]:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    snippet = cleaned[start : end + 1]
    try:
        return t.cast(dict[str, t.Any], json.loads(snippet))
    except json.JSONDecodeError:
        # Some models accidentally double-escape JSON.
        if '\\"' in snippet or "\\n" in snippet:
            unescaped = snippet.encode("utf-8").decode("unicode_escape")
            return t.cast(dict[str, t.Any], json.loads(unescaped))
        raise


def _reasoning_param() -> dict[str, t.Any]:
    effort_env = os.getenv("REASONING_EFFORT", "").strip().lower()
    if effort_env == "xhigh":
        effort_env = "high"
    if effort_env in {"none", "minimal", "low", "medium", "high"}:
        return {"effort": effort_env}
    return {"effort": "high"}


def _gemini_director_choose_shots(
    *,
    api_key: str,
    model: str,
    segments: list[t.Any],
    candidates_by_seg_id: dict[int, list[dict[str, t.Any]]],
    max_candidates_per_seg: int,
    timeout_s: float,
    site_url: str | None = None,
    app_name: str | None = None,
    debug_dir: Path | None = None,
) -> GeminiDirectorPickResult:
    """
    Experimental: ask Gemini to choose the best shot_id per segment from a curated candidate list.
    Output is compiled into deterministic edit decisions (micro-editor + renderer remain code-only).
    """
    from .openrouter_client import OpenRouterError, chat_completions_budgeted

    # Validation/gating: a "successful" JSON parse can still be editorially meaningless
    # if the model returns an empty/partial choices list and we silently fill the rest
    # with deterministic fallbacks. Benchmarks must treat that as failure.
    min_coverage = float(_float_env("VARIANT_GEMINI_MIN_COVERAGE", "0.90"))
    require_all = _truthy_env("VARIANT_GEMINI_REQUIRE_ALL", "0")
    disallow_invalid_ids = _truthy_env("VARIANT_GEMINI_DISALLOW_INVALID_IDS", "1")

    # Prompt-time stability filter: if stable options exist, only show those candidates.
    # This makes "stability must not drop" easier and shrinks tokens.
    prompt_stable_only = _truthy_env("VARIANT_GEMINI_PROMPT_STABLE_ONLY", "1")
    prompt_shake_gate = float(_float_env("VARIANT_GEMINI_SHAKE_GATE", "0.20"))
    prompt_sharp_min = float(_float_env("VARIANT_GEMINI_SHARP_MIN", "120.0"))

    # Keep director prompts conservative:
    # - omit any text that can trigger safety filters (overlay copy, detailed descriptions)
    # - prefer tags + numeric signals so the director can still reason about match/stability
    def _seg_row(seg: t.Any) -> dict[str, t.Any]:
        def _r(x: t.Any, nd: int) -> float | None:
            v = _safe_float(x)
            if v is None:
                return None
            try:
                return float(round(float(v), int(nd)))
            except Exception:
                return float(v)

        return {
            "id": int(getattr(seg, "id", 0) or 0),
            "duration_s": _r(getattr(seg, "duration_s", 0.0), 3),
            "beat_goal": str(getattr(seg, "beat_goal", "") or ""),
            # Keep compact: long tag lists explode prompt tokens and can cause empty model output.
            "desired_tags": [str(x) for x in (getattr(seg, "desired_tags", []) or [])][:8],
            "ref_luma": _r(getattr(seg, "ref_luma", None), 4),
            "ref_dark_frac": _r(getattr(seg, "ref_dark_frac", None), 4),
            "ref_rgb_mean": (
                [float(round(float(x), 4)) for x in getattr(seg, "ref_rgb_mean", [])[:3]]
                if isinstance(getattr(seg, "ref_rgb_mean", None), list)
                else None
            ),
        }

    def _cand_row(c: dict[str, t.Any]) -> dict[str, t.Any]:
        def _r(x: t.Any, nd: int) -> float | None:
            v = _safe_float(x)
            if v is None:
                return None
            try:
                return float(round(float(v), int(nd)))
            except Exception:
                return float(v)

        def _rgb(v0: t.Any) -> list[float] | None:
            if not isinstance(v0, list) or not v0:
                return None
            out: list[float] = []
            for x in v0[:3]:
                try:
                    out.append(float(round(float(x), 4)))
                except Exception:
                    return None
            return out if out else None

        return {
            "shot_id": str(c.get("id") or ""),
            "asset_id": str(c.get("asset_id") or ""),
            "sequence_group_id": str(c.get("sequence_group_id") or ""),
            "start_s": _r(c.get("start_s"), 3),
            "duration_s": _r(c.get("duration_s"), 3),
            "luma_mean": _r(c.get("luma_mean"), 4),
            "dark_frac": _r(c.get("dark_frac"), 4),
            "rgb_mean": _rgb(c.get("rgb_mean")),
            "motion_score": _r(c.get("motion_score"), 4),
            "shake_score": _r(c.get("shake_score"), 4),
            # Sharpness is a relative proxy; keep numeric but compact.
            "sharpness": _r(c.get("sharpness"), 1),
            # Keep tags short; omit other descriptive strings to avoid prompt blowups.
            "tags": [str(x) for x in (c.get("tags") or [])[:8]] if isinstance(c.get("tags"), list) else [],
        }

    def _build_payload(max_k: int) -> list[dict[str, t.Any]]:
        seg_payload: list[dict[str, t.Any]] = []
        for seg in segments:
            sid = int(getattr(seg, "id", 0) or 0)
            if sid <= 0:
                continue
            cands = candidates_by_seg_id.get(int(sid)) or []
            cands = [c for c in cands if isinstance(c, dict) and str(c.get("id") or "").strip()]
            # Deterministic stability pre-filter (when possible).
            if prompt_stable_only and cands:
                stable: list[dict[str, t.Any]] = []
                for c in cands:
                    sh = _safe_float(c.get("shake_score"))
                    sp = _safe_float(c.get("sharpness"))
                    ok = True
                    if sh is not None and float(sh) > float(prompt_shake_gate):
                        ok = False
                    if sp is not None and float(sp) < float(prompt_sharp_min):
                        ok = False
                    if ok:
                        stable.append(c)
                if stable:
                    cands = stable
            seg_payload.append(
                {
                    "segment": _seg_row(seg),
                    "candidates": [_cand_row(c) for c in cands[: max(1, int(max_k))]],
                }
            )
        return seg_payload

    system_prompt = "\n".join(
        [
            "You are an ELITE short-form video editor.",
            "You must choose the best candidate shot for each segment.",
            "You are NOT allowed to invent shots. You MUST pick from the provided candidate shot_id values.",
            "",
            "Scope:",
            "- Focus ONLY on editing craft (hook, arc, rhythm, continuity, stability, look).",
            "- Do NOT describe explicit content or sensitive personal attributes.",
            "",
            "Goals (in order):",
            "1) Strong hook and escalation arc (hook -> setup -> build -> payoff).",
            "2) Professional watchability: avoid distracting shake/blur when alternatives exist.",
            "3) Visual continuity that feels intentional (use contrast only when it adds impact).",
            "4) Match the reference intent for lighting and tags where possible.",
            "",
            "Rules:",
            "- Prefer stable, sharp shots unless the segment explicitly needs chaotic motion.",
            "- Avoid repeating the same asset_id more than twice across the whole edit if possible.",
            "- Return ONLY strict JSON, no markdown.",
            "",
            "Return JSON with this schema:",
            "{",
            '  \"choices\": [',
            '    {\"segment_id\": 1, \"shot_id\": \"...\"}',
            "  ]",
            "}",
        ]
    ).strip()

    # Shrink prompt on retries if the provider returns empty content (sometimes happens on safety or
    # backend errors despite HTTP 200). This keeps the director reliable for benchmarks.
    ks = [int(max(2, max_candidates_per_seg)), min(6, int(max_candidates_per_seg)), 4]
    seg_payload0 = _build_payload(ks[0])
    # Per-attempt allowed id sets (enforces "pick from the provided candidates").
    allowed_by_seg: dict[int, set[str]] = {}
    for it in seg_payload0:
        if not isinstance(it, dict):
            continue
        seg0 = it.get("segment") if isinstance(it.get("segment"), dict) else {}
        try:
            sid0 = int(seg0.get("id") or 0)
        except Exception:
            sid0 = 0
        if sid0 <= 0:
            continue
        cands0 = it.get("candidates") if isinstance(it.get("candidates"), list) else []
        allowed: set[str] = set()
        for c in cands0:
            if not isinstance(c, dict):
                continue
            shot_id = str(c.get("shot_id") or "").strip()
            if shot_id:
                allowed.add(shot_id)
        allowed_by_seg[int(sid0)] = allowed
    expected_seg_ids = sorted([sid for sid, allowed in allowed_by_seg.items() if allowed])

    user_text0 = json.dumps({"segments": seg_payload0}, ensure_ascii=True)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text0}]

    out: list[dict[str, t.Any]] = []
    last_text = ""
    last_err: Exception | None = None
    last_raw: dict[str, t.Any] | None = None
    last_meta: dict[str, t.Any] | None = None
    parsed_ok = False
    snippet: str | None = None
    for attempt in range(1, 4):
        # On retries, reduce candidate count to keep the input compact.
        if attempt >= 2 and int(ks[min(attempt - 1, len(ks) - 1)]) != int(ks[0]):
            seg_payload_i = _build_payload(int(ks[min(attempt - 1, len(ks) - 1)]))
            user_text_i = json.dumps({"segments": seg_payload_i}, ensure_ascii=True)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text_i}]
            allowed_by_seg = {}
            for it in seg_payload_i:
                if not isinstance(it, dict):
                    continue
                seg0 = it.get("segment") if isinstance(it.get("segment"), dict) else {}
                try:
                    sid0 = int(seg0.get("id") or 0)
                except Exception:
                    sid0 = 0
                if sid0 <= 0:
                    continue
                cands0 = it.get("candidates") if isinstance(it.get("candidates"), list) else []
                allowed: set[str] = set()
                for c in cands0:
                    if not isinstance(c, dict):
                        continue
                    shot_id = str(c.get("shot_id") or "").strip()
                    if shot_id:
                        allowed.add(shot_id)
                allowed_by_seg[int(sid0)] = allowed
            expected_seg_ids = sorted([sid for sid, allowed in allowed_by_seg.items() if allowed])

        result = chat_completions_budgeted(
            api_key=api_key,
            model=str(model),
            messages=messages,
            temperature=0.0,
            max_tokens=int(float(os.getenv("VARIANT_GEMINI_DIRECTOR_MAX_TOKENS", "1800") or 1800)),
            timeout_s=float(timeout_s),
            site_url=site_url,
            app_name=app_name,
            # Director selection must reliably return JSON. Reasoning output can consume
            # the token budget (finish_reason=length) and yield empty content.
            include_reasoning=False,
            reasoning={"effort": "minimal"},
            retries=3,
            retry_delay_s=1.5,
            extra_body={"response_format": {"type": "json_object"}},
        )
        last_text = result.content or ""
        last_raw = result.raw if isinstance(result.raw, dict) else None
        # Best-effort response metadata for logs/debug.
        try:
            fr = None
            msg_keys: list[str] | None = None
            if isinstance(last_raw, dict):
                choices = last_raw.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    fr = choices[0].get("finish_reason")
                    m0 = choices[0].get("message") or choices[0].get("delta") or {}
                    if isinstance(m0, dict):
                        msg_keys = sorted([str(k) for k in m0.keys()])
            last_meta = {
                "finish_reason": fr,
                "message_keys": msg_keys,
                "has_error": bool(isinstance(last_raw, dict) and last_raw.get("error")),
            }
        except Exception:
            last_meta = None

        if not (last_text or "").strip():
            # Empty content is almost always a provider-side issue or safety refusal; retry with
            # a smaller prompt to reduce the chance of silent drops.
            last_err = OpenRouterError(f"Empty model content (meta={last_meta})")
            continue
        try:
            parsed = _extract_json_object(last_text)
            choices_raw = parsed.get("choices")
            if not isinstance(choices_raw, list):
                raise OpenRouterError("Gemini director returned invalid JSON: missing choices[]")

            out = []
            seen_seg_ids: set[int] = set()
            invalid = 0
            for item in choices_raw:
                if not isinstance(item, dict):
                    continue
                try:
                    sid = int(item.get("segment_id") or 0)
                except Exception:
                    continue
                shot_id = str(item.get("shot_id") or "").strip()
                if sid <= 0 or not shot_id:
                    continue
                if expected_seg_ids and int(sid) not in set(expected_seg_ids):
                    # Ignore spurious ids.
                    continue
                if sid in seen_seg_ids:
                    continue
                # Validate the pick was in the provided candidate set for this segment.
                allowed = allowed_by_seg.get(int(sid))
                if disallow_invalid_ids and isinstance(allowed, set) and allowed and shot_id not in allowed:
                    invalid += 1
                    continue
                seen_seg_ids.add(int(sid))
                out.append({"segment_id": int(sid), "shot_id": shot_id})

            # Coverage sanity: if the director didn't meaningfully participate, fail and retry.
            expected_n = int(len(expected_seg_ids))
            got_n = int(len(out))
            if expected_n > 0:
                coverage = float(got_n) / float(expected_n)
            else:
                coverage = 0.0
            if invalid > 0 and disallow_invalid_ids:
                raise OpenRouterError(f"Gemini director returned invalid shot_id values (invalid={invalid}/{expected_n})")
            if require_all and expected_n > 0 and got_n < expected_n:
                raise OpenRouterError(f"Gemini director returned incomplete coverage (got={got_n}, expected={expected_n})")
            if expected_n > 0 and coverage + 1e-9 < float(min_coverage):
                raise OpenRouterError(f"Gemini director coverage too low (got={got_n}, expected={expected_n}, coverage={coverage:.2f})")
            parsed_ok = True
            break
        except Exception as e:
            last_err = e
            if attempt >= 3:
                break
            # Ask for a strict JSON reprint.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages[1]["content"]},
                {
                    "role": "user",
                    "content": "Your last response was not valid JSON. Reprint ONLY strict JSON with {\"choices\":[{\"segment_id\":1,\"shot_id\":\"...\"}]} and nothing else.",
                },
            ]

    if not parsed_ok:
        snippet = (last_text or "").strip()[:800] or None
        if debug_dir is not None:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "gemini_director_last_request.json").write_text(
                    json.dumps({"system": system_prompt, "user": messages[1]["content"]}, indent=2) + "\n",
                    encoding="utf-8",
                )
                if isinstance(last_raw, dict):
                    (debug_dir / "gemini_director_last_response.json").write_text(
                        json.dumps(last_raw, indent=2) + "\n",
                        encoding="utf-8",
                    )
            except Exception:
                pass
        # Degrade gracefully to deterministic fallback instead of crashing the whole job.
        print(
            f"[warn] Gemini director choose_shots failed; falling back to top candidates. {type(last_err).__name__ if last_err else 'Error'}: {last_err}. meta={last_meta}. text={snippet!r}",
            flush=True,
        )

    # Ensure all segments have a choice; fill missing deterministically with best candidate[0].
    by_seg: dict[int, str] = {int(x["segment_id"]): str(x["shot_id"]) for x in out if int(x.get("segment_id") or 0) > 0 and str(x.get("shot_id") or "")}
    for seg in segments:
        sid = int(getattr(seg, "id", 0) or 0)
        if sid <= 0 or sid in by_seg:
            continue
        cands = candidates_by_seg_id.get(int(sid)) or []
        if cands:
            by_seg[int(sid)] = str(cands[0].get("id") or "")

    picks = [{"segment_id": int(sid), "shot_id": str(shot_id)} for sid, shot_id in sorted(by_seg.items(), key=lambda kv: kv[0])]
    err_s = None
    if not parsed_ok and last_err is not None:
        try:
            err_s = f"{type(last_err).__name__}: {last_err}"
        except Exception:
            err_s = "unknown_error"
    return GeminiDirectorPickResult(ok=bool(parsed_ok), picks=picks, error=err_s, text_snippet=snippet, meta=last_meta)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate many Folder Edit variants (no video-gen; cuts from local footage).")
    ap.add_argument("--reel", required=True, help="Instagram reel URL or local .mp4/.mov path")
    ap.add_argument("--folder", required=True, help="Folder containing videos/images to use as footage")
    ap.add_argument("--niche", default=str(os.getenv("FOLDER_EDIT_NICHE", "") or ""), help="Optional niche/topic hint (used by story planning).")
    ap.add_argument("--vibe", default=str(os.getenv("FOLDER_EDIT_VIBE", "") or ""), help="Optional vibe/tonal hint (used by story planning).")
    ap.add_argument("--ref", help="Reference image path (identity anchor; optional)")
    ap.add_argument("--model", default=os.getenv("DIRECTOR_MODEL", "google/gemini-3-pro-preview"))
    ap.add_argument("--variants", type=int, default=100)
    ap.add_argument(
        "--finals",
        type=int,
        help="How many finalists to write into finals/ (default: same as --variants). When < --variants, the system generates more candidates and selects the best finals.",
    )
    ap.add_argument("--pro", action="store_true", help="Use shot-level (pro) variant search.")
    ap.add_argument(
        "--director",
        choices=["code", "gemini", "auto"],
        default=str(os.getenv("VARIANT_DIRECTOR", "code") or "code").strip().lower(),
        help="Variant director: code (deterministic heuristics/optimizer), gemini (LLM chooses shots among candidates; compiled into deterministic edits), or auto (try gemini, fall back to code if gemini fails/gated).",
    )
    ap.add_argument(
        "--pro-macro",
        choices=["sample", "beam"],
        default=str(os.getenv("VARIANT_PRO_MACRO", "sample") or "sample").strip().lower(),
        help="Pro macro planning mode: sample (per-slot sampling) or beam (sample from global optimizer beam).",
    )
    ap.add_argument(
        "--fix-iters",
        type=int,
        default=int(float(os.getenv("VARIANT_FIX_ITERS", "0") or 0)),
        help="Deterministic segment-level fix iterations applied to finalists (0 disables).",
    )
    ap.add_argument(
        "--critic-fix-iters",
        type=int,
        default=int(float(os.getenv("VARIANT_CRITIC_FIX_ITERS", "0") or 0)),
        help="LLM critic-driven fix iterations applied to finalists (0 disables). Uses compare-video critic JSON to deterministically recast/fix worst segments.",
    )
    ap.add_argument(
        "--critic-model",
        default=str(os.getenv("CRITIC_MODEL", "") or "").strip(),
        help="OpenRouter model for compare-video critic (default: CRITIC_MODEL env or fallback-to-pro policy when empty).",
    )
    ap.add_argument("--critic-max-mb", type=float, default=float(os.getenv("CRITIC_MAX_MB", "8.0") or 8.0), help="Max compare proxy size to inline (MB)")
    cg = ap.add_mutually_exclusive_group()
    cg.add_argument("--critic-pro-mode", dest="critic_pro_mode", action="store_true", help="Allow critic to emit pro-only deltas (story_beat / transition_hint)")
    cg.add_argument("--no-critic-pro-mode", dest="critic_pro_mode", action="store_false", help="Disallow pro-only critic deltas even in --pro mode")
    ap.set_defaults(critic_pro_mode=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start", type=int, default=1, help="Start variant index (for resume)")
    ap.add_argument("--out", help="Output project folder (for resume). If omitted, a new one is created.")
    ap.add_argument("--min-scenes", type=int, default=7)
    ap.add_argument("--max-scenes", type=int, default=9)
    ap.add_argument("--no-reel-audio", action="store_true", help="Do not use extracted reel audio")
    ap.add_argument("--audio", help="Optional custom audio file to use instead of reel audio")
    ap.add_argument("--burn-overlays", action="store_true", help="Burn overlay captions into the output video (experimental)")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--tau", type=float, default=float(os.getenv("VARIANT_TAU", "3.0")))
    ap.add_argument("--top-n", type=int, default=int(os.getenv("VARIANT_TOP_N", "28")))
    ap.add_argument("--inpoint-top-k", type=int, default=int(os.getenv("VARIANT_INPOINT_TOP_K", "2")))
    ap.add_argument(
        "--min-jaccard-dist",
        type=float,
        default=float(os.getenv("VARIANT_MIN_JACCARD_DIST", "0.35")),
        help="Minimum Jaccard distance between asset_id sets across variants. 0 disables.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("VARIANT_MAX_RETRIES", "10")),
        help="Resampling attempts per variant to satisfy diversity constraints.",
    )
    ap.add_argument(
        "--usage-penalty",
        type=float,
        default=float(os.getenv("VARIANT_USAGE_PENALTY", "0.35")),
        help="Penalty per prior usage (in rank units) to encourage asset diversity across variants. 0 disables.",
    )
    args = ap.parse_args()

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (env) or ./openrouter file")

    # Pro mode defaults: keep CLI ergonomics in line with the legacy single-edit pro pipeline.
    # These only apply when --pro is passed AND the user hasn't explicitly set env vars.
    if bool(getattr(args, "pro", False)):
        os.environ.setdefault("FOLDER_EDIT_BEAT_SYNC", "1")
        os.environ.setdefault("FOLDER_EDIT_STORY_PLANNER", "1")
        os.environ.setdefault("SHOT_INDEX_MODE", "scene")
        os.environ.setdefault("SHOT_INDEX_WORKERS", "4")
        # Shot-level tagging (VLM on thumbnails) is a key part of editorial intelligence in pro mode.
        # It is capped by SHOT_TAG_MAX to keep local runs bounded on huge libraries.
        os.environ.setdefault("SHOT_TAGGING", "1")
        os.environ.setdefault("SHOT_TAG_MAX", "250")
        os.environ.setdefault("REF_SEGMENT_FRAME_COUNT", "3")
        os.environ.setdefault("REASONING_EFFORT", "high")
        # Variant pro pipeline defaults (directed search + fix loop).
        os.environ.setdefault("VARIANT_PRO_MACRO", "beam")
        os.environ.setdefault("VARIANT_FIX_ITERS", "1")

    variants_n = max(1, int(getattr(args, "variants", 0) or 1))
    finals_n = int(getattr(args, "finals", 0) or 0)
    if getattr(args, "finals", None) is None:
        finals_n = variants_n
    finals_n = max(1, min(int(finals_n), int(variants_n)))

    timeout_s = float(args.timeout)
    burn_overlays = bool(getattr(args, "burn_overlays", False))
    media_folder = Path(args.folder).expanduser().resolve()
    if not media_folder.exists() or not media_folder.is_dir():
        raise SystemExit(f"Media folder not found: {media_folder}")

    # Optional: exclude specific source files from selection (used by benchmarks that reuse a shared
    # library containing the reference reel itself). This does NOT affect deterministic transforms,
    # only candidate eligibility.
    exclude_raw = str(os.getenv("FOLDER_EDIT_EXCLUDE_BASENAMES", "") or "").strip()
    exclude_basenames: set[str] = set()
    if exclude_raw:
        for part in exclude_raw.replace("\n", ",").replace(";", ",").split(","):
            name = part.strip()
            if name:
                exclude_basenames.add(name)

    project_root = Path(args.out).expanduser().resolve() if args.out else (Path("Outputs") / f"VariantSearch_{utc_timestamp()}")
    reference_dir = project_root / "reference"
    library_dir = project_root / "library"
    variants_dir = project_root / "variants"
    finals_dir = project_root / "finals"
    for p in [reference_dir, library_dir, variants_dir, finals_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) Acquire reel + analysis clip + audio (cached on disk for resume).
    candidate_path = Path(args.reel).expanduser()
    analysis_clip = reference_dir / "analysis_clip.mp4"
    src_path_txt = reference_dir / "source_path.txt"
    if src_path_txt.exists():
        src_path = Path(src_path_txt.read_text(encoding="utf-8", errors="replace").strip()).expanduser()
    else:
        if candidate_path.exists():
            src_path = candidate_path.resolve()
        else:
            src_path = download_reel(args.reel, output_dir=reference_dir, timeout_s=timeout_s)
        src_path_txt.write_text(str(src_path), encoding="utf-8")

    if not analysis_clip.exists():
        compress_for_analysis(src_path, dst=analysis_clip, timeout_s=timeout_s)

    reel_audio: Path | None = None
    extracted_audio = reference_dir / "reel_audio.m4a"
    try:
        if not extracted_audio.exists():
            extract_audio(video_path=analysis_clip, output_audio_path=extracted_audio, timeout_s=timeout_s)
        reel_audio = extracted_audio
    except Exception:
        reel_audio = None

    # 2) Detect cuts and extract segment frames (cached for resume).
    beat_sync_desired = _truthy_env("FOLDER_EDIT_BEAT_SYNC", "0")
    music_doc: dict[str, t.Any] | None = None
    beat_sync_used = False
    music_analysis_path = reference_dir / "music_analysis.json"

    segments_path = reference_dir / "segments.json"
    if segments_path.exists():
        seg_doc = json.loads(segments_path.read_text(encoding="utf-8"))
        segments = [(int(x["id"]), float(x["start_s"]), float(x["end_s"])) for x in (seg_doc.get("segments") or [])]
        beat_sync_used = bool(seg_doc.get("beat_sync") or False)
        if beat_sync_used and music_analysis_path.exists():
            try:
                music_doc = json.loads(music_analysis_path.read_text(encoding="utf-8"))
            except Exception:
                music_doc = None
        if beat_sync_desired and not beat_sync_used:
            # Resume-safe behavior: don't change segment timing mid-project.
            # Users can delete segments.json (or use a new --out) to regenerate with beat sync.
            print("[warn] FOLDER_EDIT_BEAT_SYNC=1 but cached segments.json has beat_sync=false; keeping cached segments for resume safety.", flush=True)
    else:
        cut_times, duration_s = detect_scene_cuts(
            video_path=analysis_clip,
            min_scenes=int(args.min_scenes),
            max_scenes=int(args.max_scenes),
            target_scenes=int(args.max_scenes),
            timeout_s=timeout_s,
        )
        segments = _ensure_segment_count(
            cut_times=cut_times,
            duration_s=float(duration_s),
            min_scenes=int(args.min_scenes),
            max_scenes=int(args.max_scenes),
        )

        # Optional: beat/onset snap segment boundaries to match music timing more closely.
        if beat_sync_desired:
            music_src: Path | None = None
            if args.audio:
                music_src = Path(args.audio).expanduser().resolve()
            elif (not args.no_reel_audio) and reel_audio and Path(reel_audio).exists():
                music_src = Path(reel_audio).expanduser().resolve()
            if music_src and music_src.exists():
                music_doc = _maybe_analyze_music(audio_path=music_src, output_dir=reference_dir, timeout_s=min(timeout_s, 240.0))
                if music_doc:
                    beat_sync_used = True
                    segments = _beat_snap_segments(segments=segments, duration_s=float(duration_s), music_doc=music_doc)

        segments_path.write_text(
            json.dumps(
                {
                    "segments": [{"id": sid, "start_s": s, "end_s": e} for sid, s, e in segments],
                    "duration_s": float(duration_s),
                    "beat_sync": bool(beat_sync_used),
                    "music_analysis": str(music_analysis_path) if (beat_sync_used and music_doc) else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    frames_dir = project_root / "reel_segment_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    segment_frames: list[tuple[int, float, float, Path]] = []
    segment_frames_multi: list[tuple[int, float, float, list[tuple[float, Path]]]] = []
    frame_count = int(max(1, min(5, float(os.getenv("REF_SEGMENT_FRAME_COUNT", "3")))))

    def _sample_times(start_s: float, end_s: float, n: int) -> list[float]:
        if n <= 1:
            return [float((start_s + end_s) / 2.0)]
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
        sample_paths: list[tuple[float, Path]] = []
        for j, t_s in enumerate(_sample_times(float(start_s), float(end_s), frame_count), start=1):
            if abs(float(t_s) - float(mid)) <= 0.02:
                sample_paths.append((float(t_s), frame_path))
                continue
            p = frames_dir / f"seg_{seg_id:02d}_f{j:02d}.jpg"
            _extract_frame(video_path=analysis_clip, at_s=float(t_s), out_path=p, timeout_s=timeout_s)
            sample_paths.append((float(t_s), p))
        segment_frames_multi.append((seg_id, start_s, end_s, sample_paths))
    ref_frame_by_id = {sid: p for sid, _s, _e, p in segment_frames}

    # 3) Analyze reference reel once (AI) (cached for resume).
    reference_data_url = encode_image_data_url(Path(args.ref)) if args.ref else None
    ref_analysis_path = reference_dir / "ref_analysis.json"
    if ref_analysis_path.exists():
        raw = json.loads(ref_analysis_path.read_text(encoding="utf-8"))
        analysis = raw.get("analysis") or {}
        segs = raw.get("segments") or []
        ref_segments: list[ReferenceSegmentPlan] = []
        for item in segs:
            if not isinstance(item, dict):
                continue
            ref_segments.append(
                ReferenceSegmentPlan(
                    id=int(item.get("id") or 0),
                    start_s=float(item.get("start_s") or 0.0),
                    end_s=float(item.get("end_s") or 0.0),
                    duration_s=float(item.get("duration_s") or 0.0),
                    beat_goal=str(item.get("beat_goal") or "setup"),
                    overlay_text=str(item.get("overlay_text") or "").strip(),
                    reference_visual=str(item.get("reference_visual") or "").strip(),
                    desired_tags=[str(x) for x in (item.get("desired_tags") or [])][:16],
                    ref_luma=_safe_float(item.get("ref_luma")),
                    ref_dark_frac=_safe_float(item.get("ref_dark_frac")),
                    ref_rgb_mean=(item.get("ref_rgb_mean") if isinstance(item.get("ref_rgb_mean"), list) else None),
                    music_energy=_safe_float(item.get("music_energy")),
                    start_beat=(int(item.get("start_beat")) if isinstance(item.get("start_beat"), int) else None),
                    end_beat=(int(item.get("end_beat")) if isinstance(item.get("end_beat"), int) else None),
                )
            )
        ref_analysis = ReferenceAnalysisPlan(analysis=analysis, segments=sorted(ref_segments, key=lambda s: s.id), raw=raw)
    else:
        # Attach luma/dark/rgb metrics to segments (used for both LLM and offline fallback).
        # Multi-frame medians are more stable than single-frame midpoints, and reduce grade flicker.
        use_multi_ref = _truthy_env("FOLDER_EDIT_REF_METRICS_MULTI", "1")
        seg_luma: dict[int, float | None] = {}
        seg_dark: dict[int, float | None] = {}
        seg_rgb: dict[int, list[float] | None] = {}
        if use_multi_ref and segment_frames_multi:
            for sid, _s, _e, paths0 in segment_frames_multi:
                fps: list[Path] = []
                for _t, p in (paths0 or []):
                    try:
                        pp = Path(p).expanduser()
                    except Exception:
                        continue
                    if pp.exists():
                        fps.append(pp)
                stats = _robust_frame_stats(fps)
                l0 = stats.get("luma")
                d0 = stats.get("dark_frac")
                seg_luma[int(sid)] = float(l0) if isinstance(l0, (int, float)) else None
                seg_dark[int(sid)] = float(d0) if isinstance(d0, (int, float)) else None
                seg_rgb[int(sid)] = (stats.get("rgb_mean") if isinstance(stats.get("rgb_mean"), list) else None)
        else:
            seg_luma = {sid: _luma_mean(p) for sid, _s, _e, p in segment_frames}
            seg_dark = {sid: _dark_frac(p) for sid, _s, _e, p in segment_frames}
            seg_rgb = {sid: _rgb_mean(p) for sid, _s, _e, p in segment_frames}
        beats = music_doc.get("beat_times") if isinstance(music_doc, dict) else None
        if not isinstance(beats, list):
            beats = []

        def _fallback_beat_goal(i: int, n: int) -> str:
            if i <= 1:
                return "hook"
            if i == n:
                return "payoff"
            if i == n - 1:
                return "cta"
            if i == 2:
                return "setup"
            return "escalation"

        ref_analysis0 = None
        llm_err: Exception | None = None
        try:
            ref_analysis0 = analyze_reference_reel_segments(
                api_key=api_key,
                model=args.model,
                segment_frames=t.cast(list[tuple[int, float, float, t.Any]], segment_frames_multi) if segment_frames_multi else segment_frames,
                reference_image_data_url=reference_data_url,
                timeout_s=timeout_s,
            )
        except Exception as e:
            llm_err = e
            ref_analysis0 = None
            print(f"[warn] reference analysis failed ({type(e).__name__}); using offline fallback. {e}", flush=True)

        seg_n = max(1, len(segments))
        llm_by_id: dict[int, t.Any] = {}
        llm_analysis: dict[str, t.Any] = {}
        llm_raw: dict[str, t.Any] = {}
        if ref_analysis0 is not None:
            try:
                llm_by_id = {int(s.id): s for s in (ref_analysis0.segments or [])}
                llm_analysis = t.cast(dict[str, t.Any], ref_analysis0.analysis or {})
                llm_raw = t.cast(dict[str, t.Any], ref_analysis0.raw or {})
            except Exception:
                llm_by_id = {}
                llm_analysis = {}
                llm_raw = {}

        ref_segments_out: list[ReferenceSegmentPlan] = []
        for idx, (sid, start_s, end_s) in enumerate(segments, start=1):
            llm_seg = llm_by_id.get(int(sid))
            beat_goal = str(getattr(llm_seg, "beat_goal", "") or "").strip() if llm_seg is not None else ""
            overlay_text = str(getattr(llm_seg, "overlay_text", "") or "").strip() if llm_seg is not None else ""
            reference_visual = str(getattr(llm_seg, "reference_visual", "") or "").strip() if llm_seg is not None else ""
            desired_tags = list(getattr(llm_seg, "desired_tags", None) or []) if llm_seg is not None else []
            beat_goal = beat_goal or _fallback_beat_goal(idx, seg_n)
            # Keep tags small and generic in offline fallback; the optimizer works without them.
            desired_tags = [str(x) for x in desired_tags if str(x).strip()][:16]

            ref_segments_out.append(
                ReferenceSegmentPlan(
                    id=int(sid),
                    start_s=float(start_s),
                    end_s=float(end_s),
                    duration_s=float(end_s - start_s),
                    beat_goal=beat_goal,
                    overlay_text=overlay_text,
                    reference_visual=reference_visual,
                    desired_tags=desired_tags,
                    ref_luma=seg_luma.get(int(sid)),
                    ref_dark_frac=seg_dark.get(int(sid)),
                    ref_rgb_mean=seg_rgb.get(int(sid)),
                    music_energy=_segment_music_energy(start_s=float(start_s), end_s=float(end_s), beats=t.cast(list[dict[str, t.Any]], beats)),
                    start_beat=_beat_floor_index(float(start_s), t.cast(list[dict[str, t.Any]], beats)),
                    end_beat=_beat_floor_index(float(end_s), t.cast(list[dict[str, t.Any]], beats)),
                )
            )

        if ref_analysis0 is not None:
            ref_analysis = ReferenceAnalysisPlan(analysis=llm_analysis, raw=llm_raw, segments=ref_segments_out)
        else:
            ref_analysis = ReferenceAnalysisPlan(
                analysis={
                    "summary": "offline_fallback_reference_analysis",
                    "what_is_happening": "",
                    "why_it_works": "",
                    "caption_style": "",
                    "pacing_notes": "",
                },
                raw={"ok": False, "fallback": True, "error": (f"{type(llm_err).__name__}: {llm_err}" if llm_err else "unknown_error")},
                segments=ref_segments_out,
            )

        # Persist a resume-friendly JSON.
        ref_analysis_path.write_text(
            json.dumps(
                {
                    "analysis": ref_analysis.analysis,
                    "segments": [
                        {
                            "id": s.id,
                            "start_s": s.start_s,
                            "end_s": s.end_s,
                            "duration_s": s.duration_s,
                            "beat_goal": s.beat_goal,
                            "overlay_text": s.overlay_text,
                            "reference_visual": s.reference_visual,
                            "desired_tags": s.desired_tags,
                            "ref_luma": s.ref_luma,
                            "ref_dark_frac": s.ref_dark_frac,
                            "ref_rgb_mean": s.ref_rgb_mean,
                            "music_energy": s.music_energy,
                            "start_beat": s.start_beat,
                            "end_beat": s.end_beat,
                        }
                        for s in ref_analysis.segments
                    ],
                    "raw": ref_analysis.raw,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # 4) Index + tag media folder once.
    index_path = library_dir / "media_index.json"
    use_cache = _truthy_env("FOLDER_EDIT_MEDIA_INDEX_CACHE", "1")
    refresh_cache = _truthy_env("FOLDER_EDIT_MEDIA_INDEX_REFRESH", "0")
    cache_root = Path(os.getenv("FOLDER_EDIT_MEDIA_INDEX_CACHE_ROOT", str(Path("Outputs") / "_media_index_cache"))).expanduser().resolve()

    if index_path.exists():
        index = load_index(index_path)
    elif use_cache:
        index, cached_index_path = load_or_build_cached_index(folder=media_folder, cache_root=cache_root, timeout_s=timeout_s, refresh=refresh_cache)
        # Make the project resume-friendly while avoiding re-ffprobe + re-thumbnailing for every project.
        try:
            index_path.symlink_to(cached_index_path)
        except Exception:
            try:
                shutil.copyfile(cached_index_path, index_path)
            except Exception:
                pass
        cached_thumbs = cached_index_path.parent / "asset_thumbs"
        local_thumbs = library_dir / "asset_thumbs"
        if cached_thumbs.exists() and not local_thumbs.exists():
            try:
                local_thumbs.symlink_to(cached_thumbs)
            except Exception:
                pass
    else:
        index = index_media_folder(folder=media_folder, output_dir=library_dir, timeout_s=timeout_s)
        save_index(index, index_path)

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

    TAG_CACHE_VERSION = 2
    global_tagged_path = Path(os.getenv("FOLDER_EDIT_TAG_CACHE_PATH", str(Path("Outputs") / "_asset_tag_cache.json"))).expanduser().resolve()
    local_tagged_path = library_dir / "media_index_tagged.json"
    tags: dict[str, t.Any] = {}
    need_write_global = False

    if global_tagged_path.exists():
        try:
            cached = json.loads(global_tagged_path.read_text(encoding="utf-8"))
            cached_tags = cached.get("tags") or {}
            if isinstance(cached_tags, dict):
                tags.update(cached_tags)
            v = cached.get("version", None)
            if v is None:
                need_write_global = True
            else:
                try:
                    if int(v) != TAG_CACHE_VERSION:
                        need_write_global = True
                except Exception:
                    need_write_global = True
        except Exception:
            pass

    if local_tagged_path.exists():
        try:
            local_doc = json.loads(local_tagged_path.read_text(encoding="utf-8"))
            local_tags = local_doc.get("tags") or {}
            if isinstance(local_tags, dict):
                tags.update(local_tags)
        except Exception:
            pass

    # Asset-level thumbnail tagging is cheap editorial intelligence that the pro pipeline
    # can inherit into its shot index (especially when SHOT_TAG_MAX caps shot-level tagging).
    do_asset_tag = bool(api_key and str(args.model or "").strip()) and _truthy_env("FOLDER_EDIT_ASSET_TAGGING", "1")
    if do_asset_tag:
        # Deterministic cap: align tagging coverage with shot-index capping so pro benchmarks
        # don't accidentally trigger hundreds of VLM calls on huge libraries.
        max_videos = 0
        try:
            max_videos = int(float(os.getenv("SHOT_INDEX_MAX_VIDEOS", "0") or 0))
        except Exception:
            max_videos = 0
        max_videos = int(max(0, max_videos))

        # Optional explicit cap (0 = no extra cap beyond SHOT_INDEX_MAX_VIDEOS filtering).
        max_tag = None
        raw_max = str(os.getenv("FOLDER_EDIT_ASSET_TAG_MAX", "") or "").strip()
        if raw_max:
            try:
                mv = int(float(raw_max))
                if mv > 0:
                    max_tag = int(mv)
            except Exception:
                max_tag = None
        if max_tag is None and bool(args.pro):
            # Safe default for pro mode: tag only the shot-index working set (or a small bound).
            max_tag = int(max_videos) if max_videos > 0 else 120

        # Tag only video assets, deterministically ordered by path (matches shot-index cap behavior).
        assets_sorted = [a for a in assets_for_tagger if str(a.get("id") or "").strip() and str(a.get("kind") or "").strip() == "video"]
        assets_sorted.sort(key=lambda a: str(a.get("path") or ""))
        if max_videos > 0 and len(assets_sorted) > max_videos:
            assets_sorted = assets_sorted[: int(max_videos)]

        missing = [a for a in assets_sorted if str(a.get("id") or "") and str(a["id"]) not in tags]
        if max_tag is not None and max_tag > 0:
            missing = missing[: int(max_tag)]

        if missing:
            tag_model = str(os.getenv("FOLDER_EDIT_ASSET_TAG_MODEL", "") or "").strip() or str(args.model)
            new_tags = tag_assets_from_thumbnails(api_key=api_key, model=tag_model, assets=missing, timeout_s=timeout_s)
            for aid, tinfo in new_tags.items():
                tags[str(aid)] = {
                    "description": tinfo.description,
                    "tags": tinfo.tags,
                    "shot_type": tinfo.shot_type,
                    "setting": tinfo.setting,
                    "mood": tinfo.mood,
                }
            local_tagged_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            need_write_global = True

        if tags and (need_write_global or not global_tagged_path.exists()):
            try:
                global_tagged_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = global_tagged_path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
                tmp.replace(global_tagged_path)
            except Exception:
                pass

    # Planner asset metas.
    assets_for_planner: list[dict[str, t.Any]] = []
    asset_by_id: dict[str, dict[str, t.Any]] = {}
    for a in index.assets:
        try:
            if exclude_basenames and Path(str(a.path)).name in exclude_basenames:
                continue
        except Exception:
            pass
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

        lumas: list[float] = []
        darks: list[float] = []
        for p in thumb_paths:
            l = _luma_mean(p)
            d = _dark_frac(p)
            if isinstance(l, (int, float)):
                lumas.append(float(l))
            if isinstance(d, (int, float)):
                darks.append(float(d))
        meta = {
            "id": a.id,
            "kind": a.kind,
            "path": a.path,
            "duration_s": a.duration_s,
            "luma_mean": (sum(lumas) / len(lumas)) if lumas else None,
            "luma_min": min(lumas) if lumas else None,
            "luma_max": max(lumas) if lumas else None,
            "dark_frac": (sum(darks) / len(darks)) if darks else None,
            "dark_frac_min": min(darks) if darks else None,
            "dark_frac_max": max(darks) if darks else None,
            "motion_score": _motion_score_from_thumbs(thumb_paths) if thumb_paths else None,
        }
        if a.id in tags:
            meta.update(tags[a.id])
        assets_for_planner.append(meta)
        asset_by_id[a.id] = meta

    # 5) Generate variants.
    params = VariantParams(tau=float(args.tau), top_n=int(args.top_n), inpoint_top_k=int(args.inpoint_top_k))
    use_auto_grade = _truthy_env("FOLDER_EDIT_AUTO_GRADE", "1")
    index_header = "variant_id\tseed\tmin_jaccard_dist\tuniq_assets\tavg_abs_luma_diff\tavg_abs_dark_diff\tfinal_video"

    # Pro mode: build/load a shot index once and precompute per-segment shot candidates.
    pro_mode = bool(args.pro)
    director_mode = str(getattr(args, "director", "code") or "code").strip().lower()
    pro_macro_mode = str(getattr(args, "pro_macro", "sample") or "sample").strip().lower()
    if director_mode not in {"code", "gemini", "auto"}:
        director_mode = "code"
    if pro_macro_mode not in {"sample", "beam"}:
        pro_macro_mode = "sample"
    if director_mode in {"gemini", "auto"} and not pro_mode:
        if director_mode == "gemini":
            raise SystemExit("--director=gemini requires --pro (shot candidates).")
        # Auto mode without --pro is identical to code mode; keep UX forgiving.
        director_mode = "code"
    if director_mode in {"gemini", "auto"} and variants_n > 12 and not _truthy_env("ALLOW_GEMINI_DIRECTOR_MANY", "0"):
        raise SystemExit("--director=gemini/auto is expensive. Limit --variants to <=12 or set ALLOW_GEMINI_DIRECTOR_MANY=1.")
    shot_index_obj = None
    shots: list[dict[str, t.Any]] = []
    shot_by_id: dict[str, dict[str, t.Any]] = {}
    shot_candidates_by_seg: dict[int, list[dict[str, t.Any]]] = {}
    # Optional: story planner produces multiple coherent plans (constraints) shared across variants.
    story_plans: list[t.Any] = []
    story_segments_by_plan: dict[str, list[t.Any]] = {}
    shot_candidates_by_seg_by_plan: dict[str, dict[int, list[dict[str, t.Any]]]] = {}
    beam_sequences_by_plan: dict[str, list[dict[str, t.Any]]] = {}
    if pro_mode:
        from .shot_index import build_or_load_shot_index
        from .edit_optimizer import shortlist_shots_for_segment

        shot_index_obj = build_or_load_shot_index(
            media_index_path=index_path,
            cache_dir=(Path("Outputs") / "_shot_index_cache"),
            api_key=api_key,
            model=args.model,
            timeout_s=timeout_s,
        )
        shots = list(shot_index_obj.shots or [])
        shot_by_id = {str(s.get("id") or ""): s for s in shots if str(s.get("id") or "")}
        if exclude_basenames:
            shots = [s for s in shots if Path(str(s.get("asset_path") or "")).name not in exclude_basenames]
            shot_by_id = {str(s.get("id") or ""): s for s in shots if str(s.get("id") or "")}

        # Optional: story planner (agentic storyline over the library).
        if _truthy_env("FOLDER_EDIT_STORY_PLANNER", "0"):
            try:
                from .story_planner import load_story_plans, plan_story_plans, save_story_plans, summarize_sequence_groups

                story_path = reference_dir / "story_plans.json"
                refresh = _truthy_env("FOLDER_EDIT_STORY_REFRESH", "0")
                story_plans = [] if refresh else (load_story_plans(story_path) if story_path.exists() else [])
                if not story_plans:
                    group_summaries = summarize_sequence_groups(shots=shots)
                    story_plans = plan_story_plans(
                        api_key=api_key,
                        model=str(args.model),
                        segments=list(ref_analysis.segments),
                        group_summaries=group_summaries,
                        niche=str(getattr(args, "niche", "") or ""),
                        vibe=str(getattr(args, "vibe", "") or ""),
                        music_doc=music_doc,
                        timeout_s=min(timeout_s, 240.0),
                        num_plans=3,
                    )
                    if story_plans:
                        save_story_plans(story_plans, story_path)

                # Build plan-specific segment lists and candidate lists.
                for p in story_plans:
                    pid = str(getattr(p, "plan_id", "") or "").strip() or "A"
                    by_id = {int(s.id): s for s in (getattr(p, "segments", []) or [])}
                    segs_new: list[ReferenceSegmentPlan] = []
                    for seg in ref_analysis.segments:
                        sp = by_id.get(int(seg.id))
                        if sp:
                            segs_new.append(
                                ReferenceSegmentPlan(
                                    id=seg.id,
                                    start_s=seg.start_s,
                                    end_s=seg.end_s,
                                    duration_s=seg.duration_s,
                                    beat_goal=seg.beat_goal,
                                    overlay_text=seg.overlay_text,
                                    reference_visual=seg.reference_visual,
                                    desired_tags=list(getattr(sp, "desired_tags", None) or seg.desired_tags),
                                    ref_luma=seg.ref_luma,
                                    ref_dark_frac=seg.ref_dark_frac,
                                    ref_rgb_mean=getattr(seg, "ref_rgb_mean", None),
                                    music_energy=getattr(seg, "music_energy", None),
                                    start_beat=getattr(seg, "start_beat", None),
                                    end_beat=getattr(seg, "end_beat", None),
                                    story_beat=str(getattr(sp, "story_beat", "") or "").strip() or None,
                                    preferred_sequence_group_ids=list(getattr(sp, "preferred_sequence_group_ids", None) or []),
                                    transition_hint=str(getattr(sp, "transition_hint", "") or "").strip() or None,
                                )
                            )
                        else:
                            segs_new.append(seg)
                    story_segments_by_plan[pid] = segs_new
                    shot_candidates_by_seg_by_plan[pid] = {}
                    for seg in segs_new:
                        shot_candidates_by_seg_by_plan[pid][int(seg.id)] = shortlist_shots_for_segment(
                            segment=seg,
                            shots=shots,
                            limit=max(int(params.top_n), 40),
                            max_per_asset=2,
                        )
            except Exception:
                story_plans = []
                story_segments_by_plan = {}
                shot_candidates_by_seg_by_plan = {}

        # Default candidates (no story constraints).
        for seg in ref_analysis.segments:
            shot_candidates_by_seg[int(seg.id)] = shortlist_shots_for_segment(
                segment=seg,
                shots=shots,
                limit=max(int(params.top_n), 40),
                max_per_asset=2,
            )

        # Persist segment-level candidate alternatives so later "fix loops" and graders can
        # deterministically recast specific failing segments without re-indexing.
        try:
            plans_doc: dict[str, t.Any] = {
                "default": {
                    "segments": {
                        str(int(seg.id)): [_compact_shot_candidate(s) for s in (shot_candidates_by_seg.get(int(seg.id)) or [])[:80]]
                        for seg in ref_analysis.segments
                    }
                }
            }
            for pid, by_seg in (shot_candidates_by_seg_by_plan or {}).items():
                plans_doc[str(pid)] = {
                    "segments": {
                        str(int(seg.id)): [_compact_shot_candidate(s) for s in (by_seg.get(int(seg.id)) or [])[:80]]
                        for seg in (story_segments_by_plan.get(str(pid)) or ref_analysis.segments)
                    }
                }
            write_json(
                project_root / "segment_candidates.json",
                {
                    "schema_version": 1,
                    "project_root": str(project_root),
                    "pro_mode": True,
                    "generated_at": utc_timestamp(),
                    "plans": plans_doc,
                },
            )
        except Exception:
            pass

        # Optional: precompute a global optimizer beam per plan for higher-coherence variants.
        # This makes the iOS pipeline match or exceed the single-edit pro pipeline (global sequencing),
        # while still allowing diverse variants by sampling from the beam.
        if pro_macro_mode == "beam":
            try:
                from .edit_optimizer import optimize_shot_sequence

                export_k = int(max(4, min(120, float(os.getenv("VARIANT_BEAM_TOPK", "40") or 40))))
                beam_size = int(max(16, min(240, float(os.getenv("VARIANT_BEAM_SIZE", str(max(40, export_k))) or max(40, export_k)))))
                cand_per_slot = int(max(16, min(120, float(os.getenv("VARIANT_BEAM_CANDS_PER_SLOT", str(max(int(params.top_n), 40))) or max(int(params.top_n), 40)))))

                # Temporarily override optimizer env for this precompute.
                keys = {
                    "OPT_EXPORT_BEAM_TOPK": str(export_k),
                    "OPT_BEAM_SIZE": str(beam_size),
                    "OPT_CANDIDATES_PER_SLOT": str(cand_per_slot),
                }
                old_env = {k: os.environ.get(k) for k in keys}
                os.environ.update(keys)
                try:
                    # Default (no story constraints).
                    seq, diag = optimize_shot_sequence(segments=list(ref_analysis.segments), shots=shots)
                    beam_items = diag.get("beam_final") if isinstance(diag, dict) else None
                    if not isinstance(beam_items, list) or not beam_items:
                        beam_items = [
                            {
                                "rank": 1,
                                "cost": float(diag.get("cost") or 0.0) if isinstance(diag, dict) else 0.0,
                                "shot_ids": [str(s.get("id") or "") for s in (seq or [])],
                                "asset_ids": [str(s.get("asset_id") or "") for s in (seq or [])],
                            }
                        ]
                    beam_sequences_by_plan["default"] = t.cast(list[dict[str, t.Any]], beam_items)

                    # Story plans (if any).
                    for pid, segs_new in (story_segments_by_plan or {}).items():
                        if not segs_new:
                            continue
                        seq2, diag2 = optimize_shot_sequence(segments=list(segs_new), shots=shots)
                        beam2 = diag2.get("beam_final") if isinstance(diag2, dict) else None
                        if not isinstance(beam2, list) or not beam2:
                            beam2 = [
                                {
                                    "rank": 1,
                                    "cost": float(diag2.get("cost") or 0.0) if isinstance(diag2, dict) else 0.0,
                                    "shot_ids": [str(s.get("id") or "") for s in (seq2 or [])],
                                    "asset_ids": [str(s.get("asset_id") or "") for s in (seq2 or [])],
                                }
                            ]
                        beam_sequences_by_plan[str(pid)] = t.cast(list[dict[str, t.Any]], beam2)
                finally:
                    for k, prev in old_env.items():
                        if prev is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = prev

                try:
                    write_json(
                        project_root / "beam_sequences.json",
                        {
                            "schema_version": 1,
                            "generated_at": utc_timestamp(),
                            "plans": {k: v[: min(len(v), export_k)] for k, v in beam_sequences_by_plan.items()},
                            "config": {"beam_topk": export_k, "beam_size": beam_size, "candidates_per_slot": cand_per_slot},
                        },
                    )
                except Exception:
                    pass
            except Exception:
                beam_sequences_by_plan = {}

    # Choose audio.
    final_audio: Path | None = None
    if args.audio:
        final_audio = Path(args.audio).expanduser().resolve()
    elif (not args.no_reel_audio) and reel_audio:
        final_audio = reel_audio

    start = max(1, int(args.start))
    end = start + int(variants_n) - 1
    index_path = project_root / "index.tsv"

    # Load existing variant asset sets (for resume) so we can enforce diversity.
    prior_sets: list[set[str]] = []
    asset_usage: dict[str, int] = {}
    try:
        for vdir in sorted([p for p in variants_dir.iterdir() if p.is_dir() and p.name.startswith("v")]):
            aset = _load_variant_asset_set(vdir)
            if not aset:
                continue
            prior_sets.append(aset)
            for aid in aset:
                asset_usage[aid] = asset_usage.get(aid, 0) + 1
    except Exception:
        prior_sets = []
        asset_usage = {}

    min_jaccard_dist = max(0.0, float(args.min_jaccard_dist))
    max_retries = max(1, int(args.max_retries))
    usage_penalty = max(0.0, float(args.usage_penalty))

    for vidx in range(start, end + 1):
        base_seed = int(args.seed) + vidx
        variant_id = f"v{vidx:03d}"
        vroot = variants_dir / variant_id
        final_video = vroot / "final_video.mov"
        if final_video.exists() and (vroot / "timeline.json").exists():
            if vidx % 10 == 0:
                print(f"[{vidx-start+1}/{variants_n}] skip existing {final_video}")
            continue
        vroot.mkdir(parents=True, exist_ok=True)

        # Optional: pick a story plan (coherent storyline) for this variant.
        plan_id: str | None = None
        plan_concept: str | None = None
        segments_for_variant = list(ref_analysis.segments)
        if pro_mode and story_segments_by_plan:
            plan_ids = sorted(story_segments_by_plan.keys())
            if plan_ids:
                plan_id = plan_ids[(vidx - start) % len(plan_ids)]
                segments_for_variant = list(story_segments_by_plan.get(plan_id) or segments_for_variant)
                # Best-effort concept lookup (for manifest/debug).
                for p in story_plans or []:
                    if str(getattr(p, "plan_id", "") or "") == str(plan_id):
                        plan_concept = str(getattr(p, "concept", "") or "").strip() or None
                        break

        # Plan: code-only sampling from the per-segment shortlist (still uses AI-derived desired_tags).
        # To make variants meaningfully different, we:
        # - penalize assets used many times across variants (usage_penalty)
        # - resample if the asset_id set is too similar to previous variants (min_jaccard_dist)
        best: tuple[float, list[dict[str, t.Any]], set[str]] | None = None  # (min_dist, decisions, aset)
        for attempt in range(max_retries):
            rng = random.Random((base_seed * 1000) + attempt)
            used: set[str] = set()
            used_shots: set[str] = set()
            decisions_try: list[dict[str, t.Any]] = []
            # Within-variant cap to prevent the same source clip dominating an edit.
            max_per_asset = max(1, _int_env("VARIANT_MAX_SEGMENTS_PER_ASSET", "2"))
            asset_counts: dict[str, int] = {}
            if pro_mode and max_per_asset > 1 and _truthy_env("VARIANT_MAX_SEGMENTS_PER_ASSET_AUTO", "1"):
                # If we have a rich candidate pool, prefer not repeating the same source clip at all.
                # This is a soft cap: downstream selection will still reuse when the library is small.
                try:
                    if plan_id and plan_id in (shot_candidates_by_seg_by_plan or {}):
                        cand_map0 = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan or {}).get(plan_id) or {})
                    else:
                        cand_map0 = t.cast(dict[int, list[dict[str, t.Any]]], shot_candidates_by_seg)
                    uniq_assets: set[str] = set()
                    per_seg_k = max(3, _int_env("VARIANT_MAX_SEGMENTS_PER_ASSET_AUTO_K", "10"))
                    for seg in segments_for_variant:
                        sid0 = int(getattr(seg, "id", 0) or 0)
                        if sid0 <= 0:
                            continue
                        for c in (cand_map0.get(int(sid0)) or [])[:per_seg_k]:
                            if not isinstance(c, dict):
                                continue
                            aid0 = str(c.get("asset_id") or "").strip()
                            if aid0:
                                uniq_assets.add(aid0)
                    ratio = float(_float_env("VARIANT_MAX_SEGMENTS_PER_ASSET_AUTO_RATIO", "1.7"))
                    if len(uniq_assets) >= int(max(1, len(segments_for_variant))) * float(ratio):
                        max_per_asset = 1
                except Exception:
                    pass

            # Experimental: Gemini chooses the macro shot sequence (compiled into deterministic edits).
            # Auto mode attempts Gemini, but falls back to code planning if Gemini fails or violates
            # stability gates.
            if pro_mode and director_mode in {"gemini", "auto"}:
                if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                    cand_map = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan.get(plan_id) or {}))
                else:
                    cand_map = shot_candidates_by_seg
                # Default lower: improves Gemini-director reliability by shrinking prompt tokens.
                cand_k = int(max(4, min(24, float(os.getenv("VARIANT_GEMINI_CAND_K", "8") or 8))))
                pick_res = _gemini_director_choose_shots(
                    api_key=api_key,
                    model=str(args.model),
                    segments=segments_for_variant,
                    candidates_by_seg_id=cand_map,
                    max_candidates_per_seg=cand_k,
                    timeout_s=min(timeout_s, 240.0),
                    debug_dir=(vroot / "director_debug"),
                )

                if (not bool(pick_res.ok)) and director_mode == "auto":
                    # Auto mode: don't use fallback picks; defer to code planning.
                    print(
                        f"[warn] Gemini director failed in auto mode; falling back to code. error={pick_res.error!r} meta={pick_res.meta}",
                        flush=True,
                    )
                elif (not bool(pick_res.ok)) and director_mode == "gemini" and _truthy_env("VARIANT_GEMINI_STRICT", "0"):
                    # Benchmarks should not silently degrade to fallback picks (it makes comparisons meaningless).
                    raise RuntimeError(
                        f"Gemini director failed in strict mode: error={pick_res.error!r} meta={pick_res.meta} snippet={pick_res.text_snippet!r}"
                    )
                else:
                    by_sid = {
                        int(p.get("segment_id") or 0): str(p.get("shot_id") or "")
                        for p in (pick_res.picks or [])
                        if int(p.get("segment_id") or 0) > 0
                    }

                    # Stability gates: keep Gemini from trading away viewer comfort.
                    # Tight defaults: Gemini director must not trade away viewer comfort.
                    shake_max = float(_float_env("VARIANT_GEMINI_SHAKE_MAX", "0.22"))
                    shake_gate = float(_float_env("VARIANT_GEMINI_SHAKE_GATE", "0.20"))
                    sharp_min = float(_float_env("VARIANT_GEMINI_SHARP_MIN", "120.0"))
                    # "Must not drop" guardrail relative to the segment's top candidate baseline.
                    shake_delta = float(_float_env("VARIANT_GEMINI_SHAKE_DELTA", "0.03"))
                    sharp_delta = float(_float_env("VARIANT_GEMINI_SHARP_DELTA", "10.0"))

                    violations: list[int] = []
                    director_source = "gemini" if bool(pick_res.ok) else "gemini_fallback"

                    def _under_asset_cap(aid: str) -> bool:
                        if not aid:
                            return True
                        return int(asset_counts.get(aid, 0)) < int(max_per_asset)

                    def _pick_alt(cands: list[dict[str, t.Any]], *, require_stable: bool, require_gate: bool, require_unique: bool, require_cap: bool) -> dict[str, t.Any] | None:
                        for c in cands:
                            if not isinstance(c, dict):
                                continue
                            sid0 = str(c.get("id") or "").strip()
                            aid0 = str(c.get("asset_id") or "").strip()
                            if require_unique and sid0 and sid0 in used_shots:
                                continue
                            if require_cap and not _under_asset_cap(aid0):
                                continue
                            if require_stable or require_gate:
                                sh0 = _safe_float(c.get("shake_score"))
                                sp0 = _safe_float(c.get("sharpness"))
                                if sh0 is not None:
                                    th = float(shake_gate) if require_gate else float(shake_max)
                                    if float(sh0) > float(th):
                                        continue
                                if sp0 is not None and float(sp0) < float(sharp_min):
                                    continue
                            return c
                        return None

                    for seg in segments_for_variant:
                        sid = int(seg.id)
                        shot_id = str(by_sid.get(int(sid)) or "").strip()
                        cands_for_seg = [c for c in (cand_map.get(int(sid)) or []) if isinstance(c, dict)]
                        if not shot_id:
                            # Deterministic fallback: top candidate for this segment.
                            shot_id = str((cands_for_seg or [{}])[0].get("id") or "")

                        # Resolve chosen candidate.
                        shot = shot_by_id.get(shot_id) or (next((c for c in cands_for_seg if str(c.get("id") or "") == shot_id), None))
                        if not isinstance(shot, dict):
                            shot = cands_for_seg[0] if cands_for_seg else None
                        if not isinstance(shot, dict):
                            continue

                        baseline = cands_for_seg[0] if cands_for_seg else shot

                        # Candidate lists for deterministic swaps.
                        stable_soft: list[dict[str, t.Any]] = []
                        stable_gate: list[dict[str, t.Any]] = []
                        for c in cands_for_seg:
                            sh = _safe_float(c.get("shake_score"))
                            sp = _safe_float(c.get("sharpness"))
                            ok_soft = True
                            ok_gate = True
                            if sh is not None and float(sh) > float(shake_max):
                                ok_soft = False
                            if sh is not None and float(sh) > float(shake_gate):
                                ok_gate = False
                            if sp is not None and float(sp) < float(sharp_min):
                                ok_soft = False
                                ok_gate = False
                            if ok_soft:
                                stable_soft.append(c)
                            if ok_gate:
                                stable_gate.append(c)

                        chosen_shake = _safe_float(shot.get("shake_score"))
                        chosen_sharp = _safe_float(shot.get("sharpness"))
                        base_shake = _safe_float(baseline.get("shake_score")) if isinstance(baseline, dict) else None
                        base_sharp = _safe_float(baseline.get("sharpness")) if isinstance(baseline, dict) else None
                        clamped = False
                        stability_swapped = False
                        repeat_swapped = False

                        # Gate 1: hard clamp to stable shots if Gemini picked something clearly worse.
                        if stable_soft:
                            if chosen_shake is not None and float(chosen_shake) > float(shake_max):
                                clamped = True
                            if chosen_sharp is not None and float(chosen_sharp) < float(sharp_min):
                                clamped = True
                            if clamped:
                                shot = stable_soft[0]

                        # Gate 2: "stability must not drop" vs the code baseline pick (within the shortlist).
                        if isinstance(shot, dict):
                            chosen_shake = _safe_float(shot.get("shake_score"))
                            chosen_sharp = _safe_float(shot.get("sharpness"))
                        if stable_gate and chosen_shake is not None and float(chosen_shake) > float(shake_gate):
                            # Deterministic: upgrade shaky picks when a gate-stable option exists.
                            shot = stable_gate[0]
                            stability_swapped = True
                        if (
                            isinstance(baseline, dict)
                            and stable_soft
                            and base_shake is not None
                            and chosen_shake is not None
                            and float(chosen_shake) > float(base_shake) + float(shake_delta)
                        ):
                            shot = stable_soft[0]
                            stability_swapped = True
                        if (
                            isinstance(baseline, dict)
                            and stable_soft
                            and base_sharp is not None
                            and chosen_sharp is not None
                            and float(chosen_sharp) + float(sharp_delta) < float(base_sharp)
                        ):
                            shot = stable_soft[0]
                            stability_swapped = True

                        # Gate 3: within-variant repetition caps (unique shot_id + per-asset cap).
                        if isinstance(shot, dict):
                            aid0 = str(shot.get("asset_id") or "").strip()
                            sid0 = str(shot.get("id") or "").strip()
                            # Prefer stable alternatives that satisfy caps, but don't fail if the library is too small.
                            if (sid0 and sid0 in used_shots) or (aid0 and not _under_asset_cap(aid0)):
                                alt = _pick_alt(stable_soft or cands_for_seg, require_stable=bool(stable_soft), require_gate=False, require_unique=True, require_cap=True)
                                if alt is None:
                                    alt = _pick_alt(stable_soft or cands_for_seg, require_stable=False, require_gate=False, require_unique=True, require_cap=True)
                                if alt is not None:
                                    shot = alt
                                    repeat_swapped = True

                        if not isinstance(shot, dict):
                            continue
                        shot_id = str(shot.get("id") or "").strip() or shot_id
                        aid = str(shot.get("asset_id") or "").strip()

                        if shot_id:
                            used_shots.add(shot_id)
                        if aid:
                            used.add(aid)
                            asset_counts[aid] = int(asset_counts.get(aid, 0)) + 1

                        # Auto rejection: if even after swaps the pick is beyond our hard stability thresholds.
                        chosen_shake = _safe_float(shot.get("shake_score"))
                        chosen_sharp = _safe_float(shot.get("sharpness"))
                        if chosen_shake is not None and float(chosen_shake) > float(shake_max):
                            violations.append(int(sid))
                        if chosen_sharp is not None and float(chosen_sharp) < float(sharp_min):
                            violations.append(int(sid))

                        speed = 1.0
                        if _truthy_env("FOLDER_EDIT_ENERGY_SPEEDUP", "0"):
                            try:
                                e = getattr(seg, "music_energy", None)
                                energy = float(e) if isinstance(e, (int, float)) else float(_segment_energy_hint(seg.duration_s))
                            except Exception:
                                energy = float(_segment_energy_hint(seg.duration_s))
                            base_th = _float_env("FOLDER_EDIT_SPEEDUP_ENERGY_TH", "0.55")
                            slope = _float_env("FOLDER_EDIT_SPEEDUP_SLOPE", "0.20")
                            max_speed = _float_env("FOLDER_EDIT_SPEEDUP_MAX", "1.12")
                            if float(energy) > float(base_th):
                                speed = min(float(max_speed), 1.0 + (float(energy) - float(base_th)) * float(slope))

                        decisions_try.append(
                            {
                                "segment_id": seg.id,
                                "asset_id": aid,
                                "shot_id": str(shot.get("id") or ""),
                                "in_s": float(shot.get("start_s") or 0.0),
                                "speed": float(speed),
                                "crop_mode": "center",
                                "director_source": director_source,
                                "director_ok": bool(pick_res.ok),
                                "director_clamped": bool(clamped),
                                "director_stability_swap": bool(stability_swapped),
                                "director_repeat_swap": bool(repeat_swapped),
                            }
                        )

                    if director_mode == "auto" and violations:
                        # Reject Gemini macro if it still violates hard stability thresholds; fall back to code planning.
                        print(
                            f"[warn] Gemini director rejected by stability gate in auto mode; falling back to code. violations={sorted(set(violations))}",
                            flush=True,
                        )
                        decisions_try = []
                    else:
                        aset = {str(d.get("asset_id") or "") for d in decisions_try if str(d.get("asset_id") or "").strip()}
                        best = (1.0, decisions_try, aset)
                        break

            # Pro macro (beam): sample a globally coherent sequence from the optimizer beam.
            if pro_mode and director_mode == "code" and pro_macro_mode == "beam" and beam_sequences_by_plan:
                pid0 = str(plan_id or "default")
                beam_items = list(beam_sequences_by_plan.get(pid0) or beam_sequences_by_plan.get("default") or [])
                # Keep only sequences that match the segment count.
                beam_items = [it for it in beam_items if isinstance(it, dict) and isinstance(it.get("shot_ids"), list) and len(t.cast(list[t.Any], it.get("shot_ids") or [])) == len(segments_for_variant)]
                if beam_items:
                    denom = max(0.25, float(params.tau))
                    weights: list[float] = []
                    for it in beam_items:
                        try:
                            rank0 = int(it.get("rank") or 1) - 1
                        except Exception:
                            rank0 = 0
                        aset0 = {str(x) for x in (it.get("asset_ids") or []) if isinstance(x, str) and x}
                        usage_sum = sum(int(asset_usage.get(a, 0)) for a in aset0) if aset0 else 0
                        eff_rank = float(rank0) + (float(usage_sum) * float(usage_penalty) * 0.15)
                        weights.append(math.exp(-eff_rank / denom))
                    picked = rng.choices(beam_items, weights=weights, k=1)[0]
                    shot_ids = [str(x) for x in (picked.get("shot_ids") or [])]
                    for seg, sid in zip(segments_for_variant, shot_ids, strict=False):
                        shot = shot_by_id.get(str(sid))
                        if not isinstance(shot, dict):
                            # Fallback: top candidate for this segment.
                            if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                                shortlist = list((shot_candidates_by_seg_by_plan.get(plan_id) or {}).get(int(seg.id)) or [])
                            else:
                                shortlist = list(shot_candidates_by_seg.get(int(seg.id)) or [])
                            shot = shortlist[0] if shortlist else None
                        else:
                            # Load the per-segment shortlist only if we need to swap for diversity caps.
                            shortlist = []
                        if not isinstance(shot, dict):
                            continue
                        # Avoid repeating the exact same shot_id in a single edit when possible.
                        shot_id0 = str(shot.get("id") or "").strip()
                        if shot_id0 and shot_id0 in used_shots:
                            if not shortlist:
                                if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                                    shortlist = list((shot_candidates_by_seg_by_plan.get(plan_id) or {}).get(int(seg.id)) or [])
                                else:
                                    shortlist = list(shot_candidates_by_seg.get(int(seg.id)) or [])
                            alt0 = next(
                                (
                                    s
                                    for s in shortlist
                                    if isinstance(s, dict)
                                    and str(s.get("id") or "").strip()
                                    and str(s.get("id") or "").strip() not in used_shots
                                ),
                                None,
                            )
                            if isinstance(alt0, dict):
                                shot = alt0
                                shot_id0 = str(shot.get("id") or "").strip()
                        aid = str(shot.get("asset_id") or "")
                        if aid and int(asset_counts.get(aid, 0)) >= int(max_per_asset):
                            # Deterministic: swap to the first alternative under the per-asset cap.
                            if not shortlist:
                                if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                                    shortlist = list((shot_candidates_by_seg_by_plan.get(plan_id) or {}).get(int(seg.id)) or [])
                                else:
                                    shortlist = list(shot_candidates_by_seg.get(int(seg.id)) or [])
                            alt = next(
                                (s for s in shortlist if isinstance(s, dict) and str(s.get("asset_id") or "").strip() and int(asset_counts.get(str(s.get("asset_id") or "").strip(), 0)) < int(max_per_asset)),
                                None,
                            )
                            if isinstance(alt, dict):
                                shot = alt
                                aid = str(shot.get("asset_id") or "")
                                shot_id0 = str(shot.get("id") or "").strip()
                        if aid:
                            used.add(aid)
                            asset_counts[aid] = int(asset_counts.get(aid, 0)) + 1
                        if shot_id0:
                            used_shots.add(shot_id0)
                        speed = 1.0
                        if _truthy_env("FOLDER_EDIT_ENERGY_SPEEDUP", "0"):
                            try:
                                e = getattr(seg, "music_energy", None)
                                energy = float(e) if isinstance(e, (int, float)) else float(_segment_energy_hint(seg.duration_s))
                            except Exception:
                                energy = float(_segment_energy_hint(seg.duration_s))
                            base_th = _float_env("FOLDER_EDIT_SPEEDUP_ENERGY_TH", "0.55")
                            slope = _float_env("FOLDER_EDIT_SPEEDUP_SLOPE", "0.20")
                            max_speed = _float_env("FOLDER_EDIT_SPEEDUP_MAX", "1.12")
                            if float(energy) > float(base_th):
                                speed = min(float(max_speed), 1.0 + (float(energy) - float(base_th)) * float(slope))
                        decisions_try.append(
                            {
                                "segment_id": seg.id,
                                "asset_id": aid,
                                "shot_id": str(shot.get("id") or ""),
                                "in_s": float(shot.get("start_s") or 0.0),
                                "speed": float(speed),
                                "crop_mode": "center",
                            }
                        )

            # Default: per-slot sampling from the per-segment shortlist (fast exploration).
            if not decisions_try:
                used_sem: dict[str, int] = {}
                prev_sem: str | None = None
                prev_sem_run = 0
                prev_seg: t.Any | None = None
                for seg in segments_for_variant:
                    if pro_mode:
                        if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                            shortlist = list((shot_candidates_by_seg_by_plan.get(plan_id) or {}).get(int(seg.id)) or [])
                        else:
                            shortlist = list(shot_candidates_by_seg.get(int(seg.id)) or [])
                        pool0 = shortlist[: max(4, int(params.top_n))]

                        # Avoid over-reusing the same underlying asset within a single variant if possible.
                        min_pool = max(4, int(params.top_n) // 3)
                        pool_unused: list[dict[str, t.Any]] = []
                        pool_cap: list[dict[str, t.Any]] = []
                        for s in pool0:
                            aid0 = str(s.get("asset_id") or "").strip()
                            if not aid0:
                                continue
                            if int(asset_counts.get(aid0, 0)) <= 0:
                                pool_unused.append(s)
                            if int(asset_counts.get(aid0, 0)) < int(max_per_asset):
                                pool_cap.append(s)
                        if len(pool_unused) >= min_pool:
                            pool = pool_unused
                        elif len(pool_cap) >= min_pool:
                            pool = pool_cap
                        else:
                            pool = pool0
                        # Avoid repeating the same shot_id within a single edit when possible.
                        pool_unique = [s for s in pool if str(s.get("id") or "").strip() and str(s.get("id") or "").strip() not in used_shots]
                        if pool_unique:
                            pool = pool_unique

                        # Weighted pick by rank with cross-variant usage penalty (tracked by asset_id).
                        weights: list[float] = []
                        denom = max(0.25, float(params.tau))
                        # Within-variant semantic diversity (soft):
                        # - Allow 1 consecutive repeat (continuity), but discourage 3+ in a row.
                        # - Keep weights conservative so we don't sacrifice reference matching.
                        sem_pen = _float_env("VARIANT_SEM_PENALTY", "0.25")
                        consec_pen = _float_env("VARIANT_CONSEC_SEM_PENALTY", "2.00")
                        mode = _segment_mode(prev_seg, seg)
                        if mode == "continuity":
                            sem_pen *= 0.20
                            consec_pen *= 0.0
                        elif mode == "contrast":
                            sem_pen *= 1.10
                            consec_pen *= 1.10
                        else:
                            consec_pen *= 0.75
                        for i, s in enumerate(pool):
                            aid = str(s.get("asset_id") or "")
                            u = int(asset_usage.get(aid, 0)) if aid else 0
                            sem_key = _sem_key_for_shot(s)
                            sem_u = int(used_sem.get(sem_key, 0))
                            # Penalize only after we've already repeated the same semantic key once
                            # (i.e., discourage 3+ in a row, but allow 2 in a row).
                            consec_u = 0
                            if prev_sem and sem_key == prev_sem:
                                consec_u = max(0, int(prev_sem_run) - 1)
                            eff_rank = (
                                float(i)
                                + (float(u) * float(usage_penalty))
                                + (float(sem_u) * float(sem_pen))
                                + (float(consec_u) * float(consec_pen))
                            )
                            weights.append(math.exp(-eff_rank / denom))
                        chosen = rng.choices(pool, weights=weights, k=1)[0]
                        shot_id = str(chosen.get("id") or "")
                        aid = str(chosen.get("asset_id") or "")
                        if aid:
                            used.add(aid)
                            asset_counts[aid] = int(asset_counts.get(aid, 0)) + 1
                        if shot_id:
                            used_shots.add(shot_id)
                        chosen_sem = _sem_key_for_shot(chosen)
                        used_sem[chosen_sem] = int(used_sem.get(chosen_sem, 0)) + 1
                        if prev_sem and chosen_sem == prev_sem:
                            prev_sem_run += 1
                        else:
                            prev_sem = chosen_sem
                            prev_sem_run = 1
                        # Optional: slight speed-up for high-energy segments to increase perceived cadence
                        # without changing the segment duration (rendering compensates by sampling a longer source window).
                        speed = 1.0
                        if _truthy_env("FOLDER_EDIT_ENERGY_SPEEDUP", "0"):
                            try:
                                e = getattr(seg, "music_energy", None)
                                energy = float(e) if isinstance(e, (int, float)) else float(_segment_energy_hint(seg.duration_s))
                            except Exception:
                                energy = float(_segment_energy_hint(seg.duration_s))
                            base_th = _float_env("FOLDER_EDIT_SPEEDUP_ENERGY_TH", "0.55")
                            slope = _float_env("FOLDER_EDIT_SPEEDUP_SLOPE", "0.20")
                            max_speed = _float_env("FOLDER_EDIT_SPEEDUP_MAX", "1.12")
                            if float(energy) > float(base_th):
                                speed = min(float(max_speed), 1.0 + (float(energy) - float(base_th)) * float(slope))
                        decisions_try.append(
                            {
                                "segment_id": seg.id,
                                "asset_id": aid,
                                "shot_id": shot_id,
                                "in_s": float(chosen.get("start_s") or 0.0),
                                "speed": float(speed),
                                "crop_mode": "center",
                            }
                        )
                        prev_seg = seg
                    else:
                        shortlist = _shortlist_assets_for_segment(
                            segment=seg, assets=assets_for_planner, used_asset_ids=used, limit=int(params.top_n)
                        )
                        # Avoid repeats if possible; cap per-asset usage within a variant.
                        min_pool = max(4, int(params.top_n) // 3)
                        pool_unused: list[dict[str, t.Any]] = []
                        pool_cap: list[dict[str, t.Any]] = []
                        for a in shortlist:
                            aid0 = str(a.get("id") or "").strip()
                            if not aid0:
                                continue
                            if int(asset_counts.get(aid0, 0)) <= 0:
                                pool_unused.append(a)
                            if int(asset_counts.get(aid0, 0)) < int(max_per_asset):
                                pool_cap.append(a)
                        if len(pool_unused) >= min_pool:
                            pool = pool_unused
                        elif len(pool_cap) >= min_pool:
                            pool = pool_cap
                        else:
                            pool = shortlist
                        chosen = _pick_weighted(pool, tau=float(params.tau), rng=rng, usage=asset_usage, usage_penalty=usage_penalty)
                        aid = str(chosen.get("id") or "")
                        if aid:
                            used.add(aid)
                            asset_counts[aid] = int(asset_counts.get(aid, 0)) + 1
                        decisions_try.append(
                            {"segment_id": seg.id, "asset_id": aid, "in_s": 0.0, "speed": 1.0, "crop_mode": "center"}
                        )
                        prev_seg = seg

            aset = {str(d.get("asset_id") or "") for d in decisions_try if str(d.get("asset_id") or "").strip()}
            if not prior_sets or min_jaccard_dist <= 0.0:
                best = (1.0, decisions_try, aset)
                break

            # Enforce diversity vs all prior variants.
            md = min((_jaccard_distance(aset, s) for s in prior_sets), default=1.0)
            if best is None or md > best[0]:
                best = (md, decisions_try, aset)
            if md >= min_jaccard_dist:
                break

        assert best is not None
        min_dist, decisions, aset = best
        # Update global usage only for the accepted variant.
        for aid in aset:
            asset_usage[aid] = asset_usage.get(aid, 0) + 1
        prior_sets.append(aset)

        # Refine in-points (code-only; no LLM calls).
        inpoint_meta_by_seg: dict[int, dict[str, t.Any]] = {}
        for seg in segments_for_variant:
            dec = next((d for d in decisions if int(d["segment_id"]) == int(seg.id)), None)
            if not dec:
                continue
            aid = str(dec.get("asset_id") or "")
            meta = asset_by_id.get(aid) or {}
            if str(meta.get("kind") or "") != "video":
                continue
            asset_path = Path(str(meta.get("path") or ""))
            if not asset_path.exists():
                continue
            if pro_mode:
                from .micro_editor import pick_inpoint_for_shot

                shot_id = str(dec.get("shot_id") or "")
                shot = shot_by_id.get(shot_id)
                if not isinstance(shot, dict):
                    continue
                chosen_t, info = pick_inpoint_for_shot(
                    segment=seg,
                    shot=shot,
                    work_dir=vroot / "refine",
                    timeout_s=timeout_s,
                )
                dec["in_s"] = float(chosen_t)
                inpoint_meta_by_seg[int(seg.id)] = info
            else:
                chosen_t, info = _auto_pick_inpoint(
                    asset_path=asset_path,
                    asset_duration_s=(float(meta.get("duration_s")) if isinstance(meta.get("duration_s"), (int, float)) else None),
                    segment_duration_s=float(seg.duration_s),
                    initial_in_s=float(dec.get("in_s") or 0.0),
                    ref_luma=getattr(seg, "ref_luma", None),
                    ref_dark=getattr(seg, "ref_dark_frac", None),
                    energy_hint=_segment_energy_hint(seg.duration_s),
                    sample_dir=vroot / "refine" / f"seg_{seg.id:02d}",
                    timeout_s=timeout_s,
                    rng=rng,
                    choose_top_k=int(params.inpoint_top_k),
                )
                dec["in_s"] = float(chosen_t)
                inpoint_meta_by_seg[int(seg.id)] = info

        # Render.
        segments_dir = vroot / "segments"
        grade_samples_dir = vroot / "grade_samples"
        segment_paths: list[Path] = []
        timeline_segments: list[dict[str, t.Any]] = []
        fade_tail = _float_env("FOLDER_EDIT_FADE_OUT_S", "0.18")
        prev_seg_for_grade = None
        prev_grade: dict[str, float] | None = None

        # Optional: compute a global grade anchor across the whole edit.
        # This improves segment-to-segment look continuity by applying a shared baseline
        # and then letting per-segment grades deviate only partially.
        global_grade: dict[str, float] | None = None
        global_w = float(_float_env("FOLDER_EDIT_GRADE_GLOBAL_RESIDUAL_W", "0.75"))
        global_w = max(0.0, min(1.0, global_w))
        if use_auto_grade and _truthy_env("FOLDER_EDIT_GRADE_GLOBAL_ANCHOR", "1"):
            try:
                ref_lumas: list[float] = []
                ref_darks: list[float] = []
                ref_rgbs: list[list[float]] = []
                out_lumas: list[float] = []
                out_darks: list[float] = []
                out_rgbs: list[list[float]] = []
                for seg in segments_for_variant:
                    rl = getattr(seg, "ref_luma", None)
                    rd = getattr(seg, "ref_dark_frac", None)
                    rr = getattr(seg, "ref_rgb_mean", None)
                    if isinstance(rl, (int, float)):
                        ref_lumas.append(float(rl))
                    if isinstance(rd, (int, float)):
                        ref_darks.append(float(rd))
                    if isinstance(rr, list) and len(rr) == 3 and all(isinstance(x, (int, float)) for x in rr):
                        ref_rgbs.append([float(rr[0]), float(rr[1]), float(rr[2])])

                    chosen0 = (inpoint_meta_by_seg.get(int(getattr(seg, "id", 0))) or {}).get("chosen") or {}
                    ol = chosen0.get("luma")
                    od = chosen0.get("dark_frac")
                    or0 = chosen0.get("rgb_mean")
                    if isinstance(ol, (int, float)):
                        out_lumas.append(float(ol))
                    if isinstance(od, (int, float)):
                        out_darks.append(float(od))
                    if isinstance(or0, list) and len(or0) == 3 and all(isinstance(x, (int, float)) for x in or0):
                        out_rgbs.append([float(or0[0]), float(or0[1]), float(or0[2])])

                ref_l = _median(ref_lumas)
                ref_d = _median(ref_darks)
                out_l = _median(out_lumas)
                out_d = _median(out_darks)
                ref_rgb = _median_rgb(ref_rgbs) if ref_rgbs else None
                out_rgb = _median_rgb(out_rgbs) if out_rgbs else None
                global_grade = _compute_eq_grade(ref_luma=ref_l, ref_dark=ref_d, out_luma=out_l, out_dark=out_d, ref_rgb=ref_rgb, out_rgb=out_rgb)
            except Exception:
                global_grade = None

        def _blend_grade_anchor(global_g: dict[str, float] | None, per_g: dict[str, float] | None) -> dict[str, float] | None:
            if global_g is None or not isinstance(global_g, dict) or not global_g:
                return per_g
            if per_g is None or not isinstance(per_g, dict) or not per_g:
                return dict(global_g)
            out: dict[str, float] = {}
            keys = set(global_g.keys()) | set(per_g.keys())
            for k in keys:
                gv = global_g.get(k)
                pv = per_g.get(k)
                if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                    out[k] = float(gv) + (float(pv) - float(gv)) * float(global_w)
                elif isinstance(pv, (int, float)):
                    out[k] = float(pv)
                elif isinstance(gv, (int, float)):
                    out[k] = float(gv)
            return out or None

        for i, seg in enumerate(segments_for_variant, start=1):
            dec = next((d for d in decisions if int(d["segment_id"]) == int(seg.id)), None)
            if not dec:
                continue
            aid = str(dec.get("asset_id") or "")
            meta = asset_by_id.get(aid) or {}
            asset_path = Path(str(meta.get("path") or ""))
            asset_kind = str(meta.get("kind") or "video")

            grade: dict[str, float] | None = None
            chosen_info = (inpoint_meta_by_seg.get(int(seg.id)) or {}).get("chosen") or {}
            out_luma = chosen_info.get("luma")
            out_dark = chosen_info.get("dark_frac")
            out_rgb = chosen_info.get("rgb_mean")
            out_luma_std: float | None = None
            out_chroma: float | None = None
            out_frame_path: Path | None = None
            try:
                fp = chosen_info.get("frame_path")
                if isinstance(fp, str) and fp.strip():
                    out_frame_path = Path(fp).expanduser()
            except Exception:
                out_frame_path = None
            if use_auto_grade:
                # Optional multi-frame sampling across the segment window to reduce grade noise and flicker.
                if asset_kind == "video" and asset_path.exists() and _truthy_env("FOLDER_EDIT_GRADE_MULTI_SAMPLE", "1"):
                    try:
                        grade_samples_dir.mkdir(parents=True, exist_ok=True)
                        seg_sample_dir = grade_samples_dir / f"seg_{int(seg.id):02d}"
                        seg_sample_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            dur0 = float(meta.get("duration_s") or 0.0) if isinstance(meta.get("duration_s"), (int, float)) else None
                        except Exception:
                            dur0 = None

                        def _clamp_t(t_s: float) -> float:
                            tt = float(max(0.0, t_s))
                            if dur0 is None:
                                return tt
                            return float(min(tt, max(0.0, float(dur0) - 0.15)))

                        in_s0 = float(dec.get("in_s") or 0.0)
                        times = [float(in_s0)]
                        if float(seg.duration_s) >= 0.25:
                            times.extend([float(in_s0) + float(seg.duration_s) * 0.50, float(in_s0) + float(seg.duration_s) * 0.85])
                        seen_t: set[float] = set()
                        times2: list[float] = []
                        for t_s in times:
                            t3 = round(_clamp_t(float(t_s)), 3)
                            if t3 in seen_t:
                                continue
                            seen_t.add(t3)
                            times2.append(float(t3))

                        sample_paths: list[Path] = []
                        # Reuse the micro-editor extracted frame for the chosen in-point when available.
                        if isinstance(out_frame_path, Path) and out_frame_path.exists():
                            sample_paths.append(out_frame_path)
                        for t_s in times2:
                            safe = f"{float(t_s):.3f}".replace(".", "p")
                            sp = seg_sample_dir / f"t_{safe}.jpg"
                            if sp.exists():
                                sample_paths.append(sp)
                                continue
                            _extract_frame(video_path=asset_path, at_s=float(t_s), out_path=sp, timeout_s=min(timeout_s, 120.0))
                            sample_paths.append(sp)

                        stats = _robust_frame_stats([p for p in sample_paths if isinstance(p, Path) and p.exists()])
                        l0 = stats.get("luma")
                        d0 = stats.get("dark_frac")
                        r0 = stats.get("rgb_mean")
                        ls0 = stats.get("luma_std")
                        c0 = stats.get("chroma")
                        out_luma = float(l0) if isinstance(l0, (int, float)) else out_luma
                        out_dark = float(d0) if isinstance(d0, (int, float)) else out_dark
                        out_rgb = r0 if isinstance(r0, list) else out_rgb
                        out_luma_std = float(ls0) if isinstance(ls0, (int, float)) else None
                        out_chroma = float(c0) if isinstance(c0, (int, float)) else None

                        # Low-key references are extremely sensitive to brief bright flares.
                        # When our multi-sample stats show a bright spike, bias grading slightly toward it
                        # (without fully chasing the max), to reduce perceived flicker.
                        if _truthy_env("FOLDER_EDIT_GRADE_LOWKEY_SPIKE_HANDLE", "1"):
                            try:
                                ref_luma = getattr(seg, "ref_luma", None)
                                ref_dark = getattr(seg, "ref_dark_frac", None)
                                lowkey_luma_max = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_REF_LUMA_MAX", "0.03"))
                                lowkey_dark_min = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_REF_DARK_MIN", "0.93"))
                                is_low_key = False
                                if isinstance(ref_luma, (int, float)) and float(ref_luma) <= float(lowkey_luma_max):
                                    is_low_key = True
                                if isinstance(ref_dark, (int, float)) and float(ref_dark) >= float(lowkey_dark_min):
                                    is_low_key = True
                                if is_low_key:
                                    med_l = float(out_luma) if isinstance(out_luma, (int, float)) else None
                                    max_l = stats.get("luma_max")
                                    if not isinstance(max_l, (int, float)):
                                        max_l = stats.get("luma_p75")
                                    if isinstance(med_l, (int, float)) and isinstance(max_l, (int, float)):
                                        spike_delta = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_SPIKE_LUMA_DELTA", "0.08"))
                                        spike_min = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_SPIKE_LUMA_MIN", "0.12"))
                                        if float(max_l) >= float(spike_min) and (float(max_l) - float(med_l)) >= float(spike_delta):
                                            w = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_SPIKE_BLEND", "0.35"))
                                            w = max(0.0, min(1.0, w))
                                            out_luma = float(med_l) + (float(max_l) - float(med_l)) * float(w)

                                            med_d = float(out_dark) if isinstance(out_dark, (int, float)) else None
                                            min_d = stats.get("dark_min")
                                            if not isinstance(min_d, (int, float)):
                                                min_d = stats.get("dark_p25")
                                            dark_delta = float(_float_env("FOLDER_EDIT_GRADE_LOWKEY_SPIKE_DARK_DELTA", "0.10"))
                                            if isinstance(med_d, (int, float)) and isinstance(min_d, (int, float)) and (float(med_d) - float(min_d)) >= float(dark_delta):
                                                out_dark = float(med_d) + (float(min_d) - float(med_d)) * float(w)

                                            rgb_med = out_rgb if isinstance(out_rgb, list) and len(out_rgb) == 3 else None
                                            rgb_max = stats.get("rgb_at_luma_max")
                                            if (
                                                isinstance(rgb_med, list)
                                                and isinstance(rgb_max, list)
                                                and len(rgb_max) == 3
                                                and all(isinstance(x, (int, float)) for x in rgb_med)
                                                and all(isinstance(x, (int, float)) for x in rgb_max)
                                            ):
                                                out_rgb = [
                                                    float(float(rgb_med[0]) + (float(rgb_max[0]) - float(rgb_med[0])) * float(w)),
                                                    float(float(rgb_med[1]) + (float(rgb_max[1]) - float(rgb_med[1])) * float(w)),
                                                    float(float(rgb_med[2]) + (float(rgb_max[2]) - float(rgb_med[2])) * float(w)),
                                                ]
                            except Exception:
                                pass
                        # Anchor frame for debug/fallbacks.
                        if sample_paths:
                            out_frame_path = sample_paths[0]
                    except Exception:
                        pass

                grade = _compute_eq_grade(
                    ref_luma=getattr(seg, "ref_luma", None),
                    ref_dark=getattr(seg, "ref_dark_frac", None),
                    out_luma=(float(out_luma) if isinstance(out_luma, (int, float)) else None),
                    out_dark=(float(out_dark) if isinstance(out_dark, (int, float)) else None),
                    ref_rgb=getattr(seg, "ref_rgb_mean", None),
                    out_rgb=(out_rgb if isinstance(out_rgb, list) else None),
                    out_luma_std=out_luma_std,
                    out_chroma=out_chroma,
                    ref_frame_path=ref_frame_by_id.get(int(seg.id)),
                    out_frame_path=(out_frame_path if out_frame_path is not None and out_frame_path.exists() else None),
                )
                grade = _blend_grade_anchor(global_grade, grade)
                grade = _smooth_grade_step(prev_segment=prev_seg_for_grade, prev_grade=prev_grade, segment=seg, grade=grade)
                if isinstance(grade, dict):
                    prev_seg_for_grade = seg
                    prev_grade = grade

            shot_id0 = str(dec.get("shot_id") or "") if pro_mode else ""
            shot0 = shot_by_id.get(shot_id0) if (pro_mode and shot_id0) else None
            if not isinstance(shot0, dict):
                shot0 = None

            stabilize = False
            zoom = _float_env("FOLDER_EDIT_ZOOM", "1.0")
            if asset_kind == "video" and _truthy_env("FOLDER_EDIT_STABILIZE", "1"):
                # Prefer shake_score when available; motion proxy is last resort.
                shake_score = None
                if isinstance(shot0, dict):
                    shake_score = shot0.get("shake_score")
                if shake_score is None:
                    shake_score = meta.get("shake_score")
                shake_th = _float_env("FOLDER_EDIT_STABILIZE_SHAKE_SCORE_THRESHOLD", "0.22")
                shake_max = _float_env("FOLDER_EDIT_STABILIZE_SHAKE_SCORE_MAX", "0.60")
                if (
                    isinstance(shake_score, (int, float))
                    and float(shake_score) >= float(shake_th)
                    and float(shake_score) <= float(shake_max)
                    and float(seg.duration_s) >= 1.0
                ):
                    stabilize = True
                    zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.08"))
                else:
                    motion = None
                    if isinstance(shot0, dict):
                        motion = shot0.get("motion_score")
                    if motion is None:
                        motion = meta.get("motion_score")
                    motion_th = _float_env("FOLDER_EDIT_STABILIZE_MOTION_THRESHOLD", "0.17")
                    if isinstance(motion, (int, float)) and float(motion) >= float(motion_th) and float(seg.duration_s) >= 1.0:
                        stabilize = True
                        zoom = max(zoom, _float_env("FOLDER_EDIT_STABILIZE_ZOOM", "1.08"))

            # Optional: small punch-in on high-energy segments to reduce "too wide/static" feel.
            if _truthy_env("FOLDER_EDIT_ENERGY_PUNCHIN", "0"):
                try:
                    e = getattr(seg, "music_energy", None)
                    energy = float(e) if isinstance(e, (int, float)) else float(_segment_energy_hint(seg.duration_s))
                except Exception:
                    energy = float(_segment_energy_hint(seg.duration_s))
                punch_th = _float_env("FOLDER_EDIT_PUNCHIN_ENERGY_TH", "0.55")
                punch_zoom = _float_env("FOLDER_EDIT_PUNCHIN_ZOOM", "1.04")
                if float(energy) >= float(punch_th):
                    zoom = max(float(zoom), float(punch_zoom))

            fade_in_s = None
            fade_in_color = None
            fade_out_s = (fade_tail if i == len(segments_for_variant) else None)
            fade_out_color = "black" if fade_out_s is not None else None

            out_path = segments_dir / f"seg_{seg.id:02d}.mp4"
            _render_segment(
                asset_path=asset_path,
                asset_kind=asset_kind,
                in_s=float(dec.get("in_s") or 0.0),
                duration_s=float(seg.duration_s),
                speed=float(dec.get("speed") or 1.0),
                crop_mode=str(dec.get("crop_mode") or "center"),
                overlay_text=str(getattr(seg, "overlay_text", "") or ""),
                grade=grade,
                stabilize=stabilize,
                stabilize_cache_dir=(project_root / "library" / "stabilized") if stabilize else None,
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

            timeline_segments.append(
                {
                    "id": seg.id,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "duration_s": seg.duration_s,
                    "ref_luma": getattr(seg, "ref_luma", None),
                    "ref_dark_frac": getattr(seg, "ref_dark_frac", None),
                    "ref_rgb_mean": getattr(seg, "ref_rgb_mean", None),
                    "music_energy": getattr(seg, "music_energy", None),
                    "start_beat": getattr(seg, "start_beat", None),
                    "end_beat": getattr(seg, "end_beat", None),
                    "beat_goal": getattr(seg, "beat_goal", ""),
                    "overlay_text": getattr(seg, "overlay_text", ""),
                    "reference_visual": getattr(seg, "reference_visual", ""),
                    "desired_tags": getattr(seg, "desired_tags", []),
                    "story_beat": getattr(seg, "story_beat", None),
                    "preferred_sequence_group_ids": getattr(seg, "preferred_sequence_group_ids", None),
                    "transition_hint": getattr(seg, "transition_hint", None),
                    "shot_id": (shot_id0 if pro_mode else None),
                    "sequence_group_id": (shot0.get("sequence_group_id") if isinstance(shot0, dict) else None),
                    # Shot-level objective signals (used by fix loops and graders).
                    "shot_motion_score": (shot0.get("motion_score") if isinstance(shot0, dict) else None),
                    "shot_shake_score": (shot0.get("shake_score") if isinstance(shot0, dict) else None),
                    "shot_sharpness": (shot0.get("sharpness") if isinstance(shot0, dict) else None),
                    "shot_luma_mean": (shot0.get("luma_mean") if isinstance(shot0, dict) else None),
                    "shot_dark_frac": (shot0.get("dark_frac") if isinstance(shot0, dict) else None),
                    "shot_rgb_mean": (shot0.get("rgb_mean") if isinstance(shot0, dict) else None),
                    "shot_cam_motion_mag": (shot0.get("cam_motion_mag") if isinstance(shot0, dict) else None),
                    "shot_cam_motion_angle_deg": (shot0.get("cam_motion_angle_deg") if isinstance(shot0, dict) else None),
                    "asset_id": aid,
                    "asset_path": str(asset_path),
                    "asset_kind": asset_kind,
                    "asset_in_s": float(dec.get("in_s") or 0.0),
                    "asset_out_s": float(dec.get("in_s") or 0.0) + float(seg.duration_s),
                    "crop_mode": str(dec.get("crop_mode") or "center"),
                    "speed": float(dec.get("speed") or 1.0),
                    "grade": grade,
                    "stabilize": stabilize,
                    "zoom": zoom,
                    "fade_in_s": fade_in_s,
                    "fade_in_color": fade_in_color,
                    "fade_out_s": fade_out_s,
                    "fade_out_color": fade_out_color,
                    "inpoint_auto": inpoint_meta_by_seg.get(int(seg.id)),
                    "director_source": str(dec.get("director_source") or ("code" if director_mode == "code" else "")) or None,
                    "director_clamped": (bool(dec.get("director_clamped")) if dec.get("director_clamped") is not None else None),
                }
            )

        silent_video = vroot / "final_video_silent.mp4"
        # Concat helper is in pipeline; do a simple concat via ffmpeg demuxer here.
        from .folder_edit_pipeline import _concat_segments  # type: ignore

        _concat_segments(segment_paths=segment_paths, output_path=silent_video, timeout_s=timeout_s)

        final_video = vroot / "final_video.mov"
        shutil.copyfile(silent_video, final_video)
        if final_audio:
            merge_audio(video_path=final_video, audio_path=final_audio)

        avg_dl = 0.0
        avg_dd = 0.0
        palette_rgb_jump_p95: float | None = None
        blur_lap_var_p05: float | None = None
        shake_score_p95: float | None = None
        segment_objectives: list[dict[str, t.Any]] = []
        if _truthy_env("VARIANT_PREVIEW_FRAMES", "1"):
            # Preview frames + cheap mismatch metrics (luma/dark) for later filtering.
            preview_dir = vroot / "preview_frames"
            preview_dir.mkdir(parents=True, exist_ok=True)
            luma_diffs: list[float] = []
            dark_diffs: list[float] = []
            rgb_diffs: list[float] = []
            blur_vals: list[float] = []
            out_rgb_by_seg_id: dict[int, list[float] | None] = {}
            obj_by_seg_id: dict[int, dict[str, t.Any]] = {}
            for seg_id, start_s, end_s, ref_frame in segment_frames:
                mid = (start_s + end_s) / 2.0
                out_frame = preview_dir / f"out_{seg_id:02d}.jpg"
                _extract_frame(video_path=silent_video, at_s=mid, out_path=out_frame, timeout_s=min(timeout_s, 120.0))
                rl = _luma_mean(ref_frame)
                rd = _dark_frac(ref_frame)
                rrgb = _rgb_mean(ref_frame)
                ol = _luma_mean(out_frame)
                od = _dark_frac(out_frame)
                orgb = _rgb_mean(out_frame)
                blur = _blur_lap_var(out_frame)
                if isinstance(rl, (int, float)) and isinstance(ol, (int, float)):
                    luma_diffs.append(abs(float(rl) - float(ol)))
                if isinstance(rd, (int, float)) and isinstance(od, (int, float)):
                    dark_diffs.append(abs(float(rd) - float(od)))
                if (
                    isinstance(rrgb, list)
                    and isinstance(orgb, list)
                    and len(rrgb) == 3
                    and len(orgb) == 3
                    and all(isinstance(x, (int, float)) for x in rrgb)
                    and all(isinstance(x, (int, float)) for x in orgb)
                ):
                    rgb_diffs.append(sum(abs(float(rrgb[i]) - float(orgb[i])) for i in range(3)) / 3.0)
                if isinstance(blur, (int, float)):
                    blur_vals.append(float(blur))

                out_rgb_by_seg_id[int(seg_id)] = orgb if isinstance(orgb, list) else None
                obj_by_seg_id[int(seg_id)] = {
                    "segment_id": int(seg_id),
                    "mid_t_s": float(mid),
                    "out_frame": str(out_frame),
                    "out_luma_mid": (float(ol) if isinstance(ol, (int, float)) else None),
                    "out_dark_mid": (float(od) if isinstance(od, (int, float)) else None),
                    "out_rgb_mid": (orgb if isinstance(orgb, list) else None),
                    "ref_luma_mid": (float(rl) if isinstance(rl, (int, float)) else None),
                    "ref_dark_mid": (float(rd) if isinstance(rd, (int, float)) else None),
                    "ref_rgb_mid": (rrgb if isinstance(rrgb, list) else None),
                    "blur_lap_var_mid": (float(blur) if isinstance(blur, (int, float)) else None),
                }

            avg_dl = (sum(luma_diffs) / len(luma_diffs)) if luma_diffs else 0.0
            avg_dd = (sum(dark_diffs) / len(dark_diffs)) if dark_diffs else 0.0
            blur_lap_var_p05 = _percentile(blur_vals, 0.05) if blur_vals else None

            # Palette continuity proxy: rgb mean jumps across adjacent segments.
            rgb_jumps: list[float] = []
            seg_order = [int(sid) for sid, _s, _e, _fp in segment_frames]
            for a, b in zip(seg_order[:-1], seg_order[1:], strict=False):
                ra = out_rgb_by_seg_id.get(int(a))
                rb = out_rgb_by_seg_id.get(int(b))
                if not (isinstance(ra, list) and isinstance(rb, list) and len(ra) == 3 and len(rb) == 3):
                    continue
                try:
                    rgb_jumps.append(sum(abs(float(rb[i]) - float(ra[i])) for i in range(3)) / 3.0)
                except Exception:
                    continue
            palette_rgb_jump_p95 = _percentile(rgb_jumps, 0.95) if rgb_jumps else None

            # Patch timeline_segments with mid-frame objectives (segment-local evidence).
            for row in timeline_segments:
                if not isinstance(row, dict):
                    continue
                try:
                    sid = int(row.get("id") or 0)
                except Exception:
                    continue
                patch = obj_by_seg_id.get(int(sid))
                if not patch:
                    continue
                row.update(patch)

            segment_objectives = [obj_by_seg_id[int(sid)] for sid in seg_order if int(sid) in obj_by_seg_id]
            try:
                write_json(vroot / "segment_objectives.json", {"schema_version": 1, "variant_id": variant_id, "segments": segment_objectives})
            except Exception:
                pass

        # Shot-index-based stability proxy (available in pro mode without any extra compute).
        shake_vals2: list[float] = []
        for row in timeline_segments:
            if not isinstance(row, dict):
                continue
            sh = row.get("shot_shake_score")
            if isinstance(sh, (int, float)):
                shake_vals2.append(float(sh))
        shake_score_p95 = _percentile(shake_vals2, 0.95) if shake_vals2 else None

        # Write timeline.json for the variant.
        timeline_doc = {
            "mode": "folder_edit_variant",
            "project_root": str(project_root),
            "variant_id": variant_id,
            "director": director_mode,
            "seed": int(args.seed) + vidx,
            "variant_params": params.__dict__,
            "reel_url_or_path": args.reel,
            "source_video": str(src_path),
            "analysis_clip": str(analysis_clip),
            "media_folder": str(media_folder),
            "reference_image": str(Path(args.ref)) if args.ref else None,
            "beat_sync": bool(beat_sync_used),
            "music_analysis": str(music_analysis_path) if (beat_sync_used and music_analysis_path.exists()) else None,
            "story_plan_id": plan_id,
            "story_plan_concept": plan_concept,
            "segments_detected": [{"id": sid, "start_s": s, "end_s": e} for sid, s, e in segments],
            "analysis": ref_analysis.analysis,
            "timeline_segments": timeline_segments,
            "audio": str(final_audio) if final_audio else None,
            "metrics": {
                "min_jaccard_dist": float(min_dist) if isinstance(min_dist, (int, float)) else None,
                "uniq_assets": len(aset) if isinstance(aset, set) else None,
                "avg_abs_luma_diff": avg_dl,
                "avg_abs_dark_diff": avg_dd,
                "palette_rgb_jump_p95": palette_rgb_jump_p95,
                "blur_lap_var_p05": blur_lap_var_p05,
                "shake_score_p95": shake_score_p95,
            },
        }
        write_json(vroot / "timeline.json", timeline_doc)

        # Persist a per-variant segment dossier and append to the global segment_runs log.
        try:
            from .segment_dossier import append_segment_runs, build_segment_dossier

            cand_for_variant: dict[int, list[dict[str, t.Any]]] | None = None
            if pro_mode:
                if plan_id and plan_id in (shot_candidates_by_seg_by_plan or {}):
                    cand_for_variant = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan or {}).get(plan_id) or {})
                else:
                    cand_for_variant = shot_candidates_by_seg

            dossier = build_segment_dossier(
                timeline_doc=timeline_doc,
                candidates_by_seg_id=cand_for_variant,
                max_candidates_per_seg=int(max(0, float(os.getenv("SEGMENT_DOSSIER_CAND_K", "24") or 24))),
            )
            write_json(vroot / "segment_dossier.json", dossier)
            append_segment_runs(
                dossier_doc=dossier,
                extra={
                    "pipeline": "run_folder_edit_variants",
                    "phase": "render",
                    "pro_mode": bool(pro_mode),
                    "analysis_model": str(args.model),
                },
            )
        except Exception:
            pass

        if (vidx - start + 1) % 5 == 0 or vidx == end:
            print(f"[{vidx-start+1}/{variants_n}] wrote {final_video} (avg_dl={avg_dl:.4f} avg_dd={avg_dd:.4f})")

    # Write (or rewrite) index.tsv by reading variant timelines (robust across script versions).
    vids = sorted([p.name for p in variants_dir.iterdir() if p.is_dir() and p.name.startswith("v")])
    vids = [vid for vid in vids if (variants_dir / vid / "timeline.json").exists()]
    sets_by_vid: dict[str, set[str]] = {vid: _load_variant_asset_set(variants_dir / vid) for vid in vids}
    variant_meta: dict[str, dict[str, t.Any]] = {}
    out_lines: list[str] = [index_header]
    for vid in vids:
        vdir = variants_dir / vid
        tpath = vdir / "timeline.json"
        seed_val: str = ""
        avg_dl = 0.0
        avg_dd = 0.0
        palette_p95: float | None = None
        blur_p05: float | None = None
        shake_p95: float | None = None
        story_plan_id: str | None = None
        if tpath.exists():
            try:
                doc = json.loads(tpath.read_text(encoding="utf-8", errors="replace"))
                seed_val = str(doc.get("seed") or "")
                story_plan_id = str(doc.get("story_plan_id") or "").strip() or None
                metrics = doc.get("metrics") or {}
                if isinstance(metrics, dict):
                    avg_dl = float(metrics.get("avg_abs_luma_diff") or 0.0)
                    avg_dd = float(metrics.get("avg_abs_dark_diff") or 0.0)
                    palette_p95 = _safe_float(metrics.get("palette_rgb_jump_p95"))
                    blur_p05 = _safe_float(metrics.get("blur_lap_var_p05"))
                    shake_p95 = _safe_float(metrics.get("shake_score_p95"))
            except Exception:
                pass

        aset = sets_by_vid.get(vid) or set()
        uniq_assets = len(aset)
        if len(vids) <= 1 or not aset:
            md = 1.0
        else:
            md = min((_jaccard_distance(aset, sets_by_vid[o]) for o in vids if o != vid), default=1.0)

        out_lines.append(
            f"{vid}\t{seed_val}\t{md:.3f}\t{uniq_assets}\t{avg_dl:.6f}\t{avg_dd:.6f}\t{(vdir / 'final_video.mov').resolve().as_posix()}"
        )
        variant_meta[vid] = {
            "variant_id": vid,
            "seed": seed_val or None,
            "min_jaccard_dist": float(md) if isinstance(md, (int, float)) else None,
            "uniq_assets": int(uniq_assets),
            "avg_abs_luma_diff": float(avg_dl),
            "avg_abs_dark_diff": float(avg_dd),
            "palette_rgb_jump_p95": palette_p95,
            "blur_lap_var_p05": blur_p05,
            "shake_score_p95": shake_p95,
            "story_plan_id": story_plan_id,
            "final_video": str((vdir / "final_video.mov").resolve()),
        }

    index_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    if not vids:
        raise RuntimeError("No variants found; nothing to select for finals/")

    # --- Finals selection + (optional) deterministic segment fix loop ---
    director_mode = str(getattr(args, "director", "code") or "code").strip().lower()
    if director_mode not in {"code", "gemini", "auto"}:
        director_mode = "code"
    fix_iters = max(0, int(getattr(args, "fix_iters", 0) or 0))

    # Optional: learned grader predictions (used as a small tie-breaker/steering signal).
    learned_grader = None
    learned_grader_info: dict[str, t.Any] | None = None
    if pro_mode and shot_by_id and _truthy_env("VARIANT_LEARNED_GRADER", "1"):
        try:
            from .grader_steering import find_latest_grader_dir, fast_features_for_sequence
            from .learned_grader import GeminiGrader

            grader_dir_env = str(os.getenv("VARIANT_LEARNED_GRADER_DIR", "") or os.getenv("FOLDER_EDIT_LEARNED_GRADER_DIR", "") or "").strip()
            grader_dir: Path | None = None
            if grader_dir_env:
                grader_dir = Path(grader_dir_env).expanduser().resolve()
            else:
                # Prefer searching next to the project when it lives under Outputs/.
                search_root = project_root.parent if project_root.parent.exists() else Path("Outputs")
                grader_dir = find_latest_grader_dir(search_root) or find_latest_grader_dir(Path("Outputs"))

            if grader_dir and grader_dir.exists():
                learned_grader = GeminiGrader.load(grader_dir)
                learned_grader_info = {"ok": True, "grader_dir": str(grader_dir)}
        except Exception as e:
            learned_grader = None
            learned_grader_info = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if learned_grader is not None:
        try:
            # Build plan-specific segment lists (semantic features depend on desired_tags/story fields).
            segments_by_plan_id: dict[str, list[t.Any]] = {"default": list(ref_analysis.segments)}
            for pid, segs0 in (story_segments_by_plan or {}).items():
                segments_by_plan_id[str(pid)] = list(segs0 or [])

            preds: dict[str, dict[str, float]] = {}
            for vid in vids:
                try:
                    doc = json.loads((variants_dir / vid / "timeline.json").read_text(encoding="utf-8", errors="replace") or "{}")
                except Exception:
                    continue
                seg_rows = doc.get("timeline_segments") if isinstance(doc.get("timeline_segments"), list) else []
                if not isinstance(seg_rows, list) or not seg_rows:
                    continue

                plan_id0 = str(doc.get("story_plan_id") or "").strip() or "default"
                seg_objs = segments_by_plan_id.get(plan_id0) or segments_by_plan_id.get("default") or []
                shot_ids0 = [str(r.get("shot_id") or "").strip() for r in seg_rows if isinstance(r, dict)]
                if not shot_ids0 or len(shot_ids0) != len(seg_objs):
                    continue
                seq: list[dict[str, t.Any]] = []
                ok = True
                for sid in shot_ids0:
                    sh = shot_by_id.get(sid)
                    if not isinstance(sh, dict):
                        ok = False
                        break
                    seq.append(sh)
                if not ok:
                    continue

                speeds = [float(r.get("speed") or 1.0) for r in seg_rows if isinstance(r, dict) and isinstance(r.get("speed"), (int, float))]
                zooms = [float(r.get("zoom") or 1.0) for r in seg_rows if isinstance(r, dict) and isinstance(r.get("zoom"), (int, float))]
                default_speed = float(sum(speeds) / len(speeds)) if speeds else 1.0
                default_zoom = float(sum(zooms) / len(zooms)) if zooms else float(_float_env("FOLDER_EDIT_ZOOM", "1.0"))

                feats = fast_features_for_sequence(
                    segments=seg_objs,
                    shots=seq,
                    default_speed=default_speed,
                    default_zoom=default_zoom,
                    stabilize_enabled=_truthy_env("FOLDER_EDIT_STABILIZE", "1"),
                    stabilize_shake_th=_float_env("FOLDER_EDIT_STEER_STABILIZE_SHAKE_TH", "0.25"),
                )
                pred = learned_grader.predict(feats)
                preds[str(vid)] = pred
                try:
                    variant_meta[vid]["learned_pred"] = pred
                except Exception:
                    pass

            write_json(project_root / "learned_grader_predictions.json", {"schema_version": 1, "grader": learned_grader_info, "preds": preds})
        except Exception:
            pass

    # Score function: deterministic composite that favors reference match + stability + look continuity.
    def _rank_score(m: dict[str, t.Any]) -> float:
        dl0 = _safe_float(m.get("avg_abs_luma_diff")) or 0.0
        dd0 = _safe_float(m.get("avg_abs_dark_diff")) or 0.0
        cheap = float(dl0 + (0.8 * dd0))
        md0 = _safe_float(m.get("min_jaccard_dist")) or 0.0
        uniq0 = _safe_float(m.get("uniq_assets")) or 0.0
        shake0 = _safe_float(m.get("shake_score_p95"))
        blur0 = _safe_float(m.get("blur_lap_var_p05"))
        pal0 = _safe_float(m.get("palette_rgb_jump_p95"))

        score = -cheap
        # Prefer diversity, but keep weight modest so we don't reward random nonsense.
        score += float(md0) * 0.18
        score += min(12.0, float(uniq0)) * 0.015
        # Penalize shakiness; even if it "matches" reference style, it hurts publishability.
        if shake0 is not None:
            score -= float(shake0) * 2.8
        # Prefer sharper frames (blur metric is Laplacian variance, higher is better).
        if blur0 is not None:
            score += min(600.0, float(blur0)) / 600.0 * 0.10
        # Penalize big palette jumps (proxy for look inconsistency).
        if pal0 is not None:
            score -= float(pal0) * 0.65

        # Learned grader (if present): light tie-breaker toward predicted publishability and stability.
        pred = m.get("learned_pred") if isinstance(m.get("learned_pred"), dict) else None
        if isinstance(pred, dict):
            po = _safe_float(pred.get("overall_score"))
            ps = _safe_float(pred.get("stability"))
            if po is not None:
                score += (float(po) / 10.0) * float(_float_env("RANK_LEARNED_OVERALL_W", "0.12"))
            if ps is not None:
                score += (float(ps) / 5.0) * float(_float_env("RANK_LEARNED_STABILITY_W", "0.06"))
        return float(score)

    ranked_vids = sorted(vids, key=lambda vid: _rank_score(variant_meta.get(vid, {})), reverse=True)
    # Enforce finals diversity (looser than generation-time constraints so we always fill finals).
    finals_min_jacc = float(_float_env("FINAL_MIN_JACCARD_DIST", "0.12"))
    chosen: list[str] = []
    for vid in ranked_vids:
        if len(chosen) >= int(finals_n):
            break
        aset = sets_by_vid.get(vid) or set()
        if not chosen:
            chosen.append(vid)
            continue
        ok = True
        for prev in chosen:
            dist = _jaccard_distance(aset, sets_by_vid.get(prev) or set())
            if dist < float(finals_min_jacc):
                ok = False
                break
        if ok:
            chosen.append(vid)
    # If we couldn't satisfy diversity, fill remaining by rank.
    for vid in ranked_vids:
        if len(chosen) >= int(finals_n):
            break
        if vid in chosen:
            continue
        chosen.append(vid)

    # Deterministic segment-level fix loop for finalists (cheap, no API calls).
    if fix_iters > 0 and pro_mode:
        try:
            from .micro_editor import pick_inpoint_for_shot
        except Exception:
            pick_inpoint_for_shot = None  # type: ignore[assignment]

        # Segment objects by plan_id for micro-editor/in-point selection.
        segments_by_plan_id: dict[str, list[t.Any]] = {"default": list(ref_analysis.segments)}
        for pid, segs0 in (story_segments_by_plan or {}).items():
            segments_by_plan_id[str(pid)] = list(segs0 or [])

        def _segment_badness(*, seg_obj: t.Any, shot: dict[str, t.Any]) -> float:
            # Target: reduce shake and extreme blur; keep reference lighting roughly matched.
            ref_l = _safe_float(getattr(seg_obj, "ref_luma", None))
            ref_d = _safe_float(getattr(seg_obj, "ref_dark_frac", None))
            sl = _safe_float(shot.get("luma_mean"))
            sd = _safe_float(shot.get("dark_frac"))
            dl = abs(float(sl) - float(ref_l)) if (ref_l is not None and sl is not None) else 0.18
            dd = abs(float(sd) - float(ref_d)) if (ref_d is not None and sd is not None) else 0.25
            sh = _safe_float(shot.get("shake_score")) or 0.0
            sharp = _safe_float(shot.get("sharpness")) or 120.0
            # Shake dominates perceived quality; blur penalty is mild.
            bad = (dl * 1.1) + (dd * 0.9)
            bad += max(0.0, float(sh) - float(_float_env("FIX_SHAKE_START", "0.22"))) * 3.0
            bad += max(0.0, float(_float_env("FIX_SHARP_MIN", "80.0")) - float(sharp)) * 0.002
            return float(bad)

        def _fix_one_variant(vid: str) -> None:
            vdir = variants_dir / vid
            tpath = vdir / "timeline.json"
            if not tpath.exists():
                return
            doc = json.loads(tpath.read_text(encoding="utf-8", errors="replace") or "{}")
            seg_rows = doc.get("timeline_segments") if isinstance(doc.get("timeline_segments"), list) else []
            if not isinstance(seg_rows, list) or not seg_rows:
                return
            plan_id0 = str(doc.get("story_plan_id") or "").strip() or "default"
            seg_objs = segments_by_plan_id.get(plan_id0) or segments_by_plan_id.get("default") or []
            seg_obj_by_id = {int(getattr(s, "id", 0) or 0): s for s in seg_objs}
            if not seg_obj_by_id:
                return

            # Build current usage for diversity caps.
            used_assets: dict[str, int] = {}
            for row in seg_rows:
                if not isinstance(row, dict):
                    continue
                aid = str(row.get("asset_id") or "").strip()
                if aid:
                    used_assets[aid] = used_assets.get(aid, 0) + 1

            # Candidate lists for this plan.
            cand_by_seg = shot_candidates_by_seg
            if plan_id0 != "default" and plan_id0 in (shot_candidates_by_seg_by_plan or {}):
                cand_by_seg = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan or {}).get(plan_id0) or cand_by_seg)

            for it in range(int(fix_iters)):
                # Find worst segments by badness.
                scored: list[tuple[float, int]] = []
                for row in seg_rows:
                    if not isinstance(row, dict):
                        continue
                    sid = int(row.get("id") or 0)
                    if sid <= 0:
                        continue
                    shot_id = str(row.get("shot_id") or "").strip()
                    if not shot_id:
                        continue
                    shot = shot_by_id.get(shot_id) or {}
                    seg_obj = seg_obj_by_id.get(int(sid))
                    if seg_obj is None or not isinstance(shot, dict):
                        continue
                    scored.append((_segment_badness(seg_obj=seg_obj, shot=shot), int(sid)))
                scored.sort(reverse=True)
                worst = [sid for _b, sid in scored[:2] if _b > float(_float_env("FIX_BADNESS_MIN", "0.42"))]
                if not worst:
                    break

                changed = False
                for sid in worst:
                    seg_obj = seg_obj_by_id.get(int(sid))
                    if seg_obj is None:
                        continue
                    row = next((r for r in seg_rows if isinstance(r, dict) and int(r.get("id") or 0) == int(sid)), None)
                    if not isinstance(row, dict):
                        continue
                    cur_shot_id = str(row.get("shot_id") or "").strip()
                    cur_shot = shot_by_id.get(cur_shot_id) if cur_shot_id else None
                    if not isinstance(cur_shot, dict):
                        continue
                    cur_bad = _segment_badness(seg_obj=seg_obj, shot=cur_shot)

                    cands = list(cand_by_seg.get(int(sid)) or [])
                    best = None
                    best_bad = cur_bad
                    for cand in cands[:80]:
                        if not isinstance(cand, dict):
                            continue
                        csid = str(cand.get("id") or "").strip()
                        if not csid or csid == cur_shot_id:
                            continue
                        aid = str(cand.get("asset_id") or "").strip()
                        # Avoid overusing the same source asset in finals.
                        if aid and used_assets.get(aid, 0) >= 2:
                            continue
                        b0 = _segment_badness(seg_obj=seg_obj, shot=cand)
                        if b0 + 0.04 < best_bad:
                            best_bad = b0
                            best = cand
                    if best is None:
                        continue

                    # Apply replacement.
                    new_shot_id = str(best.get("id") or "").strip()
                    new_aid = str(best.get("asset_id") or "").strip()
                    new_path = str(best.get("asset_path") or "").strip()
                    if not (new_shot_id and new_aid and new_path):
                        continue
                    # Update usage counts.
                    if new_aid:
                        used_assets[new_aid] = used_assets.get(new_aid, 0) + 1

                    chosen_t = float(_safe_float(best.get("start_s")) or 0.0)
                    in_meta: dict[str, t.Any] | None = None
                    if callable(pick_inpoint_for_shot):
                        try:
                            chosen_t, in_meta = pick_inpoint_for_shot(segment=seg_obj, shot=best, work_dir=vdir / "fix_refine", timeout_s=timeout_s)
                        except Exception:
                            chosen_t, in_meta = float(_safe_float(best.get("start_s")) or 0.0), None

                    row["shot_id"] = new_shot_id
                    row["sequence_group_id"] = best.get("sequence_group_id")
                    row["asset_id"] = new_aid
                    row["asset_path"] = new_path
                    row["asset_in_s"] = float(chosen_t)
                    try:
                        row["asset_out_s"] = float(chosen_t) + float(row.get("duration_s") or 0.0)
                    except Exception:
                        pass
                    # Update shot objective fields.
                    for k in (
                        "motion_score",
                        "shake_score",
                        "sharpness",
                        "luma_mean",
                        "dark_frac",
                        "rgb_mean",
                        "cam_motion_mag",
                        "cam_motion_angle_deg",
                    ):
                        row_key = f"shot_{k}"
                        row[row_key] = best.get(k)
                    if in_meta is not None:
                        row["inpoint_auto"] = in_meta

                    # Re-render this segment deterministically.
                    seg_path = vdir / "segments" / f"seg_{int(sid):02d}.mp4"
                    try:
                        grade = row.get("grade") if isinstance(row.get("grade"), dict) else None
                        # Recompute grade from inpoint sample metrics when available.
                        chosen_info = (in_meta or {}).get("chosen") if isinstance(in_meta, dict) else None
                        if isinstance(chosen_info, dict):
                            grade = _compute_eq_grade(
                                ref_luma=_safe_float(getattr(seg_obj, "ref_luma", None)),
                                ref_dark=_safe_float(getattr(seg_obj, "ref_dark_frac", None)),
                                out_luma=_safe_float(chosen_info.get("luma")),
                                out_dark=_safe_float(chosen_info.get("dark_frac")),
                                ref_rgb=getattr(seg_obj, "ref_rgb_mean", None),
                                out_rgb=(chosen_info.get("rgb_mean") if isinstance(chosen_info.get("rgb_mean"), list) else None),
                            )
                            row["grade"] = grade
                        fade_in_s = None
                        fade_in_color = None
                        try:
                            fade_in_s = float(row.get("fade_in_s")) if row.get("fade_in_s") is not None else None
                        except Exception:
                            fade_in_s = None
                        if isinstance(row.get("fade_in_color"), str):
                            fade_in_color = str(row.get("fade_in_color") or "").strip().lower() or None
                        fade_out_s = None
                        fade_out_color = None
                        try:
                            fade_out_s = float(row.get("fade_out_s")) if row.get("fade_out_s") is not None else None
                        except Exception:
                            fade_out_s = None
                        if isinstance(row.get("fade_out_color"), str):
                            fade_out_color = str(row.get("fade_out_color") or "").strip().lower() or None
                        _render_segment(
                            asset_path=Path(new_path),
                            asset_kind="video",
                            in_s=float(chosen_t),
                            duration_s=float(row.get("duration_s") or 0.0),
                            speed=float(row.get("speed") or 1.0),
                            crop_mode=str(row.get("crop_mode") or "center"),
                            overlay_text=str(row.get("overlay_text") or ""),
                            grade=(grade if isinstance(grade, dict) else None),
                            stabilize=bool(row.get("stabilize") or False),
                            stabilize_cache_dir=(project_root / "library" / "stabilized") if bool(row.get("stabilize") or False) else None,
                            zoom=float(_safe_float(row.get("zoom")) or 1.0),
                            fade_in_s=fade_in_s,
                            fade_in_color=fade_in_color,
                            fade_out_s=fade_out_s,
                            fade_out_color=fade_out_color,
                            output_path=seg_path,
                            burn_overlay=burn_overlays,
                            timeout_s=timeout_s,
                        )
                        changed = True
                    except Exception:
                        # Keep timeline changes but skip render failures (best-effort).
                        pass

                if not changed:
                    break

                # Re-concat + update final video after each fix iteration.
                try:
                    from .folder_edit_pipeline import _concat_segments  # type: ignore

                    seg_paths = [vdir / "segments" / f"seg_{int(getattr(s, 'id', 0)):02d}.mp4" for s in seg_objs]
                    seg_paths = [p for p in seg_paths if p.exists()]
                    silent_out = vdir / "final_video_silent_fixed.mp4"
                    _concat_segments(segment_paths=seg_paths, output_path=silent_out, timeout_s=timeout_s)
                    fixed_out = vdir / "final_video_fixed.mov"
                    shutil.copyfile(silent_out, fixed_out)
                    if final_audio:
                        merge_audio(video_path=fixed_out, audio_path=final_audio)
                    shutil.copyfile(fixed_out, vdir / "final_video.mov")
                    doc["metrics"] = dict(doc.get("metrics") or {})
                    doc["metrics"]["fix_iters_applied"] = int(it + 1)
                    write_json(tpath, doc)
                except Exception:
                    pass

        try:
            for vid in chosen:
                _fix_one_variant(vid)
        except Exception:
            pass

    # Optional: critic-driven fix loop for finalists (uses compare-video critic JSON, compiled into deterministic changes).
    critic_fix_iters = max(0, int(getattr(args, "critic_fix_iters", 0) or 0))
    if critic_fix_iters > 0 and pro_mode:
        try:
            from .folder_edit_pipeline import _default_critic_model  # type: ignore
            from .compare_video_critic import critique_compare_video
            from .critic_schema import severity_rank
            from .fix_actions import apply_fix_actions, apply_segment_deltas_to_timeline, apply_transition_deltas
            from .review_render import render_compare_video, render_labeled_review_video
            from types import SimpleNamespace
        except Exception as e:
            print(f"[warn] critic fix loop disabled (missing deps). {type(e).__name__}: {e}", flush=True)
        else:
            critic_model_eff = _default_critic_model(analysis_model=str(args.model), critic_model=str(getattr(args, "critic_model", "") or ""))
            critic_max_mb = float(getattr(args, "critic_max_mb", 8.0) or 8.0)
            critic_pro_mode = bool(getattr(args, "critic_pro_mode", None)) if getattr(args, "critic_pro_mode", None) is not None else bool(pro_mode)

            max_fix_finals = int(max(1, float(os.getenv("VARIANT_CRITIC_FIX_MAX_FINALS", "2") or 2)))
            max_recasts = int(max(0, float(os.getenv("VARIANT_CRITIC_RECAST_MAX", "2") or 2)))
            recast_min_overall = float(os.getenv("VARIANT_CRITIC_RECAST_OVERALL_MAX", "2.4") or 2.4)
            recast_min_stab = float(os.getenv("VARIANT_CRITIC_RECAST_STABILITY_MAX", "2.2") or 2.2)

            def _timeline_summary_for_critic(doc0: dict[str, t.Any]) -> dict[str, t.Any]:
                segs0 = doc0.get("timeline_segments") if isinstance(doc0.get("timeline_segments"), list) else []
                out0: list[dict[str, t.Any]] = []
                for s0 in segs0:
                    if not isinstance(s0, dict):
                        continue
                    try:
                        sid0 = int(s0.get("id") or 0)
                    except Exception:
                        sid0 = 0
                    if sid0 <= 0:
                        continue
                    overlay = str(s0.get("overlay_text") or "").replace("\n", " ").strip()
                    if len(overlay) > 60:
                        overlay = overlay[:57] + "..."
                    tags0 = s0.get("desired_tags") if isinstance(s0.get("desired_tags"), list) else []
                    tags = [str(x).strip().lower() for x in (tags0 or []) if str(x).strip()][:6]
                    sb = str(s0.get("story_beat") or "").strip()
                    if len(sb) > 60:
                        sb = sb[:57] + "..."
                    out0.append(
                        {
                            "segment_id": sid0,
                            "beat_goal": s0.get("beat_goal"),
                            "overlay_text": overlay,
                            "desired_tags": tags,
                            "story_beat": (sb or None),
                            "transition_hint": s0.get("transition_hint"),
                        }
                    )
                return {"segments": out0}

            def _tag_set_for_shot(shot: dict[str, t.Any]) -> set[str]:
                out: set[str] = set()
                tags0 = shot.get("tags") if isinstance(shot.get("tags"), list) else []
                for tg0 in tags0 or []:
                    nt = " ".join(str(tg0 or "").strip().lower().split())
                    if nt:
                        out.add(nt)
                for k0 in ("shot_type", "setting", "mood"):
                    v0 = shot.get(k0)
                    nv = " ".join(str(v0 or "").strip().lower().split())
                    if nv:
                        out.add(nv)
                return out

            def _segment_energy(row: dict[str, t.Any]) -> float:
                me = _safe_float(row.get("music_energy"))
                try:
                    dur = float(row.get("duration_s") or 1.0)
                except Exception:
                    dur = 1.0
                dur_e = float(_segment_energy_hint(float(dur)))
                if me is None:
                    return float(dur_e)
                # Blend instead of replacing: music_energy can be conservative early in tracks.
                me_e = float(max(0.0, min(1.0, float(me))))
                return float(max(0.0, min(1.0, (0.65 * float(dur_e)) + (0.35 * float(me_e)))))

            def _recast_cost(*, seg_row: dict[str, t.Any], shot: dict[str, t.Any], prev_row: dict[str, t.Any] | None, next_row: dict[str, t.Any] | None) -> float:
                # Lighting distance.
                rl = _safe_float(seg_row.get("ref_luma"))
                rd = _safe_float(seg_row.get("ref_dark_frac"))
                sl = _safe_float(shot.get("luma_mean"))
                sd = _safe_float(shot.get("dark_frac"))
                dl = abs(float(sl) - float(rl)) if (rl is not None and sl is not None) else 0.18
                dd = abs(float(sd) - float(rd)) if (rd is not None and sd is not None) else 0.25
                lighting = (dl * 1.2) + (dd * 1.0)

                # Color distance (mild).
                color = 0.0
                rrgb = seg_row.get("ref_rgb_mean")
                sgb = shot.get("rgb_mean")
                if (
                    isinstance(rrgb, list)
                    and isinstance(sgb, list)
                    and len(rrgb) == 3
                    and len(sgb) == 3
                    and all(isinstance(x, (int, float)) for x in rrgb)
                    and all(isinstance(x, (int, float)) for x in sgb)
                ):
                    try:
                        color = (sum(abs(float(rrgb[i]) - float(sgb[i])) for i in range(3)) / 3.0) * 0.55
                    except Exception:
                        color = 0.0

                # Energy/motion match.
                energy = _segment_energy(seg_row)
                motion = _safe_float(shot.get("motion_score"))
                mv = float(motion) if motion is not None else 0.35
                motion_w = float(_float_env("OPT_MOTION_W", "0.45"))
                motion_dist = abs(float(mv) - float(energy)) * float(motion_w)

                # Tags.
                desired = seg_row.get("desired_tags") if isinstance(seg_row.get("desired_tags"), list) else []
                desired_set = {" ".join(str(x or "").strip().lower().split()) for x in desired if str(x or "").strip()}
                overlap = len(desired_set.intersection(_tag_set_for_shot(shot))) if desired_set else 0
                tag_bonus = -0.10 * float(min(6, overlap))

                # Quality: blur + shake.
                sharp = _safe_float(shot.get("sharpness"))
                shake = _safe_float(shot.get("shake_score"))
                sharp_pen = 0.0
                if sharp is not None:
                    if float(sharp) < 60.0:
                        sharp_pen = 0.12
                    elif float(sharp) < 120.0:
                        sharp_pen = 0.06
                shake_pen = 0.0
                if shake is not None:
                    start0 = float(_float_env("FIX_SHAKE_START", "0.22"))
                    if float(shake) > float(start0):
                        shake_pen = min(0.85, (float(shake) - float(start0)) * 3.0)

                # Continuity: avoid introducing huge jumps when the reference doesn't jump.
                cont = 0.0
                if isinstance(prev_row, dict):
                    prl = _safe_float(prev_row.get("ref_luma"))
                    rrl = _safe_float(seg_row.get("ref_luma"))
                    psl = _safe_float(prev_row.get("shot_luma_mean"))
                    sl2 = _safe_float(shot.get("luma_mean"))
                    if prl is not None and rrl is not None and psl is not None and sl2 is not None:
                        ref_jump = abs(float(rrl) - float(prl))
                        out_jump = abs(float(sl2) - float(psl))
                        if ref_jump < 0.06 and out_jump > 0.14:
                            cont += (out_jump - 0.14) * 0.7
                if isinstance(next_row, dict):
                    # Keep this mild; next segment may get recast too.
                    nsl = _safe_float(next_row.get("shot_luma_mean"))
                    sl3 = _safe_float(shot.get("luma_mean"))
                    if nsl is not None and sl3 is not None:
                        out_jump2 = abs(float(nsl) - float(sl3))
                        if out_jump2 > 0.18:
                            cont += (out_jump2 - 0.18) * 0.25

                return float(lighting + color + motion_dist + tag_bonus + sharp_pen + shake_pen + cont)

            def _choose_recast_shot(
                *,
                seg_row: dict[str, t.Any],
                candidates: list[dict[str, t.Any]],
                used_assets: dict[str, int],
                prev_row: dict[str, t.Any] | None,
                next_row: dict[str, t.Any] | None,
            ) -> dict[str, t.Any] | None:
                cur_id = str(seg_row.get("shot_id") or "").strip()
                cur = shot_by_id.get(cur_id) if cur_id else None
                cur_cost = _recast_cost(seg_row=seg_row, shot=cur, prev_row=prev_row, next_row=next_row) if isinstance(cur, dict) else 9.9

                max_per_asset = max(1, _int_env("VARIANT_MAX_SEGMENTS_PER_ASSET", "2"))
                shake_max = float(_float_env("VARIANT_GEMINI_SHAKE_MAX", "0.22"))
                sharp_min = float(_float_env("VARIANT_GEMINI_SHARP_MIN", "120.0"))
                shake_delta = float(_float_env("VARIANT_GEMINI_SHAKE_DELTA", "0.03"))
                sharp_delta = float(_float_env("VARIANT_GEMINI_SHARP_DELTA", "10.0"))

                cur_shake = _safe_float(cur.get("shake_score")) if isinstance(cur, dict) else None
                cur_sharp = _safe_float(cur.get("sharpness")) if isinstance(cur, dict) else None
                # If the current shot violates hard stability thresholds, force a replacement
                # when a stable alternative exists (even if other costs are similar).
                force_stability = False
                if cur_shake is not None and float(cur_shake) > float(shake_max):
                    force_stability = True
                if cur_sharp is not None and float(cur_sharp) < float(sharp_min):
                    force_stability = True

                best: dict[str, t.Any] | None = None
                best_cost = float("inf") if force_stability else float(cur_cost)
                for c in candidates[:80]:
                    if not isinstance(c, dict):
                        continue
                    sid = str(c.get("id") or "").strip()
                    if not sid or sid == cur_id:
                        continue
                    aid = str(c.get("asset_id") or "").strip()
                    if aid and int(used_assets.get(aid, 0)) >= int(max_per_asset):
                        continue
                    sh = _safe_float(c.get("shake_score"))
                    sp = _safe_float(c.get("sharpness"))
                    if sh is not None and float(sh) > float(shake_max):
                        continue
                    if sp is not None and float(sp) < float(sharp_min):
                        continue
                    # Stability must not drop versus current shot when comparable alternatives exist.
                    if cur_shake is not None and sh is not None and float(sh) > float(cur_shake) + float(shake_delta):
                        continue
                    if cur_sharp is not None and sp is not None and float(sp) + float(sharp_delta) < float(cur_sharp):
                        continue
                    cost = _recast_cost(seg_row=seg_row, shot=c, prev_row=prev_row, next_row=next_row)
                    if force_stability:
                        if cost < best_cost:
                            best_cost = float(cost)
                            best = c
                    else:
                        if cost + 0.04 < best_cost:
                            best_cost = float(cost)
                            best = c
                return best

            def _critic_fix_one_variant(vid: str) -> None:
                vdir = variants_dir / vid
                tpath = vdir / "timeline.json"
                if not tpath.exists():
                    return
                doc0 = json.loads(tpath.read_text(encoding="utf-8", errors="replace") or "{}")
                seg_rows0 = doc0.get("timeline_segments") if isinstance(doc0.get("timeline_segments"), list) else []
                if not isinstance(seg_rows0, list) or not seg_rows0:
                    return

                # Candidate lists for this plan.
                plan_id0 = str(doc0.get("story_plan_id") or "").strip() or "default"
                cand_by_seg = shot_candidates_by_seg
                if plan_id0 != "default" and plan_id0 in (shot_candidates_by_seg_by_plan or {}):
                    cand_by_seg = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan or {}).get(plan_id0) or cand_by_seg)

                # Reuse the canonical segment render paths.
                seg_dir = vdir / "segments"
                seg_dir.mkdir(parents=True, exist_ok=True)

                # Current segment order for concat.
                seg_order = [int(r.get("id") or 0) for r in seg_rows0 if isinstance(r, dict)]
                seg_order = [sid for sid in seg_order if sid > 0]

                # Resolve media inputs for critic.
                analysis_clip_path = Path(str(doc0.get("analysis_clip") or (project_root / "reference" / "analysis_clip.mp4"))).expanduser()
                out_video_path = vdir / "final_video.mov"
                if not analysis_clip_path.exists() or not out_video_path.exists():
                    return

                # Precompute segment boundaries for labeled review renders.
                segs_for_review: list[dict[str, t.Any]] = []
                for r in seg_rows0:
                    if not isinstance(r, dict):
                        continue
                    try:
                        sid = int(r.get("id") or 0)
                        s = float(r.get("start_s") or 0.0)
                        e = float(r.get("end_s") or 0.0)
                    except Exception:
                        continue
                    if sid > 0 and e > s + 1e-3:
                        segs_for_review.append({"id": int(sid), "start_s": float(s), "end_s": float(e)})

                # Track asset usage for repetition caps.
                used_assets: dict[str, int] = {}
                for r in seg_rows0:
                    if not isinstance(r, dict):
                        continue
                    aid = str(r.get("asset_id") or "").strip()
                    if aid:
                        used_assets[aid] = used_assets.get(aid, 0) + 1

                for it in range(int(critic_fix_iters)):
                    iter_root = vdir / "critic_fix" / f"iter_{it:02d}"
                    iter_root.mkdir(parents=True, exist_ok=True)
                    ref_review = iter_root / "reference_review.mp4"
                    out_review = iter_root / "output_review.mp4"
                    compare_path = iter_root / "compare.mp4"
                    try:
                        render_labeled_review_video(
                            src_video=analysis_clip_path,
                            segments=segs_for_review,
                            out_path=ref_review,
                            label_prefix="S",
                            keep_audio=False,
                            timeout_s=min(timeout_s, 240.0),
                        )
                        render_labeled_review_video(
                            src_video=out_video_path,
                            segments=segs_for_review,
                            out_path=out_review,
                            label_prefix="S",
                            keep_audio=True,
                            timeout_s=min(timeout_s, 240.0),
                        )
                        render_compare_video(ref_review=ref_review, out_review=out_review, out_path=compare_path, timeout_s=min(timeout_s, 240.0))
                    except Exception as e:
                        print(f"[warn] critic fix review render failed for {vid}. {type(e).__name__}: {e}", flush=True)
                        break

                    try:
                        critic_res = critique_compare_video(
                            api_key=api_key,
                            model=str(critic_model_eff),
                            compare_video_path=compare_path,
                            timeline_summary=_timeline_summary_for_critic(doc0),
                            niche=str(getattr(args, "niche", "") or ""),
                            vibe=str(getattr(args, "vibe", "") or ""),
                            critic_pro_mode=bool(critic_pro_mode),
                            max_mb=float(critic_max_mb),
                            tmp_dir=(iter_root / "tmp"),
                            timeout_s=min(timeout_s, 240.0),
                        )
                        report = critic_res.report
                        (iter_root / "critique.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
                    except Exception as e:
                        # Make failures visible: without this, benchmarks can silently skip the critic loop.
                        err_doc = {"ok": False, "error": f"{type(e).__name__}: {e}", "model": str(critic_model_eff)}
                        try:
                            (iter_root / "critique_error.json").write_text(json.dumps(err_doc, indent=2) + "\n", encoding="utf-8")
                        except Exception:
                            pass
                        print(f"[warn] critic fix critic call failed for {vid}. {type(e).__name__}: {e}", flush=True)
                        break

                    # Convert Lane A actions to dicts for executor.
                    lane_a_actions: list[dict[str, t.Any]] = []
                    for a in report.lane_a_actions:
                        item: dict[str, t.Any] = {"type": a.type, "segment_id": int(a.segment_id)}
                        if a.seconds is not None:
                            item["seconds"] = float(a.seconds)
                        if a.value is not None:
                            item["value"] = a.value
                        lane_a_actions.append(item)

                    transition_deltas = [
                        {"boundary_after_segment_id": int(td.boundary_after_segment_id), "type": td.type, "seconds": float(td.seconds)}
                        for td in report.transition_deltas
                    ]

                    # Keep <=2 Lane B deltas (highest severity).
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

                    patched_doc, lane_a_report = apply_fix_actions(doc0, lane_a_actions)
                    patched_doc, trans_report = apply_transition_deltas(patched_doc, transition_deltas)
                    patched_doc, lane_b_report = apply_segment_deltas_to_timeline(patched_doc, lane_b_deltas, allow_pro_fields=bool(critic_pro_mode))

                    applied_lane_a = lane_a_report.get("applied") if isinstance(lane_a_report.get("applied"), list) else []
                    applied_lane_b = lane_b_report.get("applied") if isinstance(lane_b_report.get("applied"), list) else []

                    # Decide recast targets: lane_b segments + worst segment_scores.
                    recast_ids: set[int] = set()
                    for b in applied_lane_b:
                        if not isinstance(b, dict):
                            continue
                        try:
                            recast_ids.add(int(b.get("segment_id") or 0))
                        except Exception:
                            pass
                    # Worst by critic segment_scores (0..5).
                    by_sid_score: dict[int, tuple[float | None, float | None]] = {}
                    for sc in report.segment_scores:
                        try:
                            sid = int(sc.segment_id)
                        except Exception:
                            continue
                        by_sid_score[int(sid)] = (_safe_float(sc.overall), _safe_float(sc.stability))
                    scored_ids: list[tuple[float, float, int]] = []
                    for sid in seg_order:
                        ov, st = by_sid_score.get(int(sid), (None, None))
                        o0 = float(ov) if ov is not None else 3.0
                        s0 = float(st) if st is not None else 3.0
                        scored_ids.append((o0, s0, int(sid)))
                    scored_ids.sort(key=lambda x: (x[0], x[1]))
                    for o0, s0, sid in scored_ids[: max(1, max_recasts)]:
                        if float(o0) <= float(recast_min_overall) or float(s0) <= float(recast_min_stab):
                            recast_ids.add(int(sid))

                    # Cap recasts.
                    recast_list = [sid for sid in sorted(recast_ids) if sid > 0][: max(0, int(max_recasts))]

                    # Patch judge fields into timeline_segments for downstream logging.
                    seg_rows1 = patched_doc.get("timeline_segments") if isinstance(patched_doc.get("timeline_segments"), list) else []
                    if isinstance(seg_rows1, list):
                        for r in seg_rows1:
                            if not isinstance(r, dict):
                                continue
                            try:
                                sid = int(r.get("id") or 0)
                            except Exception:
                                continue
                            if sid <= 0:
                                continue
                            ov, st = by_sid_score.get(int(sid), (None, None))
                            sev = severity_by_id.get(int(sid))
                            r["judge"] = {
                                "critic_segment_overall": ov,
                                "critic_segment_stability": st,
                                "critic_severity": sev,
                            }

                    # Recast + rerender changed segments.
                    changed_ids: set[int] = set()
                    for a in applied_lane_a:
                        if not isinstance(a, dict):
                            continue
                        try:
                            changed_ids.add(int(a.get("segment_id") or 0))
                        except Exception:
                            pass
                    for td in transition_deltas:
                        try:
                            after = int(td.get("boundary_after_segment_id") or 0)
                        except Exception:
                            after = 0
                        if after > 0:
                            changed_ids.add(int(after))
                            changed_ids.add(int(after + 1))
                    for sid in recast_list:
                        changed_ids.add(int(sid))

                    # Helper for stable segment row lookup.
                    seg_by_id: dict[int, dict[str, t.Any]] = {}
                    for r in seg_rows1 if isinstance(seg_rows1, list) else []:
                        if not isinstance(r, dict):
                            continue
                        try:
                            sid = int(r.get("id") or 0)
                        except Exception:
                            continue
                        if sid > 0:
                            seg_by_id[int(sid)] = r

                    for sid in recast_list:
                        row = seg_by_id.get(int(sid))
                        if not isinstance(row, dict):
                            continue
                        prev_row = seg_by_id.get(int(sid - 1))
                        next_row = seg_by_id.get(int(sid + 1))
                        best = _choose_recast_shot(
                            seg_row=row,
                            candidates=list(cand_by_seg.get(int(sid)) or []),
                            used_assets=used_assets,
                            prev_row=prev_row,
                            next_row=next_row,
                        )
                        if not isinstance(best, dict):
                            continue

                        seg_obj = SimpleNamespace(
                            id=int(row.get("id") or 0),
                            duration_s=float(row.get("duration_s") or 0.0),
                            ref_luma=_safe_float(row.get("ref_luma")),
                            ref_dark_frac=_safe_float(row.get("ref_dark_frac")),
                            ref_rgb_mean=(row.get("ref_rgb_mean") if isinstance(row.get("ref_rgb_mean"), list) else None),
                            music_energy=_safe_float(row.get("music_energy")),
                            beat_goal=str(row.get("beat_goal") or ""),
                            desired_tags=list(row.get("desired_tags") or []) if isinstance(row.get("desired_tags"), list) else [],
                            reference_visual=str(row.get("reference_visual") or ""),
                            story_beat=row.get("story_beat"),
                            transition_hint=row.get("transition_hint"),
                            preferred_sequence_group_ids=row.get("preferred_sequence_group_ids"),
                        )
                        chosen_t = float(_safe_float(best.get("start_s")) or 0.0)
                        in_meta: dict[str, t.Any] | None = None
                        try:
                            from .micro_editor import pick_inpoint_for_shot

                            chosen_t, in_meta = pick_inpoint_for_shot(segment=seg_obj, shot=best, work_dir=iter_root / "refine", timeout_s=timeout_s)
                        except Exception:
                            chosen_t, in_meta = float(_safe_float(best.get("start_s")) or 0.0), None

                        row["shot_id"] = str(best.get("id") or "")
                        row["sequence_group_id"] = best.get("sequence_group_id")
                        row["asset_id"] = str(best.get("asset_id") or "")
                        row["asset_path"] = str(best.get("asset_path") or "")
                        row["asset_kind"] = "video"
                        row["asset_in_s"] = float(chosen_t)
                        row["asset_out_s"] = float(chosen_t) + float(row.get("duration_s") or 0.0)
                        for k in (
                            "motion_score",
                            "shake_score",
                            "sharpness",
                            "luma_mean",
                            "dark_frac",
                            "rgb_mean",
                            "cam_motion_mag",
                            "cam_motion_angle_deg",
                        ):
                            row[f"shot_{k}"] = best.get(k)
                        if in_meta is not None:
                            row["inpoint_auto"] = in_meta

                        # Update grade from refined inpoint sample when available.
                        chosen_info = (in_meta or {}).get("chosen") if isinstance(in_meta, dict) else None
                        if isinstance(chosen_info, dict):
                            out_fp = None
                            if isinstance(chosen_info.get("frame_path"), str):
                                out_fp = Path(str(chosen_info.get("frame_path"))).expanduser()
                            row["grade"] = _compute_eq_grade(
                                ref_luma=_safe_float(row.get("ref_luma")),
                                ref_dark=_safe_float(row.get("ref_dark_frac")),
                                out_luma=_safe_float(chosen_info.get("luma")),
                                out_dark=_safe_float(chosen_info.get("dark_frac")),
                                ref_rgb=(row.get("ref_rgb_mean") if isinstance(row.get("ref_rgb_mean"), list) else None),
                                out_rgb=(chosen_info.get("rgb_mean") if isinstance(chosen_info.get("rgb_mean"), list) else None),
                                ref_frame_path=ref_frame_by_id.get(int(sid)),
                                out_frame_path=(out_fp if out_fp is not None and out_fp.exists() else None),
                            )

                        # Update usage counts.
                        aid = str(row.get("asset_id") or "").strip()
                        if aid:
                            used_assets[aid] = used_assets.get(aid, 0) + 1

                    # Apply overlay rewrites immediately (render-time).
                    for b in applied_lane_b:
                        if not isinstance(b, dict):
                            continue
                        try:
                            sid = int(b.get("segment_id") or 0)
                        except Exception:
                            sid = 0
                        if sid <= 0:
                            continue
                        if b.get("overlay_text_rewrite") is not None:
                            row = seg_by_id.get(int(sid))
                            if isinstance(row, dict):
                                row["overlay_text"] = str(b.get("overlay_text_rewrite") or "")
                                changed_ids.add(int(sid))

                    # Rerender changed segments in-place.
                    for sid in sorted([x for x in changed_ids if int(x) > 0]):
                        row = seg_by_id.get(int(sid))
                        if not isinstance(row, dict):
                            continue
                        out_path = seg_dir / f"seg_{int(sid):02d}.mp4"
                        try:
                            fade_in_s = None
                            fade_in_color = None
                            try:
                                fade_in_s = float(row.get("fade_in_s")) if row.get("fade_in_s") is not None else None
                            except Exception:
                                fade_in_s = None
                            if isinstance(row.get("fade_in_color"), str):
                                fade_in_color = str(row.get("fade_in_color") or "").strip().lower() or None
                            fade_out_s = None
                            fade_out_color = None
                            try:
                                fade_out_s = float(row.get("fade_out_s")) if row.get("fade_out_s") is not None else None
                            except Exception:
                                fade_out_s = None
                            if isinstance(row.get("fade_out_color"), str):
                                fade_out_color = str(row.get("fade_out_color") or "").strip().lower() or None
                            _render_segment(
                                asset_path=Path(str(row.get("asset_path") or "")),
                                asset_kind=str(row.get("asset_kind") or "video"),
                                in_s=float(row.get("asset_in_s") or 0.0),
                                duration_s=float(row.get("duration_s") or 0.0),
                                speed=float(row.get("speed") or 1.0),
                                crop_mode=str(row.get("crop_mode") or "center"),
                                overlay_text=str(row.get("overlay_text") or ""),
                                grade=(row.get("grade") if isinstance(row.get("grade"), dict) else None),
                                stabilize=bool(row.get("stabilize") or False),
                                stabilize_cache_dir=(project_root / "library" / "stabilized") if bool(row.get("stabilize") or False) else None,
                                zoom=float(_safe_float(row.get("zoom")) or 1.0),
                                fade_in_s=fade_in_s,
                                fade_in_color=fade_in_color,
                                fade_out_s=fade_out_s,
                                fade_out_color=fade_out_color,
                                output_path=out_path,
                                burn_overlay=burn_overlays,
                                timeout_s=timeout_s,
                            )
                        except Exception:
                            continue

                    # Re-concat + update final video.
                    try:
                        from .folder_edit_pipeline import _concat_segments  # type: ignore

                        seg_paths = [seg_dir / f"seg_{int(sid):02d}.mp4" for sid in seg_order]
                        seg_paths = [p for p in seg_paths if p.exists()]
                        silent_out = vdir / "final_video_silent_critic_fixed.mp4"
                        _concat_segments(segment_paths=seg_paths, output_path=silent_out, timeout_s=timeout_s)
                        fixed_out = vdir / "final_video_critic_fixed.mov"
                        shutil.copyfile(silent_out, fixed_out)
                        if final_audio:
                            merge_audio(video_path=fixed_out, audio_path=final_audio)
                        shutil.copyfile(fixed_out, out_video_path)
                    except Exception:
                        pass

                    # Persist patched timeline + summary critic score.
                    patched_doc["metrics"] = dict(patched_doc.get("metrics") or {})
                    patched_doc["metrics"]["critic_fix_iters_applied"] = int(it + 1)
                    patched_doc["critic"] = {
                        "overall_score": float(report.overall_score),
                        "subscores": report.subscores,
                        "summary_nl": report.summary_nl,
                        "model": str(report.model),
                    }
                    write_json(tpath, patched_doc)
                    doc0 = patched_doc

                    # Update dossier + global segment_runs log with judge annotations.
                    try:
                        from .segment_dossier import append_segment_runs, build_segment_dossier

                        dossier2 = build_segment_dossier(
                            timeline_doc=patched_doc,
                            candidates_by_seg_id=cand_by_seg,
                            max_candidates_per_seg=int(max(0, float(os.getenv("SEGMENT_DOSSIER_CAND_K", "24") or 24))),
                        )
                        write_json(vdir / "segment_dossier.json", dossier2)
                        append_segment_runs(
                            dossier_doc=dossier2,
                            extra={"pipeline": "run_folder_edit_variants", "phase": f"critic_fix_iter{it+1:02d}", "critic_model": str(critic_model_eff)},
                        )
                    except Exception:
                        pass

            try:
                for vid in chosen[: int(max_fix_finals)]:
                    _critic_fix_one_variant(vid)
            except Exception:
                pass

    # Clear and populate finals/ with selected winners (renumbered v001..vNN for client UX).
    try:
        for p in finals_dir.glob("v*.mov"):
            p.unlink()
        for p in finals_dir.glob("v*.mp4"):
            p.unlink()
    except Exception:
        pass

    winners: list[dict[str, t.Any]] = []
    for i, src_vid in enumerate(chosen, start=1):
        final_id = f"v{i:03d}"
        src_path = variants_dir / src_vid / "final_video.mov"
        dst_path = finals_dir / f"{final_id}.mov"
        if not src_path.exists():
            continue
        shutil.copyfile(src_path, dst_path)
        winners.append(
            {
                "final_id": final_id,
                "source_variant_id": src_vid,
                "rank_score": _rank_score(variant_meta.get(src_vid, {})),
                "metrics": variant_meta.get(src_vid, {}),
                "source_video": str(src_path),
                "final_video": str(dst_path),
            }
        )

    write_json(
        finals_dir / "finals_manifest.json",
        {
            "schema_version": 1,
            "project_root": str(project_root),
            "generated_at": utc_timestamp(),
            "director": director_mode,
            "variants_generated": int(len(vids)),
            "finals_written": int(len(winners)),
            "finals_requested": int(finals_n),
            "winners": winners,
        },
    )

    print(f"Done. Project: {project_root}")
    print(f"Finals folder: {finals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
