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
    _dark_frac,
    _rgb_mean,
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
) -> list[dict[str, t.Any]]:
    """
    Experimental: ask Gemini to choose the best shot_id per segment from a curated candidate list.
    Output is compiled into deterministic edit decisions (micro-editor + renderer remain code-only).
    """
    from .openrouter_client import OpenRouterError, chat_completions

    def _seg_row(seg: t.Any) -> dict[str, t.Any]:
        return {
            "id": int(getattr(seg, "id", 0) or 0),
            "duration_s": float(getattr(seg, "duration_s", 0.0) or 0.0),
            "beat_goal": str(getattr(seg, "beat_goal", "") or ""),
            "overlay_text": str(getattr(seg, "overlay_text", "") or ""),
            "reference_visual": str(getattr(seg, "reference_visual", "") or ""),
            "desired_tags": [str(x) for x in (getattr(seg, "desired_tags", []) or [])][:12],
            "ref_luma": _safe_float(getattr(seg, "ref_luma", None)),
            "ref_dark_frac": _safe_float(getattr(seg, "ref_dark_frac", None)),
            "ref_rgb_mean": getattr(seg, "ref_rgb_mean", None),
        }

    def _cand_row(c: dict[str, t.Any]) -> dict[str, t.Any]:
        return {
            "shot_id": str(c.get("id") or ""),
            "asset_id": str(c.get("asset_id") or ""),
            "sequence_group_id": str(c.get("sequence_group_id") or ""),
            "start_s": _safe_float(c.get("start_s")),
            "end_s": _safe_float(c.get("end_s")),
            "duration_s": _safe_float(c.get("duration_s")),
            "luma_mean": _safe_float(c.get("luma_mean")),
            "dark_frac": _safe_float(c.get("dark_frac")),
            "rgb_mean": c.get("rgb_mean") if isinstance(c.get("rgb_mean"), list) else None,
            "motion_score": _safe_float(c.get("motion_score")),
            "shake_score": _safe_float(c.get("shake_score")),
            "sharpness": _safe_float(c.get("sharpness")),
            "shot_type": str(c.get("shot_type") or ""),
            "setting": str(c.get("setting") or ""),
            "mood": str(c.get("mood") or ""),
            "tags": [str(x) for x in (c.get("tags") or [])][:12] if isinstance(c.get("tags"), list) else [],
            "description": str(c.get("description") or "")[:220],
        }

    seg_payload: list[dict[str, t.Any]] = []
    for seg in segments:
        sid = int(getattr(seg, "id", 0) or 0)
        if sid <= 0:
            continue
        cands = candidates_by_seg_id.get(int(sid)) or []
        cands = [c for c in cands if isinstance(c, dict) and str(c.get("id") or "").strip()]
        seg_payload.append(
            {
                "segment": _seg_row(seg),
                "candidates": [_cand_row(c) for c in cands[: max(1, int(max_candidates_per_seg))]],
            }
        )

    system_prompt = "\n".join(
        [
            "You are an ELITE short-form video editor.",
            "You must choose the best candidate shot for each segment.",
            "You are NOT allowed to invent shots. You MUST pick from the provided candidate shot_id values.",
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

    user_text = json.dumps({"segments": seg_payload}, ensure_ascii=True)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

    result = chat_completions(
        api_key=api_key,
        model=str(model),
        messages=messages,
        temperature=0.0,
        max_tokens=1800,
        timeout_s=float(timeout_s),
        site_url=site_url,
        app_name=app_name,
        reasoning=_reasoning_param(),
        retries=2,
        retry_delay_s=1.5,
        extra_body={"response_format": {"type": "json_object"}},
    )
    parsed = _extract_json_object(result.content or "")
    choices_raw = parsed.get("choices")
    if not isinstance(choices_raw, list):
        raise OpenRouterError("Gemini director returned invalid JSON: missing choices[]")

    out: list[dict[str, t.Any]] = []
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
        out.append({"segment_id": int(sid), "shot_id": shot_id})

    # Ensure all segments have a choice; fill missing deterministically with best candidate[0].
    by_seg: dict[int, str] = {int(x["segment_id"]): str(x["shot_id"]) for x in out if int(x.get("segment_id") or 0) > 0 and str(x.get("shot_id") or "")}
    for seg in segments:
        sid = int(getattr(seg, "id", 0) or 0)
        if sid <= 0 or sid in by_seg:
            continue
        cands = candidates_by_seg_id.get(int(sid)) or []
        if cands:
            by_seg[int(sid)] = str(cands[0].get("id") or "")

    return [{"segment_id": int(sid), "shot_id": str(shot_id)} for sid, shot_id in sorted(by_seg.items(), key=lambda kv: kv[0])]


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
        choices=["code", "gemini"],
        default=str(os.getenv("VARIANT_DIRECTOR", "code") or "code").strip().lower(),
        help="Variant director: code (deterministic heuristics/optimizer) or gemini (LLM chooses shots among candidates; compiled into deterministic edits).",
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
        os.environ.setdefault("SHOT_TAGGING", "1")
        # IMPORTANT: tagging every single shot can take a long time on huge libraries.
        # Set SHOT_TAG_MAX=0 to tag everything (e.g., on AWS).
        os.environ.setdefault("SHOT_TAG_MAX", "400")
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
        # Attach luma/dark metrics to segments (used for both LLM and offline fallback).
        seg_luma: dict[int, float | None] = {sid: _luma_mean(p) for sid, _s, _e, p in segment_frames}
        seg_dark: dict[int, float | None] = {sid: _dark_frac(p) for sid, _s, _e, p in segment_frames}
        seg_rgb: dict[int, list[float] | None] = {sid: _rgb_mean(p) for sid, _s, _e, p in segment_frames}
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
    global_tagged_path = Path("Outputs") / "_asset_tag_cache.json"
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

    if not args.pro:
        missing = [a for a in assets_for_tagger if str(a.get("id") or "") and str(a["id"]) not in tags]
        if missing:
            new_tags = tag_assets_from_thumbnails(api_key=api_key, model=args.model, assets=missing, timeout_s=timeout_s)
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
                global_tagged_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Planner asset metas.
    assets_for_planner: list[dict[str, t.Any]] = []
    asset_by_id: dict[str, dict[str, t.Any]] = {}
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
    if director_mode not in {"code", "gemini"}:
        director_mode = "code"
    if pro_macro_mode not in {"sample", "beam"}:
        pro_macro_mode = "sample"
    if director_mode == "gemini" and not pro_mode:
        raise SystemExit("--director=gemini requires --pro (shot candidates).")
    if director_mode == "gemini" and variants_n > 12 and not _truthy_env("ALLOW_GEMINI_DIRECTOR_MANY", "0"):
        raise SystemExit("--director=gemini is expensive. Limit --variants to <=12 or set ALLOW_GEMINI_DIRECTOR_MANY=1.")
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
            decisions_try: list[dict[str, t.Any]] = []

            # Experimental: Gemini chooses the macro shot sequence (compiled into deterministic edits).
            if pro_mode and director_mode == "gemini":
                if plan_id and plan_id in shot_candidates_by_seg_by_plan:
                    cand_map = t.cast(dict[int, list[dict[str, t.Any]]], (shot_candidates_by_seg_by_plan.get(plan_id) or {}))
                else:
                    cand_map = shot_candidates_by_seg
                cand_k = int(max(4, min(24, float(os.getenv("VARIANT_GEMINI_CAND_K", "12") or 12))))
                picks = _gemini_director_choose_shots(
                    api_key=api_key,
                    model=str(args.model),
                    segments=segments_for_variant,
                    candidates_by_seg_id=cand_map,
                    max_candidates_per_seg=cand_k,
                    timeout_s=min(timeout_s, 240.0),
                )
                by_sid = {int(p.get("segment_id") or 0): str(p.get("shot_id") or "") for p in picks if int(p.get("segment_id") or 0) > 0}
                for seg in segments_for_variant:
                    sid = int(seg.id)
                    shot_id = str(by_sid.get(int(sid)) or "").strip()
                    if not shot_id:
                        # Deterministic fallback: top candidate for this segment.
                        shot_id = str((cand_map.get(int(sid)) or [{}])[0].get("id") or "")
                    shot = shot_by_id.get(shot_id) or (cand_map.get(int(sid)) or [{}])[0]
                    if not isinstance(shot, dict):
                        continue
                    aid = str(shot.get("asset_id") or "")
                    if aid:
                        used.add(aid)
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
                        if not isinstance(shot, dict):
                            continue
                        aid = str(shot.get("asset_id") or "")
                        if aid:
                            used.add(aid)
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
                        # Avoid reusing the same underlying asset within a single variant if possible.
                        non_repeats = [s for s in pool0 if str(s.get("asset_id") or "") and str(s.get("asset_id")) not in used]
                        pool = non_repeats if len(non_repeats) >= max(4, int(params.top_n) // 3) else pool0

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
                        # Avoid repeats if possible.
                        non_repeats = [a for a in shortlist if str(a.get("id") or "") not in used]
                        pool = non_repeats if len(non_repeats) >= max(4, int(params.top_n) // 3) else shortlist
                        chosen = _pick_weighted(pool, tau=float(params.tau), rng=rng, usage=asset_usage, usage_penalty=usage_penalty)
                        aid = str(chosen.get("id") or "")
                        if aid:
                            used.add(aid)
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
        segment_paths: list[Path] = []
        timeline_segments: list[dict[str, t.Any]] = []
        fade_tail = _float_env("FOLDER_EDIT_FADE_OUT_S", "0.18")
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
            if use_auto_grade:
                grade = _compute_eq_grade(
                    ref_luma=getattr(seg, "ref_luma", None),
                    ref_dark=getattr(seg, "ref_dark_frac", None),
                    out_luma=(float(out_luma) if isinstance(out_luma, (int, float)) else None),
                    out_dark=(float(out_dark) if isinstance(out_dark, (int, float)) else None),
                    ref_rgb=getattr(seg, "ref_rgb_mean", None),
                    out_rgb=(out_rgb if isinstance(out_rgb, list) else None),
                )

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
                fade_out_s=(fade_tail if i == len(segments_for_variant) else None),
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
                    "inpoint_auto": inpoint_meta_by_seg.get(int(seg.id)),
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
    fix_iters = max(0, int(getattr(args, "fix_iters", 0) or 0))

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
