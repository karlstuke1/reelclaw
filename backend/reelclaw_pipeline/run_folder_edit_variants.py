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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate many Folder Edit variants (no video-gen; cuts from local footage).")
    ap.add_argument("--reel", required=True, help="Instagram reel URL or local .mp4/.mov path")
    ap.add_argument("--folder", required=True, help="Folder containing videos/images to use as footage")
    ap.add_argument("--ref", help="Reference image path (identity anchor; optional)")
    ap.add_argument("--model", default=os.getenv("DIRECTOR_MODEL", "google/gemini-3-pro-preview"))
    ap.add_argument("--variants", type=int, default=100)
    ap.add_argument("--pro", action="store_true", help="Use shot-level (pro) variant search.")
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
    shot_index_obj = None
    shots: list[dict[str, t.Any]] = []
    shot_by_id: dict[str, dict[str, t.Any]] = {}
    shot_candidates_by_seg: dict[int, list[dict[str, t.Any]]] = {}
    # Optional: story planner produces multiple coherent plans (constraints) shared across variants.
    story_plans: list[t.Any] = []
    story_segments_by_plan: dict[str, list[t.Any]] = {}
    shot_candidates_by_seg_by_plan: dict[str, dict[int, list[dict[str, t.Any]]]] = {}
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

    # Choose audio.
    final_audio: Path | None = None
    if args.audio:
        final_audio = Path(args.audio).expanduser().resolve()
    elif (not args.no_reel_audio) and reel_audio:
        final_audio = reel_audio

    start = max(1, int(args.start))
    end = start + int(args.variants) - 1
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
        dst = finals_dir / f"{variant_id}.mov"
        if dst.exists():
            if vidx % 10 == 0:
                print(f"[{vidx-start+1}/{args.variants}] skip existing {dst}")
            continue
        vroot = variants_dir / variant_id
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
            used_sem: dict[str, int] = {}
            prev_sem: str | None = None
            prev_sem_run = 0
            decisions_try: list[dict[str, t.Any]] = []
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
                    shortlist = _shortlist_assets_for_segment(segment=seg, assets=assets_for_planner, used_asset_ids=used, limit=int(params.top_n))
                    # Avoid repeats if possible.
                    non_repeats = [a for a in shortlist if str(a.get("id") or "") not in used]
                    pool = non_repeats if len(non_repeats) >= max(4, int(params.top_n) // 3) else shortlist
                    chosen = _pick_weighted(pool, tau=float(params.tau), rng=rng, usage=asset_usage, usage_penalty=usage_penalty)
                    aid = str(chosen.get("id") or "")
                    if aid:
                        used.add(aid)
                    decisions_try.append({"segment_id": seg.id, "asset_id": aid, "in_s": 0.0, "speed": 1.0, "crop_mode": "center"})
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

            stabilize = False
            zoom = _float_env("FOLDER_EDIT_ZOOM", "1.0")
            if asset_kind == "video" and _truthy_env("FOLDER_EDIT_STABILIZE", "1"):
                motion = None
                if pro_mode:
                    shot_id = str(dec.get("shot_id") or "")
                    shot = shot_by_id.get(shot_id) if shot_id else None
                    if isinstance(shot, dict):
                        motion = shot.get("motion_score")
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
                    "shot_id": (str(dec.get("shot_id") or "") if pro_mode else None),
                    "sequence_group_id": (
                        (shot_by_id.get(str(dec.get("shot_id") or "")) or {}).get("sequence_group_id") if pro_mode else None
                    ),
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
        if _truthy_env("VARIANT_PREVIEW_FRAMES", "1"):
            # Preview frames + cheap mismatch metrics (luma/dark) for later filtering.
            preview_dir = vroot / "preview_frames"
            preview_dir.mkdir(parents=True, exist_ok=True)
            luma_diffs: list[float] = []
            dark_diffs: list[float] = []
            for seg_id, start_s, end_s, ref_frame in segment_frames:
                mid = (start_s + end_s) / 2.0
                out_frame = preview_dir / f"out_{seg_id:02d}.jpg"
                _extract_frame(video_path=silent_video, at_s=mid, out_path=out_frame, timeout_s=min(timeout_s, 120.0))
                rl = _luma_mean(ref_frame)
                rd = _dark_frac(ref_frame)
                ol = _luma_mean(out_frame)
                od = _dark_frac(out_frame)
                if isinstance(rl, (int, float)) and isinstance(ol, (int, float)):
                    luma_diffs.append(abs(float(rl) - float(ol)))
                if isinstance(rd, (int, float)) and isinstance(od, (int, float)):
                    dark_diffs.append(abs(float(rd) - float(od)))

            avg_dl = (sum(luma_diffs) / len(luma_diffs)) if luma_diffs else 0.0
            avg_dd = (sum(dark_diffs) / len(dark_diffs)) if dark_diffs else 0.0

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
            },
        }
        write_json(vroot / "timeline.json", timeline_doc)

        # Copy final to a single folder for quick scoring.
        shutil.copyfile(final_video, dst)

        if (vidx - start + 1) % 5 == 0 or vidx == end:
            print(f"[{vidx-start+1}/{args.variants}] wrote {dst} (avg_dl={avg_dl:.4f} avg_dd={avg_dd:.4f})")

    # Write (or rewrite) index.tsv by reading variant timelines (robust across script versions).
    vids = sorted([p.stem for p in finals_dir.glob("v*.mov")])
    sets_by_vid: dict[str, set[str]] = {vid: _load_variant_asset_set(variants_dir / vid) for vid in vids}
    out_lines: list[str] = [index_header]
    for vid in vids:
        vdir = variants_dir / vid
        tpath = vdir / "timeline.json"
        seed_val: str = ""
        avg_dl = 0.0
        avg_dd = 0.0
        if tpath.exists():
            try:
                doc = json.loads(tpath.read_text(encoding="utf-8", errors="replace"))
                seed_val = str(doc.get("seed") or "")
                metrics = doc.get("metrics") or {}
                if isinstance(metrics, dict):
                    avg_dl = float(metrics.get("avg_abs_luma_diff") or 0.0)
                    avg_dd = float(metrics.get("avg_abs_dark_diff") or 0.0)
            except Exception:
                pass

        aset = sets_by_vid.get(vid) or set()
        uniq_assets = len(aset)
        if len(vids) <= 1 or not aset:
            md = 1.0
        else:
            md = min((_jaccard_distance(aset, sets_by_vid[o]) for o in vids if o != vid), default=1.0)

        out_lines.append(
            f"{vid}\t{seed_val}\t{md:.3f}\t{uniq_assets}\t{avg_dl:.6f}\t{avg_dd:.6f}\t{(finals_dir / (vid + '.mov')).resolve().as_posix()}"
        )

    index_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Done. Project: {project_root}")
    print(f"Finals folder: {finals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
