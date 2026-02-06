from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import typing as t

from .folder_edit_planner import ReferenceSegmentPlan
from .grader_steering import fast_features_for_sequence
from .micro_editor import micro_edit_sequence
from .pipeline_output import utc_timestamp, write_json
from .shot_index import build_or_load_shot_index
from .video_tools import merge_audio

# Reuse proven rendering helpers from the main pipeline implementation.
from .folder_edit_pipeline import (  # type: ignore
    _compute_eq_grade,
    _concat_segments,
    _dark_frac,
    _estimate_shake_jitter_norm_p95,
    _extract_frame,
    _luma_mean,
    _render_segment,
    _rgb_mean,
)

try:
    from .issue_detector import IssueDetector
except Exception:  # pragma: no cover
    IssueDetector = None  # type: ignore[assignment]


def _load_json(path: Path) -> dict[str, t.Any]:
    return t.cast(dict[str, t.Any], json.loads(path.read_text(encoding="utf-8", errors="replace") or "{}"))


def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() not in {"", "0", "false", "no", "off"}


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class FinalWinner:
    final_id: str
    source_variant_id: str


def _parse_finals_manifest(path: Path) -> list[FinalWinner]:
    if not path.exists():
        return []
    doc = _load_json(path)
    winners = doc.get("winners") or []
    out: list[FinalWinner] = []
    if isinstance(winners, list):
        for w in winners:
            if not isinstance(w, dict):
                continue
            fid = str(w.get("final_id") or "").strip()
            src = str(w.get("source_variant_id") or "").strip()
            if fid and src:
                out.append(FinalWinner(final_id=fid, source_variant_id=src))
    return out


def _build_segments(timeline_doc: dict[str, t.Any]) -> list[ReferenceSegmentPlan]:
    segs = timeline_doc.get("timeline_segments") or []
    if not isinstance(segs, list) or not segs:
        raise ValueError("timeline.json missing timeline_segments[]")
    out: list[ReferenceSegmentPlan] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        out.append(
            ReferenceSegmentPlan(
                id=int(s.get("id") or 0),
                start_s=float(s.get("start_s") or 0.0),
                end_s=float(s.get("end_s") or 0.0),
                duration_s=float(s.get("duration_s") or 0.0),
                beat_goal=str(s.get("beat_goal") or ""),
                overlay_text=str(s.get("overlay_text") or ""),
                reference_visual=str(s.get("reference_visual") or ""),
                desired_tags=list(s.get("desired_tags") or []),
                ref_luma=(float(s["ref_luma"]) if s.get("ref_luma") is not None else None),
                ref_dark_frac=(float(s["ref_dark_frac"]) if s.get("ref_dark_frac") is not None else None),
                ref_rgb_mean=(s.get("ref_rgb_mean") if isinstance(s.get("ref_rgb_mean"), list) else None),
                music_energy=(float(s["music_energy"]) if s.get("music_energy") is not None else None),
                start_beat=(int(s["start_beat"]) if s.get("start_beat") is not None else None),
                end_beat=(int(s["end_beat"]) if s.get("end_beat") is not None else None),
                story_beat=(str(s.get("story_beat") or "") or None),
                preferred_sequence_group_ids=(list(s.get("preferred_sequence_group_ids") or []) or None),
                transition_hint=(str(s.get("transition_hint") or "") or None),
            )
        )
    out = [s for s in out if int(s.id) > 0 and float(s.duration_s) > 0.05]
    out.sort(key=lambda s: int(s.id))
    return out


def _load_music_doc(project_root: Path, timeline_doc: dict[str, t.Any]) -> dict[str, t.Any] | None:
    """
    Best-effort: load music analysis to let micro-edit decisions attach beat indices.
    This mirrors the local `scripts/polish_winners.py` behavior.
    """
    try:
        mp = timeline_doc.get("music_analysis")
        if isinstance(mp, str) and mp.strip():
            p = Path(mp).expanduser()
            if p.exists():
                try:
                    return _load_json(p)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        p2 = project_root / "reference" / "music_analysis.json"
        if p2.exists():
            try:
                return _load_json(p2)
            except Exception:
                return None
    except Exception:
        return None
    return None


def _load_shot_by_id(project_root: Path, *, timeout_s: float) -> dict[str, dict[str, t.Any]]:
    """
    Best-effort: load the cached shot index created during variant generation.
    """
    media_index_path = project_root / "library" / "media_index.json"
    if not media_index_path.exists():
        return {}

    cache_dir = Path(os.getenv("FOLDER_EDIT_SHOT_INDEX_CACHE_ROOT", str(Path("Outputs") / "_shot_index_cache"))).expanduser().resolve()
    mode0 = (os.getenv("SHOT_INDEX_MODE", "scene") or "scene").strip().lower() or "scene"
    modes = [mode0] + [m for m in ("scene", "fast") if m != mode0]

    for mode in modes:
        try:
            os.environ["SHOT_INDEX_MODE"] = mode
            idx = build_or_load_shot_index(
                media_index_path=media_index_path,
                cache_dir=cache_dir,
                api_key=None,
                model=None,
                timeout_s=float(timeout_s),
            )
            shot_by_id = {str(s.get("id") or ""): s for s in (idx.shots or []) if str(s.get("id") or "")}
            if shot_by_id:
                return shot_by_id
        except Exception:
            continue
    return {}


def _apply_issue_fixes_to_decisions(
    *,
    probs: dict[str, float],
    decisions: list[t.Any],
    segments: list[ReferenceSegmentPlan],
    shots_by_id: dict[str, dict[str, t.Any]],
) -> tuple[list[t.Any], dict[str, t.Any]]:
    """
    Deterministic heuristics that map predicted issues -> small knobs:
    - pacing: slightly increase playback speed
    - hook: tighten segment 1 (speed + inpoint shift)
    - clarity: enforce smart crop when available + modest zoom
    - stability: force stabilization (cheap insurance)
    """
    from dataclasses import replace as _dc_replace

    diag: dict[str, t.Any] = {"applied": []}
    if not probs:
        return list(decisions), diag

    out_decs: list[t.Any] = list(decisions)

    thr = float(os.getenv("ISSUE_FIX_THRESHOLD", "0.55") or 0.55)

    def p(name: str) -> float:
        try:
            return float(probs.get(name) or 0.0)
        except Exception:
            return 0.0

    pacing_p = max(p("pacing"), p("slow_pacing"), p("too_slow"))
    hook_p = max(p("hook"), p("weak_hook"), p("no_hook"))
    clarity_p = max(p("clarity"), p("low_clarity"))
    stability_p = max(p("stability"), p("shaky"), p("shake"))

    # Apply global pacing speed-up (kept conservative to avoid looking \"sped up\").
    if pacing_p >= thr:
        sp = float(os.getenv("ISSUE_FIX_PACING_SPEED", "1.10") or 1.10)
        sp = max(0.90, min(1.20, sp))
        for idx, d in enumerate(out_decs):
            try:
                out_decs[idx] = _dc_replace(d, speed=float(max(float(getattr(d, "speed", 1.0) or 1.0), sp)))
            except Exception:
                continue
        diag["applied"].append({"issue": "pacing", "action": "min_speed", "value": sp, "p": pacing_p})

    # Hook: tighten segment 1.
    if hook_p >= thr and out_decs:
        d0 = out_decs[0]
        seg0 = segments[0] if segments else None
        shot = shots_by_id.get(str(getattr(d0, "shot_id", "") or ""))
        # Speed nudge.
        speed0 = float(getattr(d0, "speed", 1.0) or 1.0)
        try:
            hs = float(os.getenv("ISSUE_FIX_HOOK_SPEED", "1.15") or 1.15)
            hs = max(0.95, min(1.25, hs))
            speed0 = float(max(float(speed0), hs))
        except Exception:
            pass
        # Inpoint shift (clamped to shot window).
        in0 = float(getattr(d0, "in_s", 0.0) or 0.0)
        try:
            shift = float(os.getenv("ISSUE_FIX_HOOK_INPOINT_SHIFT_S", "0.22") or 0.22)
            shift = max(-0.35, min(0.35, shift))
            start_s = _safe_float(shot.get("start_s")) if isinstance(shot, dict) else None
            end_s = _safe_float(shot.get("end_s")) if isinstance(shot, dict) else None
            dur = float(getattr(seg0, "duration_s", 0.0) or 0.0) if seg0 else float(getattr(d0, "duration_s", 0.0) or 0.0)
            span = float(dur) * float(speed0)
            new_in = float(in0) + float(shift)
            if start_s is not None and end_s is not None and end_s > start_s + 1e-3:
                max_in = float(end_s) - float(span)
                if max_in >= start_s:
                    new_in = max(float(start_s), min(float(new_in), float(max_in)))
            else:
                new_in = max(0.0, float(new_in))
            in0 = float(new_in)
        except Exception:
            pass
        try:
            out_decs[0] = _dc_replace(d0, speed=float(speed0), in_s=float(in0))
        except Exception:
            pass
        diag["applied"].append({"issue": "hook", "action": "tighten_seg1", "p": hook_p})

    # Clarity: prefer smart crop (requires cv2; fallback is a mild zoom).
    if clarity_p >= thr:
        zoom = float(os.getenv("ISSUE_FIX_CLARITY_ZOOM", "1.08") or 1.08)
        zoom = max(1.0, min(1.15, zoom))
        for idx, d in enumerate(out_decs):
            try:
                if getattr(d, "reframe", None) is None:
                    continue
                out_decs[idx] = _dc_replace(d, crop_mode="smart")
            except Exception:
                continue
        diag["applied"].append({"issue": "clarity", "action": "prefer_smart_crop", "zoom_min": zoom, "p": clarity_p})
        # zoom is applied later at render-time; stored in diag for caller.
        diag["clarity_zoom_min"] = zoom

    # Stability: force stabilize flags at render time (adds zoom).
    if stability_p >= thr:
        diag["applied"].append({"issue": "stability", "action": "force_stabilize", "p": stability_p})
        diag["force_stabilize"] = True
        diag["stabilize_zoom_min"] = float(os.getenv("ISSUE_FIX_STABILIZE_ZOOM", "1.10") or 1.10)

    return out_decs, diag


def polish_finals(
    *,
    project_root: Path,
    burn_overlays: bool,
    timeout_s: float = 360.0,
    issue_detector_dir: Path | None = None,
    progress_cb: t.Callable[[str, int, int], None] | None = None,
) -> dict[str, t.Any]:
    """
    Post-process a variant search project:
    - read finals/finals_manifest.json
    - for each winner, re-render from source assets with micro DP + smart crop + stabilization
    - overwrite finals/v###.mov with polished outputs
    """
    project_root = project_root.expanduser().resolve()
    finals_dir = project_root / "finals"
    variants_dir = project_root / "variants"

    manifest_path = finals_dir / "finals_manifest.json"
    winners = _parse_finals_manifest(manifest_path)
    if not winners:
        return {"ok": False, "error": f"Missing or empty finals manifest: {manifest_path}"}

    shot_by_id = _load_shot_by_id(project_root, timeout_s=float(timeout_s))
    if not shot_by_id:
        return {"ok": False, "error": "Failed to load shot index (shot_by_id empty)."}

    issue_detector = None
    if issue_detector_dir and IssueDetector is not None:
        try:
            issue_detector = IssueDetector.load(issue_detector_dir)
        except Exception:
            issue_detector = None

    out: dict[str, t.Any] = {"ok": True, "project_root": str(project_root), "winners_polished": []}

    # Keep polish artifacts separate from finals/ so the client only sees videos.
    polish_root = project_root / "polish"
    polish_root.mkdir(parents=True, exist_ok=True)

    # Enable stabilizations in the polish stage regardless of draft settings used during search.
    old_env: dict[str, str | None] = {}
    for k, v in {"FOLDER_EDIT_STABILIZE": "1"}.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        for i, w in enumerate(winners, start=1):
            if progress_cb:
                progress_cb("Polishing", i, len(winners))

            vdir = variants_dir / w.source_variant_id
            timeline_path = vdir / "timeline.json"
            if not timeline_path.exists():
                out["winners_polished"].append({"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": "missing_timeline"})
                continue

            timeline_doc = _load_json(timeline_path)
            try:
                segments = _build_segments(timeline_doc)
            except Exception as e:
                out["winners_polished"].append(
                    {"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": f"bad_segments:{type(e).__name__}:{e}"}
                )
                continue

            seg_rows = timeline_doc.get("timeline_segments") if isinstance(timeline_doc.get("timeline_segments"), list) else []
            if not isinstance(seg_rows, list) or not seg_rows:
                out["winners_polished"].append({"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": "missing_timeline_segments"})
                continue

            # Resolve chosen shots by shot_id.
            chosen_shots: list[dict[str, t.Any]] = []
            missing = False
            for r in seg_rows:
                if not isinstance(r, dict):
                    missing = True
                    break
                sid = str(r.get("shot_id") or "").strip()
                if not sid:
                    missing = True
                    break
                sh = shot_by_id.get(sid)
                if not isinstance(sh, dict):
                    missing = True
                    break
                chosen_shots.append(sh)
            if missing or len(chosen_shots) != len(segments):
                out["winners_polished"].append({"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": "shot_lookup_failed"})
                continue

            # Compute fast features for ML detectors/steering.
            speeds = [float(r.get("speed") or 1.0) for r in seg_rows if isinstance(r, dict) and isinstance(r.get("speed"), (int, float))]
            zooms = [float(r.get("zoom") or 1.0) for r in seg_rows if isinstance(r, dict) and isinstance(r.get("zoom"), (int, float))]
            default_speed = float(sum(speeds) / len(speeds)) if speeds else float(os.getenv("FOLDER_EDIT_DEFAULT_SPEED", "1.0") or 1.0)
            default_zoom = float(sum(zooms) / len(zooms)) if zooms else float(os.getenv("FOLDER_EDIT_ZOOM", "1.0") or 1.0)

            feats = fast_features_for_sequence(
                segments=segments,
                shots=chosen_shots,
                default_speed=default_speed,
                default_zoom=default_zoom,
                stabilize_enabled=True,
                stabilize_shake_th=float(os.getenv("FOLDER_EDIT_STEER_STABILIZE_SHAKE_TH", "0.25") or 0.25),
            )

            issue_probs: dict[str, float] = {}
            if issue_detector is not None:
                try:
                    issue_probs = issue_detector.predict_proba(feats)
                except Exception:
                    issue_probs = {}

            # Micro DP + smart crop decisions.
            polish_dir = polish_root / w.final_id
            micro_dir = polish_dir / "micro"
            music_doc = _load_music_doc(project_root, timeline_doc)
            try:
                micro_decisions, micro_diag = micro_edit_sequence(
                    segments=segments,
                    chosen_shots=chosen_shots,
                    music_doc=music_doc,
                    work_dir=micro_dir,
                    timeout_s=float(timeout_s),
                    default_crop_mode=str(os.getenv("FOLDER_EDIT_DEFAULT_CROP", "center") or "center"),
                    default_speed=float(os.getenv("FOLDER_EDIT_DEFAULT_SPEED", "1.0") or 1.0),
                )
            except Exception as e:
                out["winners_polished"].append(
                    {"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": f"micro_failed:{type(e).__name__}:{e}"}
                )
                continue

            # Apply issue-driven deterministic tweaks (speed/crop/stabilize).
            decs_mut, fix_diag = _apply_issue_fixes_to_decisions(
                probs=issue_probs,
                decisions=list(micro_decisions),
                segments=segments,
                shots_by_id=shot_by_id,
            )
            clarity_zoom_min = _safe_float(fix_diag.get("clarity_zoom_min"))
            force_stabilize = bool(fix_diag.get("force_stabilize") or False)
            stabilize_zoom_min = _safe_float(fix_diag.get("stabilize_zoom_min"))

            # Render segments.
            seg_dir = polish_dir / "segments"
            grade_dir = polish_dir / "grade_samples"
            seg_dir.mkdir(parents=True, exist_ok=True)
            grade_dir.mkdir(parents=True, exist_ok=True)

            segment_paths: list[Path] = []
            polished_timeline_segments: list[dict[str, t.Any]] = []
            dec_by_seg: dict[int, t.Any] = {int(d.segment_id): d for d in decs_mut}

            fade_tail = float(os.getenv("FOLDER_EDIT_FADE_OUT_S", "0.18") or 0.18)
            for idx, seg in enumerate(segments, start=1):
                md = dec_by_seg.get(int(seg.id))
                if md is None:
                    continue
                shot = shot_by_id.get(str(md.shot_id))
                if not isinstance(shot, dict):
                    continue
                asset_path = Path(str(shot.get("asset_path") or "")).expanduser()
                if not asset_path.exists():
                    continue

                out_luma = None
                out_dark = None
                out_rgb = None
                try:
                    safe = f"{float(md.in_s):.3f}".replace(".", "p")
                    sample_path = grade_dir / f"seg_{int(seg.id):02d}_t_{safe}.jpg"
                    if not sample_path.exists():
                        _extract_frame(video_path=asset_path, at_s=float(md.in_s), out_path=sample_path, timeout_s=min(float(timeout_s), 120.0))
                    out_luma = _luma_mean(sample_path)
                    out_dark = _dark_frac(sample_path)
                    out_rgb = _rgb_mean(sample_path)
                except Exception:
                    pass

                grade = _compute_eq_grade(
                    ref_luma=seg.ref_luma,
                    ref_dark=seg.ref_dark_frac,
                    out_luma=out_luma,
                    out_dark=out_dark,
                    ref_rgb=getattr(seg, "ref_rgb_mean", None),
                    out_rgb=(out_rgb if isinstance(out_rgb, list) else None),
                )

                stabilize = False
                zoom = float(os.getenv("FOLDER_EDIT_ZOOM", "1.0") or 1.0)
                if clarity_zoom_min is not None:
                    zoom = max(float(zoom), float(clarity_zoom_min))

                if _truthy_env("FOLDER_EDIT_STABILIZE", "1"):
                    try:
                        shake_p95 = _estimate_shake_jitter_norm_p95(
                            video_path=asset_path,
                            start_s=float(md.in_s),
                            end_s=float(md.in_s) + float(seg.duration_s),
                        )
                    except Exception:
                        shake_p95 = None
                    thr = float(os.getenv("FOLDER_EDIT_STABILIZE_SHAKE_P95_THRESHOLD", "0.06") or 0.06)
                    if isinstance(shake_p95, (int, float)) and float(shake_p95) >= float(thr) and float(seg.duration_s) >= 1.0:
                        stabilize = True
                        zoom = max(zoom, float(os.getenv("FOLDER_EDIT_STABILIZE_ZOOM", "1.08") or 1.08))

                if force_stabilize and float(seg.duration_s) >= 1.0:
                    stabilize = True
                    if stabilize_zoom_min is not None:
                        zoom = max(float(zoom), float(stabilize_zoom_min))

                out_path = seg_dir / f"seg_{int(seg.id):02d}.mp4"
                _render_segment(
                    asset_path=asset_path,
                    asset_kind="video",
                    in_s=float(md.in_s),
                    duration_s=float(seg.duration_s),
                    speed=float(md.speed),
                    crop_mode=str(md.crop_mode),
                    reframe=(md.reframe if isinstance(md.reframe, dict) else None),
                    overlay_text=str(seg.overlay_text or ""),
                    grade=grade,
                    stabilize=stabilize,
                    stabilize_cache_dir=(polish_dir / "stabilized") if stabilize else None,
                    zoom=zoom,
                    fade_out_s=(fade_tail if idx == len(segments) else None),
                    output_path=out_path,
                    burn_overlay=bool(burn_overlays),
                    timeout_s=float(timeout_s),
                )
                segment_paths.append(out_path)

                polished_timeline_segments.append(
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
                        "overlay_text": seg.overlay_text,
                        "reference_visual": seg.reference_visual,
                        "desired_tags": seg.desired_tags,
                        "story_beat": getattr(seg, "story_beat", None),
                        "preferred_sequence_group_ids": getattr(seg, "preferred_sequence_group_ids", None),
                        "transition_hint": getattr(seg, "transition_hint", None),
                        "shot_id": str(md.shot_id),
                        "sequence_group_id": (shot.get("sequence_group_id") if isinstance(shot, dict) else None),
                        "asset_id": str(md.asset_id),
                        "asset_path": str(asset_path),
                        "asset_kind": "video",
                        "asset_in_s": float(md.in_s),
                        "asset_out_s": float(md.in_s) + (float(seg.duration_s) * float(md.speed)),
                        "speed": float(md.speed),
                        "crop_mode": str(md.crop_mode),
                        "reframe": (md.reframe if isinstance(md.reframe, dict) else None),
                        "grade": grade,
                        "stabilize": stabilize,
                        "zoom": zoom,
                        "micro_editor": md.debug,
                    }
                )

            if not segment_paths:
                out["winners_polished"].append({"final_id": w.final_id, "source_variant_id": w.source_variant_id, "ok": False, "error": "no_segments_rendered"})
                continue

            silent = polish_dir / "final_video_silent.mp4"
            _concat_segments(segment_paths=segment_paths, output_path=silent, timeout_s=float(timeout_s))

            polished_video = polish_dir / "final_video_polished.mov"
            polished_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(silent, polished_video)

            audio = timeline_doc.get("audio")
            if isinstance(audio, str) and audio.strip():
                audio_path = Path(audio).expanduser()
                if audio_path.exists():
                    try:
                        merge_audio(video_path=polished_video, audio_path=audio_path)
                    except Exception:
                        pass

            # Write polished timeline for reproducibility.
            write_json(
                polish_dir / "timeline_polished.json",
                {
                    "mode": "folder_edit_polished",
                    "generated_at": utc_timestamp(),
                    "source_project": str(project_root),
                    "source_variant_id": w.source_variant_id,
                    "final_id": w.final_id,
                    "features_fast": feats,
                    "issue_probs": issue_probs,
                    "issue_fixes": fix_diag,
                    "micro_editor": micro_diag,
                    "timeline_segments": polished_timeline_segments,
                    "audio": str(audio) if isinstance(audio, str) else None,
                },
            )

            # Overwrite final output in finals/ for downstream upload.
            final_out = finals_dir / f"{w.final_id}.mov"
            final_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(polished_video, final_out)

            out["winners_polished"].append(
                {
                    "final_id": w.final_id,
                    "source_variant_id": w.source_variant_id,
                    "ok": True,
                    "final_video": str(final_out),
                    "polish_dir": str(polish_dir),
                }
            )

    finally:
        for k, prev in old_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    # Persist a small manifest for later debugging (and S3 upload).
    write_json(polish_root / "manifest_polished.json", out)
    return out
