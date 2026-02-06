from __future__ import annotations

import json
import os
import time
from pathlib import Path
import typing as t


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _norm_tag(tag: str) -> str:
    return " ".join((tag or "").strip().lower().split())


def _compact_shot_candidate(shot: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Compact a shot_index row (or shortlist entry) down to stable, deterministic fields.
    This is used for per-variant dossiers and deterministic recast/fix loops.
    """
    tags = shot.get("tags") if isinstance(shot.get("tags"), list) else []
    return {
        "id": str(shot.get("id") or ""),
        "asset_id": str(shot.get("asset_id") or ""),
        "sequence_group_id": str(shot.get("sequence_group_id") or ""),
        "asset_path": str(shot.get("asset_path") or ""),
        "start_s": _safe_float(shot.get("start_s")),
        "end_s": _safe_float(shot.get("end_s")),
        "duration_s": _safe_float(shot.get("duration_s")),
        "luma_mean": _safe_float(shot.get("luma_mean")),
        "dark_frac": _safe_float(shot.get("dark_frac")),
        "rgb_mean": (shot.get("rgb_mean") if isinstance(shot.get("rgb_mean"), list) else None),
        "motion_score": _safe_float(shot.get("motion_score")),
        "shake_score": _safe_float(shot.get("shake_score")),
        "sharpness": _safe_float(shot.get("sharpness")),
        "cam_motion_mag": _safe_float(shot.get("cam_motion_mag")),
        "cam_motion_angle_deg": _safe_float(shot.get("cam_motion_angle_deg")),
        "shot_type": str(shot.get("shot_type") or ""),
        "setting": str(shot.get("setting") or ""),
        "mood": str(shot.get("mood") or ""),
        "tags": [_norm_tag(str(x)) for x in tags if _norm_tag(str(x))][:24],
    }


def build_segment_dossier(
    *,
    timeline_doc: dict[str, t.Any],
    candidates_by_seg_id: dict[int, list[dict[str, t.Any]]] | None = None,
    max_candidates_per_seg: int = 24,
) -> dict[str, t.Any]:
    """
    Build a deterministic per-segment dossier for a rendered variant timeline.
    This is the "data contract" needed for compounding improvement:
    - segment-level evidence + chosen shots + candidate alternatives
    - optional judge/critic annotations can be added later
    """
    segs = timeline_doc.get("timeline_segments")
    if not isinstance(segs, list):
        segs = []

    dossier_segs: list[dict[str, t.Any]] = []
    for row in segs:
        if not isinstance(row, dict):
            continue
        try:
            sid = int(row.get("id") or 0)
        except Exception:
            sid = 0
        if sid <= 0:
            continue

        ref = {
            "duration_s": _safe_float(row.get("duration_s")),
            "beat_goal": str(row.get("beat_goal") or ""),
            "desired_tags": [str(x) for x in (row.get("desired_tags") or []) if str(x).strip()][:24]
            if isinstance(row.get("desired_tags"), list)
            else [],
            "reference_visual": str(row.get("reference_visual") or ""),
            "story_beat": (str(row.get("story_beat") or "") or None),
            "transition_hint": (str(row.get("transition_hint") or "") or None),
            "ref_luma": _safe_float(row.get("ref_luma")),
            "ref_dark_frac": _safe_float(row.get("ref_dark_frac")),
            "ref_rgb_mean": (row.get("ref_rgb_mean") if isinstance(row.get("ref_rgb_mean"), list) else None),
            "music_energy": _safe_float(row.get("music_energy")),
            "start_beat": row.get("start_beat"),
            "end_beat": row.get("end_beat"),
        }
        chosen = {
            "shot_id": str(row.get("shot_id") or ""),
            "sequence_group_id": str(row.get("sequence_group_id") or ""),
            "asset_id": str(row.get("asset_id") or ""),
            "asset_path": str(row.get("asset_path") or ""),
            "asset_kind": str(row.get("asset_kind") or ""),
            "asset_in_s": _safe_float(row.get("asset_in_s")),
            "duration_s": _safe_float(row.get("duration_s")),
            "speed": _safe_float(row.get("speed")),
            "crop_mode": str(row.get("crop_mode") or ""),
            "stabilize": bool(row.get("stabilize")) if row.get("stabilize") is not None else None,
            "zoom": _safe_float(row.get("zoom")),
            "grade": row.get("grade") if isinstance(row.get("grade"), dict) else None,
            "shot_metrics": {
                "motion_score": _safe_float(row.get("shot_motion_score")),
                "shake_score": _safe_float(row.get("shot_shake_score")),
                "sharpness": _safe_float(row.get("shot_sharpness")),
                "luma_mean": _safe_float(row.get("shot_luma_mean")),
                "dark_frac": _safe_float(row.get("shot_dark_frac")),
                "rgb_mean": (row.get("shot_rgb_mean") if isinstance(row.get("shot_rgb_mean"), list) else None),
                "cam_motion_mag": _safe_float(row.get("shot_cam_motion_mag")),
                "cam_motion_angle_deg": _safe_float(row.get("shot_cam_motion_angle_deg")),
            },
            "inpoint_auto": row.get("inpoint_auto") if isinstance(row.get("inpoint_auto"), dict) else None,
            "director_source": str(row.get("director_source") or "") or None,
            "director_clamped": bool(row.get("director_clamped")) if row.get("director_clamped") is not None else None,
            "director_stability_swap": bool(row.get("director_stability_swap")) if row.get("director_stability_swap") is not None else None,
            "director_repeat_swap": bool(row.get("director_repeat_swap")) if row.get("director_repeat_swap") is not None else None,
        }

        # Objective evidence (mid-frame diffs), when present.
        evidence = {
            "out_frame": str(row.get("out_frame") or "") or None,
            "ref_frame": str(row.get("ref_frame") or "") or None,
            "out_luma_mid": _safe_float(row.get("out_luma_mid")),
            "out_dark_mid": _safe_float(row.get("out_dark_mid")),
            "out_rgb_mid": (row.get("out_rgb_mid") if isinstance(row.get("out_rgb_mid"), list) else None),
            "ref_luma_mid": _safe_float(row.get("ref_luma_mid")),
            "ref_dark_mid": _safe_float(row.get("ref_dark_mid")),
            "ref_rgb_mid": (row.get("ref_rgb_mid") if isinstance(row.get("ref_rgb_mid"), list) else None),
            "blur_lap_var_mid": _safe_float(row.get("blur_lap_var_mid")),
        }

        alternatives: list[dict[str, t.Any]] = []
        if isinstance(candidates_by_seg_id, dict) and sid in candidates_by_seg_id:
            raw = candidates_by_seg_id.get(int(sid)) or []
            for c in raw[: max(0, int(max_candidates_per_seg))]:
                if not isinstance(c, dict):
                    continue
                alternatives.append(_compact_shot_candidate(c))

        dossier_segs.append(
            {
                "segment_id": int(sid),
                "ref": ref,
                "chosen": chosen,
                "evidence": evidence,
                # Optional judge/critic fields are added later; keep stable key.
                "judge": row.get("judge") if isinstance(row.get("judge"), dict) else None,
                "alternatives": alternatives,
            }
        )

    return {
        "schema_version": 1,
        "created_at": _now_ts(),
        "project_root": str(timeline_doc.get("project_root") or ""),
        "variant_id": str(timeline_doc.get("variant_id") or ""),
        "mode": str(timeline_doc.get("mode") or ""),
        "story_plan_id": str(timeline_doc.get("story_plan_id") or "") or None,
        "director": str((timeline_doc.get("director") or "") if isinstance(timeline_doc.get("director"), str) else "") or None,
        "segments": dossier_segs,
    }


def _append_jsonl(path: Path, rec: dict[str, t.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=True) + "\n"
    try:
        import fcntl  # type: ignore

        with path.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line)
            try:
                f.flush()
            except Exception:
                pass
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        # Best-effort fallback.
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def append_segment_runs(
    *,
    dossier_doc: dict[str, t.Any],
    extra: dict[str, t.Any] | None = None,
) -> None:
    """
    Append per-segment rows to a global JSONL log (Outputs/segment_runs.jsonl by default).
    This forms the training/eval substrate for segment-level graders and directed fix loops.
    """
    segs = dossier_doc.get("segments")
    if not isinstance(segs, list) or not segs:
        return

    log_root = str(os.getenv("SEGMENT_RUNS_LOG_DIR", "") or "").strip()
    if log_root:
        out_path = (Path(log_root).expanduser().resolve() / "segment_runs.jsonl").resolve()
    else:
        out_path = (Path("Outputs") / "segment_runs.jsonl").resolve()

    base = {
        "schema_version": 1,
        "created_at": _now_ts(),
        "project_root": str(dossier_doc.get("project_root") or ""),
        "variant_id": str(dossier_doc.get("variant_id") or ""),
        "story_plan_id": str(dossier_doc.get("story_plan_id") or "") or None,
        "director": str(dossier_doc.get("director") or "") or None,
    }
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k in base:
                continue
            base[k] = v

    for s in segs:
        if not isinstance(s, dict):
            continue
        sid = s.get("segment_id")
        try:
            sid_i = int(sid)
        except Exception:
            sid_i = 0
        if sid_i <= 0:
            continue
        rec = dict(base)
        rec["segment_id"] = int(sid_i)
        rec["ref"] = s.get("ref") if isinstance(s.get("ref"), dict) else None
        rec["chosen"] = s.get("chosen") if isinstance(s.get("chosen"), dict) else None
        rec["evidence"] = s.get("evidence") if isinstance(s.get("evidence"), dict) else None
        rec["judge"] = s.get("judge") if isinstance(s.get("judge"), dict) else None
        _append_jsonl(out_path, rec)

