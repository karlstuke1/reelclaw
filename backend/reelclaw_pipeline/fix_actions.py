from __future__ import annotations

import copy
import re
import typing as t


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _norm_tag(tag: str) -> str:
    return " ".join((tag or "").strip().lower().split())


def _grade_preset(name: str) -> dict[str, float] | None:
    """
    Translate a small set of named grade presets into deterministic grade dicts.

    This exists so the critic can return compact actions (e.g. "night_boost") that
    compile into deterministic ffmpeg parameters.
    """
    s = (name or "").strip().lower()
    if not s:
        return None
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")

    presets: dict[str, dict[str, float]] = {
        # Lift visibility for underexposed night footage without blowing highlights too hard.
        "night_boost": {"brightness": 0.12, "contrast": 1.12, "saturation": 1.06},
        # Slightly crush blacks for a more low-key look (use cautiously).
        "night_crush": {"brightness": -0.06, "contrast": 1.22, "saturation": 1.02},
        # Neutral reset (useful if a segment got over-graded).
        "reset": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0, "r_gain": 1.0, "g_gain": 1.0, "b_gain": 1.0},
    }
    return presets.get(s)


def _validate_overlay_text(value: str) -> tuple[bool, str | None]:
    s = (value or "").strip()
    if not s:
        return True, ""
    lines = s.splitlines()
    if len(lines) > 2:
        return False, "overlay_text must be <= 2 lines"
    for ln in lines:
        if len(ln) > 26:
            return False, "overlay_text lines must be <= 26 chars"
    return True, "\n".join(lines)


_SHOT_ID_RE = re.compile(r"^(?P<asset_id>[A-Za-z0-9-]+)#(?P<start_ms>\d+)_(?P<end_ms>\d+)$")
_SHOT_ID_IN_NOTES_RE = re.compile(r"\bshot_id=([A-Za-z0-9-]+#\d+_\d+)\b")


def _shot_window_s_from_shot_id(shot_id: str) -> tuple[float, float] | None:
    """
    Parse shot window boundaries from a shot_id like: <asset_id>#00001234_00005678 (ms).
    Returns (start_s, end_s) or None.
    """
    s = str(shot_id or "").strip()
    if not s:
        return None
    m = _SHOT_ID_RE.match(s)
    if not m:
        return None
    try:
        start_ms = int(m.group("start_ms"))
        end_ms = int(m.group("end_ms"))
    except Exception:
        return None
    if end_ms <= start_ms:
        return None
    return float(start_ms) / 1000.0, float(end_ms) / 1000.0


def _shot_id_from_segment(seg: dict[str, t.Any]) -> str | None:
    v = seg.get("shot_id")
    if isinstance(v, str) and v.strip():
        return v.strip()
    notes = seg.get("notes")
    if isinstance(notes, str) and notes:
        m = _SHOT_ID_IN_NOTES_RE.search(notes)
        if m:
            return str(m.group(1) or "").strip() or None
    return None


def apply_fix_actions(
    timeline_doc: dict[str, t.Any],
    actions: list[dict[str, t.Any]],
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """
    Apply executable Lane A FixActions to timeline_doc (no reselection).
    Returns (patched_timeline_doc, report).
    """
    patched = copy.deepcopy(timeline_doc)
    segs = patched.get("timeline_segments")
    if not isinstance(segs, list):
        return patched, {"applied": [], "rejected": [{"reason": "timeline_segments missing or not a list", "action": None}]}

    seg_by_id: dict[int, dict[str, t.Any]] = {}
    for s in segs:
        if not isinstance(s, dict):
            continue
        try:
            sid = int(s.get("id") or 0)
        except Exception:
            continue
        if sid > 0:
            seg_by_id[sid] = s

    applied: list[dict[str, t.Any]] = []
    rejected: list[dict[str, t.Any]] = []

    for act in actions or []:
        if not isinstance(act, dict):
            rejected.append({"reason": "action not an object", "action": act})
            continue
        atype = str(act.get("type") or "").strip()
        try:
            sid = int(act.get("segment_id") or 0)
        except Exception:
            sid = 0
        if sid <= 0 or sid not in seg_by_id:
            rejected.append({"reason": "segment_id not found", "action": act})
            continue
        seg = seg_by_id[sid]

        if atype == "set_stabilize":
            v = act.get("value")
            if isinstance(v, bool):
                seg["stabilize"] = bool(v)
                applied.append({"type": atype, "segment_id": sid, "value": bool(v)})
            else:
                rejected.append({"reason": "set_stabilize.value must be bool", "action": act})
            continue

        if atype == "set_crop_mode":
            v = str(act.get("value") or "").strip().lower()
            if v not in {"center", "top", "bottom", "face", "smart"}:
                rejected.append({"reason": "set_crop_mode.value invalid", "action": act})
                continue
            seg["crop_mode"] = v
            applied.append({"type": atype, "segment_id": sid, "value": v})
            continue

        if atype == "set_zoom":
            try:
                z = float(act.get("value"))
            except Exception:
                rejected.append({"reason": "set_zoom.value must be float", "action": act})
                continue
            z0 = float(z)
            z = _clamp(float(z), 1.0, 1.25)
            seg["zoom"] = float(z)
            rec: dict[str, t.Any] = {"type": atype, "segment_id": sid, "value": float(z)}
            if abs(z - z0) >= 1e-6:
                rec["clamped_from"] = float(z0)
            applied.append(rec)
            continue

        if atype == "shift_inpoint":
            # Move the in-point by a small delta, clamped to the selected shot window (if available).
            raw = act.get("seconds") if act.get("seconds") is not None else act.get("value")
            try:
                requested = float(raw)
            except Exception:
                rejected.append({"reason": "shift_inpoint requires seconds (float) in seconds or value", "action": act})
                continue
            delta = _clamp(float(requested), -0.60, 0.60)

            if seg.get("asset_in_s") is None:
                rejected.append({"reason": "shift_inpoint requires timeline_segments[].asset_in_s", "action": act})
                continue
            try:
                old_in = float(seg.get("asset_in_s"))
            except Exception:
                rejected.append({"reason": "shift_inpoint asset_in_s must be float", "action": act})
                continue

            if seg.get("duration_s") is None:
                rejected.append({"reason": "shift_inpoint requires timeline_segments[].duration_s", "action": act})
                continue
            try:
                dur = float(seg.get("duration_s"))
            except Exception:
                rejected.append({"reason": "shift_inpoint duration_s must be float", "action": act})
                continue
            if dur <= 0.0:
                rejected.append({"reason": "shift_inpoint duration_s must be > 0", "action": act})
                continue

            sp_raw = seg.get("speed")
            try:
                speed = float(sp_raw) if sp_raw is not None else 1.0
            except Exception:
                speed = 1.0
            if speed <= 0.0:
                speed = 1.0

            src_span = float(dur) * float(speed)
            shot_id = _shot_id_from_segment(seg)
            win = _shot_window_s_from_shot_id(shot_id) if shot_id else None

            new_in = float(old_in) + float(delta)
            unclamped_in = float(new_in)
            if win is not None:
                start_s, end_s = win
                max_start = float(end_s) - float(src_span)
                if max_start < float(start_s) - 1e-6:
                    rejected.append({"reason": "shift_inpoint cannot satisfy shot window", "action": act})
                    continue
                new_in = _clamp(float(new_in), float(start_s), float(max_start))
            else:
                new_in = max(0.0, float(new_in))

            seg["asset_in_s"] = float(new_in)
            if "asset_out_s" in seg:
                try:
                    seg["asset_out_s"] = float(new_in) + float(dur) * float(speed)
                except Exception:
                    pass

            rec3: dict[str, t.Any] = {"type": atype, "segment_id": sid, "seconds": float(new_in - old_in)}
            if abs(float(requested) - float(delta)) >= 1e-6:
                rec3["seconds_clamped_from"] = float(requested)
            if abs(float(unclamped_in) - float(new_in)) >= 1e-6:
                rec3["asset_in_s_clamped_from"] = float(unclamped_in)
            applied.append(rec3)
            continue

        if atype == "set_speed":
            raw = act.get("value") if act.get("value") is not None else act.get("seconds")
            try:
                requested = float(raw)
            except Exception:
                rejected.append({"reason": "set_speed.value must be float", "action": act})
                continue
            speed = _clamp(float(requested), 0.85, 1.25)

            if seg.get("asset_in_s") is None:
                rejected.append({"reason": "set_speed requires timeline_segments[].asset_in_s", "action": act})
                continue
            try:
                in_s = float(seg.get("asset_in_s"))
            except Exception:
                rejected.append({"reason": "set_speed asset_in_s must be float", "action": act})
                continue

            if seg.get("duration_s") is None:
                rejected.append({"reason": "set_speed requires timeline_segments[].duration_s", "action": act})
                continue
            try:
                dur = float(seg.get("duration_s"))
            except Exception:
                rejected.append({"reason": "set_speed duration_s must be float", "action": act})
                continue
            if dur <= 0.0:
                rejected.append({"reason": "set_speed duration_s must be > 0", "action": act})
                continue

            # Clamp in-point to shot window if known (avoid looping).
            shot_id = _shot_id_from_segment(seg)
            win = _shot_window_s_from_shot_id(shot_id) if shot_id else None
            new_in = float(in_s)
            new_in0 = float(new_in)
            if win is not None:
                start_s, end_s = win
                max_start = float(end_s) - (float(dur) * float(speed))
                if max_start < float(start_s) - 1e-6:
                    rejected.append({"reason": "set_speed cannot satisfy shot window", "action": act})
                    continue
                new_in = _clamp(float(new_in), float(start_s), float(max_start))
            else:
                new_in = max(0.0, float(new_in))

            seg["speed"] = float(speed)
            seg["asset_in_s"] = float(new_in)
            if "asset_out_s" in seg:
                try:
                    seg["asset_out_s"] = float(new_in) + float(dur) * float(speed)
                except Exception:
                    pass

            rec4: dict[str, t.Any] = {"type": atype, "segment_id": sid, "value": float(speed)}
            if abs(float(requested) - float(speed)) >= 1e-6:
                rec4["clamped_from"] = float(requested)
            if abs(float(new_in0) - float(new_in)) >= 1e-6:
                rec4["asset_in_s_adjusted_from"] = float(new_in0)
            applied.append(rec4)
            continue

        if atype == "set_grade":
            v = act.get("value")
            if isinstance(v, str):
                preset = _grade_preset(v)
                if preset is None:
                    rejected.append({"reason": "set_grade preset unknown", "action": act})
                    continue
                v = preset
            if not isinstance(v, dict):
                rejected.append({"reason": "set_grade.value must be object or known preset string", "action": act})
                continue
            cur = seg.get("grade")
            out: dict[str, float] = dict(cur) if isinstance(cur, dict) else {}
            # Conservative clamps aligned with _compute_eq_grade().
            for k, lo, hi in (
                ("brightness", -0.30, 0.30),
                ("contrast", 0.75, 1.55),
                ("saturation", 0.60, 1.70),
                ("gamma", 0.75, 1.55),
                ("r_gain", 0.70, 1.40),
                ("g_gain", 0.70, 1.40),
                ("b_gain", 0.70, 1.40),
            ):
                if k not in v:
                    continue
                try:
                    x = float(v.get(k))
                except Exception:
                    continue
                out[k] = float(_clamp(float(x), float(lo), float(hi)))
            seg["grade"] = out or None
            applied.append({"type": atype, "segment_id": sid, "value": out})
            continue

        if atype == "set_fade_out":
            try:
                sec = float(act.get("seconds"))
            except Exception:
                rejected.append({"reason": "set_fade_out.seconds must be float", "action": act})
                continue
            sec0 = float(sec)
            sec = _clamp(sec0, 0.0, 0.5)
            seg["fade_out_s"] = float(sec) if float(sec) > 1e-6 else None
            seg.setdefault("fade_out_color", "black")
            rec2: dict[str, t.Any] = {"type": atype, "segment_id": sid, "seconds": float(sec)}
            if abs(sec - sec0) >= 1e-6:
                rec2["clamped_from"] = float(sec0)
            applied.append(rec2)
            continue

        if atype == "set_overlay_text":
            v = act.get("value")
            if not isinstance(v, str):
                rejected.append({"reason": "set_overlay_text.value must be string", "action": act})
                continue
            ok, cleaned = _validate_overlay_text(v)
            if not ok or cleaned is None:
                rejected.append({"reason": "overlay_text invalid (max 2 lines, 26 chars/line)", "action": act})
                continue
            seg["overlay_text"] = cleaned
            applied.append({"type": atype, "segment_id": sid, "value": cleaned})
            continue

        rejected.append({"reason": f"unsupported action type: {atype}", "action": act})

    return patched, {"applied": applied, "rejected": rejected}


def apply_transition_deltas(
    timeline_doc: dict[str, t.Any],
    deltas: list[dict[str, t.Any]],
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """
    Apply TransitionDelta by mapping boundary changes to per-segment fade-in/fade-out params.
    """
    patched = copy.deepcopy(timeline_doc)
    segs = patched.get("timeline_segments")
    if not isinstance(segs, list):
        return patched, {"applied": [], "rejected": [{"reason": "timeline_segments missing or not a list", "delta": None}]}

    # Build timeline order and lookup.
    ids: list[int] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        try:
            ids.append(int(s.get("id") or 0))
        except Exception:
            ids.append(0)
    id_to_index = {sid: i for i, sid in enumerate(ids) if sid > 0}

    applied: list[dict[str, t.Any]] = []
    rejected: list[dict[str, t.Any]] = []

    for d in deltas or []:
        if not isinstance(d, dict):
            rejected.append({"reason": "delta not an object", "delta": d})
            continue
        try:
            after = int(d.get("boundary_after_segment_id") or 0)
        except Exception:
            after = 0
        if after <= 0 or after not in id_to_index:
            rejected.append({"reason": "boundary_after_segment_id not found", "delta": d})
            continue
        idx = id_to_index[after]
        if idx >= len(segs) - 1:
            rejected.append({"reason": "boundary_after_segment_id has no next segment", "delta": d})
            continue
        prev = segs[idx] if isinstance(segs[idx], dict) else None
        nxt = segs[idx + 1] if isinstance(segs[idx + 1], dict) else None
        if prev is None or nxt is None:
            rejected.append({"reason": "timeline segments malformed at boundary", "delta": d})
            continue

        ttype = str(d.get("type") or "").strip()
        try:
            sec = float(d.get("seconds"))
        except Exception:
            rejected.append({"reason": "seconds must be float", "delta": d})
            continue
        sec = _clamp(sec, 0.04, 0.20)

        if ttype == "hard_cut":
            prev["fade_out_s"] = None
            nxt["fade_in_s"] = None
            applied.append({"boundary_after_segment_id": after, "type": ttype, "seconds": float(sec)})
            continue

        if ttype in {"dip_to_black", "dip_to_white"}:
            color = "black" if ttype == "dip_to_black" else "white"
            prev["fade_out_s"] = float(sec)
            prev["fade_out_color"] = color
            nxt["fade_in_s"] = float(sec)
            nxt["fade_in_color"] = color
            applied.append({"boundary_after_segment_id": after, "type": ttype, "seconds": float(sec)})
            continue

        rejected.append({"reason": f"unsupported transition type: {ttype}", "delta": d})

    return patched, {"applied": applied, "rejected": rejected}


def apply_segment_deltas_to_timeline(
    timeline_doc: dict[str, t.Any],
    deltas: list[dict[str, t.Any]],
    *,
    allow_pro_fields: bool,
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """
    Apply Lane B SegmentDelta constraints to the timeline (no reselection here).
    This updates desired_tags/story_beat/transition_hint/overlay_text fields to steer later planning.
    """
    patched = copy.deepcopy(timeline_doc)
    segs = patched.get("timeline_segments")
    if not isinstance(segs, list):
        return patched, {"applied": [], "rejected": [{"reason": "timeline_segments missing or not a list", "delta": None}]}

    seg_by_id: dict[int, dict[str, t.Any]] = {}
    for s in segs:
        if not isinstance(s, dict):
            continue
        try:
            sid = int(s.get("id") or 0)
        except Exception:
            continue
        if sid > 0:
            seg_by_id[sid] = s

    applied: list[dict[str, t.Any]] = []
    rejected: list[dict[str, t.Any]] = []

    def _map_transition_hint(h: str) -> str | None:
        hh = (h or "").strip().lower()
        if not hh or hh == "neutral":
            return None
        if hh == "continuity":
            return "match cut"
        if hh == "contrast":
            return "hard contrast"
        return None

    for d in deltas or []:
        if not isinstance(d, dict):
            rejected.append({"reason": "delta not an object", "delta": d})
            continue
        try:
            sid = int(d.get("segment_id") or 0)
        except Exception:
            sid = 0
        if sid <= 0 or sid not in seg_by_id:
            rejected.append({"reason": "segment_id not found", "delta": d})
            continue
        seg = seg_by_id[sid]

        desired = seg.get("desired_tags") if isinstance(seg.get("desired_tags"), list) else []
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
        seg["desired_tags"] = next_tags

        rec: dict[str, t.Any] = {"segment_id": sid, "desired_tags_add": add, "desired_tags_remove": sorted(list(rm))}

        # Overlay text rewrite (always allowed).
        if "overlay_text_rewrite" in d and d.get("overlay_text_rewrite") is not None:
            v = d.get("overlay_text_rewrite")
            if isinstance(v, str):
                ok, cleaned = _validate_overlay_text(v)
                if ok and cleaned is not None:
                    seg["overlay_text"] = cleaned
                    rec["overlay_text_rewrite"] = cleaned
                else:
                    rejected.append({"reason": "overlay_text_rewrite invalid", "delta": d})
                    continue

        # Pro-only fields.
        if d.get("story_beat") is not None:
            if not allow_pro_fields:
                rejected.append({"reason": "story_beat not allowed (critic_pro_mode=false)", "delta": d})
                continue
            sb = str(d.get("story_beat") or "").strip()
            if len(sb) > 80:
                sb = sb[:80]
            seg["story_beat"] = sb
            rec["story_beat"] = sb

        if d.get("transition_hint") is not None:
            if not allow_pro_fields:
                rejected.append({"reason": "transition_hint not allowed (critic_pro_mode=false)", "delta": d})
                continue
            mapped = _map_transition_hint(str(d.get("transition_hint") or ""))
            seg["transition_hint"] = mapped
            rec["transition_hint"] = str(d.get("transition_hint") or "").strip().lower()

        applied.append(rec)

    return patched, {"applied": applied, "rejected": rejected}
