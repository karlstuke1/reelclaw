from __future__ import annotations

import dataclasses
import typing as t


SEVERITIES = ("low", "med", "high")
TRANSITION_TYPES = ("hard_cut", "dip_to_black", "dip_to_white")
FIX_ACTION_TYPES = (
    "set_stabilize",
    "set_crop_mode",
    "set_zoom",
    "set_grade",
    "shift_inpoint",
    "set_speed",
    "set_fade_out",
    "set_overlay_text",
)


def _is_str_list(x: t.Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _coerce_int(x: t.Any, *, field: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Invalid int for {field}: {x!r}") from e


def _coerce_float(x: t.Any, *, field: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid float for {field}: {x!r}") from e


def _coerce_bool(x: t.Any, *, field: str) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    raise ValueError(f"Invalid bool for {field}: {x!r}")


def _validate_overlay_text(value: str) -> str:
    """
    Enforce overlay caps (max 2 lines, <=26 chars/line). Raise on violation.
    """
    s = (value or "").strip()
    if not s:
        return ""
    lines = s.splitlines()
    _require(len(lines) <= 2, "overlay_text must be <= 2 lines")
    for ln in lines:
        _require(len(ln) <= 26, "overlay_text lines must be <= 26 chars")
    return "\n".join(lines)


def _validate_tags(value: t.Any, *, field: str, max_items: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        # Keep validator tolerant; executor/enforcer layers will apply caps.
        return []
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        tag = raw.strip().lower()
        if not tag:
            continue
        out.append(tag)
        if len(out) >= int(max_items):
            break
    return out


@dataclasses.dataclass(frozen=True)
class CritiqueSegment:
    segment_id: int
    issues: list[str]
    suggestions: list[str]
    severity: str


@dataclasses.dataclass(frozen=True)
class SegmentScore:
    segment_id: int
    overall: float | None = None
    stability: float | None = None
    rhythm: float | None = None
    look: float | None = None


@dataclasses.dataclass(frozen=True)
class FixAction:
    type: str
    segment_id: int
    value: t.Any = None
    seconds: float | None = None


@dataclasses.dataclass(frozen=True)
class SegmentDelta:
    segment_id: int
    desired_tags_add: list[str]
    desired_tags_remove: list[str]
    story_beat: str | None = None
    transition_hint: str | None = None
    overlay_text_rewrite: str | None = None


@dataclasses.dataclass(frozen=True)
class TransitionDelta:
    boundary_after_segment_id: int
    type: str
    seconds: float


@dataclasses.dataclass(frozen=True)
class CritiqueReport:
    version: int
    model: str
    overall_score: float
    subscores: dict[str, float]
    summary_nl: str
    segments: list[CritiqueSegment]
    segment_scores: list[SegmentScore]
    lane_a_actions: list[FixAction]
    lane_b_deltas: list[SegmentDelta]
    transition_deltas: list[TransitionDelta]

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "version": int(self.version),
            "model": str(self.model),
            "overall_score": float(self.overall_score),
            "subscores": {k: float(v) for k, v in self.subscores.items()},
            "summary_nl": str(self.summary_nl),
            "segments": [
                {
                    "segment_id": int(s.segment_id),
                    "issues": list(s.issues),
                    "suggestions": list(s.suggestions),
                    "severity": str(s.severity),
                }
                for s in self.segments
            ],
            "segment_scores": [
                {
                    "segment_id": int(s.segment_id),
                    **({"overall": float(s.overall)} if s.overall is not None else {}),
                    **({"stability": float(s.stability)} if s.stability is not None else {}),
                    **({"rhythm": float(s.rhythm)} if s.rhythm is not None else {}),
                    **({"look": float(s.look)} if s.look is not None else {}),
                }
                for s in self.segment_scores
            ],
            "lane_a_actions": [
                {
                    "type": a.type,
                    "segment_id": int(a.segment_id),
                    **({"value": a.value} if a.value is not None else {}),
                    **({"seconds": float(a.seconds)} if a.seconds is not None else {}),
                }
                for a in self.lane_a_actions
            ],
            "lane_b_deltas": [
                {
                    "segment_id": int(d.segment_id),
                    "desired_tags_add": list(d.desired_tags_add),
                    "desired_tags_remove": list(d.desired_tags_remove),
                    **({"story_beat": d.story_beat} if d.story_beat is not None else {}),
                    **({"transition_hint": d.transition_hint} if d.transition_hint is not None else {}),
                    **({"overlay_text_rewrite": d.overlay_text_rewrite} if d.overlay_text_rewrite is not None else {}),
                }
                for d in self.lane_b_deltas
            ],
            "transition_deltas": [
                {"boundary_after_segment_id": int(td.boundary_after_segment_id), "type": td.type, "seconds": float(td.seconds)}
                for td in self.transition_deltas
            ],
        }


def severity_rank(severity: str) -> int:
    s = str(severity or "").strip().lower()
    if s == "high":
        return 3
    if s == "med":
        return 2
    return 1


def validate_critique_report(doc: t.Any, *, model: str) -> CritiqueReport:
    _require(isinstance(doc, dict), "CritiqueReport must be an object")
    d = t.cast(dict[str, t.Any], doc)

    version = _coerce_int(d.get("version"), field="version")
    _require(version == 1, "CritiqueReport.version must be 1")

    overall = _coerce_float(d.get("overall_score"), field="overall_score")
    _require(0.0 <= float(overall) <= 10.0, "overall_score must be 0..10")

    subs = d.get("subscores")
    _require(isinstance(subs, dict), "subscores must be an object")
    subscores: dict[str, float] = {}
    for key in ("story_arc", "rhythm", "continuity", "stability", "framing", "look"):
        v = _coerce_float(t.cast(dict[str, t.Any], subs).get(key), field=f"subscores.{key}")
        _require(0.0 <= float(v) <= 5.0, f"subscores.{key} must be 0..5")
        subscores[key] = float(v)

    # Enforce stability->overall hard caps (matches our compare critic rubric).
    try:
        stability = float(subscores.get("stability") or 0.0)
        if stability <= 1.0:
            overall = float(min(float(overall), 3.0))
        elif stability <= 2.0:
            overall = float(min(float(overall), 5.0))
    except Exception:
        pass

    summary = str(d.get("summary_nl") or "").strip()
    _require(len(summary) <= 600, "summary_nl must be <= 600 chars")

    segs_raw = d.get("segments")
    _require(isinstance(segs_raw, list), "segments must be a list")
    segs: list[CritiqueSegment] = []
    for item in segs_raw:
        _require(isinstance(item, dict), "segments items must be objects")
        it = t.cast(dict[str, t.Any], item)
        seg_id = _coerce_int(it.get("segment_id"), field="segments[].segment_id")
        issues = it.get("issues") or []
        suggestions = it.get("suggestions") or []
        _require(_is_str_list(issues), "segments[].issues must be list[str]")
        _require(_is_str_list(suggestions), "segments[].suggestions must be list[str]")
        sev = str(it.get("severity") or "").strip().lower()
        _require(sev in SEVERITIES, f"segments[].severity must be one of {SEVERITIES}")
        segs.append(CritiqueSegment(segment_id=int(seg_id), issues=list(issues), suggestions=list(suggestions), severity=sev))

    # Optional: per-segment numeric scores (helps segment-level fix loops).
    seg_scores_raw = d.get("segment_scores")
    seg_scores: list[SegmentScore] = []
    if isinstance(seg_scores_raw, list):
        for item in seg_scores_raw:
            if not isinstance(item, dict):
                continue
            it = t.cast(dict[str, t.Any], item)
            try:
                seg_id = _coerce_int(it.get("segment_id"), field="segment_scores[].segment_id")
            except Exception:
                continue
            if int(seg_id) <= 0:
                continue

            def _score_field(name: str) -> float | None:
                if it.get(name) is None:
                    return None
                try:
                    v0 = _coerce_float(it.get(name), field=f"segment_scores[].{name}")
                except Exception:
                    return None
                if not (0.0 <= float(v0) <= 5.0):
                    return None
                return float(v0)

            seg_scores.append(
                SegmentScore(
                    segment_id=int(seg_id),
                    overall=_score_field("overall"),
                    stability=_score_field("stability"),
                    rhythm=_score_field("rhythm"),
                    look=_score_field("look"),
                )
            )

    lane_a_raw = d.get("lane_a_actions")
    if not isinstance(lane_a_raw, list):
        lane_a_raw = []
    lane_a: list[FixAction] = []
    for item in lane_a_raw:
        if not isinstance(item, dict):
            continue
        it = t.cast(dict[str, t.Any], item)
        at = str(it.get("type") or "").strip()
        if not at:
            continue
        try:
            seg_id = _coerce_int(it.get("segment_id"), field="lane_a_actions[].segment_id")
        except Exception:
            continue

        value = it.get("value")
        seconds: float | None = None
        if it.get("seconds") is not None:
            try:
                seconds = _coerce_float(it.get("seconds"), field="lane_a_actions[].seconds")
            except Exception:
                seconds = None
        # Common failure: model sometimes puts fade seconds in "value".
        if at == "set_fade_out" and seconds is None and isinstance(value, (int, float)):
            seconds = float(value)

        lane_a.append(FixAction(type=at, segment_id=int(seg_id), value=value, seconds=seconds))

    lane_b_raw = d.get("lane_b_deltas")
    if not isinstance(lane_b_raw, list):
        lane_b_raw = []
    lane_b: list[SegmentDelta] = []
    for item in lane_b_raw:
        if not isinstance(item, dict):
            continue
        it = t.cast(dict[str, t.Any], item)
        try:
            seg_id = _coerce_int(it.get("segment_id"), field="lane_b_deltas[].segment_id")
        except Exception:
            continue
        tags_add = _validate_tags(it.get("desired_tags_add"), field="desired_tags_add", max_items=8)
        tags_rm = _validate_tags(it.get("desired_tags_remove"), field="desired_tags_remove", max_items=8)

        story_beat_raw = str(it.get("story_beat") or "").strip()
        story_beat = story_beat_raw[:80] if story_beat_raw else None

        transition_hint_raw = str(it.get("transition_hint") or "").strip()
        transition_hint = transition_hint_raw if transition_hint_raw in {"continuity", "contrast", "neutral"} else None

        overlay_text_rewrite: str | None = None
        overlay_rewrite = it.get("overlay_text_rewrite")
        if isinstance(overlay_rewrite, str):
            try:
                overlay_text_rewrite = _validate_overlay_text(overlay_rewrite)
            except Exception:
                overlay_text_rewrite = None

        lane_b.append(
            SegmentDelta(
                segment_id=int(seg_id),
                desired_tags_add=tags_add,
                desired_tags_remove=tags_rm,
                story_beat=story_beat,
                transition_hint=transition_hint,
                overlay_text_rewrite=overlay_text_rewrite,
            )
        )

    trans_raw = d.get("transition_deltas")
    if not isinstance(trans_raw, list):
        trans_raw = []
    trans: list[TransitionDelta] = []
    for item in trans_raw:
        if not isinstance(item, dict):
            continue
        it = t.cast(dict[str, t.Any], item)
        try:
            boundary = _coerce_int(it.get("boundary_after_segment_id"), field="transition_deltas[].boundary_after_segment_id")
        except Exception:
            continue
        ttype = str(it.get("type") or "").strip()
        try:
            seconds = _coerce_float(it.get("seconds"), field="transition_deltas[].seconds")
        except Exception:
            continue
        trans.append(TransitionDelta(boundary_after_segment_id=int(boundary), type=ttype, seconds=float(seconds)))

    return CritiqueReport(
        version=int(version),
        model=str(d.get("model") or model),
        overall_score=float(overall),
        subscores=subscores,
        summary_nl=summary,
        segments=segs,
        segment_scores=seg_scores,
        lane_a_actions=lane_a,
        lane_b_deltas=lane_b,
        transition_deltas=trans,
    )
