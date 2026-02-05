from __future__ import annotations

import typing as t


NUMERIC_0_10_KEYS: set[str] = {
    "overall_score",
    "match_score",
}

NUMERIC_0_5_KEYS: set[str] = {
    "story_arc",
    "rhythm",
    "continuity",
    "stability",
    "stability_match",
    "framing",
    "look",
    "look_match",
    "exposure_match",
}


def _to_float(x: t.Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def normalize_judge_result(result: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Normalize/clip numeric fields from a Gemini "judge" JSON result.

    Purpose:
    - Make label ranges consistent for downstream ranking/training.
    - Be robust to occasional scale confusion (some models output 0..10 for 0..5 subscores).
    - Enforce hard stability caps on overall_score (even if model ignores the rubric).
    """
    if not isinstance(result, dict):
        return {}

    out: dict[str, t.Any] = dict(result)

    for k in sorted(NUMERIC_0_10_KEYS):
        v = _to_float(out.get(k))
        if v is None:
            continue
        out[k] = _clip(v, 0.0, 10.0)

    for k in sorted(NUMERIC_0_5_KEYS):
        v0 = _to_float(out.get(k))
        if v0 is None:
            continue
        v = float(v0)
        # If a model accidentally outputs a 0..10 subscore, map it back to 0..5.
        if v > 5.0 and v <= 10.0:
            v = v / 2.0
        out[k] = _clip(v, 0.0, 5.0)

    # Enforce stability->overall hard caps (matches our prompt rubric).
    stability = _to_float(out.get("stability"))
    overall = _to_float(out.get("overall_score"))
    if stability is not None and overall is not None:
        cap: float | None = None
        if float(stability) <= 1.0:
            cap = 3.0
        elif float(stability) <= 2.0:
            cap = 5.0
        if cap is not None:
            out["overall_score"] = float(min(float(overall), float(cap)))

    return out

