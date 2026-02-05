from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import typing as t

from .gemini_score_normalize import normalize_judge_result
from .openrouter_client import OpenRouterError, chat_completions


def _strip_code_fences(text: str) -> str:
    s = text.strip()
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
        return json.loads(snippet)
    except json.JSONDecodeError:
        # Some models accidentally double-escape JSON.
        if '\\"' in snippet or "\\n" in snippet:
            try:
                unescaped = snippet.encode("utf-8").decode("unicode_escape")
                return json.loads(unescaped)
            except Exception:
                pass
        raise


def _encode_image_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _encode_video_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "m4v": "video/mp4",
        "webm": "video/webm",
    }.get(ext, "video/mp4")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _reasoning_param() -> dict[str, t.Any]:
    effort_env = os.getenv("REASONING_EFFORT", "").strip().lower()
    if effort_env == "xhigh":
        effort_env = "high"
    if effort_env in {"none", "minimal", "low", "medium", "high"}:
        return {"effort": effort_env}
    # Default to high to get more consistent/nuanced critique; override via REASONING_EFFORT.
    return {"effort": "high"}


@dataclass(frozen=True)
class SegmentCritique:
    segment_id: int
    issues: list[str]
    suggestions: list[str]


@dataclass(frozen=True)
class EditEvaluation:
    overall_score: float
    overall_notes: str
    planner_guidance: str
    segment_critiques: list[SegmentCritique]
    raw: dict[str, t.Any]


@dataclass(frozen=True)
class FullVideoEvaluation:
    result: dict[str, t.Any]
    raw: dict[str, t.Any]
    usage: dict[str, t.Any] | None
    model_requested: str
    model_used: str | None


def _build_compare_system_prompt(*, criteria: str, variant: str) -> str:
    variant = (variant or "").strip().lower() or "compare_general"
    if variant not in {"compare_general", "compare_detailed"}:
        variant = "compare_general"

    base = [
        "You are a ruthless but fair SHORT-FORM REEL CRITIC and senior video editor.",
        "You will be given TWO full videos (in this order):",
        "- OUTPUT (our edit to judge)",
        "- REFERENCE (Instagram reel; style target)",
        "",
        "Task:",
        "- Score OUTPUT on absolute professional quality AND on match-to-reference.",
        "",
        "Safety/Scope:",
        "- Focus ONLY on editing craft (storyline, rhythm, continuity, stability, framing, look).",
        "- Do NOT describe nudity/sexual content or any sensitive personal attributes.",
        "",
        "Scoring rules (important):",
        "- Be strict and consistent. Use the FULL range (do not cluster everything in 7-9).",
        "- overall_score is ABSOLUTE publishability/pro quality (0-10). match_score is how well OUTPUT matches REFERENCE (0-10).",
        "- Subscores (0-5) are diagnostic and NOT additive. Do NOT sum or average subscores to compute overall_score.",
        "- match_score must NEVER excuse bad execution (e.g., shaky footage can match reference style and still be low stability/low overall).",
        "- overall_score MUST be decided from OUTPUT alone. Treat REFERENCE as style guidance only, not as a grading curve.",
        "",
        "Evaluation procedure (follow this order):",
        "1) Watch OUTPUT and score overall_score + subscores based on professional publishability ONLY.",
        "2) Then compare OUTPUT vs REFERENCE and set match_score, stability_match, look_match, and differences.",
        "",
        "Important scoring guidance (stability):",
        "- stability is ABSOLUTE viewer comfort regardless of reference.",
        "- stability_match is whether OUTPUT matches the reference's camera motion style.",
        "If REFERENCE is shaky, OUTPUT can match it and still have low stability.",
        "",
        "Stability rubric (use this):",
        "- 5 = gimbal/tripod stable, no distracting shake/warp.",
        "- 4 = minor handheld motion, not distracting on mobile.",
        "- 3 = noticeable shake, borderline distracting, but tolerable.",
        "- 2 = distracting shake/warp/jitter that harms watchability.",
        "- 1 = severe shake/warp; uncomfortable to watch.",
        "- 0 = unwatchable due to instability.",
        "Hard constraint: if stability <= 2, overall_score MUST be <= 5. If stability == 1, overall_score MUST be <= 3.",
        "",
        f"Criteria focus: {criteria}",
        "",
    ]

    if variant == "compare_detailed":
        body = [
            "Return ONLY strict JSON with these keys:",
            "{",
            '  "overall_score": 0-10,',
            '  "match_score": 0-10,',
            '  "story_arc": 0-5,',
            '  "rhythm": 0-5,',
            '  "continuity": 0-5,',
            '  "stability": 0-5,',
            '  "stability_match": 0-5,',
            '  "framing": 0-5,',
            '  "look": 0-5,',
            '  "look_match": 0-5,',
            '  "exposure_match": 0-5,',
            '  "top_issues": [string, ...],',
            '  "actionable_fixes": [string, ...]',
            "}",
        ]
    else:
        body = [
            "Return ONLY strict JSON with these keys:",
            "{",
            '  "overall_score": 0-10,',
            '  "match_score": 0-10,',
            '  "story_arc": 0-5,',
            '  "rhythm": 0-5,',
            '  "continuity": 0-5,',
            '  "stability": 0-5,',
            '  "framing": 0-5,',
            '  "look": 0-5,',
            '  "exposure_match": 0-5,',
            '  "top_issues": [string, ...],',
            '  "actionable_fixes": [string, ...]',
            "}",
        ]

    return "\n".join(base + body).strip()


def evaluate_edit_full_video_compare(
    *,
    api_key: str,
    model: str,
    output_video_path: Path,
    reference_video_path: Path,
    criteria: str,
    prompt_variant: str = "compare_general",
    timeout_s: float = 240.0,
    site_url: str | None = None,
    app_name: str | None = None,
) -> FullVideoEvaluation:
    """
    Full-video judge: upload OUTPUT and REFERENCE (OUTPUT first) and request strict rubric scores.
    """
    if not output_video_path.exists():
        raise FileNotFoundError(f"Output video not found: {output_video_path}")
    if not reference_video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video_path}")

    system_prompt = _build_compare_system_prompt(criteria=criteria, variant=prompt_variant)
    content: list[dict[str, t.Any]] = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    "Videos will be provided in this exact order: OUTPUT then REFERENCE.",
                    "Return JSON only.",
                ]
            ),
        },
        {"type": "text", "text": "OUTPUT VIDEO"},
        {"type": "video_url", "video_url": {"url": _encode_video_data_url(output_video_path)}},
        {"type": "text", "text": "REFERENCE VIDEO"},
        {"type": "video_url", "video_url": {"url": _encode_video_data_url(reference_video_path)}},
    ]

    messages: list[dict[str, t.Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    last_text = ""
    data: dict[str, t.Any] | None = None
    for attempt in range(1, 4):
        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.0,
            timeout_s=timeout_s,
            site_url=site_url,
            app_name=app_name,
            reasoning=_reasoning_param(),
            retries=3,
            retry_delay_s=2.0,
        )
        last_text = result.content or ""
        try:
            data = _extract_json_object(last_text)
            normalized = normalize_judge_result(t.cast(dict[str, t.Any], data))
            if normalized != data:
                normalized = dict(normalized)
                normalized["_normalization_applied"] = True
                normalized["_raw_model_result"] = data
            return FullVideoEvaluation(
                result=normalized if normalized != data else data,
                raw=result.raw,
                usage=result.usage,
                model_requested=str(model),
                model_used=(str(result.raw.get("model")) if isinstance(result.raw, dict) and isinstance(result.raw.get("model"), str) else None),
            )
        except Exception as e:
            if attempt >= 3:
                snippet = (last_text or "").strip()[:800]
                raise OpenRouterError(f"Failed to parse full-video judge JSON: {type(e).__name__}: {e}. text={snippet!r}")
            # Ask for a strict JSON reprint.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your last response was not valid JSON. Reprint ONLY valid JSON (no markdown, no extra prose). Keep strings short.",
                        }
                    ],
                },
            ]

    raise OpenRouterError("Full-video judge failed unexpectedly")


def evaluate_edit_similarity(
    *,
    api_key: str,
    model: str,
    reference_frames: list[tuple[int, Path]],
    output_frames: list[tuple[int, Path]],
    segment_summaries: list[str],
    timeout_s: float = 180.0,
    site_url: str | None = None,
    app_name: str | None = None,
) -> EditEvaluation:
    """
    Compare reference vs output using per-segment frames and return meta guidance (NOT video-specific prompts).
    """
    if len(reference_frames) != len(output_frames):
        raise ValueError("reference_frames and output_frames must have same length")

    system_prompt = "\n".join(
        [
            "You are a ruthless but helpful SHORT-FORM EDIT CRITIC.",
            "You will be shown paired frames: REFERENCE (from a viral reel) vs OUTPUT (our recreated edit).",
            "",
            "Task:",
            "1) Judge how well the OUTPUT matches the REFERENCE structure and style.",
            "2) Identify where the OUTPUT diverges (wrong setting, wrong shot type, wrong energy, unclear motif, poor readability).",
            "3) Produce META guidance to improve the planner/editor universally for ANY reel (not specific to this video).",
            "",
            "Extra focus areas (to make this feel like a professional human editor):",
            "- Storyline coherence: each segment should feel like it belongs in an intentional arc (hook -> build -> peak).",
            "- Shot language: variety of shot sizes + purposeful continuity/contrast (not random switching).",
            "- Visual continuity: avoid jarring jumps in color temperature / exposure unless the reference also jumps.",
            "",
            "Rules:",
            "- Do NOT guess identities or names.",
            "- Do NOT propose prompts tailored to this specific reel; keep guidance generic and reusable.",
            "- Focus on principles: shot continuity, motif consistency, pacing, overlay text brevity/placement, asset selection logic.",
            "- Keep outputs SHORT to avoid truncation:",
            "  - overall_notes <= 450 chars",
            "  - planner_guidance <= 650 chars",
            "  - each segment: <= 4 issues and <= 4 suggestions, short phrases",
            "",
            "Return ONLY strict JSON:",
            "{",
            '  \"overall_score\": 0-10,',
            '  \"overall_notes\": string,',
            '  \"planner_guidance\": string,',
            '  \"segments\": [',
            "    {",
            '      \"segment_id\": 1,',
            '      \"issues\": [string, ...],',
            '      \"suggestions\": [string, ...]',
            "    }",
            "  ]",
            "}",
        ]
    ).strip()

    content: list[dict[str, t.Any]] = [
        {
            "type": "text",
            "text": "Compare each segment pair and then output JSON only.",
        }
    ]

    for (sid_r, ref_path), (sid_o, out_path), summary in zip(reference_frames, output_frames, segment_summaries, strict=True):
        sid = sid_r
        if sid_o != sid_r:
            sid = sid_r
        content.append({"type": "text", "text": f"SEGMENT {sid} SUMMARY: {summary}"})
        content.append({"type": "text", "text": f"SEGMENT {sid} REFERENCE FRAME"})
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(ref_path)}})
        content.append({"type": "text", "text": f"SEGMENT {sid} OUTPUT FRAME"})
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(out_path)}})

    messages: list[dict[str, t.Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    last_text = ""
    last_error: Exception | None = None
    # Evaluation should be stable for benchmarking; allow forcing temperature via env.
    forced_temp = os.getenv("FOLDER_EDIT_EVAL_TEMPERATURE", "").strip()
    forced_temp_f: float | None = None
    if forced_temp:
        try:
            forced_temp_f = float(forced_temp)
        except Exception:
            forced_temp_f = None
    for attempt in range(1, 4):
        temp = 0.0
        if forced_temp_f is not None:
            temp = float(forced_temp_f)
        else:
            # First attempt can be slightly creative, but this makes scores noisy. Prefer forcing
            # FOLDER_EDIT_EVAL_TEMPERATURE=0 for deterministic regression testing.
            temp = 0.2 if attempt == 1 else 0.0
        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=float(temp),
            max_tokens=1600,
            timeout_s=timeout_s,
            site_url=site_url,
            app_name=app_name,
            reasoning=_reasoning_param(),
            retries=3,
            retry_delay_s=2.0,
        )
        last_text = result.content or ""
        try:
            data = _extract_json_object(last_text)
            break
        except Exception as e:
            last_error = e
            if attempt >= 3:
                snippet = (last_text or "").strip()[:800]
                raise OpenRouterError(f"Failed to parse evaluation JSON: {type(e).__name__}: {e}. text={snippet!r}")
            # Ask for a strict JSON reprint.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your last response was not valid JSON. Reprint ONLY valid JSON (no markdown, no extra prose). Keep strings short.",
                        }
                    ],
                },
            ]

    segs = data.get("segments") or []
    critiques: list[SegmentCritique] = []
    if isinstance(segs, list):
        for item in segs:
            if not isinstance(item, dict):
                continue
            sid = int(item.get("segment_id") or 0)
            issues = item.get("issues") or []
            suggestions = item.get("suggestions") or []
            if not isinstance(issues, list) or not isinstance(suggestions, list):
                continue
            critiques.append(
                SegmentCritique(
                    segment_id=sid,
                    issues=[str(x) for x in issues[:6]],
                    suggestions=[str(x) for x in suggestions[:6]],
                )
            )

    try:
        score = float(data.get("overall_score") or 0.0)
    except Exception:
        score = 0.0

    return EditEvaluation(
        overall_score=score,
        overall_notes=str(data.get("overall_notes") or "").strip(),
        planner_guidance=str(data.get("planner_guidance") or "").strip(),
        segment_critiques=sorted(critiques, key=lambda c: c.segment_id),
        raw=data,
    )
