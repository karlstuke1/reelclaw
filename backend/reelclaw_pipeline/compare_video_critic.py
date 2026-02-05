from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
import typing as t

from .critic_schema import CritiqueReport, validate_critique_report
from .openrouter_client import OpenRouterError, chat_completions, normalize_model
from .video_proxy import encode_video_data_url, ensure_inlineable_video


def _reasoning_param() -> dict[str, t.Any]:
    effort_env = os.getenv("REASONING_EFFORT", "").strip().lower()
    if effort_env == "xhigh":
        effort_env = "high"
    if effort_env in {"none", "minimal", "low", "medium", "high"}:
        return {"effort": effort_env}
    return {"effort": "high"}


def _strip_code_fences(text: str) -> str:
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
            try:
                unescaped = snippet.encode("utf-8").decode("unicode_escape")
                return t.cast(dict[str, t.Any], json.loads(unescaped))
            except Exception:
                pass
        raise


@dataclasses.dataclass(frozen=True)
class CompareVideoCriticResult:
    report: CritiqueReport
    usage: dict[str, t.Any] | None
    model_requested: str
    model_used: str | None
    raw_text: str
    video_meta: dict[str, t.Any]


def _build_system_prompt(*, critic_pro_mode: bool) -> str:
    pro_line = (
        "- You MAY use story_beat and transition_hint in lane_b_deltas (critic_pro_mode=true)."
        if critic_pro_mode
        else "- You MUST NOT use story_beat or transition_hint in lane_b_deltas (critic_pro_mode=false)."
    )
    return "\n".join(
        [
            "You are a ruthless but fair SHORT-FORM EDIT CRITIC.",
            "You will be shown ONE compare video (side-by-side):",
            "- LEFT = REFERENCE (style/structure target)",
            "- RIGHT = OUTPUT (our edit)",
            "",
            "Both sides have burned-in segment labels (S01, S02, ...). Use these IDs when referring to problems.",
            "",
            "Task: critique the OUTPUT and propose strict, executable improvements.",
            "",
            "Critical rule:",
            "- Do NOT apply free-form edits. Output only schema-valid JSON deltas/actions.",
            "",
            "Exposure/look rule:",
            "- Judge exposure/brightness/contrast RELATIVE to the REFERENCE.",
            "- If the REFERENCE is intentionally dark/low-key, do NOT penalize darkness unless OUTPUT deviates from it.",
            "",
            "Stability rule (important):",
            "- stability is ABSOLUTE viewer comfort regardless of reference style.",
            "- If REFERENCE is shaky, OUTPUT can match it and still have low stability.",
            "Hard constraint: if subscores.stability <= 2, overall_score MUST be <= 5. If subscores.stability <= 1, overall_score MUST be <= 3.",
            "",
            "Return ONLY strict JSON with this exact schema (no extra keys):",
            "{",
            '  "version": 1,',
            '  "model": string,',
            '  "overall_score": 0-10,  // ABSOLUTE publishability/pro quality',
            '  "subscores": {',
            '    "story_arc": 0-5, "rhythm": 0-5, "continuity": 0-5, "stability": 0-5, "framing": 0-5, "look": 0-5',
            "  },",
            '  "summary_nl": "short paragraph (<=600 chars)",',
            '  "segments": [',
            '    {"segment_id": 1, "issues": [..], "suggestions": [..], "severity": "low|med|high"}',
            "  ],",
            '  "segment_scores": [',
            '    {"segment_id": 1, "overall": 0-5, "stability"?: 0-5, "rhythm"?: 0-5, "look"?: 0-5}',
            "  ],",
            '  "lane_a_actions": [',
            '    {"type": "...", "segment_id": 1, "value": ...} OR {"type":"set_fade_out","segment_id":1,"seconds":0.18}',
            "  ],",
            '  "lane_b_deltas": [',
            '    {"segment_id": 1, "desired_tags_add": [...], "desired_tags_remove": [...], "story_beat"?: str, "transition_hint"?: "continuity|contrast|neutral", "overlay_text_rewrite"?: str}',
            "  ],",
            '  "transition_deltas": [',
            '    {"boundary_after_segment_id": 1, "type": "hard_cut|dip_to_black|dip_to_white", "seconds": 0.04-0.20}',
            "  ]",
            "}",
            "",
            "Constraints:",
            "- Keep outputs short to avoid truncation.",
            "- lane_b_deltas MUST modify at most 2 segments total (pick highest severity).",
            "- segment_scores: include ALL segments shown if possible. Keep as numeric only (no extra strings).",
            "- desired_tags_add/remove: <=8 tags each, lowercase.",
            "- overlay_text_rewrite and lane_a set_overlay_text: max 2 lines, max 26 chars/line.",
            "- lane_a_actions allowed types: set_stabilize, set_crop_mode(center|top|bottom|face|smart), set_zoom(1.0..1.25), set_grade, set_fade_out(0..0.5), set_overlay_text.",
            pro_line,
        ]
    ).strip()


def _build_user_text(*, niche: str, vibe: str, timeline_summary: dict[str, t.Any]) -> str:
    # Keep this compact; the video carries most of the signal.
    segs = timeline_summary.get("segments") if isinstance(timeline_summary, dict) else None
    lines: list[str] = [
        f"Target niche/topic: {' '.join((niche or '').split()) or 'N/A'}",
        f"Vibe: {vibe}",
        "",
        "Timeline summary (for labels/intent only):",
    ]
    if isinstance(segs, list):
        for s in segs[:12]:
            if not isinstance(s, dict):
                continue
            sid = s.get("segment_id")
            beat = str(s.get("beat_goal") or "")
            overlay = str(s.get("overlay_text") or "")
            tags = s.get("desired_tags") if isinstance(s.get("desired_tags"), list) else []
            hint = str(s.get("transition_hint") or "")
            sb = str(s.get("story_beat") or "")
            # Keep one-line summaries.
            line = f"S{int(sid):02d} beat={beat} overlay={overlay!r} tags={tags} hint={hint!r} story_beat={sb!r}"
            lines.append(line[:240])
    return "\n".join(lines).strip()


def critique_compare_video(
    *,
    api_key: str,
    model: str,
    compare_video_path: Path,
    timeline_summary: dict[str, t.Any],
    niche: str,
    vibe: str,
    critic_pro_mode: bool,
    max_mb: float,
    tmp_dir: Path,
    timeout_s: float = 240.0,
    site_url: str | None = None,
    app_name: str | None = None,
) -> CompareVideoCriticResult:
    if not compare_video_path.exists():
        raise FileNotFoundError(f"Compare video not found: {compare_video_path}")

    model_norm = normalize_model(str(model))
    system_prompt = _build_system_prompt(critic_pro_mode=bool(critic_pro_mode))
    user_text = _build_user_text(niche=niche, vibe=vibe, timeline_summary=timeline_summary)

    inline_path, video_meta = ensure_inlineable_video(compare_video_path, max_mb=float(max_mb), tmp_dir=tmp_dir, allow_proxy=True)
    url = encode_video_data_url(inline_path, max_mb=float(max_mb))

    content: list[dict[str, t.Any]] = [
        {"type": "text", "text": user_text},
        {"type": "video_url", "video_url": {"url": url}},
    ]

    messages: list[dict[str, t.Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    last_text = ""
    last_usage: dict[str, t.Any] | None = None
    last_model_used: str | None = None
    for attempt in range(1, 4):
        result = chat_completions(
            api_key=api_key,
            model=model_norm,
            messages=messages,
            temperature=0.0,
            max_tokens=1600,
            timeout_s=float(timeout_s),
            site_url=site_url,
            app_name=app_name,
            reasoning=_reasoning_param(),
            retries=2,
            retry_delay_s=1.5,
            extra_body={"response_format": {"type": "json_object"}},
        )
        last_text = (result.content or "").strip()
        last_usage = result.usage
        last_model_used = (str(result.raw.get("model")) if isinstance(result.raw, dict) and isinstance(result.raw.get("model"), str) else None)

        try:
            parsed = _extract_json_object(last_text)
            # Ensure schema-valid, stable structure.
            report = validate_critique_report(parsed, model=(last_model_used or model_norm))
            return CompareVideoCriticResult(
                report=report,
                usage=last_usage,
                model_requested=str(model_norm),
                model_used=last_model_used,
                raw_text=last_text,
                video_meta=video_meta,
            )
        except Exception as e:
            if attempt >= 3:
                snippet = (last_text or "").strip()[:800]
                raise OpenRouterError(f"Failed to parse/validate compare-video critic JSON: {type(e).__name__}: {e}. text={snippet!r}")
            # Ask for a strict JSON reprint that matches the schema exactly.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your last response was not valid schema JSON. Reprint ONLY valid JSON matching the schema exactly (no markdown, no extra prose). Keep strings short.",
                        }
                    ],
                },
            ]

    raise OpenRouterError("Compare-video critic failed unexpectedly")
