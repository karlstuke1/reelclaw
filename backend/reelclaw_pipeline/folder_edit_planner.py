from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import typing as t

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
        # Some models accidentally return an object with quotes/backslashes escaped
        # (e.g. {\"key\": \"value\"}). Try a conservative unescape pass.
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


def _reasoning_param() -> dict[str, t.Any]:
    # Gemini 3 models map reasoning.effort -> thinkingLevel in OpenRouter.
    effort_env = os.getenv("REASONING_EFFORT", "").strip().lower()
    if effort_env == "xhigh":
        effort_env = "high"
    if effort_env in {"none", "minimal", "low", "medium", "high"}:
        return {"effort": effort_env}
    # Default to high to improve planning quality; override via REASONING_EFFORT.
    return {"effort": "high"}


@dataclass(frozen=True)
class TaggedAsset:
    asset_id: str
    description: str
    tags: list[str]
    shot_type: str | None = None
    setting: str | None = None
    mood: str | None = None


def tag_assets_from_thumbnails(
    *,
    api_key: str,
    model: str,
    assets: list[dict[str, t.Any]],
    timeout_s: float = 120.0,
    site_url: str | None = None,
    app_name: str | None = None,
    batch_size: int = 6,
) -> dict[str, TaggedAsset]:
    """
    Caption/tag assets using ONLY their thumbnails (cheap + general).

    assets: list of dicts with keys: id, filename, kind, duration_s (optional), thumbnail_path (required)
    """
    tagged: dict[str, TaggedAsset] = {}
    if not assets:
        return tagged

    max_thumbs = int(max(1, min(5, float(os.getenv("TAGGER_MAX_THUMBS", "5")))))
    # If we send more images per asset, reduce batch size to avoid huge multi-image requests.
    # Keep a conservative budget (roughly <=18 images/call).
    img_budget = int(max(6, float(os.getenv("TAGGER_IMAGE_BUDGET", "18"))))
    eff_batch_size = max(1, min(int(batch_size), int(max(1, img_budget // max_thumbs))))

    system_prompt = "\\n".join(
        [
            "You are a media librarian helping pick clips for short-form edits.",
            "You will be given several thumbnails, each preceded by an ASSET header.",
            "Some assets include multiple thumbnails (start/mid/end) from the same clip.",
            "",
            "Rules:",
            "- Describe only what is visible. Don't guess identities or names.",
            "- Do NOT infer age, race, ethnicity, religion, politics, health, or other sensitive traits.",
            "- The description should be 1-2 sentences describing what happens across the thumbnails (action/progression) and the key visual motif.",
            "- Keep tags short, lowercase, and general (e.g., 'outdoor', 'selfie', 'city street', 'close-up', 'car interior').",
            "- Include tags for: setting, shot type, subject action, props, mood, camera angle, camera motion hints.",
            "- ALSO tag lighting/contrast and composition so an editor can match a reel's aesthetic:",
            "  - lighting: 'low-key lighting', 'high-key lighting', 'neon', 'backlit', 'silhouette', 'golden hour', 'night'",
            "  - contrast/color: 'high contrast', 'muted colors', 'warm', 'cool', 'monochrome'",
            "  - composition: 'minimalist', 'negative space', 'centered subject', 'wide shot', 'close-up'",
            "- If thumbnails vary, tag what is CONSISTENT across the clip and include a tag like 'varied scene' if needed.",
            "",
            "Return ONLY strict JSON with this shape:",
            "{",
            '  \"assets\": {',
            '    \"<asset_id>\": {',
            '      \"description\": string,',
            '      \"tags\": [string, ...],',
            '      \"shot_type\": string,',
            '      \"setting\": string,',
            '      \"mood\": string',
            "    }",
            "  }",
            "}",
        ]
    ).strip()

    for i in range(0, len(assets), max(1, eff_batch_size)):
        batch = assets[i : i + eff_batch_size]
        content: list[dict[str, t.Any]] = [
            {
                "type": "text",
                "text": "Tag each asset based on its thumbnails. Output JSON only.",
            }
        ]
        for a in batch:
            aid = str(a.get("id") or "")
            thumbs = a.get("thumbnail_paths")
            if isinstance(thumbs, list):
                thumbs = [str(x) for x in thumbs if str(x)]
            else:
                thumbs = []

            if not thumbs:
                single = a.get("thumbnail_path")
                if single:
                    thumbs = [str(single)]

            if not aid or not thumbs:
                continue
            filename = str(a.get("filename") or Path(str(a.get("path") or "")).name)
            kind = str(a.get("kind") or "")
            dur = a.get("duration_s")
            header = f"ASSET {aid} | {kind} | {filename} | thumbs={len(thumbs)}"
            if isinstance(dur, (int, float)) and dur:
                header += f" | duration_s={float(dur):.2f}"
            content.append({"type": "text", "text": header})
            show = thumbs[: max_thumbs]
            for ti, tpath in enumerate(show, start=1):
                content.append({"type": "text", "text": f"THUMBNAIL {ti}/{len(show)}"})
                content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(Path(tpath))}})

        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=1800,
            timeout_s=timeout_s,
            site_url=site_url,
            app_name=app_name,
            reasoning=_reasoning_param(),
            retries=3,
            retry_delay_s=2.0,
        )

        try:
            data = _extract_json_object(result.content)
        except Exception as e:
            # The most common failure mode here is truncated JSON (token limits) for larger batches.
            # Retry by shrinking the batch (meta + deterministic) instead of hard-failing the pipeline.
            if len(batch) > 1 and eff_batch_size > 1:
                sub = max(1, int(eff_batch_size) // 2)
                if sub >= int(eff_batch_size):
                    sub = 1
                sub_tags = tag_assets_from_thumbnails(
                    api_key=api_key,
                    model=model,
                    assets=batch,
                    timeout_s=timeout_s,
                    site_url=site_url,
                    app_name=app_name,
                    batch_size=sub,
                )
                tagged.update(sub_tags)
                continue

            snippet = (result.content or "").strip()[:800]
            raise OpenRouterError(f"Failed to parse asset tagger JSON: {type(e).__name__}: {e}. text={snippet!r}")

        assets_map = data.get("assets") or {}
        if not isinstance(assets_map, dict):
            continue
        for aid, v in assets_map.items():
            if not isinstance(v, dict):
                continue
            desc = str(v.get("description") or "").strip()
            tags = v.get("tags") or []
            if not desc or not isinstance(tags, list):
                continue
            cleaned_tags: list[str] = []
            for tag in tags:
                s = " ".join(str(tag).strip().lower().split())
                if s and s not in cleaned_tags:
                    cleaned_tags.append(s)
            tagged[str(aid)] = TaggedAsset(
                asset_id=str(aid),
                description=desc,
                tags=cleaned_tags[:18],
                shot_type=(str(v.get("shot_type") or "").strip() or None),
                setting=(str(v.get("setting") or "").strip() or None),
                mood=(str(v.get("mood") or "").strip() or None),
            )

    return tagged


@dataclass(frozen=True)
class ReferenceSegmentPlan:
    id: int
    start_s: float
    end_s: float
    duration_s: float
    beat_goal: str
    overlay_text: str
    reference_visual: str
    desired_tags: list[str]
    ref_luma: float | None = None
    ref_dark_frac: float | None = None
    # Optional extra matching signals (computed from reference midpoint frame + music analysis).
    ref_rgb_mean: list[float] | None = None  # [r,g,b] in 0..1
    music_energy: float | None = None  # 0..1 (rough), derived from beat strengths
    start_beat: int | None = None
    end_beat: int | None = None
    # Optional story planning constraints (generated by our story planner).
    story_beat: str | None = None
    preferred_sequence_group_ids: list[str] | None = None
    transition_hint: str | None = None


@dataclass(frozen=True)
class ReferenceAnalysisPlan:
    analysis: dict[str, t.Any]
    segments: list[ReferenceSegmentPlan]
    raw: dict[str, t.Any]


def analyze_reference_reel_segments(
    *,
    api_key: str,
    model: str,
    segment_frames: list[tuple[int, float, float, t.Any]],
    reference_image_data_url: str | None,
    timeout_s: float = 180.0,
    site_url: str | None = None,
    app_name: str | None = None,
) -> ReferenceAnalysisPlan:
    """
    Analyze a reel from multiple per-segment frames (e.g., start/mid/end).
    Returns analysis + per-segment caption/intent/tag targets.
    """
    if not segment_frames:
        raise ValueError("segment_frames is empty")

    system_prompt = "\\n".join(
        [
            "You are a short-form video EDIT ANALYST.",
            "You receive a sequence of frames sampled from a reference reel across each edit segment (not just a single midpoint).",
            "You also receive the segment start/end times, so you can infer motion/transition intent and pacing.",
            "",
            "Task:",
            "1) Explain what the reel is about and why it works (hook, escalation, payoff, caption style, pacing).",
            "2) For each segment, propose the overlay_text we should burn-in (max 2 lines) AND what the viewer should see in that segment.",
            "3) Provide desired_tags for retrieval from a user's media library (short, generic, lowercase).",
            "   desired_tags MUST include at least one tag for each:",
            "   - setting (e.g., 'car interior', 'rooftop bar', 'street at night')",
            "   - shot type (e.g., 'wide shot', 'close-up', 'over-the-shoulder')",
            "   - lighting/contrast (e.g., 'low-key lighting', 'night', 'high contrast', 'silhouette')",
            "",
            "Rules:",
            "- Do NOT guess identities or names from people in frames.",
            "- Do NOT infer sensitive attributes (age, race, etc.).",
            "- overlay_text MUST be max 2 lines; use a single \\n for the line break.",
            "- Keep overlay_text short: aim <= 26 chars per line.",
            "",
            "Return ONLY strict JSON:",
            "{",
            '  \"analysis\": {',
            '    \"summary\": string,',
            '    \"what_is_happening\": string,',
            '    \"why_it_works\": string,',
            '    \"caption_style\": string,',
            '    \"pacing_notes\": string',
            "  },",
            '  \"segments\": [',
            "    {",
            '      \"id\": 1,',
            '      \"beat_goal\": \"hook|setup|escalation|twist|payoff|cta\",',
            '      \"overlay_text\": \"LINE1\\nLINE2\",',
            '      \"reference_visual\": string,',
            '      \"desired_tags\": [string, ...]',
            "    }",
            "  ]",
            "}",
        ]
    ).strip()

    user_text = "Frames are in chronological order. Use them to infer what each segment is communicating. Output JSON only."

    content: list[dict[str, t.Any]] = [{"type": "text", "text": user_text}]
    if reference_image_data_url:
        content.append({"type": "image_url", "image_url": {"url": reference_image_data_url}})

    for seg_id, start_s, end_s, frame_path in segment_frames:
        header = f"SEGMENT {seg_id} | start_s={start_s:.2f} end_s={end_s:.2f} duration_s={(end_s-start_s):.2f}"
        content.append({"type": "text", "text": header})
        # segment_frames may provide a single Path (legacy) or a list of (time_s, Path) samples.
        if isinstance(frame_path, list):
            frames_list = frame_path
            for j, item in enumerate(frames_list, start=1):
                try:
                    t_s, p = item
                except Exception:
                    continue
                content.append({"type": "text", "text": f"FRAME {j}/{len(frames_list)} at t={float(t_s):.2f}s"})
                content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(Path(p))}})
        else:
            content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(Path(frame_path))}})

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    last_error: Exception | None = None
    last_text = ""
    for attempt in range(1, 4):
        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.4 if attempt == 1 else 0.2,
            max_tokens=2400,
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
                snippet = last_text.strip()[:800]
                raise OpenRouterError(f"Failed to parse reference analysis JSON: {type(e).__name__}: {e}. text={snippet!r}")
            # Ask for a strict JSON reprint.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your last response was not strict JSON. Reprint ONLY valid JSON (no markdown, no extra text, no escaped quotes like \\\\\"key\\\\\").",
                        }
                    ],
                },
            ]

    analysis = data.get("analysis")
    segments = data.get("segments")
    if not isinstance(analysis, dict) or not isinstance(segments, list):
        raise OpenRouterError("Reference analysis output missing required keys: analysis, segments")

    # Map segment timing from the input list to ensure correctness.
    timing: dict[int, tuple[float, float]] = {sid: (s, e) for sid, s, e, _ in segment_frames}
    planned: list[ReferenceSegmentPlan] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        sid = int(item.get("id") or 0)
        if sid not in timing:
            continue
        start_s, end_s = timing[sid]
        desired_tags = item.get("desired_tags") or []
        tags_clean: list[str] = []
        if isinstance(desired_tags, list):
            for tag in desired_tags:
                s = " ".join(str(tag).strip().lower().split())
                if s and s not in tags_clean:
                    tags_clean.append(s)
        planned.append(
            ReferenceSegmentPlan(
                id=sid,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
                beat_goal=str(item.get("beat_goal") or "setup"),
                overlay_text=str(item.get("overlay_text") or "").strip(),
                reference_visual=str(item.get("reference_visual") or "").strip(),
                desired_tags=tags_clean[:16],
                ref_luma=None,
                ref_dark_frac=None,
            )
        )

    planned = sorted(planned, key=lambda s: s.id)
    return ReferenceAnalysisPlan(analysis=analysis, segments=planned, raw=data)


@dataclass(frozen=True)
class EditDecision:
    segment_id: int
    asset_id: str
    in_s: float
    duration_s: float
    speed: float = 1.0
    crop_mode: str = "center"  # center|top|bottom|face
    notes: str | None = None


@dataclass(frozen=True)
class FolderEditPlan:
    analysis: dict[str, t.Any]
    decisions: list[EditDecision]
    raw: dict[str, t.Any]


@dataclass(frozen=True)
class InpointRefinement:
    chosen_time_s: float
    crop_mode: str
    speed: float
    reason: str | None = None


def refine_inpoint_for_segment(
    *,
    api_key: str,
    model: str,
    segment: ReferenceSegmentPlan,
    reference_frame_path: Path,
    asset_meta: dict[str, t.Any],
    candidate_frames: list[tuple[float, Path, float | None, float | None, float | None]],
    timeout_s: float = 120.0,
    site_url: str | None = None,
    app_name: str | None = None,
) -> InpointRefinement:
    """
    Use the model to pick the best in-point among candidate frames for a given segment.

    This keeps prompts meta: it's selecting between concrete options, not inventing story-specific text.
    """
    if not candidate_frames:
        raise ValueError("candidate_frames is empty")

    system_prompt = "\n".join(
        [
            "You are a short-form video EDITOR selecting the best cut for a segment.",
            "",
            "You will be shown:",
            "- A REFERENCE segment frame from the viral reel.",
            "- Several CANDIDATE frames from ONE local asset at different timestamps.",
            "",
            "Task:",
            "- Choose the best candidate timestamp to match the reference segment's",
            "  (1) shot type, (2) lighting/contrast, (3) composition/negative space, (4) mood/energy.",
            "",
            "Rules:",
            "- You MUST choose one of the provided candidate_time_s values exactly.",
            "- Prefer aesthetic match over literal subject match.",
            "- Use luma + dark_frac numbers as sanity checks:",
            "  - lower ref_luma -> prefer lower luma candidate",
            "  - higher ref_dark (closer to 1.0) -> prefer candidates with more negative space / darker frame",
            "- crop_mode must be one of: center, top, bottom, face.",
            "- speed should usually be 1.0; only adjust if necessary to match energy (0.75–1.5).",
            "- Keep reason <= 200 characters.",
            "",
            "Return ONLY strict JSON:",
            "{",
            '  \"candidate_time_s\": 0.0,',
            '  \"crop_mode\": \"center\",',
            '  \"speed\": 1.0,',
            '  \"reason\": \"short\"',
            "}",
        ]
    ).strip()

    # Keep the text input compact to reduce truncation risk.
    asset_tags = asset_meta.get("tags") or []
    if not isinstance(asset_tags, list):
        asset_tags = []

    user_text = "\n".join(
        [
            f"SEGMENT id={segment.id} beat={segment.beat_goal} ref_luma={segment.ref_luma} ref_dark={segment.ref_dark_frac}",
            f"desired_tags={segment.desired_tags}",
            f"overlay_text={segment.overlay_text!r}",
            "",
            f"ASSET id={asset_meta.get('id')} kind={asset_meta.get('kind')} duration_s={asset_meta.get('duration_s')} luma={asset_meta.get('luma_mean')} dark={asset_meta.get('dark_frac')} motion={asset_meta.get('motion_score')}",
            f"asset_tags={asset_tags[:14]}",
            (f"asset_desc={str(asset_meta.get('description') or '')[:160]}" if asset_meta.get("description") else ""),
        ]
    ).strip()

    content: list[dict[str, t.Any]] = [{"type": "text", "text": user_text}]
    content.append({"type": "text", "text": "REFERENCE FRAME"})
    content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(reference_frame_path)}})

    # Candidates.
    times = [t for t, _p, _l, _d, _m in candidate_frames]
    packed = [(round(float(t), 3), l, d, m) for t, _p, l, d, m in candidate_frames]
    content.append({"type": "text", "text": f"candidate options: {packed}"})
    for t_s, p, luma, dark, motion in candidate_frames:
        content.append(
            {
                "type": "text",
                "text": f"CANDIDATE time_s={t_s:.3f} luma={luma} dark={dark} motion={motion}",
            }
        )
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(p)}})

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    last_text = ""
    for attempt in range(1, 4):
        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=240,
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
            if attempt >= 3:
                snippet = last_text.strip()[:800]
                raise OpenRouterError(f"Failed to parse inpoint refinement JSON: {type(e).__name__}: {e}. text={snippet!r}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Reprint ONLY valid JSON. candidate_time_s must exactly match one of the provided options.",
                        }
                    ],
                },
            ]

    chosen = data.get("candidate_time_s")
    try:
        chosen_f = float(chosen)
    except Exception:
        chosen_f = float(candidate_frames[0][0])

    # Snap to nearest provided time if the model slightly deviates.
    best_t = min(times, key=lambda x: abs(float(x) - chosen_f))

    crop_mode = str(data.get("crop_mode") or "center").strip().lower()
    if crop_mode not in {"center", "top", "bottom", "face"}:
        crop_mode = "center"
    try:
        speed = float(data.get("speed") or 1.0)
    except Exception:
        speed = 1.0
    speed = min(1.5, max(0.75, speed))

    reason = str(data.get("reason") or "").strip() or None
    return InpointRefinement(chosen_time_s=float(best_t), crop_mode=crop_mode, speed=speed, reason=reason)


def plan_folder_edit_edl(
    *,
    api_key: str,
    model: str,
    segments: list[ReferenceSegmentPlan],
    assets: list[dict[str, t.Any]],
    reference_image_data_url: str | None,
    timeout_s: float = 180.0,
    site_url: str | None = None,
    app_name: str | None = None,
    extra_guidance: str | None = None,
) -> FolderEditPlan:
    """
    Create an edit decision list (EDL) that maps each reference segment to a local asset.

    assets must include: id, kind, duration_s (optional), tags (optional), description (optional)
    """
    if not segments:
        raise ValueError("segments is empty")
    if not assets:
        raise ValueError("assets is empty")

    system_prompt = "\\n".join(
        [
            "You are a short-form EDITOR who recreates viral reels using a user's existing footage library.",
            "",
            "You will be given:",
            "- A list of reference reel segments with exact timings and desired_tags.",
            "- A library of available local assets (videos/images) with tags/descriptions.",
            "",
            "Task:",
            "For EACH segment, choose ONE best asset_id from the library and specify an in-point (in_s) and duration_s.",
            "The goal is to match the reference pacing and beat structure as closely as possible.",
            "",
            "Rules:",
            "- Be concise: analysis strings must be <= 200 characters; per-segment notes must be <= 120 characters.",
            "- Use ONLY asset_id values that exist in the provided library list.",
            "- Keep a single main character/persona consistent. If the library includes that character, prefer those shots.",
            "- Prefer clips that naturally match desired_tags. If no perfect match exists, pick the closest and explain in notes.",
            "- Aesthetic matching is critical: if the reference segment is 'night/low-key/high contrast/silhouette', do NOT pick bright daylight footage.",
            "- Match shot type first (wide vs close-up), then lighting/contrast, then setting details.",
            "- Use the provided luma + dark_frac numbers as sanity checks:",
            "  - if ref_luma is low (dark), prefer assets with low luma",
            "  - if ref_dark is high (closer to 1.0), prefer assets with lots of negative space / mostly-dark frames",
            "- Match ENERGY: shorter segments generally feel higher-energy; prefer higher motion_score for short segments and lower motion_score for longer segments (use sparingly).",
            "- Don't invent new story content; preserve the reference structure.",
            "- crop_mode must be one of: center, top, bottom, face.",
            "- speed can be 0.75 to 1.5 when needed (use sparingly).",
            "",
            "Output ONLY strict JSON:",
            "{",
            '  \"analysis\": {',
            '    \"overall_edit_strategy\": string,',
            '    \"risk_notes\": string',
            "  },",
            '  \"edl\": [',
            "    {",
            '      \"segment_id\": 1,',
            '      \"asset_id\": \"<id>\",',
            '      \"in_s\": 0.0,',
            '      \"duration_s\": 2.0,',
            '      \"speed\": 1.0,',
            '      \"crop_mode\": \"center\",',
            '      \"notes\": string',
            "    }",
            "  ]",
            "}",
        ]
    ).strip()

    user_lines: list[str] = []
    if extra_guidance:
        user_lines.append(f"Extra guidance: {extra_guidance.strip()}")

    content: list[dict[str, t.Any]] = [{"type": "text", "text": "\n".join(user_lines).strip() or "Output JSON only."}]
    if reference_image_data_url:
        content.append({"type": "image_url", "image_url": {"url": reference_image_data_url}})

    # Provide the reference segments.
    seg_text_lines: list[str] = ["REFERENCE SEGMENTS:"]
    for seg in segments:
        seg_text_lines.append(
            f"- id={seg.id} start={seg.start_s:.2f} end={seg.end_s:.2f} dur={seg.duration_s:.2f} beat={seg.beat_goal} ref_luma={seg.ref_luma} ref_dark={seg.ref_dark_frac} tags={seg.desired_tags} overlay={seg.overlay_text!r} visual={seg.reference_visual!r}"
        )
    content.append({"type": "text", "text": "\n".join(seg_text_lines)})

    # Provide the asset library in text (avoid sending all thumbnails).
    lib_lines: list[str] = ["ASSET LIBRARY:"]
    for a in assets:
        aid = str(a.get("id") or "")
        if not aid:
            continue
        kind = str(a.get("kind") or "")
        dur = a.get("duration_s")
        desc = str(a.get("description") or "").strip()
        tags = a.get("tags") or []
        luma = a.get("luma_mean")
        dark = a.get("dark_frac")
        motion = a.get("motion_score")
        tag_str = ", ".join([str(x) for x in tags[:10]]) if isinstance(tags, list) else ""
        dur_str = f"{float(dur):.2f}s" if isinstance(dur, (int, float)) and dur else "?"
        luma_str = f"{float(luma):.3f}" if isinstance(luma, (int, float)) else "?"
        dark_str = f"{float(dark):.3f}" if isinstance(dark, (int, float)) else "?"
        motion_str = f"{float(motion):.3f}" if isinstance(motion, (int, float)) else "?"
        lib_lines.append(f"- {aid} | {kind} | dur={dur_str} | luma={luma_str} dark={dark_str} motion={motion_str} | tags=[{tag_str}] | {desc[:120]}")
    content.append({"type": "text", "text": "\n".join(lib_lines)})

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    last_text = ""
    for attempt in range(1, 4):
        result = chat_completions(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.1 if attempt == 1 else 0.0,
            max_tokens=4096,
            timeout_s=timeout_s,
            site_url=site_url,
            app_name=app_name,
            reasoning=_reasoning_param(),
            extra_body={"response_format": {"type": "json_object"}},
            retries=3,
            retry_delay_s=2.0,
        )
        last_text = result.content or ""
        try:
            data = _extract_json_object(last_text)
            break
        except Exception as e:
            if attempt >= 3:
                snippet = last_text.strip()[:800]
                raise OpenRouterError(f"Failed to parse folder edit plan JSON: {type(e).__name__}: {e}. text={snippet!r}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your last response was not strict JSON. Reprint ONLY valid JSON (no markdown, no trailing commas, no extra commentary).",
                        }
                    ],
                },
            ]

    analysis = data.get("analysis") or {}
    edl = data.get("edl") or []
    if not isinstance(analysis, dict) or not isinstance(edl, list):
        raise OpenRouterError("Folder edit planner output missing required keys: analysis, edl")

    # Validate + coerce.
    asset_ids = {str(a.get("id") or "") for a in assets}
    decisions: list[EditDecision] = []
    seg_by_id = {s.id: s for s in segments}
    for item in edl:
        if not isinstance(item, dict):
            continue
        sid = int(item.get("segment_id") or 0)
        if sid not in seg_by_id:
            continue
        aid = str(item.get("asset_id") or "")
        if aid not in asset_ids:
            continue
        seg = seg_by_id[sid]
        try:
            in_s = float(item.get("in_s") or 0.0)
        except Exception:
            in_s = 0.0
        duration_s = seg.duration_s
        try:
            # Allow model to shorten slightly, but never lengthen beyond segment.
            req = float(item.get("duration_s") or duration_s)
            if 0.2 < req < duration_s:
                duration_s = req
        except Exception:
            pass
        try:
            speed = float(item.get("speed") or 1.0)
        except Exception:
            speed = 1.0
        speed = min(1.5, max(0.75, speed))
        crop_mode = str(item.get("crop_mode") or "center").strip().lower()
        if crop_mode not in {"center", "top", "bottom", "face"}:
            crop_mode = "center"
        notes = str(item.get("notes") or "").strip() or None
        decisions.append(
            EditDecision(
                segment_id=sid,
                asset_id=aid,
                in_s=max(0.0, in_s),
                duration_s=max(0.2, float(duration_s)),
                speed=speed,
                crop_mode=crop_mode,
                notes=notes,
            )
        )

    decisions = sorted(decisions, key=lambda d: d.segment_id)
    return FolderEditPlan(analysis=analysis, decisions=decisions, raw=data)
