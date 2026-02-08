from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import typing as t
import urllib.error
import urllib.request
from pathlib import Path


class OpenRouterError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class OpenRouterChatResult:
    content: str
    raw: dict[str, t.Any]
    usage: dict[str, t.Any] | None
    images: list[str] | None = None


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, "").strip()
    return v or default


def _usage_log_path() -> Path | None:
    raw = _env("REELCLAW_OPENROUTER_USAGE_PATH") or _env("OPENROUTER_USAGE_PATH")
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except Exception:
        return None


def _count_image_parts(messages: list[dict[str, t.Any]]) -> int:
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                n += 1
    return n


def _extract_cost_usd(headers: dict[str, str] | None) -> float | None:
    if not headers:
        return None
    h = {str(k).lower(): str(v) for k, v in headers.items() if k}
    for key in (
        "x-openrouter-cost",
        "x-openrouter-cost-usd",
        "x-openrouter-usage-cost",
        "x-openrouter-credits-used",
    ):
        raw = (h.get(key) or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return None


def _append_jsonl(path: Path, obj: dict[str, t.Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
    except Exception:
        # Best-effort: never crash the pipeline for telemetry.
        return


def _maybe_record_usage(
    *,
    model: str,
    messages: list[dict[str, t.Any]],
    usage: dict[str, t.Any] | None,
    headers: dict[str, str] | None,
    latency_ms: int | None,
) -> None:
    log_path = _usage_log_path()
    if not log_path:
        return
    rec: dict[str, t.Any] = {
        "ts_ms": int(time.time() * 1000),
        "model": str(model or "").strip(),
        "image_parts": int(_count_image_parts(messages)),
        "usage": usage,
    }
    if latency_ms is not None:
        rec["latency_ms"] = int(latency_ms)
    cost = _extract_cost_usd(headers)
    if cost is not None:
        rec["cost_usd"] = float(cost)
    _append_jsonl(log_path, rec)


def normalize_model(model: str) -> str:
    """
    Normalize a model id string (trim). Model routing policy lives at call sites.
    """
    return str(model or "").strip()


def _extract_message(response_json: dict[str, t.Any]) -> dict[str, t.Any]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError(f"Unexpected response: missing choices (keys={list(response_json.keys())})")

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise OpenRouterError("Unexpected response: choices[0] is not an object")

    message = choice0.get("message")
    if isinstance(message, dict):
        return message

    delta = choice0.get("delta")
    if isinstance(delta, dict):
        return delta

    raise OpenRouterError("Unexpected response: missing message")


def _extract_content(response_json: dict[str, t.Any]) -> str:
    message = _extract_message(response_json)
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Some providers return response_format JSON as an object instead of a string.
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=True)
        except Exception:
            return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            # Providers vary: sometimes type is "text", sometimes "output_text", etc.
            if isinstance(item.get("text"), str) and item.get("text"):
                parts.append(t.cast(str, item["text"]))
                continue
            # Some providers embed JSON parts directly.
            if item.get("type") == "json" and isinstance(item.get("json"), dict):
                try:
                    parts.append(json.dumps(item["json"], ensure_ascii=True))
                except Exception:
                    parts.append(str(item["json"]))
        return "\n".join(p for p in parts if p.strip()).strip()
    # Some OpenRouter-compatible responses use a separate refusal field.
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()
    return ""


def _extract_images(response_json: dict[str, t.Any]) -> list[str] | None:
    message = _extract_message(response_json)
    images = message.get("images")
    if not isinstance(images, list) or not images:
        images = None
    urls: list[str] = []
    if isinstance(images, list):
        for item in images:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)

    # Some OpenAI-compatible responses put images in message.content as parts.
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls or None


def chat_completions(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, t.Any]],
    temperature: float = 0.4,
    max_tokens: int | None = None,
    timeout_s: float = 90.0,
    site_url: str | None = None,
    app_name: str | None = None,
    modalities: list[str] | None = None,
    reasoning: dict[str, t.Any] | None = None,
    include_reasoning: bool | None = None,
    extra_body: dict[str, t.Any] | None = None,
    retries: int = 3,
    retry_delay_s: float = 2.0,
) -> OpenRouterChatResult:
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = normalize_model(model)

    payload: dict[str, t.Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if modalities:
        payload["modalities"] = modalities
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if include_reasoning is not None:
        payload["include_reasoning"] = include_reasoning
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            resp_headers: dict[str, str] | None = None
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    resp_headers = {str(k): str(v) for k, v in resp.headers.items()}
                except Exception:
                    resp_headers = None
            latency_ms = int(round((time.time() - t0) * 1000.0))
            response_json = json.loads(body)

            content = _extract_content(response_json)
            usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else None
            images = _extract_images(response_json)
            _maybe_record_usage(model=model, messages=messages, usage=usage, headers=resp_headers, latency_ms=latency_ms)
            return OpenRouterChatResult(content=content, raw=response_json, usage=usage, images=images)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            last_error = OpenRouterError(f"HTTP {e.code}: {err_body or e.reason}")
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
        except (json.JSONDecodeError, OpenRouterError) as e:
            last_error = e

        if attempt < retries:
            time.sleep(retry_delay_s * attempt)

    raise OpenRouterError(str(last_error) if last_error else "Unknown OpenRouter error")


_AFFORD_RE = re.compile(r"requested up to\\s+(\\d+)\\s+tokens,\\s+but can only afford\\s+(\\d+)", re.IGNORECASE)


def _parse_affordable_max_tokens(err_text: str) -> int | None:
    """
    Best-effort parse for OpenRouter 402 "can only afford N tokens" errors.
    Returns the affordable max_tokens value when present.
    """
    s = str(err_text or "")
    m = _AFFORD_RE.search(s)
    if not m:
        return None
    try:
        afford = int(m.group(2))
    except Exception:
        return None
    if afford <= 0:
        return None
    return afford


def chat_completions_budgeted(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, t.Any]],
    temperature: float = 0.4,
    max_tokens: int | None = None,
    timeout_s: float = 90.0,
    site_url: str | None = None,
    app_name: str | None = None,
    modalities: list[str] | None = None,
    reasoning: dict[str, t.Any] | None = None,
    include_reasoning: bool | None = None,
    extra_body: dict[str, t.Any] | None = None,
    retries: int = 3,
    retry_delay_s: float = 2.0,
    min_tokens: int = 256,
) -> OpenRouterChatResult:
    """
    Wrapper around chat_completions that auto-retries on 402 "can only afford N tokens" errors
    by reducing max_tokens. This prevents hard-failing pro pipelines when the account's credit
    is low, while keeping deterministic temperature=0 behavior intact.
    """
    mt = int(max_tokens) if max_tokens is not None else None
    for attempt in range(1, 4):
        try:
            return chat_completions(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=mt,
                timeout_s=timeout_s,
                site_url=site_url,
                app_name=app_name,
                modalities=modalities,
                reasoning=reasoning,
                include_reasoning=include_reasoning,
                extra_body=extra_body,
                retries=retries,
                retry_delay_s=retry_delay_s,
            )
        except OpenRouterError as e:
            msg = str(e)
            if "HTTP 402" not in msg or mt is None:
                raise
            afford = _parse_affordable_max_tokens(msg)
            if afford is not None:
                mt2 = int(max(min_tokens, int(float(afford) * 0.88)))
            else:
                mt2 = int(max(min_tokens, int(float(mt) * 0.85)))
            if mt2 >= int(mt):
                # Can't reduce further in a meaningful way.
                raise
            mt = mt2
            if attempt >= 3:
                raise
            time.sleep(0.4 * attempt)

    raise OpenRouterError("chat_completions_budgeted failed unexpectedly")
