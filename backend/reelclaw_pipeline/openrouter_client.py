from __future__ import annotations

import dataclasses
import json
import time
import typing as t
import urllib.error
import urllib.request


class OpenRouterError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class OpenRouterChatResult:
    content: str
    raw: dict[str, t.Any]
    usage: dict[str, t.Any] | None
    images: list[str] | None = None


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
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(p for p in parts if p.strip()).strip()
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
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            response_json = json.loads(body)

            content = _extract_content(response_json)
            usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else None
            images = _extract_images(response_json)
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
