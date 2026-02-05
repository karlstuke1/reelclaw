from __future__ import annotations

import base64
from pathlib import Path


def _detect_mime_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    if ext in {"heic", "heif"}:
        return "image/heic"
    return "application/octet-stream"


def encode_image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = _detect_mime_type(path)
    return f"data:{mime};base64,{b64}"

