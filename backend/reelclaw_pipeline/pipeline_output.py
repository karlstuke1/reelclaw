from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
import typing as t


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images_dir: Path
    timeline_path: Path
    video_path: Path


def utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def sanitize_filename(value: str) -> str:
    trimmed = (value or "").strip()
    if not trimmed:
        return "item"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", trimmed)
    safe = safe.strip("._-") or "item"
    return safe[:120]


def ensure_output_dirs(base_dir: Path) -> OutputPaths:
    stamp = utc_timestamp()
    root = base_dir / stamp
    images_dir = root / "generated_images"
    timeline_path = root / "timeline.json"
    video_path = root / "final_video.mov"
    images_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, images_dir=images_dir, timeline_path=timeline_path, video_path=video_path)


def write_json(path: Path, data: dict[str, t.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

