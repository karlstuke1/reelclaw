from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import typing as t
import hashlib


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def _run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def _ffprobe_json(path: Path, *, timeout_s: float) -> dict[str, t.Any]:
    ffprobe = os.getenv("FFPROBE", "ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout_s=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "ffprobe failed")
    return json.loads(result.stdout or "{}")


def _best_stream(streams: list[dict[str, t.Any]], codec_type: str) -> dict[str, t.Any] | None:
    best: dict[str, t.Any] | None = None
    for s in streams:
        if s.get("codec_type") != codec_type:
            continue
        if best is None:
            best = s
            continue
        # Prefer the highest resolution video stream if multiple exist.
        if codec_type == "video":
            try:
                bw = int(best.get("width") or 0) * int(best.get("height") or 0)
                sw = int(s.get("width") or 0) * int(s.get("height") or 0)
                if sw > bw:
                    best = s
            except Exception:
                pass
    return best


def _sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def folder_cache_key(folder: Path) -> str:
    return _sha1_id(str(folder.expanduser().resolve()))


def load_or_build_cached_index(
    *,
    folder: Path,
    cache_root: Path,
    timeout_s: float = 60.0,
    progress_cb: t.Callable[[int, int, str], None] | None = None,
    refresh: bool = False,
) -> tuple[MediaIndex, Path]:
    """
    Global cache for MediaIndex (ffprobe + thumbnails) keyed by the resolved folder path.

    This avoids re-running ffprobe/ffmpeg thumbnail extraction per project when you run
    many reference reels against the same footage library.
    """
    cache_root = cache_root.expanduser().resolve()
    cache_dir = cache_root / folder_cache_key(folder)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "media_index.json"

    if (not refresh) and index_path.exists():
        try:
            existing = load_index(index_path)
            updated, changed = update_cached_index(
                existing=existing,
                folder=folder,
                output_dir=cache_dir,
                timeout_s=timeout_s,
                progress_cb=progress_cb,
            )
            if changed:
                save_index(updated, index_path)
            return updated, index_path
        except Exception:
            # Fall through and rebuild.
            pass

    index = index_media_folder(folder=folder, output_dir=cache_dir, timeout_s=timeout_s, progress_cb=progress_cb)
    save_index(index, index_path)
    return index, index_path


def update_cached_index(
    *,
    existing: MediaIndex,
    folder: Path,
    output_dir: Path,
    timeout_s: float = 60.0,
    progress_cb: t.Callable[[int, int, str], None] | None = None,
) -> tuple[MediaIndex, bool]:
    """
    Incrementally update a cached MediaIndex for `folder`.

    This is much faster than a full rebuild when only a few files were added/changed.
    Returns (updated_index, changed).
    """
    folder = folder.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    thumbs_dir = output_dir / "asset_thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Fast lookup by asset id.
    by_id: dict[str, MediaAsset] = {a.id: a for a in (existing.assets or []) if getattr(a, "id", None)}

    files = sorted([p for p in folder.iterdir() if p.is_file()])
    media_files = [p for p in files if p.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS)]
    total = len(media_files)

    new_assets: list[MediaAsset] = []
    changed = False

    def _thumbs_exist(asset: MediaAsset) -> bool:
        if asset.thumbnail_path:
            try:
                if not Path(asset.thumbnail_path).exists():
                    return False
            except Exception:
                return False
        if getattr(asset, "thumbnail_paths", None):
            try:
                for raw in (asset.thumbnail_paths or [])[:3]:
                    if raw and (not Path(str(raw)).exists()):
                        return False
            except Exception:
                return False
        return True

    for idx, path in enumerate(media_files, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"Indexing {path.name}")
        st = path.stat()
        ext = path.suffix.lower()
        kind = "video" if ext in VIDEO_EXTS else "image"
        asset_id = _sha1_id(str(path))

        prev = by_id.get(asset_id)
        if prev is not None:
            try:
                same_size = int(prev.size_bytes) == int(st.st_size)
            except Exception:
                same_size = False
            try:
                same_mtime = abs(float(prev.mtime) - float(st.st_mtime)) <= 1e-6
            except Exception:
                same_mtime = False
            if same_size and same_mtime and prev.kind == kind and _thumbs_exist(prev):
                new_assets.append(prev)
                continue

        # New or changed -> re-index this file.
        changed = True
        duration_s: float | None = None
        width: int | None = None
        height: int | None = None
        has_audio: bool | None = None
        thumb_path: Path | None = None
        thumb_paths: list[Path] = []

        try:
            info = _ffprobe_json(path, timeout_s=timeout_s)
            streams = t.cast(list[dict[str, t.Any]], info.get("streams") or [])
            fmt = t.cast(dict[str, t.Any], info.get("format") or {})
            if kind == "video":
                v = _best_stream(streams, "video") or {}
                a = _best_stream(streams, "audio")
                try:
                    width = int(v.get("width") or 0) or None
                    height = int(v.get("height") or 0) or None
                except Exception:
                    width = height = None
                try:
                    duration_s = float(fmt.get("duration") or 0) or None
                except Exception:
                    duration_s = None
                has_audio = a is not None
            else:
                v = _best_stream(streams, "video") or {}
                try:
                    width = int(v.get("width") or 0) or None
                    height = int(v.get("height") or 0) or None
                except Exception:
                    width = height = None
        except Exception:
            # Keep going; the asset is still usable even if ffprobe fails.
            pass

        # Thumbnails (same logic as index_media_folder).
        try:
            if kind == "video" and duration_s and duration_s > 0.8:
                def _clamp_t(t_s: float) -> float:
                    return min(max(t_s, 0.1), max(duration_s - 0.1, 0.1))

                t_start = _clamp_t(duration_s * 0.15)
                t_mid = _clamp_t(duration_s * 0.50)
                t_end = _clamp_t(duration_s * 0.85)

                p_mid = thumbs_dir / f"{asset_id}.jpg"
                p_start = thumbs_dir / f"{asset_id}_start.jpg"
                p_end = thumbs_dir / f"{asset_id}_end.jpg"

                _extract_thumbnail(src=path, dst=p_start, kind=kind, at_s=t_start, timeout_s=max(30.0, timeout_s))
                _extract_thumbnail(src=path, dst=p_mid, kind=kind, at_s=t_mid, timeout_s=max(30.0, timeout_s))
                _extract_thumbnail(src=path, dst=p_end, kind=kind, at_s=t_end, timeout_s=max(30.0, timeout_s))
                thumb_paths = [p_start, p_mid, p_end]
                thumb_path = p_mid
            else:
                thumb_path = thumbs_dir / f"{asset_id}.jpg"
                _extract_thumbnail(src=path, dst=thumb_path, kind=kind, at_s=None, timeout_s=max(30.0, timeout_s))
                thumb_paths = [thumb_path]
        except Exception:
            thumb_path = None
            thumb_paths = []

        new_assets.append(
            MediaAsset(
                id=asset_id,
                path=str(path),
                kind=kind,
                ext=ext,
                size_bytes=int(st.st_size),
                mtime=float(st.st_mtime),
                duration_s=duration_s,
                width=width,
                height=height,
                has_audio=has_audio,
                thumbnail_path=str(thumb_path) if thumb_path else None,
                thumbnail_paths=[str(p) for p in thumb_paths if p],
            )
        )

    # If any old asset disappeared, mark changed.
    new_ids = {a.id for a in new_assets}
    old_ids = set(by_id.keys())
    if new_ids != old_ids:
        changed = True

    if progress_cb:
        progress_cb(total, total, "Index complete")

    from datetime import datetime, timezone

    updated = MediaIndex(
        source_folder=str(folder),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        assets=new_assets,
    )
    return updated, bool(changed)


def _extract_thumbnail(
    *,
    src: Path,
    dst: Path,
    kind: str,
    at_s: float | None,
    timeout_s: float,
) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate thumbnails. Please install ffmpeg and try again.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg, "-y"]
    if kind == "video" and at_s is not None:
        cmd.extend(["-ss", f"{at_s:.3f}"])
    cmd.extend(["-i", str(src), "-frames:v", "1", "-vf", "scale=480:-2:flags=lanczos", str(dst)])
    result = _run(cmd, timeout_s=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg thumbnail failed: {stderr or 'unknown error'}")


@dataclass(frozen=True)
class MediaAsset:
    id: str
    path: str
    kind: str  # "video" | "image"
    ext: str
    size_bytes: int
    mtime: float
    duration_s: float | None
    width: int | None
    height: int | None
    has_audio: bool | None
    thumbnail_path: str | None
    # Optional extra thumbnails (e.g. start/mid/end) for better tagging and selection.
    thumbnail_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MediaIndex:
    source_folder: str
    generated_at: str
    assets: list[MediaAsset]

    def to_json(self) -> dict[str, t.Any]:
        return {
            "source_folder": self.source_folder,
            "generated_at": self.generated_at,
            "assets": [asset.__dict__ for asset in self.assets],
        }


def index_media_folder(
    *,
    folder: Path,
    output_dir: Path,
    timeout_s: float = 60.0,
    progress_cb: t.Callable[[int, int, str], None] | None = None,
) -> MediaIndex:
    """
    Build a lightweight index of a folder of media (videos/images).

    This intentionally does NOT send media to an LLM. It produces enough metadata
    for downstream retrieval and for selectively showing candidates to a model.
    """
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Media folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    thumbs_dir = output_dir / "asset_thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in folder.iterdir() if p.is_file()])
    media_files = [p for p in files if p.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS)]
    total = len(media_files)

    assets: list[MediaAsset] = []
    for idx, path in enumerate(media_files, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"Indexing {path.name}")

        st = path.stat()
        ext = path.suffix.lower()
        kind = "video" if ext in VIDEO_EXTS else "image"
        asset_id = _sha1_id(str(path))

        duration_s: float | None = None
        width: int | None = None
        height: int | None = None
        has_audio: bool | None = None
        thumb_path: Path | None = None

        try:
            info = _ffprobe_json(path, timeout_s=timeout_s)
            streams = t.cast(list[dict[str, t.Any]], info.get("streams") or [])
            fmt = t.cast(dict[str, t.Any], info.get("format") or {})
            if kind == "video":
                v = _best_stream(streams, "video") or {}
                a = _best_stream(streams, "audio")
                try:
                    width = int(v.get("width") or 0) or None
                    height = int(v.get("height") or 0) or None
                except Exception:
                    width = height = None
                try:
                    duration_s = float(fmt.get("duration") or 0) or None
                except Exception:
                    duration_s = None
                has_audio = a is not None
            else:
                v = _best_stream(streams, "video") or {}
                try:
                    width = int(v.get("width") or 0) or None
                    height = int(v.get("height") or 0) or None
                except Exception:
                    width = height = None
        except Exception:
            # Keep going; the asset is still usable even if ffprobe fails.
            pass

        # Thumbnails:
        # - For video: extract start/mid/end to improve tagging + selection.
        # - For image: one scaled thumb.
        thumb_paths: list[Path] = []
        try:
            if kind == "video" and duration_s and duration_s > 0.8:
                # Clamp times away from exact ends to avoid black/invalid frames.
                def _clamp_t(t_s: float) -> float:
                    return min(max(t_s, 0.1), max(duration_s - 0.1, 0.1))

                t_start = _clamp_t(duration_s * 0.15)
                t_mid = _clamp_t(duration_s * 0.50)
                t_end = _clamp_t(duration_s * 0.85)

                p_mid = thumbs_dir / f"{asset_id}.jpg"  # keep stable primary thumb path
                p_start = thumbs_dir / f"{asset_id}_start.jpg"
                p_end = thumbs_dir / f"{asset_id}_end.jpg"

                _extract_thumbnail(src=path, dst=p_start, kind=kind, at_s=t_start, timeout_s=max(30.0, timeout_s))
                _extract_thumbnail(src=path, dst=p_mid, kind=kind, at_s=t_mid, timeout_s=max(30.0, timeout_s))
                _extract_thumbnail(src=path, dst=p_end, kind=kind, at_s=t_end, timeout_s=max(30.0, timeout_s))
                thumb_paths = [p_start, p_mid, p_end]
                thumb_path = p_mid
            else:
                thumb_path = thumbs_dir / f"{asset_id}.jpg"
                _extract_thumbnail(src=path, dst=thumb_path, kind=kind, at_s=None, timeout_s=max(30.0, timeout_s))
                thumb_paths = [thumb_path]
        except Exception:
            thumb_path = None
            thumb_paths = []

        assets.append(
            MediaAsset(
                id=asset_id,
                path=str(path),
                kind=kind,
                ext=ext,
                size_bytes=int(st.st_size),
                mtime=float(st.st_mtime),
                duration_s=duration_s,
                width=width,
                height=height,
                has_audio=has_audio,
                thumbnail_path=str(thumb_path) if thumb_path else None,
                thumbnail_paths=[str(p) for p in thumb_paths if p],
            )
        )

    if progress_cb:
        progress_cb(total, total, "Index complete")

    from datetime import datetime, timezone

    return MediaIndex(
        source_folder=str(folder),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        assets=assets,
    )


def load_index(path: Path) -> MediaIndex:
    data = json.loads(path.read_text(encoding="utf-8"))
    assets = [MediaAsset(**a) for a in data.get("assets") or []]
    return MediaIndex(source_folder=str(data.get("source_folder") or ""), generated_at=str(data.get("generated_at") or ""), assets=assets)


def save_index(index: MediaIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
