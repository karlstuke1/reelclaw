from __future__ import annotations

import json
import os
import shutil
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import typing as t

from .reel_cut_detect import detect_scene_cuts

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None


try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


SCHEMA_VERSION = 5
# Bump when shot-level tagging inputs/outputs change materially (e.g., number of thumbs).
TAG_CACHE_VERSION = 3


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}... {stderr or 'unknown error'}")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def folder_cache_key(folder: Path) -> str:
    # Deterministic cache key for a folder; changes to contents are handled by asset signatures.
    return _sha1(str(folder.expanduser().resolve()))[:12]


def _index_config_signature(config: dict[str, t.Any]) -> str:
    # Hash only stable, JSON-serializable values so cache invalidation is reliable.
    try:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = repr(sorted(config.items()))
    return _sha1(payload)[:12]


def _extract_thumb(*, video_path: Path, at_s: float, out_path: Path, timeout_s: float) -> None:
    ffmpeg = os.getenv("FFMPEG", "") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to build the shot index. Please install ffmpeg and try again.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w = int(max(160, float(os.getenv("SHOT_THUMB_WIDTH", "480"))))
    # ffmpeg can occasionally exit 0 yet produce no frame near the very end of a clip.
    # Retry slightly earlier timestamps to ensure we get a usable thumbnail.
    for attempt, dt in enumerate([0.0, -0.06, -0.12, -0.24], start=1):
        t_s = max(0.0, float(at_s) + float(dt))
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{t_s:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            # Ensure MJPEG gets a compatible full-range pixel format on iPhone footage.
            f"scale={thumb_w}:-2:flags=lanczos,format=yuvj420p",
            "-q:v",
            "2",
            str(out_path),
        ]
        _run(cmd, timeout_s=timeout_s)
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                return
        except Exception:
            pass
        if attempt >= 4:
            break
    raise RuntimeError(f"ffmpeg produced no thumbnail at ~{float(at_s):.3f}s for {video_path.name}")


def _luma_mean(path: Path) -> float | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L").resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return None
        return float(sum(pixels) / len(pixels)) / 255.0
    except Exception:
        return None


def _dark_frac(path: Path, *, threshold: int = 32) -> float | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L").resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return None
        dark = sum(1 for p in pixels if int(p) < int(threshold))
        return float(dark) / float(len(pixels))
    except Exception:
        return None


def _rgb_mean(path: Path) -> list[float] | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("RGB").resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return None
        r = sum(int(p[0]) for p in pixels) / len(pixels)
        g = sum(int(p[1]) for p in pixels) / len(pixels)
        b = sum(int(p[2]) for p in pixels) / len(pixels)
        return [float(r) / 255.0, float(g) / 255.0, float(b) / 255.0]
    except Exception:
        return None


def _motion_score_from_thumbs(paths: list[Path]) -> float | None:
    if Image is None:
        return None
    imgs: list[list[int]] = []
    for p in paths:
        try:
            img = Image.open(p).convert("L").resize((64, 64))
        except Exception:
            continue
        imgs.append(list(img.getdata()))
    if len(imgs) < 2:
        return None
    diffs: list[float] = []
    for a, b in zip(imgs, imgs[1:], strict=False):
        if not a or not b:
            continue
        n = min(len(a), len(b))
        if n <= 0:
            continue
        diffs.append(sum(abs(int(a[i]) - int(b[i])) for i in range(n)) / (n * 255.0))
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))


def _phase_correlation_shift(a: "np.ndarray", b: "np.ndarray") -> tuple[float, float] | None:
    """
    Estimate global translation between two grayscale images using phase correlation.
    Returns (dx, dy) in pixels, where positive dx means b is shifted right relative to a.
    """
    if np is None:
        return None
    try:
        if a.shape != b.shape:
            return None
        # Remove DC component to improve correlation peak.
        a0 = a.astype("float32", copy=False) - float(a.mean())
        b0 = b.astype("float32", copy=False) - float(b.mean())
        fa = np.fft.fft2(a0)
        fb = np.fft.fft2(b0)
        r = fa * np.conj(fb)
        denom = np.abs(r)
        r = r / (denom + 1e-9)
        corr = np.fft.ifft2(r)
        corr_abs = np.abs(corr)
        y, x = np.unravel_index(int(np.argmax(corr_abs)), corr_abs.shape)
        h, w = corr_abs.shape
        dy = float(y)
        dx = float(x)
        if dy > h / 2:
            dy -= float(h)
        if dx > w / 2:
            dx -= float(w)
        return dx, dy
    except Exception:
        return None


def _camera_motion_stats_from_thumbs(paths: list[Path]) -> dict[str, float] | None:
    """
    Estimate camera motion stats using a few spaced thumbnails.

    Returns:
    - shake_score: residual jitter after removing dominant translation (0..~1, normalized by image size)
    - cam_motion_dx/dy: dominant translation direction (normalized by image size)
    - cam_motion_mag: magnitude of dominant translation (normalized)
    - cam_motion_angle_deg: atan2(dy, dx) in degrees (-180..180), 0 when mag is ~0

    This is deliberately cheap; it helps:
    - avoid very shaky clips
    - reason about motion-direction continuity for transitions
    - decide when stabilization is likely to crop/warp too much
    """
    if Image is None or np is None:
        return None
    try:
        if len(paths) < 2:
            return None
        size = int(max(48, min(96, float(os.getenv("SHOT_SHAKE_SIZE", "64")))))

        frames: list["np.ndarray"] = []
        for p in paths:
            try:
                img = Image.open(p).convert("L").resize((size, size))
                arr = np.asarray(img, dtype="float32")
                frames.append(arr)
            except Exception:
                continue
        if len(frames) < 2:
            return None

        shifts: list[tuple[float, float]] = []
        for a, b in zip(frames, frames[1:], strict=False):
            sh = _phase_correlation_shift(a, b)
            if sh is None:
                continue
            shifts.append(sh)
        if not shifts:
            return None

        mdx = sum(dx for dx, _dy in shifts) / len(shifts)
        mdy = sum(dy for _dx, dy in shifts) / len(shifts)

        # Residual jitter after removing dominant motion (pans/tilts).
        res = [((dx - mdx) ** 2 + (dy - mdy) ** 2) for dx, dy in shifts]
        rms = float((sum(res) / len(res)) ** 0.5)

        denom = float(size) if size > 0 else 64.0
        cam_dx = float(mdx) / denom
        cam_dy = float(mdy) / denom
        cam_mag = float((cam_dx**2 + cam_dy**2) ** 0.5)
        cam_angle = 0.0
        if cam_mag > 1e-6:
            import math

            cam_angle = float(math.degrees(math.atan2(cam_dy, cam_dx)))

        return {
            "shake_score": float(rms / denom),
            "cam_motion_dx": float(cam_dx),
            "cam_motion_dy": float(cam_dy),
            "cam_motion_mag": float(cam_mag),
            "cam_motion_angle_deg": float(cam_angle),
        }
    except Exception:
        return None


def _shake_score_from_thumbs(paths: list[Path]) -> float | None:
    """
    Cheap camera shake proxy from a few spaced thumbnails:
    - Estimate global shifts between consecutive thumbs.
    - Remove the mean shift (pans/tilts) and measure residual jitter.

    This is not perfect (only 4 deltas for 5 thumbs), but it reliably filters "very shaky" clips
    and gives a better stabilization trigger than total motion alone.
    """
    stats = _camera_motion_stats_from_thumbs(paths)
    if not stats:
        return None
    try:
        return float(stats.get("shake_score"))
    except Exception:
        return None


def _sharpness(path: Path) -> float | None:
    """
    Very cheap sharpness proxy: variance of a Laplacian-like response on a tiny grayscale image.
    Higher is sharper. Keep as a relative score only.
    """
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L").resize((64, 64))
        px = list(img.getdata())
        if len(px) != 64 * 64:
            return None
        # 2D access helper.
        def p(x: int, y: int) -> int:
            return int(px[y * 64 + x])

        vals: list[float] = []
        for y in range(1, 63):
            for x in range(1, 63):
                c = p(x, y)
                lap = (p(x - 1, y) + p(x + 1, y) + p(x, y - 1) + p(x, y + 1) - 4 * c)
                vals.append(float(lap))
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        return float(var)
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _compute_target_scenes(duration_s: float) -> tuple[int, int, int]:
    """
    Decide how many scenes to ask ffmpeg's scene detector for.
    Keep index size bounded and shots reasonably sized for editing.
    """
    target_len = float(os.getenv("SHOT_TARGET_LEN_S", "2.0"))
    target_len = _clamp(target_len, 0.8, 6.0)
    target = int(round(float(duration_s) / target_len)) if duration_s > 0 else 1
    target = int(_clamp(float(target), 4.0, 80.0))
    min_s = max(2, target - 20)
    max_s = min(120, target + 20)
    return int(min_s), int(max_s), int(target)


def _merge_and_split_shots(
    *,
    boundaries: list[float],
    min_shot_s: float,
    max_shot_s: float,
) -> list[tuple[float, float]]:
    # Create raw shots from boundaries.
    raw: list[tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        s = float(boundaries[i])
        e = float(boundaries[i + 1])
        if e <= s:
            continue
        raw.append((s, e))
    if not raw:
        return []

    # Merge micro-shots.
    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(raw):
        s, e = raw[i]
        d = e - s
        if d >= min_shot_s or len(raw) == 1:
            merged.append((s, e))
            i += 1
            continue

        # Prefer merging into previous if possible; otherwise merge forward.
        if merged:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
            i += 1
            continue
        if i + 1 < len(raw):
            ns, ne = raw[i + 1]
            raw[i + 1] = (s, ne)
            i += 1
            continue
        merged.append((s, e))
        i += 1

    # Split overly long shots to keep candidate granularity.
    if max_shot_s <= 0:
        return merged
    out: list[tuple[float, float]] = []
    for s, e in merged:
        d = e - s
        if d <= max_shot_s + 1e-6:
            out.append((s, e))
            continue
        # Use even splitting so we don't end up with a tiny \"remainder\" shot.
        import math

        n = int(max(2, math.ceil(d / max_shot_s)))
        step = d / float(n)
        t0 = s
        for _i in range(n):
            t1 = min(e, t0 + step)
            if t1 - t0 > 1e-3:
                out.append((t0, t1))
            t0 = t1
    return out


@dataclass(frozen=True)
class ShotIndex:
    source_folder: str
    generated_at: str
    config: dict[str, t.Any]
    asset_signatures: dict[str, dict[str, t.Any]]
    shots: list[dict[str, t.Any]]

    def to_json(self) -> dict[str, t.Any]:
        return {
            "version": SCHEMA_VERSION,
            "source_folder": self.source_folder,
            "generated_at": self.generated_at,
            "config": self.config,
            "asset_signatures": self.asset_signatures,
            "shots": self.shots,
        }


def _cache_valid(media_assets: list[dict[str, t.Any]], shot_doc: dict[str, t.Any], *, config_sig: str) -> bool:
    sigs = shot_doc.get("asset_signatures")
    if not isinstance(sigs, dict):
        return False
    # Rebuild if any indexing parameters changed.
    cfg = shot_doc.get("config") or {}
    if not isinstance(cfg, dict):
        return False
    if str(cfg.get("sig") or "") != str(config_sig):
        return False
    # Require every current video asset to match signature.
    for a in media_assets:
        if a.get("kind") != "video":
            continue
        aid = str(a.get("id") or "")
        if not aid:
            continue
        s = sigs.get(aid)
        if not isinstance(s, dict):
            return False
        for k in ("mtime", "size_bytes"):
            if str(s.get(k)) != str(a.get(k)):
                return False
    return True


def load_shot_index(path: Path) -> ShotIndex:
    doc = json.loads(path.read_text(encoding="utf-8"))
    ver = int(doc.get("version") or 0)
    if ver not in {SCHEMA_VERSION, 4}:
        raise ValueError("Unsupported shot_index schema version")

    # Back-compat: schema v4 did not include camera motion fields (e.g., shake_score).
    # We accept it and upgrade in memory; callers can persist the upgraded form.
    if ver == 4:
        doc["version"] = SCHEMA_VERSION
        shots = doc.get("shots") or []
        if isinstance(shots, list):
            for r in shots:
                if not isinstance(r, dict):
                    continue
                r.setdefault("shake_score", None)
                r.setdefault("cam_motion_dx", None)
                r.setdefault("cam_motion_dy", None)
                r.setdefault("cam_motion_mag", None)
                r.setdefault("cam_motion_angle_deg", None)

    return ShotIndex(
        source_folder=str(doc.get("source_folder") or ""),
        generated_at=str(doc.get("generated_at") or ""),
        config=t.cast(dict[str, t.Any], doc.get("config") or {}),
        asset_signatures=t.cast(dict[str, dict[str, t.Any]], doc.get("asset_signatures") or {}),
        shots=t.cast(list[dict[str, t.Any]], doc.get("shots") or []),
    )


def save_shot_index(index: ShotIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")


def build_or_load_shot_index(
    *,
    media_index_path: Path,
    cache_dir: Path,
    api_key: str | None,
    model: str | None,
    timeout_s: float = 240.0,
    site_url: str | None = None,
    app_name: str | None = None,
    progress_cb: t.Callable[[int, int, str], None] | None = None,
) -> ShotIndex:
    """
    Build a shot-level index for a media folder (segmented within each video).
    Cached under cache_dir/<folder_hash>/shot_index.json.

    `media_index_path` is the per-run MediaIndex JSON written by `media_index.py`.
    We use it to detect if the cache is still valid and to resolve video paths.
    """
    media_doc = json.loads(media_index_path.read_text(encoding="utf-8"))
    assets = t.cast(list[dict[str, t.Any]], media_doc.get("assets") or [])
    source_folder = str(media_doc.get("source_folder") or "")

    mode = os.getenv("SHOT_INDEX_MODE", "fast").strip().lower()
    if mode not in {"fast", "scene"}:
        mode = "fast"

    folder_id = folder_cache_key(Path(source_folder))
    out_dir = cache_dir / f"{folder_id}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shots_path = out_dir / "shot_index.json"
    thumbs_dir = out_dir / "shot_thumbs"
    tag_cache_path = out_dir / "shot_tag_cache.json"

    # Cache invalidation signature for indexing params (so we don't silently reuse an index built
    # with different shot/window settings).
    thumb_w = int(max(160, float(os.getenv("SHOT_THUMB_WIDTH", "480"))))
    min_shot_s = float(os.getenv("SHOT_MIN_LEN_S", "0.55"))
    # Default max shot length is intentionally smaller to give the optimizer more granular choices.
    max_shot_s = float(os.getenv("SHOT_MAX_LEN_S", "4.0"))
    group_span_s = float(os.getenv("SHOT_GROUP_SPAN_S", "10.0"))
    # Default to 5 so the VLM can infer what's happening across time (action, camera movement, etc.).
    thumb_count = int(max(1, min(5, float(os.getenv("SHOT_THUMB_COUNT", "5")))))
    idx_cfg = {
        "mode": mode,
        "shot_target_len_s": float(os.getenv("SHOT_TARGET_LEN_S", "2.0")),
        "shot_min_len_s": float(min_shot_s),
        "shot_max_len_s": float(max_shot_s),
        "shot_group_span_s": float(group_span_s),
        "shot_thumb_width": int(thumb_w),
        "shot_thumb_count": int(thumb_count),
        # Fast-mode params (still part of signature so switching modes is safe).
        "shot_window_s": float(os.getenv("SHOT_WINDOW_S", "6.0")),
        "shot_fast_windows": int(max(3, min(9, float(os.getenv("SHOT_FAST_WINDOWS", "3"))))),
    }
    idx_cfg["sig"] = _index_config_signature(idx_cfg)

    def _enrich_cached_shots(rows: list[dict[str, t.Any]]) -> bool:
        """
        Fill missing camera motion fields from cached thumbnails without rebuilding cuts.
        Returns True if any row was updated.
        """
        updated = False
        for r in rows:
            thumbs = r.get("thumbnail_paths") or []
            if not isinstance(thumbs, list) or len(thumbs) < 2:
                continue
            # If we already have a shake score + motion direction, skip.
            if any(k not in r for k in ("shake_score", "cam_motion_angle_deg", "cam_motion_mag")):
                need = True
            else:
                need = (r.get("shake_score") is None) or (r.get("cam_motion_mag") is None) or (r.get("cam_motion_angle_deg") is None)
            if not need:
                continue

            paths: list[Path] = []
            for tp in thumbs[:5]:
                p = Path(str(tp))
                if p.exists():
                    paths.append(p)
            if len(paths) < 2:
                continue
            stats = _camera_motion_stats_from_thumbs(paths)
            if not stats:
                continue
            if r.get("shake_score") is None:
                r["shake_score"] = float(stats.get("shake_score", 0.0))
                updated = True
            # Always populate motion direction stats when missing.
            for k in ("cam_motion_dx", "cam_motion_dy", "cam_motion_mag", "cam_motion_angle_deg"):
                if r.get(k) is None and k in stats:
                    r[k] = float(stats[k])
                    updated = True
        return updated

    def _thumb_times_for_shot(s: float, e: float, *, count: int) -> list[tuple[str, float]]:
        d = float(e - s)
        if d <= 1e-6:
            return [("mid", float(s))]
        pad = min(0.12, max(0.02, d * 0.07))
        t_mid = (s + e) / 2.0
        t_start = min(max(s + pad, 0.0), e)
        t_end = max(s, e - pad)

        if count <= 1:
            return [("mid", float(t_mid))]
        if count == 2:
            return [("start", float(t_start)), ("end", float(t_end))]
        if count == 3:
            return [("start", float(t_start)), ("mid", float(t_mid)), ("end", float(t_end))]
        if count == 4:
            return [
                ("start", float(t_start)),
                ("p33", float(s + (d * 0.33))),
                ("p67", float(s + (d * 0.67))),
                ("end", float(t_end)),
            ]
        # count >= 5
        return [
            ("start", float(t_start)),
            ("p25", float(s + (d * 0.25))),
            ("mid", float(t_mid)),
            ("p75", float(s + (d * 0.75))),
            ("end", float(t_end)),
        ]

    # Build shots.
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    # (min/max/group/thumb_count already computed above for cache signature)

    shot_rows: list[dict[str, t.Any]] = []
    asset_sigs: dict[str, dict[str, t.Any]] = {}
    video_assets = [a for a in assets if a.get("kind") == "video"]
    total_videos = len(video_assets)

    workers = int(max(1, min(6, float(os.getenv("SHOT_INDEX_WORKERS", "2")))))

    def _index_one(a: dict[str, t.Any]) -> tuple[str, dict[str, t.Any] | None, list[dict[str, t.Any]], str]:
        aid = str(a.get("id") or "")
        apath = Path(str(a.get("path") or "")).expanduser()
        name = apath.name
        if not aid or not apath.exists():
            return aid, None, [], name
        dur = a.get("duration_s")
        try:
            dur_s = float(dur) if isinstance(dur, (int, float, str)) else None
        except Exception:
            dur_s = None
        if not isinstance(dur_s, (int, float)) or not dur_s or dur_s <= 0.5:
            return aid, None, [], name

        sig = {"mtime": a.get("mtime"), "size_bytes": a.get("size_bytes"), "duration_s": float(dur_s)}

        if mode == "scene":
            min_s, max_s, target = _compute_target_scenes(float(dur_s))
            try:
                cuts, _dur = detect_scene_cuts(
                    video_path=apath,
                    min_scenes=min_s,
                    max_scenes=max_s,
                    target_scenes=target,
                    min_spacing_s=0.18,
                    timeout_s=min(timeout_s, 180.0),
                )
            except Exception:
                cuts = []
            boundaries = [0.0] + [float(c) for c in cuts if 0.0 < float(c) < float(dur_s)] + [float(dur_s)]
            boundaries = sorted(set(boundaries))
            shots = _merge_and_split_shots(boundaries=boundaries, min_shot_s=min_shot_s, max_shot_s=max_shot_s)
        else:
            win = float(os.getenv("SHOT_WINDOW_S", "6.0"))
            win = _clamp(win, 1.0, 12.0)
            win = min(win, float(dur_s))
            n_windows = int(max(3, min(9, float(os.getenv("SHOT_FAST_WINDOWS", "3")))))
            if n_windows == 3:
                centers = [0.15, 0.50, 0.85]
            else:
                centers = [0.10 + (0.80 * (i / max(1, n_windows - 1))) for i in range(n_windows)]
            shots = []
            for c in centers:
                mid = float(dur_s) * float(c)
                s0 = max(0.0, mid - (win / 2.0))
                e0 = min(float(dur_s), s0 + win)
                if e0 - s0 >= 0.25:
                    shots.append((s0, e0))

        rows: list[dict[str, t.Any]] = []
        for s, e in shots:
            d = float(e - s)
            if d <= 0.05:
                continue
            s_ms = int(round(s * 1000.0))
            e_ms = int(round(e * 1000.0))
            shot_id = f"{aid}#{s_ms:08d}_{e_ms:08d}"

            thumb_paths: list[Path] = []
            times = _thumb_times_for_shot(float(s), float(e), count=int(thumb_count))
            seen_paths: set[str] = set()
            for label, t_s in times:
                safe_label = str(label).strip().lower() or "t"
                p = thumbs_dir / f"{shot_id}_{safe_label}.jpg"
                if not p.exists():
                    _extract_thumb(video_path=apath, at_s=float(t_s), out_path=p, timeout_s=min(timeout_s, 120.0))
                if p.exists():
                    ps = str(p)
                    if ps in seen_paths:
                        continue
                    seen_paths.add(ps)
                    thumb_paths.append(p)

            lumas = [float(x) for x in (_luma_mean(p) for p in thumb_paths) if isinstance(x, (int, float))]
            darks = [float(x) for x in (_dark_frac(p) for p in thumb_paths) if isinstance(x, (int, float))]
            rgbs = [x for x in (_rgb_mean(p) for p in thumb_paths) if isinstance(x, list) and len(x) == 3]
            luma_mean = (sum(lumas) / len(lumas)) if lumas else None
            luma_min = (min(lumas) if lumas else None)
            luma_max = (max(lumas) if lumas else None)
            dark_mean = (sum(darks) / len(darks)) if darks else None
            dark_min = (min(darks) if darks else None)
            dark_max = (max(darks) if darks else None)
            rgb_mean = ([sum(v[i] for v in rgbs) / len(rgbs) for i in range(3)] if rgbs else None)

            motion = _motion_score_from_thumbs(thumb_paths) if len(thumb_paths) >= 2 else None
            sharp = _sharpness(thumb_paths[0]) if thumb_paths else None
            cam_stats = _camera_motion_stats_from_thumbs(thumb_paths) if len(thumb_paths) >= 2 else None
            shake = float(cam_stats.get("shake_score")) if isinstance(cam_stats, dict) and "shake_score" in cam_stats else None

            g = int(s / max(1e-6, group_span_s))
            seq_group = f"{aid}_g{g:03d}"

            rows.append(
                {
                    "id": shot_id,
                    "asset_id": aid,
                    "asset_path": str(apath),
                    "start_s": float(s),
                    "end_s": float(e),
                    "duration_s": float(d),
                    "sequence_group_id": seq_group,
                    "thumbnail_paths": [str(p) for p in thumb_paths],
                    "luma_mean": luma_mean,
                    "luma_min": luma_min,
                    "luma_max": luma_max,
                    "dark_frac": dark_mean,
                    "dark_frac_min": dark_min,
                    "dark_frac_max": dark_max,
                    "rgb_mean": rgb_mean,
                    "motion_score": motion,
                    "shake_score": shake,
                    "cam_motion_dx": (float(cam_stats.get("cam_motion_dx")) if isinstance(cam_stats, dict) and "cam_motion_dx" in cam_stats else None),
                    "cam_motion_dy": (float(cam_stats.get("cam_motion_dy")) if isinstance(cam_stats, dict) and "cam_motion_dy" in cam_stats else None),
                    "cam_motion_mag": (float(cam_stats.get("cam_motion_mag")) if isinstance(cam_stats, dict) and "cam_motion_mag" in cam_stats else None),
                    "cam_motion_angle_deg": (float(cam_stats.get("cam_motion_angle_deg")) if isinstance(cam_stats, dict) and "cam_motion_angle_deg" in cam_stats else None),
                    "sharpness": sharp,
                }
            )

        return aid, sig, rows, name

    # Incremental cache: reuse existing shot rows and only index new/changed assets when possible.
    cached_idx: ShotIndex | None = None
    if shots_path.exists():
        try:
            doc = json.loads(shots_path.read_text(encoding="utf-8"))
            ver = int(doc.get("version") or 0)
            cfg = doc.get("config") or {}
            cfg_sig = str(cfg.get("sig") or "") if isinstance(cfg, dict) else ""
            if ver in {SCHEMA_VERSION, 4} and cfg_sig == str(idx_cfg["sig"]):
                cached_idx = load_shot_index(shots_path)  # may upgrade v4->v5 in memory
        except Exception:
            cached_idx = None

    to_index: list[dict[str, t.Any]] = list(video_assets)
    if cached_idx is not None:
        cached_sigs = cached_idx.asset_signatures or {}
        current_ids = {str(a.get("id") or "") for a in video_assets if str(a.get("id") or "")}
        removed = {aid for aid in cached_sigs.keys() if aid not in current_ids}

        def _sig_matches(asset: dict[str, t.Any], sig: dict[str, t.Any]) -> bool:
            for k in ("mtime", "size_bytes", "duration_s"):
                if str(sig.get(k)) != str(asset.get(k)):
                    return False
            return True

        changed_ids: set[str] = set()
        changed_assets: list[dict[str, t.Any]] = []
        for a in video_assets:
            aid = str(a.get("id") or "")
            if not aid:
                continue
            sig = cached_sigs.get(aid)
            if not isinstance(sig, dict) or (not _sig_matches(a, sig)):
                changed_ids.add(aid)
                changed_assets.append(a)

        # Keep shots for unchanged assets only.
        keep_ids = {aid for aid in current_ids if aid not in changed_ids}
        shot_rows = [t.cast(dict[str, t.Any], r) for r in (cached_idx.shots or []) if str(r.get("asset_id") or "") in keep_ids]
        asset_sigs = {aid: t.cast(dict[str, t.Any], cached_sigs[aid]) for aid in keep_ids if aid in cached_sigs}

        # Re-index only new/changed assets.
        to_index = changed_assets

        # Drop removed signatures (shots already filtered out above).
        if removed:
            # Keep mypy happy; asset_sigs already excludes removed ids.
            removed = removed

    total_to_index = len(to_index)
    if total_to_index:
        if workers <= 1 or total_to_index <= 1:
            for vi, a in enumerate(to_index, start=1):
                if progress_cb:
                    progress_cb(vi - 1, total_to_index, f"Shot indexing ({mode}) {Path(str(a.get('path') or '')).name}")
                aid, sig, rows, _name = _index_one(t.cast(dict[str, t.Any], a))
                if sig is not None and aid:
                    asset_sigs[aid] = sig
                shot_rows.extend(rows)
                if progress_cb:
                    progress_cb(vi, total_to_index, f"Indexed {vi}/{total_to_index} videos")
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_index_one, t.cast(dict[str, t.Any], a)) for a in to_index]
                for fut in as_completed(futs):
                    try:
                        aid, sig, rows, name = fut.result()
                    except Exception as e:
                        done += 1
                        if progress_cb:
                            progress_cb(done, total_to_index, f"Shot indexing failed ({mode}) {type(e).__name__}")
                        continue
                    if sig is not None and aid:
                        asset_sigs[aid] = sig
                    shot_rows.extend(rows)
                    done += 1
                    if progress_cb:
                        progress_cb(done, total_to_index, f"Indexed {done}/{total_to_index} ({mode}) {name}")

    # Stable ordering for deterministic downstream behavior.
    shot_rows.sort(key=lambda r: (str(r.get("asset_id") or ""), float(r.get("start_s") or 0.0)))

    # Fill camera motion stats from cached thumbs if missing.
    _enrich_cached_shots(shot_rows)

    # Tag shots (optional; cached). This can be expensive, so allow env cap.
    tags: dict[str, t.Any] = {}
    if tag_cache_path.exists():
        try:
            cached = json.loads(tag_cache_path.read_text(encoding="utf-8"))
            if int(cached.get("version") or 0) == TAG_CACHE_VERSION:
                cached_tags = cached.get("tags") or {}
                if isinstance(cached_tags, dict):
                    tags = cached_tags
        except Exception:
            tags = {}

    def _apply_tags() -> None:
        for row in shot_rows:
            sid = str(row.get("id") or "")
            info = tags.get(sid)
            if isinstance(info, dict):
                row.update(info)

    _apply_tags()

    do_tag = bool(api_key and model) and os.getenv("SHOT_TAGGING", "1").strip().lower() not in {"0", "false", "no", "off"}
    if do_tag:
        # Default to tagging ALL shots so the system can be truly "library-aware".
        # Users can cap this via SHOT_TAG_MAX if they need to limit cost/time.
        max_tag = int(max(0, float(os.getenv("SHOT_TAG_MAX", "0"))))
        missing = [r for r in shot_rows if str(r.get("id") or "") and str(r.get("id")) not in tags]
        if max_tag > 0:
            missing = missing[:max_tag]
        if missing:
            from .folder_edit_planner import tag_assets_from_thumbnails

            assets_for_tagger: list[dict[str, t.Any]] = []
            for r in missing:
                sid = str(r.get("id") or "")
                thumbs = r.get("thumbnail_paths") or []
                if not sid or not isinstance(thumbs, list) or not thumbs:
                    continue
                assets_for_tagger.append(
                    {
                        "id": sid,
                        "kind": "video",
                        "path": r.get("asset_path"),
                        "filename": f"{sid}.mp4",
                        "duration_s": r.get("duration_s"),
                        # Pass multiple thumbs so the VLM can infer progression across time.
                        "thumbnail_paths": thumbs,
                        "thumbnail_path": thumbs[len(thumbs) // 2] if thumbs else None,
                    }
                )

            new_tags = tag_assets_from_thumbnails(
                api_key=t.cast(str, api_key),
                model=t.cast(str, model),
                assets=assets_for_tagger,
                timeout_s=min(timeout_s, 180.0),
                site_url=site_url,
                app_name=app_name,
                batch_size=4,
            )
            for sid, tinfo in new_tags.items():
                tags[str(sid)] = {
                    "description": tinfo.description,
                    "tags": tinfo.tags,
                    "shot_type": tinfo.shot_type,
                    "setting": tinfo.setting,
                    "mood": tinfo.mood,
                }
            tag_cache_path.write_text(json.dumps({"version": TAG_CACHE_VERSION, "tags": tags}, indent=2), encoding="utf-8")
            _apply_tags()

    # Persist.
    from datetime import datetime, timezone

    idx = ShotIndex(
        source_folder=source_folder,
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        config=idx_cfg,
        asset_signatures=asset_sigs,
        shots=shot_rows,
    )
    save_shot_index(idx, shots_path)
    return idx
