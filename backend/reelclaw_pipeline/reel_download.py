from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        tail = stderr or stdout or "unknown error"
        if len(tail) > 2000:
            tail = tail[-2000:]
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{tail}")


def _is_instagram_url(url: str) -> bool:
    try:
        host = (urlparse(str(url)).hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    return host == "instagram.com" or host.endswith(".instagram.com")


def _maybe_write_cookies_file(output_dir: Path, *, required: bool = False) -> Path | None:
    """
    Return a path to a Netscape-format cookies.txt file if configured.

    Supported env vars:
      - REELCLAW_YTDLP_COOKIES_FILE: absolute/relative path inside the container
      - REELCLAW_YTDLP_COOKIES: raw cookies.txt contents (Netscape format)
      - REELCLAW_YTDLP_COOKIES_B64: base64-encoded cookies.txt contents
      - REELCLAW_YTDLP_COOKIES_SECRET_ID: Secrets Manager id holding cookies contents (raw or base64)
      - YTDLP_COOKIES_FILE / YTDLP_COOKIES_B64: aliases for local/dev
    """
    raw_path = os.getenv("REELCLAW_YTDLP_COOKIES_FILE", "").strip() or os.getenv("YTDLP_COOKIES_FILE", "").strip()
    if raw_path:
        p = Path(raw_path).expanduser()
        if p.exists() and p.is_file():
            return p

    raw_text = os.getenv("REELCLAW_YTDLP_COOKIES", "").strip() or os.getenv("YTDLP_COOKIES", "").strip()
    if not raw_text:
        raw_b64 = os.getenv("REELCLAW_YTDLP_COOKIES_B64", "").strip() or os.getenv("YTDLP_COOKIES_B64", "").strip()
        if not raw_b64:
            secret_id = os.getenv("REELCLAW_YTDLP_COOKIES_SECRET_ID", "").strip() or os.getenv("YTDLP_COOKIES_SECRET_ID", "").strip()
            if not secret_id:
                if required:
                    raise RuntimeError(
                        "Cookies are required but not configured. "
                        "Set REELCLAW_YTDLP_COOKIES_SECRET_ID (Secrets Manager id) or REELCLAW_YTDLP_COOKIES_B64 / REELCLAW_YTDLP_COOKIES."
                    )
                return None
            try:
                import boto3  # type: ignore

                region = os.getenv("REELCLAW_AWS_REGION", "").strip() or os.getenv("AWS_REGION", "").strip() or "us-east-1"
                sm = boto3.client("secretsmanager", region_name=region)
                resp = sm.get_secret_value(SecretId=secret_id)
                raw_b64 = str(resp.get("SecretString") or "").strip()
                if not raw_b64:
                    if required:
                        raise RuntimeError(
                            f"Cookies secret is empty: {secret_id}. Set it to a Netscape cookies.txt (raw or base64)."
                        )
                    return None
            except Exception as exc:
                if required:
                    exc_name = type(exc).__name__
                    if exc_name == "ResourceNotFoundException":
                        raise RuntimeError(
                            f"Cookies secret has no value set (AWSCURRENT missing): {secret_id}. "
                            "Set it to a Netscape cookies.txt (raw or base64)."
                        ) from exc
                    if exc_name == "AccessDeniedException":
                        raise RuntimeError(
                            f"Access denied reading cookies secret {secret_id}. Verify Batch IAM role has secretsmanager:GetSecretValue."
                        ) from exc
                    raise RuntimeError(
                        f"Failed to read cookies secret {secret_id}. Verify Batch role permissions and secret value. ({exc_name})"
                    ) from exc
                return None

        if not raw_b64:
            return None

        # If the secret contains raw cookies text (common), write as-is.
        if "\n" in raw_b64 or "\t" in raw_b64 or "Netscape" in raw_b64:
            raw_text = raw_b64
        else:
            try:
                data = base64.b64decode(raw_b64.encode("utf-8"), validate=False)
            except Exception:
                if required:
                    raise RuntimeError("Cookies secret was not valid base64 and did not look like raw Netscape cookies.txt.")
                return None
            if not data:
                if required:
                    raise RuntimeError("Cookies secret decoded to an empty payload.")
                return None
            try:
                raw_text = data.decode("utf-8", errors="replace")
            except Exception:
                return None
            if not raw_text.strip():
                if required:
                    raise RuntimeError("Cookies secret decoded to an empty cookies.txt.")
                return None

    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "ytdlp_cookies.txt"
    dst.write_text(raw_text, encoding="utf-8", errors="replace")
    try:
        os.chmod(dst, 0o600)
    except Exception:
        pass
    return dst


def download_reel(url: str, *, output_dir: Path, timeout_s: float = 180.0) -> Path:
    ytdlp = os.getenv("YTDLP", "").strip() or shutil.which("yt-dlp")
    if not ytdlp:
        for candidate in ("/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp"):
            if Path(candidate).exists():
                ytdlp = candidate
                break
    if not ytdlp:
        raise RuntimeError(
            "yt-dlp is required to download reference videos (Instagram/YouTube/etc). Install with `brew install yt-dlp` or `pipx install yt-dlp`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "reel_source.%(ext)s")

    # Avoid picking up stale prior downloads if a run is resumed.
    for p in output_dir.glob("reel_source.*"):
        try:
            p.unlink()
        except Exception:
            pass

    # Prefer MP4 video+M4A audio when possible (helps ffmpeg + iOS compatibility),
    # but fall back to "best" if the source doesn't offer MP4.
    fmt = os.getenv("YTDLP_FORMAT", "").strip() or "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"

    # Cookies can help reliability (especially on AWS IP ranges), but should be best-effort by default.
    # If you want to fail-fast when cookies are missing/misconfigured, set:
    #   REELCLAW_YTDLP_COOKIES_REQUIRED=1
    cookies_required = _is_instagram_url(url) and (
        _truthy_env("REELCLAW_YTDLP_COOKIES_REQUIRED", "0") or _truthy_env("YTDLP_COOKIES_REQUIRED", "0")
    )
    cookies_file = _maybe_write_cookies_file(output_dir, required=cookies_required)

    cmd = [
        ytdlp,
        "--no-playlist",
        "--no-progress",
        "--retries",
        os.getenv("YTDLP_RETRIES", "").strip() or "3",
        "--extractor-retries",
        os.getenv("YTDLP_EXTRACTOR_RETRIES", "").strip() or "3",
        "--fragment-retries",
        os.getenv("YTDLP_FRAGMENT_RETRIES", "").strip() or "3",
    ]
    if cookies_file is not None:
        cmd += ["--cookies", str(cookies_file)]
    cmd += [
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        url,
    ]
    try:
        _run(cmd, timeout_s=timeout_s)
    except Exception as exc:
        msg = str(exc)
        lower = msg.lower()
        if any(k in lower for k in ("require_login", "login required", "not-logged-in", "sign in", "cookies")):
            secret_id = os.getenv("REELCLAW_YTDLP_COOKIES_SECRET_ID", "").strip()
            secret_hint = (
                f"Set AWS Secrets Manager `{secret_id}` to a Netscape cookies.txt (raw or base64)."
                if secret_id
                else "Set REELCLAW_YTDLP_COOKIES_B64 (base64 of a Netscape cookies.txt file)."
            )
            raise RuntimeError(
                "Reference download blocked (login required). " + secret_hint + " This enables Instagram/YouTube downloads."
            ) from exc
        if any(k in lower for k in ("unavailable for certain audiences", "certain audiences", "content may be inappropriate")):
            secret_id = os.getenv("REELCLAW_YTDLP_COOKIES_SECRET_ID", "").strip()
            secret_hint = (
                f"Set AWS Secrets Manager `{secret_id}` to a Netscape cookies.txt (raw or base64)."
                if secret_id
                else "Set REELCLAW_YTDLP_COOKIES_B64 (base64 of a Netscape cookies.txt file)."
            )
            raise RuntimeError(
                "Reference download blocked (Instagram: unavailable for certain audiences). "
                + secret_hint
                + " If the reel is viewable when logged-in, the cookies account must be able to view it; otherwise use Reference → Upload."
            ) from exc
        raise

    candidates = [p for p in sorted(output_dir.glob("reel_source.*")) if p.is_file()]
    candidates = [p for p in candidates if not p.name.endswith(".part")]
    candidates = [p for p in candidates if p.suffix.lower() not in {".json", ".srt", ".vtt", ".ytdl"}]

    videos = [p for p in candidates if p.suffix.lower() in _VIDEO_EXTS]
    if videos:
        return max(videos, key=lambda p: p.stat().st_size)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size)
    raise RuntimeError("yt-dlp completed but no video output file was found")


def compress_for_analysis(src: Path, *, dst: Path, max_seconds: int = 40, timeout_s: float = 180.0) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for reel analysis. Please install ffmpeg and try again.")

    raw_max = os.getenv("REEL_ANALYSIS_MAX_SECONDS", "").strip()
    eff_max: int | None = int(max_seconds)
    if raw_max:
        try:
            v = int(float(raw_max))
            eff_max = None if v <= 0 else v
        except Exception:
            eff_max = int(max_seconds)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src)]
    if eff_max is not None:
        cmd += ["-t", str(int(eff_max))]
    cmd += [
        "-vf",
        "scale=720:-2:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(dst),
    ]
    _run(cmd, timeout_s=timeout_s)
