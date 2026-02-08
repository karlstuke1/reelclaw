from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _redact_proxy_url(proxy_url: str) -> str:
    s = str(proxy_url or "").strip()
    if not s:
        return ""
    try:
        parsed = urlparse(s)
    except Exception:
        parsed = None  # type: ignore[assignment]
    if not parsed or not getattr(parsed, "hostname", None):
        # Best-effort fallback: hide userinfo if present.
        if "@" in s and "//" in s:
            return s.split("//", 1)[0] + "//" + s.split("@", 1)[-1]
        return s

    if not (getattr(parsed, "username", None) or getattr(parsed, "password", None)):
        return s

    host = str(parsed.hostname or "")
    port = f":{parsed.port}" if getattr(parsed, "port", None) else ""
    user = str(parsed.username or "")
    if user:
        netloc = f"{user}:***@{host}{port}"
    else:
        netloc = f"{host}{port}"
    # Preserve scheme/path/query fragments while redacting credentials.
    return parsed._replace(netloc=netloc).geturl()  # type: ignore[attr-defined]


def _redact_cmd(cmd: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        part = cmd[i]
        if part == "--proxy":
            out.append(part)
            if i + 1 < len(cmd):
                out.append(_redact_proxy_url(cmd[i + 1]))
                i += 2
                continue
        if isinstance(part, str) and part.startswith("--proxy="):
            out.append("--proxy=" + _redact_proxy_url(part[len("--proxy=") :]))
            i += 1
            continue
        out.append(part)
        i += 1
    return out


def _aws_region() -> str:
    return os.getenv("REELCLAW_AWS_REGION", "").strip() or os.getenv("AWS_REGION", "").strip() or "us-east-1"


def _read_secret_string(secret_id: str) -> str:
    import boto3  # type: ignore

    sm = boto3.client("secretsmanager", region_name=_aws_region())
    resp = sm.get_secret_value(SecretId=secret_id)
    return str(resp.get("SecretString") or "").strip()


def _resolve_oxylabs_proxy_url() -> str:
    """
    Build an Oxylabs proxy URL from environment variables (compatible with /Users/work/Desktop/instauto).

    Supported env vars:
      - OXYLABS_PROXY_PROTOCOL (default: http)
      - OXYLABS_PROXY_HOST (default: pr.oxylabs.io)
      - OXYLABS_PROXY_PORT (default: 7777)
      - OXYLABS_PROXY_USERNAME
      - OXYLABS_PROXY_USERNAME_TEMPLATE (supports {accountId}, {jobId}, {username})
      - OXYLABS_PROXY_PASSWORD
    """
    password = os.getenv("OXYLABS_PROXY_PASSWORD", "").strip()
    if not password:
        return ""

    username = os.getenv("OXYLABS_PROXY_USERNAME", "").strip()
    template = os.getenv("OXYLABS_PROXY_USERNAME_TEMPLATE", "").strip()
    if template:
        account_id = os.getenv("REELCLAW_USER_ID", "").strip() or os.getenv("REELCLAW_JOB_ID", "").strip()
        job_id = os.getenv("REELCLAW_JOB_ID", "").strip()
        user_id = os.getenv("REELCLAW_USER_ID", "").strip()
        rendered = (
            template.replace("{accountId}", account_id)
            .replace("{jobId}", job_id)
            .replace("{username}", user_id)
        )
        # If unknown placeholders remain, do not guess.
        if "{" not in rendered and "}" not in rendered:
            username = rendered.strip()

    if not username:
        return ""

    protocol = os.getenv("OXYLABS_PROXY_PROTOCOL", "").strip() or "http"
    host = os.getenv("OXYLABS_PROXY_HOST", "").strip() or "pr.oxylabs.io"
    port = os.getenv("OXYLABS_PROXY_PORT", "").strip() or "7777"
    if not host or not port:
        return ""
    return f"{protocol}://{username}:{password}@{host}:{port}"


def _is_valid_proxy_url(url: str) -> bool:
    """Reject placeholder/malformed proxy URLs like ``http://:@:``."""
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    host = (parsed.hostname or "").strip()
    return bool(host)


def _resolve_proxy_url(*, required: bool) -> str:
    """
    Resolve a proxy URL to use with yt-dlp.

    Supported env vars:
      - REELCLAW_YTDLP_PROXY / YTDLP_PROXY: explicit proxy URL (ex: http://user:pass@host:port)
      - REELCLAW_YTDLP_PROXY_SECRET_ID / YTDLP_PROXY_SECRET_ID: Secrets Manager id holding proxy URL
      - DEFAULT_PROXY_URL / IG_PROXY: common proxy vars (supported by instauto)
      - OXYLABS_PROXY_*: build a proxy URL from Oxylabs creds (supported by instauto)
    """
    direct = os.getenv("REELCLAW_YTDLP_PROXY", "").strip() or os.getenv("YTDLP_PROXY", "").strip()
    if direct and _is_valid_proxy_url(direct):
        return direct

    secret_id = os.getenv("REELCLAW_YTDLP_PROXY_SECRET_ID", "").strip() or os.getenv("YTDLP_PROXY_SECRET_ID", "").strip()
    if secret_id:
        try:
            value = _read_secret_string(secret_id)
            if value and _is_valid_proxy_url(value):
                return value
            if required:
                raise RuntimeError(f"Proxy secret is empty or malformed: {secret_id}. Set it to a proxy URL string.")
        except Exception as exc:
            if required:
                exc_name = type(exc).__name__
                if exc_name == "ResourceNotFoundException":
                    raise RuntimeError(
                        f"Proxy secret has no value set (AWSCURRENT missing): {secret_id}. Set it to a proxy URL string."
                    ) from exc
                if exc_name == "AccessDeniedException":
                    raise RuntimeError(
                        f"Access denied reading proxy secret {secret_id}. Verify Batch IAM role has secretsmanager:GetSecretValue."
                    ) from exc
                raise RuntimeError(
                    f"Failed to read proxy secret {secret_id}. Verify Batch role permissions and secret value. ({exc_name})"
                ) from exc
    # Common names used by other repos/tools.
    common = os.getenv("DEFAULT_PROXY_URL", "").strip() or os.getenv("IG_PROXY", "").strip()
    if common and _is_valid_proxy_url(common):
        return common
    result = _resolve_oxylabs_proxy_url()
    if result and not _is_valid_proxy_url(result):
        return ""
    return result


def _run(cmd: list[str], *, timeout_s: float) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        tail = stderr or stdout or "unknown error"
        if len(tail) > 2000:
            tail = tail[-2000:]
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(_redact_cmd(cmd))}\n{tail}")


def _is_instagram_url(url: str) -> bool:
    try:
        host = (urlparse(str(url)).hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    return host == "instagram.com" or host.endswith(".instagram.com")


def _cookies_file_has_name(path: Path, cookie_name: str) -> bool:
    """Best-effort check for a cookie name in a Netscape cookies.txt file."""
    name = str(cookie_name or "").strip()
    if not name:
        return False
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    # Netscape cookie format is tab-separated; cookie name is the 6th field.
    needle = "\t" + name.lower() + "\t"
    return needle in raw.lower()


def _instagram_cookies_look_logged_in(cookies_path: Path) -> bool:
    """
    Heuristic: if the cookies.txt includes `sessionid`, it's very likely authenticated.

    From AWS IP ranges, Instagram often requires a logged-in session for public reels.
    """
    return _cookies_file_has_name(cookies_path, "sessionid")


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

                sm = boto3.client("secretsmanager", region_name=_aws_region())
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


def _resolve_apify_token() -> str:
    """Resolve Apify API token from env or Secrets Manager."""
    direct = os.getenv("REELCLAW_APIFY_TOKEN", "").strip() or os.getenv("APIFY_TOKEN", "").strip()
    if direct:
        return direct
    secret_id = os.getenv("REELCLAW_APIFY_TOKEN_SECRET_ID", "").strip()
    if secret_id:
        try:
            return _read_secret_string(secret_id)
        except Exception:
            return ""
    return ""


def _download_via_apify(reel_url: str, *, output_dir: Path, timeout_s: float = 300.0) -> Path | None:
    """
    Download an Instagram reel via Apify's Instagram Reel Scraper API.
    Returns the downloaded video path, or None if Apify is not configured or fails.
    """
    token = _resolve_apify_token()
    if not token:
        return None

    actor_id = "apify~instagram-reel-scraper"
    api_url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    body = json.dumps({
        "username": ["instagram"],  # required by schema; directUrls is what matters
        "directUrls": [reel_url],
        "resultsLimit": 1,
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[apify] API call failed: {type(exc).__name__}: {exc}")
        return None

    if not items or not isinstance(items, list):
        print("[apify] No results returned")
        return None

    video_url = items[0].get("videoUrl") or ""
    if not video_url:
        print(f"[apify] No videoUrl in result (keys: {list(items[0].keys())[:10]})")
        return None

    # Download the video file
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "reel_source.mp4"
    try:
        vid_req = urllib.request.Request(video_url)
        with urllib.request.urlopen(vid_req, timeout=120) as vid_resp:
            with open(dst, "wb") as f:
                while True:
                    chunk = vid_resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as exc:
        print(f"[apify] Video download failed: {type(exc).__name__}: {exc}")
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    if not dst.exists() or dst.stat().st_size < 1000:
        print(f"[apify] Downloaded file too small or missing: {dst}")
        return None

    print(f"[apify] Downloaded reel to {dst} ({dst.stat().st_size} bytes)")
    return dst


def download_reel(url: str, *, output_dir: Path, timeout_s: float = 180.0) -> Path:
    # For Instagram URLs, try Apify first (works from any IP, no cookies needed).
    if _is_instagram_url(url):
        apify_result = _download_via_apify(url, output_dir=output_dir, timeout_s=min(timeout_s, 300.0))
        if apify_result is not None:
            return apify_result
        print("[download] Apify fallback: trying yt-dlp...")

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

    # Routing through a proxy can help reliability for some sources (notably Instagram) from
    # cloud IP ranges. This is best-effort by default.
    proxy_required = _is_instagram_url(url) and (
        _truthy_env("REELCLAW_YTDLP_PROXY_REQUIRED", "0") or _truthy_env("YTDLP_PROXY_REQUIRED", "0")
    )
    proxy_url = _resolve_proxy_url(required=proxy_required)

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
    if proxy_url:
        cmd += ["--proxy", proxy_url]
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
            proxy_hint = ""
            if not proxy_url and _is_instagram_url(url):
                proxy_hint = (
                    " If this keeps happening from AWS IP ranges, you may need a residential proxy "
                    "(set REELCLAW_YTDLP_PROXY or REELCLAW_YTDLP_PROXY_SECRET_ID)."
                )

            if cookies_file is None:
                secret_hint = (
                    f"Set AWS Secrets Manager `{secret_id}` to a Netscape cookies.txt (raw or base64)."
                    if secret_id
                    else "Set REELCLAW_YTDLP_COOKIES_B64 (base64 of a Netscape cookies.txt file)."
                )
                raise RuntimeError(
                    "Reference download blocked (login required). "
                    + secret_hint
                    + " This enables Instagram/YouTube downloads."
                    + proxy_hint
                ) from exc

            # Cookies were provided but Instagram still blocked. Give a more actionable message.
            if _is_instagram_url(url) and not _instagram_cookies_look_logged_in(cookies_file):
                secret_hint2 = (
                    f"Cookies were provided, but they do not include an authenticated Instagram session (`sessionid`). "
                    f"Update `{secret_id}` with cookies exported from a logged-in Instagram account (must include `sessionid`)."
                    if secret_id
                    else "Cookies were provided, but they do not include an authenticated Instagram session (`sessionid`). Provide logged-in cookies."
                )
                raise RuntimeError(
                    "Reference download blocked (Instagram login required). "
                    + secret_hint2
                    + " Alternatively, use Reference → Upload instead of a link."
                    + proxy_hint
                ) from exc

            secret_hint3 = (
                f"Cookies were provided via `{secret_id}`, but Instagram still blocked the download. "
                "Cookies may be expired or the content may be restricted for that account."
                if secret_id
                else "Cookies were provided, but Instagram still blocked the download."
            )
            raise RuntimeError(
                "Reference download blocked (login required). " + secret_hint3 + proxy_hint
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
