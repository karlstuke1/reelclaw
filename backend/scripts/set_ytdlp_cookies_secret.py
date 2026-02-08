from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


def _filter_netscape_cookies(raw: str, *, domain_substrings: list[str]) -> tuple[str, int]:
    """
    Filter a Netscape cookies.txt payload to just the domains we care about.

    Secrets Manager SecretString has a 64KiB limit; full-browser exports can exceed this.
    """
    want = [d.strip().lower() for d in domain_substrings if d.strip()]
    if not want:
        return raw, 0

    out_lines: list[str] = []
    kept = 0
    for ln in (raw or "").splitlines():
        if not ln.strip():
            continue
        if ln.startswith("#"):
            out_lines.append(ln)
            continue
        parts = ln.split("\t")
        domain = (parts[0] if parts else "").strip().lower()
        if domain and any(d in domain for d in want):
            out_lines.append(ln)
            kept += 1
    return "\n".join(out_lines).strip() + "\n", kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Update the AWS Secrets Manager cookie secret used by yt-dlp in the Batch worker.\n\n"
            "This expects a Netscape-format cookies.txt file (commonly exported via a browser extension).\n"
            "It stores the raw file contents as SecretString (do not commit the file to git)."
        )
    )
    ap.add_argument("--env", choices=["prod", "staging"], default="prod")
    ap.add_argument("--region", default=_env("AWS_REGION", "us-east-1"))
    ap.add_argument("--cookies-file", required=True, help="Path to a Netscape cookies.txt file.")
    ap.add_argument(
        "--secret-id",
        default=None,
        help="Override secret id/name (default: reelclaw-<env>/ytdlp_cookies).",
    )
    args = ap.parse_args()

    region = str(args.region).strip() or "us-east-1"
    env_name = str(args.env).strip()
    secret_id = str(args.secret_id).strip() if args.secret_id else f"reelclaw-{env_name}/ytdlp_cookies"

    cookies_path = Path(str(args.cookies_file)).expanduser().resolve()
    if not cookies_path.exists() or not cookies_path.is_file():
        raise SystemExit(f"cookies file not found: {cookies_path}")

    raw = cookies_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        raise SystemExit("cookies file is empty")

    max_bytes = 65536
    raw_bytes = len(raw.encode("utf-8", errors="replace"))
    if raw_bytes > max_bytes:
        filtered, kept = _filter_netscape_cookies(
            raw,
            domain_substrings=[
                "instagram.com",
                "facebook.com",
                "youtube.com",
                "google.com",
            ],
        )
        filtered_bytes = len(filtered.encode("utf-8", errors="replace"))
        print(
            f"warning: cookies export is {raw_bytes} bytes (Secrets Manager limit {max_bytes}). "
            f"Filtered to IG/YT domains: kept {kept} cookie lines, now {filtered_bytes} bytes."
        )
        raw = filtered
        raw_bytes = filtered_bytes
        if raw_bytes > max_bytes:
            raise SystemExit(
                f"filtered cookies are still too large ({raw_bytes} bytes). "
                "Export fewer cookies (only IG/YT) or filter the Netscape cookies.txt before uploading."
            )

    # Gentle sanity check (non-blocking): most valid files include at least one domain line.
    if "instagram.com" not in raw and "youtube.com" not in raw and "google.com" not in raw:
        print("warning: cookies file does not mention instagram.com / youtube.com; verify it is the correct export.")

    sm = boto3.client("secretsmanager", region_name=region)
    try:
        sm.put_secret_value(SecretId=secret_id, SecretString=raw)
    except sm.exceptions.ResourceNotFoundException:
        raise SystemExit(
            f"secret not found: {secret_id} (did you apply Terraform for {env_name}?)"
        ) from None

    print(f"updated secret {secret_id} ({raw_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
