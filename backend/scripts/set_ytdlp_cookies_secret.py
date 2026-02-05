from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


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

    size = len(raw.encode("utf-8", errors="replace"))
    print(f"updated secret {secret_id} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

