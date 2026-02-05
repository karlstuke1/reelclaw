from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import jwt
import requests


VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


class E2EError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


def _req(name: str) -> str:
    v = _env(name)
    if not v:
        raise E2EError(f"Missing required env var: {name}")
    return v


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mp4":
        return "video/mp4"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".m4v":
        return "video/x-m4v"
    return "application/octet-stream"


def _read_secret(*, region: str, secret_id: str) -> str:
    sm = boto3.client("secretsmanager", region_name=region)
    resp = sm.get_secret_value(SecretId=secret_id)
    s = (resp.get("SecretString") or "").strip()
    if not s:
        raise E2EError(f"Empty secret value for {secret_id}")
    return s


def _issue_jwt(*, jwt_secret: str, user_id: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iss": "reelclaw",
        "iat": now,
        "exp": now + int(ttl_seconds),
    }
    return str(jwt.encode(payload, jwt_secret, algorithm="HS256"))


@dataclass(frozen=True)
class JobCreated:
    job_id: str
    reference_upload_url: str | None
    clip_upload_urls: list[str]


def _api_json(*, method: str, url: str, token: str, body: dict[str, Any] | None = None) -> Any:
    headers = {
        "Accept": "application/json",
        # API Gateway HTTP APIs may drop/strip headers containing "Authorization" before reaching ALB/ECS.
        "X-Reelclaw-Token": token,
        "Authorization": f"Bearer {token}",
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    resp = requests.request(method, url, headers=headers, data=data, timeout=60)
    if resp.status_code < 200 or resp.status_code >= 300:
        msg = (resp.text or "").strip()
        raise E2EError(f"{method} {url} -> HTTP {resp.status_code}: {msg[:800]}")
    if not resp.content:
        return None
    try:
        return resp.json()
    except Exception as e:
        raise E2EError(f"Failed to decode JSON from {url}: {type(e).__name__}: {resp.text[:800]}")


def _upload_put(*, upload_url: str, file_path: Path, content_type: str) -> None:
    with file_path.open("rb") as f:
        resp = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=600)
    if resp.status_code < 200 or resp.status_code >= 300:
        raise E2EError(f"PUT upload failed -> HTTP {resp.status_code}: {resp.text[:800]}")


def _create_job(
    *,
    api_base_url: str,
    token: str,
    reference_url: str | None,
    reference_upload: Path | None,
    clips: list[Path],
    variations: int,
    burn_overlays: bool,
) -> JobCreated:
    if (reference_url is None) == (reference_upload is None):
        raise E2EError("Provide exactly one of: --reference-url OR --reference-upload")
    if not clips:
        raise E2EError("No clip files found.")

    if reference_url is not None:
        ref: dict[str, Any] = {"type": "url", "url": reference_url}
    else:
        assert reference_upload is not None
        if not reference_upload.exists():
            raise E2EError(f"Missing reference file: {reference_upload}")
        ref = {
            "type": "upload",
            "filename": reference_upload.name,
            "content_type": _guess_content_type(reference_upload),
            "bytes": int(reference_upload.stat().st_size),
        }

    clip_specs: list[dict[str, Any]] = []
    for p in clips:
        clip_specs.append(
            {
                "filename": p.name,
                "content_type": _guess_content_type(p),
                "bytes": int(p.stat().st_size),
            }
        )

    body: dict[str, Any] = {
        "reference": ref,
        "variations": int(variations),
        "burn_overlays": bool(burn_overlays),
        "clips": clip_specs,
    }
    url = api_base_url.rstrip("/") + "/v1/jobs"
    doc = _api_json(method="POST", url=url, token=token, body=body)
    if not isinstance(doc, dict):
        raise E2EError("Unexpected /v1/jobs response (not an object)")
    job_id = str(doc.get("job_id") or "").strip()
    if not job_id:
        raise E2EError("Missing job_id in /v1/jobs response")
    ref_upload = doc.get("reference_upload")
    reference_upload_url: str | None = None
    if isinstance(ref_upload, dict) and isinstance(ref_upload.get("upload_url"), str):
        reference_upload_url = str(ref_upload["upload_url"])
    clip_uploads = doc.get("clip_uploads")
    if not isinstance(clip_uploads, list) or len(clip_uploads) != len(clips):
        raise E2EError(f"clip_uploads mismatch (expected {len(clips)} got {len(clip_uploads) if isinstance(clip_uploads, list) else 'non-list'})")
    upload_urls: list[str] = []
    for item in clip_uploads:
        if not isinstance(item, dict) or not isinstance(item.get("upload_url"), str):
            raise E2EError("Invalid clip_uploads item (missing upload_url)")
        upload_urls.append(str(item["upload_url"]))
    return JobCreated(job_id=job_id, reference_upload_url=reference_upload_url, clip_upload_urls=upload_urls)


def _start_job(*, api_base_url: str, token: str, job_id: str) -> None:
    url = api_base_url.rstrip("/") + f"/v1/jobs/{job_id}/start"
    _api_json(method="POST", url=url, token=token, body={})


def _get_job(*, api_base_url: str, token: str, job_id: str) -> dict[str, Any]:
    url = api_base_url.rstrip("/") + f"/v1/jobs/{job_id}"
    doc = _api_json(method="GET", url=url, token=token)
    if not isinstance(doc, dict):
        raise E2EError("Unexpected /v1/jobs/{id} response (not an object)")
    return doc


def _get_variants(*, api_base_url: str, token: str, job_id: str) -> dict[str, Any]:
    url = api_base_url.rstrip("/") + f"/v1/jobs/{job_id}/variants"
    doc = _api_json(method="GET", url=url, token=token)
    if not isinstance(doc, dict):
        raise E2EError("Unexpected /v1/jobs/{id}/variants response (not an object)")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E: mimic the iOS -> AWS flow (create job, presigned uploads, start, poll).")
    ap.add_argument("--api-base-url", default=_env("API_BASE_URL", "https://y8uexiqu08.execute-api.us-east-1.amazonaws.com"))
    ap.add_argument("--region", default=_env("AWS_REGION", "us-east-1"))
    ap.add_argument("--env", choices=["staging", "prod"], default="prod", help="Used to locate Secrets Manager ids.")
    ap.add_argument("--user-id", default="e2e-user", help="JWT sub claim used for the test job.")
    ap.add_argument("--variations", type=int, default=1)
    ap.add_argument("--burn-overlays", action="store_true")
    ap.add_argument("--clips-dir", type=str, required=True)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all clips in the folder.")
    ap.add_argument("--reference-url", type=str, default=None)
    ap.add_argument("--reference-upload", type=str, default=None)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = ap.parse_args()

    api_base_url = str(args.api_base_url).strip().rstrip("/")
    region = str(args.region).strip() or "us-east-1"

    jwt_secret = _env("REELCLAW_JWT_SECRET")
    if not jwt_secret:
        secret_id = _env("REELCLAW_JWT_SECRET_ID", f"reelclaw-{args.env}/jwt_secret")
        jwt_secret = _read_secret(region=region, secret_id=secret_id)

    token = _issue_jwt(jwt_secret=jwt_secret, user_id=str(args.user_id))

    clips_dir = Path(args.clips_dir).expanduser().resolve()
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise E2EError(f"Invalid clips dir: {clips_dir}")

    clips = [p for p in sorted(clips_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if int(args.max_clips or 0) > 0:
        clips = clips[: int(args.max_clips)]
    if not clips:
        raise E2EError(f"No clips found in {clips_dir} (expected one of: {sorted(VIDEO_EXTS)})")

    reference_upload: Path | None = None
    if args.reference_upload:
        reference_upload = Path(args.reference_upload).expanduser().resolve()

    job = _create_job(
        api_base_url=api_base_url,
        token=token,
        reference_url=(str(args.reference_url).strip() if args.reference_url else None),
        reference_upload=reference_upload,
        clips=clips,
        variations=int(args.variations),
        burn_overlays=bool(args.burn_overlays),
    )
    print("job_id:", job.job_id)

    # Upload reference (if needed).
    if job.reference_upload_url:
        if reference_upload is None:
            raise E2EError("Server requested reference upload but no --reference-upload was provided.")
        _upload_put(
            upload_url=job.reference_upload_url,
            file_path=reference_upload,
            content_type=_guess_content_type(reference_upload),
        )
        print("uploaded reference:", reference_upload.name)

    # Upload clips (in request order).
    for p, upload_url in zip(clips, job.clip_upload_urls, strict=True):
        _upload_put(upload_url=upload_url, file_path=p, content_type=_guess_content_type(p))
        print("uploaded clip:", p.name)

    _start_job(api_base_url=api_base_url, token=token, job_id=job.job_id)
    print("started job")

    deadline = time.time() + float(args.timeout_seconds)
    last_line = ""
    while time.time() < deadline:
        j = _get_job(api_base_url=api_base_url, token=token, job_id=job.job_id)
        status = str(j.get("status") or "")
        stage = str(j.get("stage") or "")
        msg = str(j.get("message") or "")
        cur = j.get("progress_current")
        tot = j.get("progress_total")
        line = f"{status} | {stage} | {cur}/{tot} | {msg}".strip()
        if line != last_line:
            print(line)
            last_line = line
        if status in {"succeeded", "failed"}:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))

    j = _get_job(api_base_url=api_base_url, token=token, job_id=job.job_id)
    status = str(j.get("status") or "")
    if status != "succeeded":
        raise E2EError(f"Job ended with status={status}: {j}")

    variants = _get_variants(api_base_url=api_base_url, token=token, job_id=job.job_id)
    print(json.dumps(variants, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
