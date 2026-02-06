from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    dev_auth: bool
    director_model: str
    reference_analysis_max_seconds: int | None
    aws_mode: bool
    aws_region: str
    uploads_bucket: str | None
    outputs_bucket: str | None
    jobs_table: str | None
    devices_table: str | None
    batch_job_queue: str | None
    batch_job_definition: str | None
    jwt_secret: str | None
    apple_audience: str | None
    enable_apns: bool
    sns_platform_application_arn: str | None
    max_clips: int
    max_clip_bytes: int
    upload_url_ttl_seconds: int
    download_url_ttl_seconds: int
    uploading_job_cleanup_seconds: int


def load_settings() -> Settings:
    data_dir = Path(os.getenv("REELCLAW_DATA_DIR", "backend/data")).expanduser().resolve()
    dev_auth = _truthy_env("REELCLAW_DEV_AUTH", "0")
    director_model = os.getenv("REELCLAW_DIRECTOR_MODEL", os.getenv("DIRECTOR_MODEL", "google/gemini-3-pro-preview"))

    # If set to 0 (or negative), use full reel length for analysis (no cap).
    raw_max = os.getenv("REELCLAW_REFERENCE_ANALYSIS_MAX_SECONDS", "").strip()
    reference_analysis_max_seconds: int | None = None
    if raw_max:
        try:
            v = int(float(raw_max))
            reference_analysis_max_seconds = None if v <= 0 else v
        except Exception:
            reference_analysis_max_seconds = None

    aws_mode = _truthy_env("REELCLAW_AWS_MODE", "0")
    aws_region = os.getenv("REELCLAW_AWS_REGION", os.getenv("AWS_REGION", "")).strip() or "us-east-1"

    uploads_bucket = os.getenv("REELCLAW_UPLOADS_BUCKET", "").strip() or None
    outputs_bucket = os.getenv("REELCLAW_OUTPUTS_BUCKET", "").strip() or None
    jobs_table = os.getenv("REELCLAW_JOBS_TABLE", "").strip() or None
    devices_table = os.getenv("REELCLAW_DEVICES_TABLE", "").strip() or None
    batch_job_queue = os.getenv("REELCLAW_BATCH_JOB_QUEUE", "").strip() or None
    batch_job_definition = os.getenv("REELCLAW_BATCH_JOB_DEFINITION", "").strip() or None

    jwt_secret = os.getenv("REELCLAW_JWT_SECRET", "").strip() or None
    apple_audience = os.getenv("REELCLAW_APPLE_AUDIENCE", "").strip() or None

    enable_apns = _truthy_env("REELCLAW_ENABLE_APNS", "0")
    sns_platform_application_arn = os.getenv("REELCLAW_SNS_PLATFORM_APPLICATION_ARN", "").strip() or None

    max_clips = max(1, _int_env("REELCLAW_MAX_CLIPS", 30))
    max_clip_bytes = max(1, _int_env("REELCLAW_MAX_CLIP_BYTES", 250 * 1024 * 1024))
    upload_url_ttl_seconds = max(60, _int_env("REELCLAW_UPLOAD_URL_TTL_SECONDS", 900))
    download_url_ttl_seconds = max(60, _int_env("REELCLAW_DOWNLOAD_URL_TTL_SECONDS", 900))
    # Stuck uploads can create lots of noise in the UI. Clean them up eventually.
    # Default matches the per-job uploads S3 lifecycle (uploads/ expires after ~7d).
    uploading_job_cleanup_seconds = max(0, _int_env("REELCLAW_UPLOADING_JOB_CLEANUP_SECONDS", 7 * 24 * 60 * 60))

    return Settings(
        data_dir=data_dir,
        dev_auth=dev_auth,
        director_model=director_model,
        reference_analysis_max_seconds=reference_analysis_max_seconds,
        aws_mode=aws_mode,
        aws_region=aws_region,
        uploads_bucket=uploads_bucket,
        outputs_bucket=outputs_bucket,
        jobs_table=jobs_table,
        devices_table=devices_table,
        batch_job_queue=batch_job_queue,
        batch_job_definition=batch_job_definition,
        jwt_secret=jwt_secret,
        apple_audience=apple_audience,
        enable_apns=enable_apns,
        sns_platform_application_arn=sns_platform_application_arn,
        max_clips=max_clips,
        max_clip_bytes=max_clip_bytes,
        upload_url_ttl_seconds=upload_url_ttl_seconds,
        download_url_ttl_seconds=download_url_ttl_seconds,
        uploading_job_cleanup_seconds=uploading_job_cleanup_seconds,
    )
