from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3

from reelclaw_backend.aws_services import sns_publish_apns
from reelclaw_backend.aws_store import DynamoDeviceStore, DynamoJobStore


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


def _req(name: str) -> str:
    v = _env(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _safe_slug(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120] or "file"


def _count_variants(finals_dir: Path) -> int:
    try:
        return len(list(finals_dir.glob("v*.mov"))) + len(list(finals_dir.glob("v*.mp4")))
    except Exception:
        return 0


def _ffmpeg_thumb(video_path: Path, out_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "thumbnail,scale=360:-2",
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _tail_text(path: Path, *, max_chars: int = 2400) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if not raw:
        return ""
    raw = raw.strip()
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


@dataclass(frozen=True)
class WorkerEnv:
    region: str
    uploads_bucket: str
    outputs_bucket: str
    jobs_table: str
    devices_table: str
    job_id: str
    user_id_hint: str | None
    enable_apns: bool
    reference_analysis_max_seconds: int | None


def _load_env() -> WorkerEnv:
    region = _env("REELCLAW_AWS_REGION", _env("AWS_REGION", "us-east-1")) or "us-east-1"
    raw_max = (_env("REELCLAW_REFERENCE_ANALYSIS_MAX_SECONDS") or "").strip()
    max_s: int | None = None
    if raw_max:
        try:
            v = int(float(raw_max))
            max_s = None if v <= 0 else v
        except Exception:
            max_s = None

    return WorkerEnv(
        region=region,
        uploads_bucket=_req("REELCLAW_UPLOADS_BUCKET"),
        outputs_bucket=_req("REELCLAW_OUTPUTS_BUCKET"),
        jobs_table=_req("REELCLAW_JOBS_TABLE"),
        devices_table=_req("REELCLAW_DEVICES_TABLE"),
        job_id=_req("REELCLAW_JOB_ID"),
        user_id_hint=_env("REELCLAW_USER_ID"),
        enable_apns=_truthy_env("REELCLAW_ENABLE_APNS", "0"),
        reference_analysis_max_seconds=max_s,
    )


def _update_job(jobs: DynamoJobStore, job_id: str, **fields: Any) -> dict[str, Any]:
    return jobs.update(job_id, **fields)


def _download_s3_dir(*, s3, bucket: str, prefix: str, dst_dir: Path) -> None:
    """
    Download all objects under prefix to dst_dir, preserving filenames.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            filename = Path(key).name
            dst = dst_dir / _safe_slug(filename)
            s3.download_file(bucket, key, str(dst))


def main() -> int:
    env = _load_env()
    jobs = DynamoJobStore(table_name=env.jobs_table, region=env.region)
    devices = DynamoDeviceStore(table_name=env.devices_table, region=env.region)
    s3 = boto3.client("s3", region_name=env.region)

    job = jobs.get(env.job_id)
    if not job:
        raise RuntimeError(f"Job not found: {env.job_id}")

    user_id = str(job.get("user_id") or env.user_id_hint or "").strip()
    if not user_id:
        raise RuntimeError("Job missing user_id")

    variations = max(1, int(job.get("variations") or 3))
    burn_overlays = bool(job.get("burn_overlays") or False)

    reference = job.get("reference") if isinstance(job.get("reference"), dict) else {}
    clips = job.get("clips") if isinstance(job.get("clips"), list) else []

    work_root = Path(_env("REELCLAW_WORK_DIR", "/tmp/reelclaw")).expanduser().resolve()
    job_root = work_root / env.job_id
    uploads_dir = job_root / "uploads"
    out_dir = job_root / "pipeline"
    finals_dir = out_dir / "finals"
    logs_dir = job_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "worker.log"

    _update_job(
        jobs,
        env.job_id,
        status="running",
        stage="Downloading",
        message="Downloading clips…",
        progress_current=0,
        progress_total=variations,
        error_code=None,
        error_detail=None,
    )

    # Download clips by their recorded S3 keys.
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for c in clips:
        if not isinstance(c, dict):
            continue
        key = str(c.get("s3_key") or "").strip()
        if not key:
            raise RuntimeError("Clip missing s3_key")
        filename = str(c.get("filename") or Path(key).name)
        dst = uploads_dir / _safe_slug(filename)
        s3.download_file(env.uploads_bucket, key, str(dst))

    # Acquire reference input.
    reel_arg = ""
    if str(reference.get("type") or "") == "url":
        reel_arg = str(reference.get("url") or "").strip()
    elif str(reference.get("type") or "") == "upload":
        rkey = str(reference.get("s3_key") or "").strip()
        if not rkey:
            raise RuntimeError("Reference upload missing s3_key")
        ref_dir = job_root / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        filename = str(reference.get("filename") or Path(rkey).name or "reference.mp4")
        ref_path = ref_dir / _safe_slug(filename)
        s3.download_file(env.uploads_bucket, rkey, str(ref_path))
        reel_arg = str(ref_path)
    else:
        raise RuntimeError("Invalid reference type")

    if not reel_arg:
        raise RuntimeError("Missing reference reel input")

    _update_job(
        jobs,
        env.job_id,
        status="running",
        stage="Starting",
        message="Booting up the editors…",
        progress_current=0,
        progress_total=variations,
    )

    # Run variant pipeline (reel replication only; no image-gen).
    backend_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "reelclaw_pipeline.run_folder_edit_variants",
        "--reel",
        reel_arg,
        "--folder",
        str(uploads_dir),
        "--variants",
        str(variations),
        "--out",
        str(out_dir),
        "--seed",
        "1337",
        "--model",
        str(_env("REELCLAW_DIRECTOR_MODEL", _env("DIRECTOR_MODEL", "google/gemini-3-pro-preview"))),
        "--timeout",
        "240",
    ]
    if burn_overlays:
        cmd.append("--burn-overlays")

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(backend_root) + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    if env.reference_analysis_max_seconds is None:
        run_env["REEL_ANALYSIS_MAX_SECONDS"] = "0"
    else:
        run_env["REEL_ANALYSIS_MAX_SECONDS"] = str(int(env.reference_analysis_max_seconds))

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=backend_root,
            env=run_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        last = -1
        while proc.poll() is None:
            cur = _count_variants(finals_dir)
            if cur != last:
                last = cur
                _update_job(
                    jobs,
                    env.job_id,
                    status="running",
                    stage=f"Rendering variations ({min(cur, variations)}/{variations})",
                    message="This can take a few minutes depending on clip length.",
                    progress_current=min(cur, variations),
                    progress_total=variations,
                )
            time.sleep(1.5)

        rc = int(proc.returncode or 0)
        produced = _count_variants(finals_dir)
        if rc != 0:
            tail = _tail_text(log_path)
            msg = f"Pipeline exited with code {rc}."
            # If the pipeline surfaced a human-friendly error, bubble it up.
            for line in reversed(tail.splitlines()):
                s = line.strip()
                if not s:
                    continue
                if "Reference download blocked" in s or "REELCLAW_YTDLP_COOKIES_B64" in s:
                    msg = s[:240]
                    break
            _update_job(
                jobs,
                env.job_id,
                status="failed",
                stage="Failed",
                message=msg,
                error_code=f"exit_{rc}",
                error_detail=tail or f"See worker log at {log_path.name}",
            )
            return 1
        if produced <= 0:
            tail = _tail_text(log_path)
            _update_job(
                jobs,
                env.job_id,
                status="failed",
                stage="No outputs",
                message="Pipeline finished but produced no variants.",
                error_code="no_outputs",
                error_detail=tail or f"See worker log at {log_path.name}",
            )
            return 1

    _update_job(
        jobs,
        env.job_id,
        status="running",
        stage="Uploading",
        message="Uploading outputs…",
        progress_current=min(produced, variations),
        progress_total=variations,
    )

    # Upload finals + thumbs.
    variants: list[dict[str, Any]] = []
    thumbs_dir = job_root / "thumbs"
    for p in sorted(finals_dir.glob("v*.mov")) + sorted(finals_dir.glob("v*.mp4")):
        vid = p.stem
        video_key = f"outputs/{user_id}/{env.job_id}/finals/{p.name}"
        s3.upload_file(str(p), env.outputs_bucket, video_key)

        thumb_key: str | None = None
        try:
            thumb_path = thumbs_dir / f"{vid}.jpg"
            _ffmpeg_thumb(p, thumb_path)
            thumb_key = f"outputs/{user_id}/{env.job_id}/thumbs/{thumb_path.name}"
            s3.upload_file(str(thumb_path), env.outputs_bucket, thumb_key)
        except Exception:
            thumb_key = None

        variants.append(
            {
                "id": vid,
                "title": f"Variation {vid.lstrip('v')}",
                "score": None,
                "video_s3_key": video_key,
                "thumb_s3_key": thumb_key,
            }
        )

    _update_job(
        jobs,
        env.job_id,
        status="succeeded",
        stage="Done",
        message="Your variations are ready.",
        variants=variants,
        progress_current=min(produced, variations),
        progress_total=variations,
        error_code=None,
        error_detail=None,
    )

    # Push notify (best-effort).
    if env.enable_apns:
        try:
            devs = devices.list_devices(user_id=user_id)
            for d in devs:
                if not isinstance(d, dict):
                    continue
                endpoint_arn = str(d.get("sns_endpoint_arn") or "").strip()
                if not endpoint_arn:
                    continue
                env_name = str(d.get("environment") or "").strip().lower()
                is_sandbox = env_name == "sandbox"
                sns_publish_apns(
                    region=env.region,
                    endpoint_arn=endpoint_arn,
                    title="ReelClaw",
                    body="Your edit is ready.",
                    job_id=env.job_id,
                    is_sandbox=is_sandbox,
                )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
