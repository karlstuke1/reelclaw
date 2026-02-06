from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

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


def _s3_io_workers(task_count: int) -> int:
    if task_count <= 1:
        return 1
    raw = (_env("REELCLAW_S3_MAX_WORKERS") or "").strip()
    try:
        configured = int(float(raw)) if raw else 4
    except Exception:
        configured = 4
    return max(1, min(configured, 16, task_count))


def _unique_filenames(names: list[str]) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for name in names:
        safe = _safe_slug(name)
        if safe not in used:
            used.add(safe)
            out.append(safe)
            continue
        p = Path(safe)
        stem = p.stem or "file"
        suffix = p.suffix
        n = 2
        while True:
            cand = f"{stem}_{n}{suffix}"
            if cand not in used:
                used.add(cand)
                out.append(cand)
                break
            n += 1
    return out


def _count_variants(finals_dir: Path) -> int:
    try:
        return len(list(finals_dir.glob("v*.mov"))) + len(list(finals_dir.glob("v*.mp4")))
    except Exception:
        return 0


def _apply_pro_defaults(env: dict[str, str]) -> None:
    """
    Keep iOS/prod pipeline in line with the legacy pro-mode scripts by turning on the
    higher-quality "editor brain" knobs, but without overriding explicit env.
    """
    env.setdefault("FOLDER_EDIT_BEAT_SYNC", "1")
    env.setdefault("FOLDER_EDIT_STORY_PLANNER", "1")
    env.setdefault("SHOT_INDEX_MODE", "scene")
    env.setdefault("SHOT_INDEX_WORKERS", "4")
    # Shot-level tagging is a key part of the pro pipeline "editor brain".
    # Cap with SHOT_TAG_MAX to keep local runs bounded on huge libraries.
    env.setdefault("SHOT_TAGGING", "1")
    env.setdefault("SHOT_TAG_MAX", "250")
    env.setdefault("REF_SEGMENT_FRAME_COUNT", "3")
    env.setdefault("REASONING_EFFORT", "high")
    # Directed quality lift: generate more internal candidates, fix worst segments on finalists.
    env.setdefault("VARIANT_FIX_ITERS", "1")
    env.setdefault("VARIANT_PRO_MACRO", "beam")


def _count_candidates(variants_dir: Path) -> int:
    """
    Count candidate renders in variants/ (independent of finals selection/copying).
    """
    try:
        return len(list(variants_dir.glob("v*/final_video.mov")))
    except Exception:
        return 0


def _internal_variant_count(*, finals: int, pro_mode: bool) -> int:
    raw = (_env("REELCLAW_INTERNAL_VARIANTS") or "").strip()
    try:
        configured = int(float(raw)) if raw else 0
    except Exception:
        configured = 0
    if configured > 0:
        return max(int(finals), int(configured))
    # Default: keep old behavior unless pro mode is enabled.
    return max(int(finals), 16) if pro_mode else int(finals)


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


def _load_final_scores(finals_dir: Path) -> dict[str, float]:
    """
    Best-effort: load finals_manifest.json and produce a per-final normalized score (0..10).
    This is UI-friendly (higher is better) and stable within a job.
    """
    manifest = finals_dir / "finals_manifest.json"
    if not manifest.exists():
        return {}
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8", errors="replace") or "{}")
    except Exception:
        return {}
    winners = doc.get("winners") or []
    raw_scores: dict[str, float] = {}
    if isinstance(winners, list):
        for w in winners:
            if not isinstance(w, dict):
                continue
            fid = str(w.get("final_id") or "").strip()
            rs = w.get("rank_score")
            if not fid or not isinstance(rs, (int, float)):
                continue
            raw_scores[fid] = float(rs)

    if not raw_scores:
        return {}
    vals = list(raw_scores.values())
    lo = min(vals)
    hi = max(vals)
    out: dict[str, float] = {}
    if abs(float(hi) - float(lo)) < 1e-9:
        for k in raw_scores.keys():
            out[k] = 10.0
        return out
    for k, v in raw_scores.items():
        out[k] = float(max(0.0, min(10.0, 10.0 * ((float(v) - float(lo)) / (float(hi) - float(lo))))))
    return out


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
    s3 = boto3.client("s3", region_name=env.region, config=Config(max_pool_connections=64))

    job = jobs.get(env.job_id)
    if not job:
        raise RuntimeError(f"Job not found: {env.job_id}")

    user_id = str(job.get("user_id") or env.user_id_hint or "").strip()
    if not user_id:
        raise RuntimeError("Job missing user_id")

    variations = max(1, int(job.get("variations") or 3))
    burn_overlays = bool(job.get("burn_overlays") or False)
    director = str(job.get("director") or "").strip().lower() or None
    if director not in {"code", "gemini", "auto"}:
        director = None

    pro_mode = _truthy_env("REELCLAW_PRO_MODE", "1")
    if director in {"gemini", "auto"}:
        pro_mode = True
    internal_variants = _internal_variant_count(finals=variations, pro_mode=pro_mode)
    if director in {"gemini", "auto"} and internal_variants > 12 and not _truthy_env("ALLOW_GEMINI_DIRECTOR_MANY", "0"):
        internal_variants = max(int(variations), 12)

    reference = job.get("reference") if isinstance(job.get("reference"), dict) else {}
    clips = job.get("clips") if isinstance(job.get("clips"), list) else []

    work_root = Path(_env("REELCLAW_WORK_DIR", "/tmp/reelclaw")).expanduser().resolve()
    job_root = work_root / env.job_id
    uploads_dir = job_root / "uploads"
    out_dir = job_root / "pipeline"
    finals_dir = out_dir / "finals"
    variants_dir = out_dir / "variants"
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
    clip_keys: list[str] = []
    clip_filenames: list[str] = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        key = str(c.get("s3_key") or "").strip()
        if not key:
            raise RuntimeError("Clip missing s3_key")
        filename = str(c.get("filename") or Path(key).name)
        clip_keys.append(key)
        clip_filenames.append(filename)

    clip_dst_filenames = _unique_filenames(clip_filenames)
    clip_tasks: list[tuple[str, Path]] = [
        (key, uploads_dir / dst_name) for key, dst_name in zip(clip_keys, clip_dst_filenames)
    ]

    # Acquire reference input (can download in parallel with clips).
    reel_arg = ""
    ref_task: tuple[str, Path] | None = None
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
        ref_task = (rkey, ref_path)
        reel_arg = str(ref_path)
    else:
        raise RuntimeError("Invalid reference type")

    if not reel_arg:
        raise RuntimeError("Missing reference reel input")

    def _download_one(key: str, dst: Path) -> None:
        s3.download_file(env.uploads_bucket, key, str(dst))

    download_tasks = clip_tasks + ([ref_task] if ref_task else [])
    if download_tasks:
        workers = _s3_io_workers(len(download_tasks))
        if workers <= 1:
            for key, dst in download_tasks:
                _download_one(key, dst)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_download_one, key, dst) for key, dst in download_tasks]
                for fut in as_completed(futures):
                    fut.result()

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
        str(internal_variants),
        "--finals",
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
    if pro_mode:
        cmd.append("--pro")
    if director:
        cmd.extend(["--director", str(director)])
    if burn_overlays:
        cmd.append("--burn-overlays")

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(backend_root) + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    if pro_mode:
        _apply_pro_defaults(run_env)
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
            cur = _count_candidates(variants_dir)
            if cur != last:
                last = cur
                scaled = 0
                try:
                    scaled = int(round((float(cur) / max(1.0, float(internal_variants))) * float(variations)))
                except Exception:
                    scaled = 0
                _update_job(
                    jobs,
                    env.job_id,
                    status="running",
                    stage=f"Exploring candidates ({min(cur, internal_variants)}/{internal_variants})",
                    message="This can take a few minutes depending on clip length.",
                    progress_current=min(max(0, scaled), variations),
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
                if (
                    "Reference download blocked" in s
                    or "REELCLAW_YTDLP_COOKIES_B64" in s
                    or "Cookies secret has no value set" in s
                    or "Cookies secret is empty" in s
                    or "Failed to read cookies secret" in s
                ):
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
    final_files = sorted(finals_dir.glob("v*.mov")) + sorted(finals_dir.glob("v*.mp4"))
    thumbs_dir = job_root / "thumbs"
    score_by_id = _load_final_scores(finals_dir)

    def _upload_variant(p: Path) -> tuple[str, str, str | None]:
        video_key = f"outputs/{user_id}/{env.job_id}/finals/{p.name}"
        s3.upload_file(str(p), env.outputs_bucket, video_key)

        thumb_key: str | None = None
        try:
            thumb_path = thumbs_dir / f"{p.stem}.jpg"
            _ffmpeg_thumb(p, thumb_path)
            thumb_key = f"outputs/{user_id}/{env.job_id}/thumbs/{thumb_path.name}"
            s3.upload_file(str(thumb_path), env.outputs_bucket, thumb_key)
        except Exception:
            thumb_key = None

        return p.name, video_key, thumb_key

    upload_results: dict[str, tuple[str, str | None]] = {}
    if final_files:
        workers = _s3_io_workers(len(final_files))
        if workers <= 1:
            for p in final_files:
                name, video_key, thumb_key = _upload_variant(p)
                upload_results[name] = (video_key, thumb_key)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_upload_variant, p) for p in final_files]
                for fut in as_completed(futures):
                    name, video_key, thumb_key = fut.result()
                    upload_results[name] = (video_key, thumb_key)

    variants: list[dict[str, Any]] = []
    for p in final_files:
        video_key, thumb_key = upload_results[p.name]
        vid = p.stem
        variants.append(
            {
                "id": vid,
                "title": f"Variation {vid.lstrip('v')}",
                "score": score_by_id.get(vid),
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
