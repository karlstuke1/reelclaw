from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from reelclaw_backend.apple_signin import verify_apple_identity_token
from reelclaw_backend.auth import JWTAuth, bearer_token
from reelclaw_backend.aws_services import presign_get, presign_put, s3_head, sns_create_endpoint, submit_batch_job
from reelclaw_backend.aws_store import DynamoDeviceStore, DynamoJobStore
from reelclaw_backend.config import Settings, load_settings
from reelclaw_backend.storage import DeviceStore, JobStore, TokenStore
from reelclaw_backend.worker import JobRunner

app = FastAPI(title="ReelClaw Backend", version="0.2.0")


############################
# Models (API contract)
############################


class AppleAuthRequest(BaseModel):
    identity_token: str
    authorization_code: str


class AppleAuthResponse(BaseModel):
    access_token: str


class ReferenceSpec(BaseModel):
    type: Literal["url", "upload"]
    url: str | None = None
    filename: str | None = None
    content_type: str | None = None
    bytes: int | None = None


class ClipSpec(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    bytes: int = Field(ge=1)


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reference: ReferenceSpec
    variations: int = 3
    burn_overlays: bool = False
    clips: list[ClipSpec] = Field(default_factory=list)


class UploadTarget(BaseModel):
    s3_key: str | None = None
    upload_url: str


class ClipUploadTarget(BaseModel):
    clip_id: str
    s3_key: str | None = None
    upload_url: str


class CreateJobResponse(BaseModel):
    job_id: str
    reference_upload: UploadTarget | None = None
    clip_uploads: list[ClipUploadTarget]


class JobStatusResponse(BaseModel):
    job_id: str
    created_at: str | None = None
    status: str
    stage: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    error_code: str | None = None
    error_detail: str | None = None


class JobSummary(BaseModel):
    job_id: str
    created_at: str | None = None
    status: str
    stage: str | None = None
    message: str | None = None


class ListJobsResponse(BaseModel):
    jobs: list[JobSummary]


class VariantsResponse(BaseModel):
    job_id: str
    variants: list[dict[str, Any]]


class RegisterDeviceRequest(BaseModel):
    device_token: str
    environment: Literal["sandbox", "production"]


class RegisterDeviceResponse(BaseModel):
    ok: bool


############################
# Internal helpers/state
############################


_STATE: dict[str, Any] = {}


def _safe_slug(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:100] or "file"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_ttl_seconds(settings: Settings) -> int:
    # Use S3 lifecycle for object expiration; Dynamo TTL keeps tables lean.
    # Defaults match infra lifecycle (uploads 7d, outputs 30d). Keep jobs slightly longer.
    return int(35 * 24 * 60 * 60)


def _decode_jwt_payload_unverified(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        doc = json.loads(raw.decode("utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _get_settings() -> Settings:
    s = _STATE.get("settings")
    if isinstance(s, Settings):
        return s
    raise RuntimeError("Backend not initialized")


def _get_jwt() -> JWTAuth:
    a = _STATE.get("jwt")
    if isinstance(a, JWTAuth):
        return a
    raise RuntimeError("Backend not initialized (missing jwt)")


def _get_jobs() -> Any:
    j = _STATE.get("jobs")
    if j is None:
        raise RuntimeError("Backend not initialized (missing jobs)")
    return j


def _get_devices() -> Any:
    d = _STATE.get("devices")
    if d is None:
        raise RuntimeError("Backend not initialized (missing devices)")
    return d


def _get_runner() -> JobRunner | None:
    r = _STATE.get("runner")
    return r if isinstance(r, JobRunner) else None


def _is_aws_mode(settings: Settings) -> bool:
    if not settings.aws_mode:
        return False
    required = [settings.uploads_bucket, settings.outputs_bucket, settings.jobs_table, settings.devices_table]
    return all(bool(x) for x in required)


def require_user(
    authorization: str | None = Header(default=None),
    x_reelclaw_authorization: str | None = Header(default=None, alias="X-Reelclaw-Authorization"),
    x_reelclaw_token: str | None = Header(default=None, alias="X-Reelclaw-Token"),
) -> str:
    # Note: API Gateway HTTP APIs can drop/strip the standard `Authorization` header on the way to ALB/ECS.
    # To keep iOS simple (no domain needed) we also accept `X-Reelclaw-Authorization`.
    # Prefer standard Bearer auth, but fall back to a custom header that API Gateway reliably forwards.
    token = bearer_token(authorization) or bearer_token(x_reelclaw_authorization) or (str(x_reelclaw_token or "").strip() or None)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing auth header (Authorization, X-Reelclaw-Authorization, or X-Reelclaw-Token)",
        )
    try:
        claims = _get_jwt().verify(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


def _require_job_owner(job_id: str, user_id: str) -> dict[str, Any]:
    job = _get_jobs().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(job.get("user_id") or "") != str(user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    return job


_VARIANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


@app.on_event("startup")
def _startup() -> None:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if not settings.jwt_secret:
        if settings.dev_auth:
            # Local dev convenience (do not use in production).
            jwt_auth = JWTAuth(secret="dev_insecure_secret_change_me")
        else:
            raise RuntimeError("Missing REELCLAW_JWT_SECRET")
    else:
        jwt_auth = JWTAuth(secret=settings.jwt_secret)

    if _is_aws_mode(settings):
        jobs = DynamoJobStore(table_name=str(settings.jobs_table), region=settings.aws_region)
        devices = DynamoDeviceStore(table_name=str(settings.devices_table), region=settings.aws_region)
        runner: JobRunner | None = None
    else:
        jobs = JobStore(settings.data_dir)
        devices = DeviceStore.load(settings.data_dir)
        runner = JobRunner.create(settings=settings, jobs=jobs)

    # Legacy token store kept for local testing only (not used for auth anymore).
    tokens = TokenStore.load(settings.data_dir / "auth_tokens.json")

    _STATE["settings"] = settings
    _STATE["jwt"] = jwt_auth
    _STATE["jobs"] = jobs
    _STATE["devices"] = devices
    _STATE["runner"] = runner
    _STATE["tokens"] = tokens


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


############################
# Auth
############################


@app.post("/v1/auth/apple", response_model=AppleAuthResponse)
def auth_apple(body: AppleAuthRequest) -> AppleAuthResponse:
    settings = _get_settings()

    if settings.dev_auth:
        claims = _decode_jwt_payload_unverified(body.identity_token)
        user_id = str(claims.get("sub") or "").strip() or "dev_user"
    else:
        if not settings.apple_audience:
            raise HTTPException(status_code=500, detail="Server misconfigured (missing REELCLAW_APPLE_AUDIENCE).")
        try:
            claims = verify_apple_identity_token(body.identity_token, audience=settings.apple_audience)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Apple token verification failed: {type(e).__name__}")
        user_id = str(claims.get("sub") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="Apple token missing sub")

    access_token = _get_jwt().issue(user_id=user_id)
    return AppleAuthResponse(access_token=access_token)


############################
# Push device registration
############################


@app.post("/v1/devices/apns", response_model=RegisterDeviceResponse)
def register_device(body: RegisterDeviceRequest, user_id: str = Depends(require_user)) -> RegisterDeviceResponse:
    settings = _get_settings()
    token = (body.device_token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Missing device_token")

    device_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
    now = _utc_now_iso()

    record: dict[str, Any] = {
        "apns_token": token,
        "environment": str(body.environment),
        "updated_at": now,
        "sns_endpoint_arn": None,
    }

    if settings.enable_apns and settings.sns_platform_application_arn:
        try:
            endpoint_arn = sns_create_endpoint(
                region=settings.aws_region,
                platform_application_arn=settings.sns_platform_application_arn,
                token=token,
                custom_user_data=str(user_id),
            )
            record["sns_endpoint_arn"] = endpoint_arn or None
        except Exception:
            # Best-effort. Device can still poll.
            record["sns_endpoint_arn"] = None

    _get_devices().upsert_device(user_id=str(user_id), device_id=device_id, record=record)
    return RegisterDeviceResponse(ok=True)


############################
# Jobs
############################


def _make_s3_key(*, user_id: str, job_id: str, kind: str, filename: str) -> str:
    safe = _safe_slug(Path(filename).name)
    if kind == "reference":
        return f"uploads/{user_id}/{job_id}/reference/reference_{safe}"
    return f"uploads/{user_id}/{job_id}/clips/{kind}_{safe}"


@app.post("/v1/jobs", response_model=CreateJobResponse)
def create_job(body: CreateJobRequest, request: Request, user_id: str = Depends(require_user)) -> CreateJobResponse:
    settings = _get_settings()

    variations = max(1, min(int(body.variations or 3), 10))
    if len(body.clips) <= 0:
        raise HTTPException(status_code=400, detail="Please provide at least 1 clip.")
    if len(body.clips) > settings.max_clips:
        raise HTTPException(status_code=400, detail=f"Too many clips (max {settings.max_clips}).")

    ref = body.reference
    if ref.type == "url":
        url = (ref.url or "").strip()
        if not (url.startswith("https://") or url.startswith("http://")):
            raise HTTPException(status_code=400, detail="reference.url must be http(s).")
        reference: dict[str, Any] = {"type": "url", "url": url}
    else:
        fn = (ref.filename or "").strip()
        ct = (ref.content_type or "").strip() or "application/octet-stream"
        size = int(ref.bytes or 0)
        if not fn or size <= 0:
            raise HTTPException(status_code=400, detail="reference upload requires filename and bytes.")
        if size > settings.max_clip_bytes:
            raise HTTPException(status_code=400, detail="reference video too large.")
        reference = {"type": "upload", "filename": fn, "content_type": ct, "bytes": size}

    clips: list[dict[str, Any]] = []
    for c in body.clips:
        if int(c.bytes) > settings.max_clip_bytes:
            raise HTTPException(status_code=400, detail=f"Clip too large: {c.filename}")
        clips.append(
            {
                "clip_id": f"c{uuid4().hex[:10]}",
                "filename": str(c.filename),
                "content_type": str(c.content_type or "application/octet-stream"),
                "bytes": int(c.bytes),
                "uploaded": False,
                "s3_key": None,
                "path": None,
            }
        )

    ttl_seconds = _job_ttl_seconds(settings)
    job = _get_jobs().create_job(
        user_id=str(user_id),
        reference=reference,
        variations=variations,
        burn_overlays=bool(body.burn_overlays),
        clips=clips,
        ttl_seconds=ttl_seconds,
    )
    job_id = str(job["job_id"])

    aws_mode = _is_aws_mode(settings)
    clip_uploads: list[ClipUploadTarget] = []
    reference_upload: UploadTarget | None = None

    if aws_mode:
        # Persist S3 keys.
        updated_ref = dict(reference)
        if updated_ref.get("type") == "upload":
            s3_key = _make_s3_key(user_id=str(user_id), job_id=job_id, kind="reference", filename=str(updated_ref.get("filename") or "reference.mp4"))
            updated_ref["s3_key"] = s3_key
            reference_upload = UploadTarget(
                s3_key=s3_key,
                upload_url=presign_put(
                    region=settings.aws_region,
                    bucket=str(settings.uploads_bucket),
                    key=s3_key,
                    content_type=str(updated_ref.get("content_type") or "application/octet-stream"),
                    expires_in=settings.upload_url_ttl_seconds,
                ),
            )

        updated_clips: list[dict[str, Any]] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            s3_key = _make_s3_key(user_id=str(user_id), job_id=job_id, kind=clip_id, filename=str(clip["filename"]))
            clip2 = dict(clip)
            clip2["s3_key"] = s3_key
            updated_clips.append(clip2)
            clip_uploads.append(
                ClipUploadTarget(
                    clip_id=clip_id,
                    s3_key=s3_key,
                    upload_url=presign_put(
                        region=settings.aws_region,
                        bucket=str(settings.uploads_bucket),
                        key=s3_key,
                        content_type=str(clip2.get("content_type") or "application/octet-stream"),
                        expires_in=settings.upload_url_ttl_seconds,
                    ),
                )
            )

        _get_jobs().update(job_id, reference=updated_ref, clips=updated_clips)
    else:
        # Local dev: upload directly to this API.
        store: JobStore = _get_jobs()
        updated_ref = dict(reference)
        if updated_ref.get("type") == "upload":
            filename = str(updated_ref.get("filename") or "reference.mp4")
            ref_path = store.job_dir(job_id) / "uploads" / _safe_slug(f"reference_{Path(filename).name}")
            updated_ref["path"] = str(ref_path)
            reference_upload = UploadTarget(
                s3_key=None,
                upload_url=str(request.url_for("upload_reference", job_id=job_id)),
            )

        updated_clips: list[dict[str, Any]] = []
        for clip in clips:
            clip_id = str(clip["clip_id"])
            p = store.reserve_upload_path(job_id, filename=str(clip["filename"]), clip_id=clip_id)
            clip2 = dict(clip)
            clip2["path"] = str(p)
            updated_clips.append(clip2)
            clip_uploads.append(
                ClipUploadTarget(
                    clip_id=clip_id,
                    s3_key=None,
                    upload_url=str(request.url_for("upload_clip", job_id=job_id, clip_id=clip_id)),
                )
            )

        store.update(job_id, reference=updated_ref, clips=updated_clips)

    return CreateJobResponse(job_id=job_id, reference_upload=reference_upload, clip_uploads=clip_uploads)


@app.put("/v1/jobs/{job_id}/clips/{clip_id}", name="upload_clip")
async def upload_clip(job_id: str, clip_id: str, request: Request, user_id: str = Depends(require_user)) -> Response:
    settings = _get_settings()
    if _is_aws_mode(settings):
        raise HTTPException(status_code=410, detail="Use presigned upload URLs.")

    job = _require_job_owner(job_id, user_id)
    clips = job.get("clips") or []
    if not isinstance(clips, list):
        raise HTTPException(status_code=400, detail="Invalid job clips")
    clip = next((c for c in clips if isinstance(c, dict) and str(c.get("clip_id") or "") == str(clip_id)), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    dst = Path(str(clip.get("path") or "")).expanduser().resolve()
    if not dst:
        raise HTTPException(status_code=500, detail="Server misconfigured (missing clip path)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0
    try:
        with dst.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                f.write(chunk)
                byte_count += len(chunk)
    except Exception:
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        raise

    if byte_count <= 0:
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty body")

    # Mark uploaded.
    clip["uploaded"] = True
    _get_jobs().update(job_id, clips=clips, status="uploading", stage="Uploading", message="Clips uploaded.")
    return Response(status_code=204)


@app.put("/v1/jobs/{job_id}/reference", name="upload_reference")
async def upload_reference(job_id: str, request: Request, user_id: str = Depends(require_user)) -> Response:
    settings = _get_settings()
    if _is_aws_mode(settings):
        raise HTTPException(status_code=410, detail="Use presigned upload URLs.")

    job = _require_job_owner(job_id, user_id)
    ref = job.get("reference") or {}
    if not isinstance(ref, dict) or str(ref.get("type") or "") != "upload":
        raise HTTPException(status_code=404, detail="Reference upload not configured")
    dst = Path(str(ref.get("path") or "")).expanduser().resolve()
    if not dst:
        raise HTTPException(status_code=500, detail="Server misconfigured (missing reference path)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0
    try:
        with dst.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                f.write(chunk)
                byte_count += len(chunk)
    except Exception:
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        raise

    if byte_count <= 0:
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty body")

    ref["uploaded"] = True
    _get_jobs().update(job_id, reference=ref, status="uploading", stage="Uploading", message="Reference uploaded.")
    return Response(status_code=204)


@app.post("/v1/jobs/{job_id}/start")
def start_job(job_id: str, user_id: str = Depends(require_user)) -> dict[str, str]:
    settings = _get_settings()
    job = _require_job_owner(job_id, user_id)

    status = str(job.get("status") or "")
    if status in {"queued", "running"}:
        return {"ok": "true"}
    if status == "succeeded":
        return {"ok": "true"}

    reference = job.get("reference") if isinstance(job.get("reference"), dict) else {}
    clips = job.get("clips") if isinstance(job.get("clips"), list) else []

    aws_mode = _is_aws_mode(settings)
    if aws_mode:
        # Validate S3 objects exist.
        for c in clips:
            if not isinstance(c, dict):
                continue
            key = str(c.get("s3_key") or "")
            if not key:
                raise HTTPException(status_code=409, detail="Missing clip upload key.")
            try:
                s3_head(region=settings.aws_region, bucket=str(settings.uploads_bucket), key=key)
            except Exception:
                raise HTTPException(status_code=409, detail=f"Clip not uploaded yet: {c.get('filename')}")

        if str(reference.get("type") or "") == "upload":
            rkey = str(reference.get("s3_key") or "")
            if not rkey:
                raise HTTPException(status_code=409, detail="Missing reference upload key.")
            try:
                s3_head(region=settings.aws_region, bucket=str(settings.uploads_bucket), key=rkey)
            except Exception:
                raise HTTPException(status_code=409, detail="Reference not uploaded yet.")

        if not settings.batch_job_queue or not settings.batch_job_definition:
            raise HTTPException(status_code=500, detail="Server misconfigured (missing Batch settings).")

        batch_job_id = submit_batch_job(
            region=settings.aws_region,
            job_queue=settings.batch_job_queue,
            job_definition=settings.batch_job_definition,
            job_name=f"reelclaw-{job_id}",
            environment={
                "REELCLAW_JOB_ID": str(job_id),
                "REELCLAW_USER_ID": str(user_id),
            },
        )

        _get_jobs().update(
            job_id,
            status="queued",
            stage="Queued",
            message="Job submitted.",
            progress_current=0,
            progress_total=max(1, int(job.get("variations") or 3)),
            batch_job_id=batch_job_id or None,
            error_code=None,
            error_detail=None,
        )
        return {"ok": "true"}

    # Local mode: ensure at least 1 upload exists.
    for c in clips:
        if not isinstance(c, dict):
            continue
        p = str(c.get("path") or "")
        if not p or not Path(p).exists():
            raise HTTPException(status_code=409, detail=f"Clip not uploaded yet: {c.get('filename')}")

    if str(reference.get("type") or "") == "upload":
        p = str(reference.get("path") or "")
        if not p or not Path(p).exists():
            raise HTTPException(status_code=409, detail="Reference not uploaded yet.")
        # Back-compat: local runner expects a reel url/path string.
        job["reference_reel_url"] = p
        _get_jobs().update(job_id, reference_reel_url=p)

    runner = _get_runner()
    if not runner:
        raise HTTPException(status_code=500, detail="Local runner not initialized")
    runner.start(job_id)
    return {"ok": "true"}


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, user_id: str = Depends(require_user)) -> JobStatusResponse:
    job = _require_job_owner(job_id, user_id)
    return JobStatusResponse(
        job_id=str(job.get("job_id") or job_id),
        created_at=str(job.get("created_at") or "") or None,
        status=str(job.get("status") or "queued"),
        stage=job.get("stage"),
        message=job.get("message"),
        progress_current=job.get("progress_current"),
        progress_total=job.get("progress_total"),
        error_code=job.get("error_code"),
        error_detail=job.get("error_detail"),
    )


@app.get("/v1/jobs", response_model=ListJobsResponse)
def list_jobs(user_id: str = Depends(require_user)) -> ListJobsResponse:
    jobs = _get_jobs().list_for_user(str(user_id), limit=50)
    out: list[JobSummary] = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        out.append(
            JobSummary(
                job_id=str(j.get("job_id") or ""),
                created_at=str(j.get("created_at") or "") or None,
                status=str(j.get("status") or "queued"),
                stage=j.get("stage"),
                message=j.get("message"),
            )
        )
    return ListJobsResponse(jobs=out)


@app.get("/v1/jobs/{job_id}/variants", response_model=VariantsResponse)
def get_variants(job_id: str, request: Request, user_id: str = Depends(require_user)) -> VariantsResponse:
    settings = _get_settings()
    job = _require_job_owner(job_id, user_id)
    status = str(job.get("status") or "")
    if status != "succeeded":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    aws_mode = _is_aws_mode(settings)
    if aws_mode:
        variants_raw = job.get("variants") or []
        variants: list[dict[str, Any]] = []
        for v in variants_raw if isinstance(variants_raw, list) else []:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("id") or "").strip()
            vkey = str(v.get("video_s3_key") or "").strip()
            if not vid or not vkey:
                continue
            video_url = presign_get(
                region=settings.aws_region,
                bucket=str(settings.outputs_bucket),
                key=vkey,
                expires_in=settings.download_url_ttl_seconds,
            )
            thumb_key = str(v.get("thumb_s3_key") or "").strip() or None
            thumb_url = (
                presign_get(
                    region=settings.aws_region,
                    bucket=str(settings.outputs_bucket),
                    key=thumb_key,
                    expires_in=settings.download_url_ttl_seconds,
                )
                if thumb_key
                else None
            )
            variants.append(
                {
                    "id": vid,
                    "title": v.get("title") or f"Variation {vid.lstrip('v')}",
                    "score": v.get("score"),
                    "video_url": video_url,
                    "thumbnail_url": thumb_url,
                }
            )
        return VariantsResponse(job_id=job_id, variants=variants)

    # Local mode: keep the old variant serving.
    pipeline_root = Path(str(job.get("pipeline_root") or _get_jobs().job_dir(job_id) / "pipeline")).expanduser().resolve()
    finals_dir = pipeline_root / "finals"

    variants: list[dict[str, Any]] = []
    for p in sorted(finals_dir.glob("v*.mov")) + sorted(finals_dir.glob("v*.mp4")):
        vid = p.stem
        video_url = request.url_for("variant_video", job_id=job_id, variant_id=vid)
        variants.append(
            {
                "id": vid,
                "title": f"Variation {vid.lstrip('v')}",
                "score": None,
                "video_url": str(video_url),
                "thumbnail_url": None,
            }
        )
    return VariantsResponse(job_id=job_id, variants=variants)


@app.get("/v1/jobs/{job_id}/variants/{variant_id}/video", name="variant_video")
def variant_video(job_id: str, variant_id: str) -> FileResponse:
    settings = _get_settings()
    if _is_aws_mode(settings):
        raise HTTPException(status_code=410, detail="Variants are served via presigned URLs.")

    if not _VARIANT_ID_RE.match(variant_id or ""):
        raise HTTPException(status_code=400, detail="Invalid variant id")

    job = _get_jobs().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pipeline_root = Path(str(job.get("pipeline_root") or _get_jobs().job_dir(job_id) / "pipeline")).expanduser().resolve()
    finals_dir = pipeline_root / "finals"

    mov = finals_dir / f"{variant_id}.mov"
    mp4 = finals_dir / f"{variant_id}.mp4"
    path = mov if mov.exists() else mp4
    if not path.exists():
        raise HTTPException(status_code=404, detail="Variant not found")

    media_type = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
    return FileResponse(path=path, media_type=media_type, filename=path.name)
