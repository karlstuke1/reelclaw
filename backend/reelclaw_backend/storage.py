from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    # Use a lexicographically sortable, stable format.
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_epoch() -> int:
    return int(time.time())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{int(time.time() * 1000)}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_slug(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:80] or "file"


@dataclass
class TokenStore:
    path: Path
    _tokens: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "TokenStore":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                tokens = {str(k): str(v) for k, v in raw.get("tokens", {}).items() if str(k) and str(v)}
            else:
                tokens = {}
        except Exception:
            tokens = {}
        return cls(path=path, _tokens=tokens)

    def save(self) -> None:
        _atomic_write_json(self.path, {"tokens": self._tokens})

    def issue(self, *, user_id: str) -> str:
        token = f"rc_{uuid4().hex}"
        self._tokens[token] = str(user_id)
        self.save()
        return token

    def user_for_token(self, token: str) -> str | None:
        return self._tokens.get(token)


@dataclass
class JobStore:
    root: Path

    def _jobs_index_path(self) -> Path:
        return self.root / "jobs_index.json"

    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir() / job_id

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def create_job(
        self,
        *,
        user_id: str,
        reference: dict[str, Any],
        variations: int,
        burn_overlays: bool,
        clips: list[dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        job_id = f"job_{uuid4().hex[:16]}"
        now_iso = _utc_now_iso()
        ttl: int | None = None
        if ttl_seconds and int(ttl_seconds) > 0:
            ttl = _utc_now_epoch() + int(ttl_seconds)

        record: dict[str, Any] = {
            "job_id": job_id,
            "user_id": str(user_id),
            "created_at": now_iso,
            "reference": reference,
            # Back-compat for the local runner which expects a reel url/path string.
            "reference_reel_url": str(reference.get("url") or reference.get("path") or ""),
            "variations": int(variations),
            "burn_overlays": bool(burn_overlays),
            "status": "uploading",
            "stage": "Uploading",
            "message": "Upload clips…",
            "progress_current": 0,
            "progress_total": max(1, int(variations)),
            "pipeline_root": None,
            "error_code": None,
            "error_detail": None,
            "clips": clips,
            "variants": [],
            "batch_job_id": None,
            "ttl": ttl,
        }
        _atomic_write_json(self.job_path(job_id), record)

        uploads_dir = self.job_dir(job_id) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_path(job_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        rec = self.get(job_id)
        if not rec:
            raise KeyError(f"job not found: {job_id}")
        rec.update(fields)
        _atomic_write_json(self.job_path(job_id), rec)
        return rec

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """
        Best-effort listing for local dev.
        """
        out: list[dict[str, Any]] = []
        jobs_dir = self.jobs_dir()
        if not jobs_dir.exists():
            return []
        for job_path in sorted(jobs_dir.glob("job_*/job.json"), reverse=True)[: max(1, int(limit))]:
            try:
                raw = json.loads(job_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and str(raw.get("user_id") or "") == str(user_id):
                    out.append(raw)
            except Exception:
                continue
        # created_at is ISO; sort newest-first.
        out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return out

    def add_upload(self, job_id: str, *, filename: str, data: bytes) -> Path:
        dst = self.reserve_upload_path(job_id, filename=filename)
        dst.write_bytes(data)
        return dst

    def reserve_upload_path(self, job_id: str, *, filename: str, clip_id: str | None = None) -> Path:
        uploads_dir = self.job_dir(job_id) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(filename).name
        base = _safe_slug(base_name)
        if clip_id:
            base = _safe_slug(f"{clip_id}_{base}")
        dst = uploads_dir / base
        if dst.exists():
            stem = dst.stem
            ext = dst.suffix
            dst = uploads_dir / f"{stem}_{uuid4().hex[:6]}{ext}"
        return dst


@dataclass
class DeviceStore:
    path: Path

    @classmethod
    def load(cls, root: Path) -> "DeviceStore":
        return cls(path=root / "devices.json")

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _write(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.path, payload)

    def upsert_device(self, *, user_id: str, device_id: str, record: dict[str, Any]) -> None:
        doc = self._read()
        u = doc.get(str(user_id)) if isinstance(doc.get(str(user_id)), dict) else {}
        u[str(device_id)] = dict(record)
        doc[str(user_id)] = u
        self._write(doc)

    def list_devices(self, *, user_id: str) -> list[dict[str, Any]]:
        doc = self._read()
        u = doc.get(str(user_id))
        if not isinstance(u, dict):
            return []
        out: list[dict[str, Any]] = []
        for _k, v in u.items():
            if isinstance(v, dict):
                out.append(v)
        return out
