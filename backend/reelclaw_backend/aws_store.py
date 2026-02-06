from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any

import boto3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_in(seconds: int) -> int:
    return int(time.time()) + int(seconds)


def _dynamo_sanitize(value: Any) -> Any:
    """
    DynamoDB (via boto3 TypeSerializer) doesn't allow floats; use Decimal.
    Also drop non-finite floats to None to avoid serialization errors.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _dynamo_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dynamo_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_dynamo_sanitize(v) for v in value]
    return value


@dataclass
class DynamoJobStore:
    table_name: str
    region: str

    def _table(self):
        return boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    def create_job(
        self,
        *,
        user_id: str,
        reference: dict[str, Any],
        variations: int,
        burn_overlays: bool,
        director: str | None = None,
        clips: list[dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        from uuid import uuid4

        job_id = f"job_{uuid4().hex[:16]}"
        created_at = _utc_now_iso()
        updated_at = created_at

        record: dict[str, Any] = {
            "job_id": job_id,
            "user_id": str(user_id),
            "created_at": created_at,
            "updated_at": updated_at,
            "queued_at": None,
            "started_at": None,
            "finished_at": None,
            "reference": reference,
            "variations": int(variations),
            "burn_overlays": bool(burn_overlays),
            "director": str(director) if director else None,
            "status": "uploading",
            "stage": "Uploading",
            "message": "Upload clips…",
            "progress_current": 0,
            "progress_total": max(1, int(variations)),
            "error_code": None,
            "error_detail": None,
            "clips": clips,
            "variants": [],
            "batch_job_id": None,
        }
        self._table().put_item(Item=_dynamo_sanitize(record))
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        resp = self._table().get_item(Key={"job_id": str(job_id)})
        item = resp.get("Item")
        return item if isinstance(item, dict) else None

    def put(self, record: dict[str, Any]) -> None:
        self._table().put_item(Item=_dynamo_sanitize(dict(record)))

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        rec = self.get(job_id)
        if not rec:
            raise KeyError(f"job not found: {job_id}")

        now_iso = _utc_now_iso()
        prev_status = str(rec.get("status") or "").strip().lower()
        next_status = str(fields.get("status") or prev_status).strip().lower()

        if "updated_at" not in fields:
            fields["updated_at"] = now_iso

        def _set_if_empty(key: str) -> None:
            cur = str(rec.get(key) or "").strip()
            if not cur:
                rec[key] = now_iso

        if "status" in fields and next_status:
            # Backfill timestamps even if the job started before these fields existed.
            if next_status == "queued":
                _set_if_empty("queued_at")
            elif next_status == "running":
                _set_if_empty("started_at")
            elif next_status in {"succeeded", "failed"}:
                _set_if_empty("finished_at")

        rec.update(fields)
        self.put(rec)
        return rec

    def list_for_user(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        tbl = self._table()
        resp = tbl.query(
            IndexName="gsi_user_created",
            KeyConditionExpression="user_id = :u",
            ExpressionAttributeValues={":u": str(user_id)},
            ScanIndexForward=False,
            Limit=max(1, int(limit)),
        )
        items = resp.get("Items") or []
        return [it for it in items if isinstance(it, dict)]

    def delete(self, job_id: str) -> None:
        self._table().delete_item(Key={"job_id": str(job_id)})


@dataclass
class DynamoDeviceStore:
    table_name: str
    region: str

    def _table(self):
        return boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    def upsert_device(self, *, user_id: str, device_id: str, record: dict[str, Any]) -> None:
        item = {"user_id": str(user_id), "device_id": str(device_id)}
        item.update(dict(record))
        self._table().put_item(Item=_dynamo_sanitize(item))

    def list_devices(self, *, user_id: str) -> list[dict[str, Any]]:
        resp = self._table().query(
            KeyConditionExpression="user_id = :u",
            ExpressionAttributeValues={":u": str(user_id)},
        )
        items = resp.get("Items") or []
        return [it for it in items if isinstance(it, dict)]
