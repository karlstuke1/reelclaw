from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import boto3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_in(seconds: int) -> int:
    return int(time.time()) + int(seconds)


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
        clips: list[dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        from uuid import uuid4

        job_id = f"job_{uuid4().hex[:16]}"
        created_at = _utc_now_iso()
        ttl: int | None = None
        if ttl_seconds and int(ttl_seconds) > 0:
            ttl = _epoch_in(int(ttl_seconds))

        record: dict[str, Any] = {
            "job_id": job_id,
            "user_id": str(user_id),
            "created_at": created_at,
            "reference": reference,
            "variations": int(variations),
            "burn_overlays": bool(burn_overlays),
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
            "ttl": ttl,
        }
        self._table().put_item(Item=record)
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        resp = self._table().get_item(Key={"job_id": str(job_id)})
        item = resp.get("Item")
        return item if isinstance(item, dict) else None

    def put(self, record: dict[str, Any]) -> None:
        self._table().put_item(Item=dict(record))

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        rec = self.get(job_id)
        if not rec:
            raise KeyError(f"job not found: {job_id}")
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


@dataclass
class DynamoDeviceStore:
    table_name: str
    region: str

    def _table(self):
        return boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    def upsert_device(self, *, user_id: str, device_id: str, record: dict[str, Any]) -> None:
        item = {"user_id": str(user_id), "device_id": str(device_id)}
        item.update(dict(record))
        self._table().put_item(Item=item)

    def list_devices(self, *, user_id: str) -> list[dict[str, Any]]:
        resp = self._table().query(
            KeyConditionExpression="user_id = :u",
            ExpressionAttributeValues={":u": str(user_id)},
        )
        items = resp.get("Items") or []
        return [it for it in items if isinstance(it, dict)]
