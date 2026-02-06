from __future__ import annotations

import argparse
import os
from typing import Any

import boto3


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


def main() -> int:
    ap = argparse.ArgumentParser(description="One-time migration: remove DynamoDB TTL attribute from ReelClaw jobs.")
    ap.add_argument("--env", choices=["prod", "staging"], default="prod")
    ap.add_argument("--region", default=_env("AWS_REGION", "us-east-1"))
    ap.add_argument(
        "--scope",
        choices=["succeeded", "all"],
        default="succeeded",
        help="Which jobs to migrate (default: succeeded only).",
    )
    ap.add_argument("--table", default=None, help="Override DynamoDB table name.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change without updating DynamoDB.")
    args = ap.parse_args()

    region = str(args.region).strip() or "us-east-1"
    env_name = str(args.env).strip()
    table_name = str(args.table).strip() if args.table else f"reelclaw_{env_name}_jobs"

    scope = str(args.scope).strip()
    dry_run = bool(args.dry_run)

    tbl = boto3.resource("dynamodb", region_name=region).Table(table_name)

    migrated = 0
    scanned = 0
    last_key: dict[str, Any] | None = None
    while True:
        scan_kwargs: dict[str, Any] = {
            "ProjectionExpression": "job_id, #s, ttl",
            "ExpressionAttributeNames": {"#s": "status"},
        }
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key
        resp = tbl.scan(**scan_kwargs)
        items = resp.get("Items") or []
        for it in items:
            scanned += 1
            job_id = str(it.get("job_id") or "").strip()
            if not job_id:
                continue
            if "ttl" not in it:
                continue
            status = str(it.get("status") or "").strip().lower()
            if scope == "succeeded" and status != "succeeded":
                continue

            migrated += 1
            if dry_run:
                print("would remove ttl:", job_id)
                continue
            tbl.update_item(Key={"job_id": job_id}, UpdateExpression="REMOVE ttl")

        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

    print(f"scanned: {scanned}")
    print(f"ttl removed: {migrated}{' (dry-run)' if dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

