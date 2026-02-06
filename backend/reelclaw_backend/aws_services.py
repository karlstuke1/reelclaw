from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import boto3


@dataclass(frozen=True)
class AWSClients:
    region: str

    def s3(self):
        return boto3.client("s3", region_name=self.region)

    def dynamodb(self):
        return boto3.resource("dynamodb", region_name=self.region)

    def batch(self):
        return boto3.client("batch", region_name=self.region)

    def sns(self):
        return boto3.client("sns", region_name=self.region)


def presign_put(
    *,
    region: str,
    bucket: str,
    key: str,
    content_type: str,
    expires_in: int,
) -> str:
    s3 = boto3.client("s3", region_name=region)
    return str(
        s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=int(expires_in),
        )
    )


def presign_get(*, region: str, bucket: str, key: str, expires_in: int) -> str:
    s3 = boto3.client("s3", region_name=region)
    return str(
        s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
    )


def s3_head(*, region: str, bucket: str, key: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    s3.head_object(Bucket=bucket, Key=key)


def s3_delete_prefix(*, region: str, bucket: str, prefix: str) -> int:
    """
    Delete all objects under the given prefix. Best-effort; returns the number of deleted objects.

    Notes:
    - This does not handle versioned delete markers.
    - Deletion is batched to 1000 keys per request (S3 API limit).
    """
    s3 = boto3.client("s3", region_name=region)
    deleted = 0

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": str(obj.get("Key") or "")} for obj in (page.get("Contents") or []) if obj.get("Key")]
        if not keys:
            continue
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})
            deleted += len(resp.get("Deleted") or [])

    return deleted


def submit_batch_job(
    *,
    region: str,
    job_queue: str,
    job_definition: str,
    job_name: str,
    environment: dict[str, str],
) -> str:
    batch = boto3.client("batch", region_name=region)
    resp = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={"environment": [{"name": k, "value": v} for k, v in environment.items()]},
    )
    return str(resp.get("jobId") or "")


def sns_create_endpoint(
    *,
    region: str,
    platform_application_arn: str,
    token: str,
    custom_user_data: str | None = None,
) -> str:
    sns = boto3.client("sns", region_name=region)
    params: dict[str, Any] = {
        "PlatformApplicationArn": platform_application_arn,
        "Token": token,
    }
    if custom_user_data:
        params["CustomUserData"] = str(custom_user_data)
    resp = sns.create_platform_endpoint(**params)
    return str(resp.get("EndpointArn") or "")


def sns_publish_apns(
    *,
    region: str,
    endpoint_arn: str,
    title: str,
    body: str,
    job_id: str,
    is_sandbox: bool,
) -> None:
    """
    Publish a basic APNs push via SNS to a single platform endpoint.
    """
    sns = boto3.client("sns", region_name=region)

    apns_payload = {
        "aps": {
            "alert": {"title": str(title), "body": str(body)},
            "sound": "default",
        },
        "job_id": str(job_id),
    }
    msg = {
        "default": f"{title}: {body}",
        ("APNS_SANDBOX" if is_sandbox else "APNS"): json.dumps(apns_payload),
    }
    sns.publish(TargetArn=endpoint_arn, MessageStructure="json", Message=json.dumps(msg))
