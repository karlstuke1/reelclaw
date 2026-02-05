from __future__ import annotations

import argparse
import json
from typing import Any

import boto3


def _pick(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k in d and d[k] is not None:
            out[k] = d[k]
    return out


def _ecs_register_task_definition_with_image(*, ecs, family: str, container_name: str, image: str) -> str:
    td = ecs.describe_task_definition(taskDefinition=family)["taskDefinition"]

    container_defs = td.get("containerDefinitions") or []
    for c in container_defs:
        if not isinstance(c, dict):
            continue
        if str(c.get("name") or "") == container_name:
            c["image"] = image

    payload = _pick(
        td,
        [
            "family",
            "taskRoleArn",
            "executionRoleArn",
            "networkMode",
            "containerDefinitions",
            "volumes",
            "placementConstraints",
            "requiresCompatibilities",
            "cpu",
            "memory",
            "pidMode",
            "ipcMode",
            "proxyConfiguration",
            "inferenceAccelerators",
            "runtimePlatform",
            "ephemeralStorage",
        ],
    )
    resp = ecs.register_task_definition(**payload)
    return str(resp["taskDefinition"]["taskDefinitionArn"])


def _ecs_deploy_service(*, ecs, cluster: str, service: str, task_def_arn: str) -> None:
    ecs.update_service(cluster=cluster, service=service, taskDefinition=task_def_arn, forceNewDeployment=True)
    waiter = ecs.get_waiter("services_stable")
    waiter.wait(cluster=cluster, services=[service], WaiterConfig={"Delay": 15, "MaxAttempts": 40})


def _batch_register_job_definition_with_image(*, batch, name: str, image: str) -> str:
    defs = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE").get("jobDefinitions") or []
    if not defs:
        raise RuntimeError(f"No active Batch job definitions found for: {name}")

    latest = max(defs, key=lambda d: int(d.get("revision") or 0))
    container_props = dict(latest.get("containerProperties") or {})
    container_props["image"] = image

    payload: dict[str, Any] = {
        "jobDefinitionName": name,
        "type": str(latest.get("type") or "container"),
        "containerProperties": container_props,
    }
    for k in ("retryStrategy", "timeout", "platformCapabilities", "propagateTags", "tags", "schedulingPriority"):
        if k in latest and latest[k] is not None:
            payload[k] = latest[k]

    resp = batch.register_job_definition(**payload)
    return str(resp.get("jobDefinitionArn") or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Runtime deploy (ECS + Batch) without Terraform apply.")
    ap.add_argument("--env", choices=["staging", "prod"], default="prod")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--api-image", required=True, help="Full ECR image URI for the API container.")
    ap.add_argument("--worker-image", required=True, help="Full ECR image URI for the Batch worker container.")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    env_name = str(args.env)
    region = str(args.region)

    cluster = f"reelclaw-{env_name}"
    service = f"reelclaw-{env_name}-api"
    task_family = f"reelclaw-{env_name}-api"
    job_def_name = f"reelclaw-{env_name}-worker"

    ecs = boto3.client("ecs", region_name=region)
    batch = boto3.client("batch", region_name=region)

    api_task_def_arn = _ecs_register_task_definition_with_image(
        ecs=ecs,
        family=task_family,
        container_name="api",
        image=str(args.api_image),
    )
    _ecs_deploy_service(ecs=ecs, cluster=cluster, service=service, task_def_arn=api_task_def_arn)

    worker_job_def_arn = _batch_register_job_definition_with_image(batch=batch, name=job_def_name, image=str(args.worker_image))

    out = {
        "cluster": cluster,
        "service": service,
        "api_task_definition": api_task_def_arn,
        "worker_job_definition": worker_job_def_arn,
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("ECS:", api_task_def_arn)
        print("Batch:", worker_job_def_arn or job_def_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

