from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("'").strip('"')
        if key:
            out[key] = val
    return out


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _capture(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(cmd))
    res = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return res.stdout


def _aws_account_id(env: dict[str, str]) -> str:
    raw = _capture(["aws", "sts", "get-caller-identity", "--output", "json"], env=env)
    doc = json.loads(raw)
    account = str(doc.get("Account") or "").strip()
    if not account:
        raise RuntimeError("Could not determine AWS account id (missing Account in sts response).")
    return account


def _ecr_login(*, region: str, registry: str, env: dict[str, str]) -> None:
    password = _capture(["aws", "ecr", "get-login-password", "--region", region], env=env)
    print("+ docker login --username AWS --password-stdin", registry)
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        text=True,
        env=env,
        check=True,
    )


def _terraform_backend_args(*, bucket: str, key: str, region: str, dynamodb_table: str) -> list[str]:
    return [
        "-reconfigure",
        f"-backend-config=bucket={bucket}",
        f"-backend-config=key={key}",
        f"-backend-config=region={region}",
        f"-backend-config=dynamodb_table={dynamodb_table}",
    ]


def _terraform_env_dir(env_name: str) -> Path:
    return _repo_root() / "infra" / "terraform" / "envs" / env_name


def _terraform_output(env_dir: Path, name: str, *, env: dict[str, str]) -> str:
    raw = _capture(["terraform", f"-chdir={str(env_dir)}", "output", "-raw", name], env=env).strip()
    if not raw:
        raise RuntimeError(f"Missing terraform output: {name}")
    return raw


def _smoke_test_healthz(api_base_url: str) -> None:
    url = api_base_url.rstrip("/") + "/healthz"
    print("+ GET", url)
    with urllib.request.urlopen(url, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"Smoke test failed: HTTP {resp.status}: {body[:400]}")
        print(body.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy ReelClaw to AWS via Terraform + ECR + ECS.")
    parser.add_argument("--env", choices=["staging", "prod"], default="staging")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--env-file", default=".env.production")
    parser.add_argument("--skip-docker", action="store_true", help="Skip docker build/push.")
    parser.add_argument("--skip-terraform", action="store_true", help="Skip terraform apply.")
    parser.add_argument("--skip-ecs", action="store_true", help="Skip ECS force-new-deployment/wait.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip /healthz smoke test.")
    args = parser.parse_args()

    root = _repo_root()
    dotenv_path = (root / args.env_file).resolve()

    env = os.environ.copy()
    env.update(_load_dotenv(dotenv_path))
    env.setdefault("AWS_REGION", args.region)
    # In some sandboxed environments, setting AWS_DEFAULT_REGION can cause AWS CLI
    # connectivity issues. AWS_REGION is sufficient for AWS CLI.
    env.pop("AWS_DEFAULT_REGION", None)

    account_id = _aws_account_id(env)
    registry = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com"

    api_repo = f"{registry}/reelclaw-{args.env}-api"
    worker_repo = f"{registry}/reelclaw-{args.env}-worker"

    if not args.skip_docker:
        _ecr_login(region=args.region, registry=registry, env=env)
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-f",
                "backend/Dockerfile.api",
                "-t",
                f"{api_repo}:{args.tag}",
                ".",
            ],
            cwd=root,
            env=env,
        )
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-f",
                "backend/Dockerfile.worker",
                "-t",
                f"{worker_repo}:{args.tag}",
                ".",
            ],
            cwd=root,
            env=env,
        )
        _run(["docker", "push", f"{api_repo}:{args.tag}"], cwd=root, env=env)
        _run(["docker", "push", f"{worker_repo}:{args.tag}"], cwd=root, env=env)

    env_dir = _terraform_env_dir(args.env)
    backend_bucket = f"reelclaw-tf-state-{account_id}-{args.region}"
    backend_key = f"reelclaw/{args.env}/terraform.tfstate"
    backend_lock_table = "reelclaw-tf-lock"

    if not args.skip_terraform:
        _run(
            [
                "terraform",
                f"-chdir={str(env_dir)}",
                "init",
                *_terraform_backend_args(
                    bucket=backend_bucket,
                    key=backend_key,
                    region=args.region,
                    dynamodb_table=backend_lock_table,
                ),
            ],
            cwd=root,
            env=env,
        )
        _run(
            [
                "terraform",
                f"-chdir={str(env_dir)}",
                "apply",
                "-auto-approve",
                "-var",
                f"api_image_tag={args.tag}",
                "-var",
                f"worker_image_tag={args.tag}",
            ],
            cwd=root,
            env=env,
        )

    if not args.skip_ecs:
        cluster = f"reelclaw-{args.env}"
        service = f"reelclaw-{args.env}-api"
        _run(
            [
                "aws",
                "ecs",
                "update-service",
                "--cluster",
                cluster,
                "--service",
                service,
                "--force-new-deployment",
            ],
            env=env,
        )
        _run(["aws", "ecs", "wait", "services-stable", "--cluster", cluster, "--services", service], env=env)

    api_base_url = _terraform_output(env_dir, "api_base_url", env=env)
    print("api_base_url:", api_base_url)

    if not args.skip_smoke:
        _smoke_test_healthz(api_base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
