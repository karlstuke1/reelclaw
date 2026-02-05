# ReelClaw AWS Infra (Terraform)

This folder provisions the AWS infrastructure for:

- **HTTPS API URL without a custom domain** via **API Gateway default `*.execute-api.<region>.amazonaws.com`**
- **ECS Fargate** for the FastAPI API service (behind an **HTTP** ALB)
- **AWS Batch (EC2)** for video processing workers
- **S3** for uploads + outputs
- **DynamoDB** for job/device state

## Quick start

1) Export AWS credentials (bootstrap only)

Use `.env.production` locally (do not commit new secrets):

```bash
set -a
source .env.production
set +a
```

2) Bootstrap Terraform state (recommended)

```bash
cd infra/terraform/bootstrap
terraform init
terraform apply
```

Take note of the outputs:
- `tf_state_bucket`
- `tf_state_lock_table`

3) Configure remote state backend (optional but recommended)

Copy `infra/terraform/envs/staging/backend.tf.example` to `backend.tf` and fill in the bucket/table outputs.

4) Apply an environment

```bash
cd infra/terraform/envs/staging
terraform init
terraform apply -var "apple_audience=YOUR_BUNDLE_ID"
```

5) Grab the API base URL

Terraform prints:
- `api_base_url` (API Gateway HTTPS URL)

Use that in the iOS app’s Settings (or set a build-time default later).

## One-command deploy (staging)

If you’ve already bootstrapped remote state + set Secrets Manager values, you can deploy via:

```bash
python3 scripts/deploy_reelclaw_aws.py --env staging
```

## Notes

- API Gateway terminates HTTPS. The ALB only listens on **port 80** and requires a secret header set by API Gateway.
- You can add a custom domain later by attaching an ACM cert to API Gateway and creating a DNS record; the rest of the stack stays the same.

## Required Secrets Manager values

The module creates two secrets per env (values are not set by Terraform):

- `reelclaw-<env>/jwt_secret`
- `reelclaw-<env>/openrouter_api_key`

Set them after the first apply, then re-apply Terraform (or restart ECS) so tasks pick up the values.
