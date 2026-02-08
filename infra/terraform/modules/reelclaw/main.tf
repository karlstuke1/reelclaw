terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.30.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  name       = "reelclaw-${var.env}"
  account_id = data.aws_caller_identity.current.account_id

  uploads_bucket = "reelclaw-${var.env}-uploads-${local.account_id}"
  outputs_bucket = "reelclaw-${var.env}-outputs-${local.account_id}"

  jobs_table    = "reelclaw_${var.env}_jobs"
  devices_table = "reelclaw_${var.env}_devices"

  # API Gateway -> ALB origin protection (no custom domain / no ACM cert).
  # We require a secret header at the ALB listener and have API Gateway add it.
  origin_auth_header_name = "X-Reelclaw-Origin-Verify"

  github_actions_enabled = var.env == "prod" && length(trimspace(var.github_repo)) > 0
  github_deploy_role_name = "${local.name}-github-deploy"
}

############################
# S3 buckets (uploads/outputs)
############################

resource "aws_s3_bucket" "uploads" {
  bucket = local.uploads_bucket
}

resource "aws_s3_bucket_public_access_block" "uploads" {
  bucket                  = aws_s3_bucket.uploads.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  rule {
    id     = "expire-uploads"
    status = "Enabled"
    filter {
      prefix = "uploads/"
    }
    expiration {
      days = 7
    }
  }
  rule {
    id     = "expire-library"
    status = "Enabled"
    filter {
      prefix = "library/"
    }
    expiration {
      days = 90
    }
  }
}

resource "aws_s3_bucket" "outputs" {
  bucket = local.outputs_bucket
}

resource "aws_s3_bucket_public_access_block" "outputs" {
  bucket                  = aws_s3_bucket.outputs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "outputs" {
  bucket = aws_s3_bucket.outputs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

############################
# DynamoDB tables
############################

resource "aws_dynamodb_table" "jobs" {
  name         = local.jobs_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "gsi_user_created"
    hash_key        = "user_id"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "devices" {
  name         = local.devices_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"
  range_key    = "device_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  attribute {
    name = "device_id"
    type = "S"
  }
}

############################
# ECR repos
############################

resource "aws_ecr_repository" "api" {
  name = "${local.name}-api"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "worker" {
  name = "${local.name}-worker"
  image_scanning_configuration {
    scan_on_push = true
  }
}

############################
# GitHub Actions OIDC deploy role (prod only)
############################

resource "aws_iam_openid_connect_provider" "github" {
  count = local.github_actions_enabled ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [var.github_oidc_thumbprint]
}

resource "aws_iam_role" "github_deploy" {
  count = local.github_actions_enabled ? 1 : 0

  name = local.github_deploy_role_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github[0].arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/${var.github_branch}"
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "github_deploy" {
  count = local.github_actions_enabled ? 1 : 0

  name = "${local.github_deploy_role_name}-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
          "ecr:BatchGetImage"
        ]
        Resource = [
          aws_ecr_repository.api.arn,
          aws_ecr_repository.worker.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:DescribeTaskDefinition",
          "ecs:RegisterTaskDefinition",
          "ecs:UpdateService",
          "ecs:DescribeServices",
          "ecs:DescribeTasks",
          "ecs:ListTasks"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "batch:DescribeJobDefinitions",
          "batch:RegisterJobDefinition"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_task_execution.arn,
          aws_iam_role.api_task.arn,
          aws_iam_role.worker_task.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "github_deploy" {
  count      = local.github_actions_enabled ? 1 : 0
  role       = aws_iam_role.github_deploy[0].name
  policy_arn = aws_iam_policy.github_deploy[0].arn
}

############################
# Secrets (values set out-of-band)
############################

resource "aws_secretsmanager_secret" "jwt_secret" {
  name = "${local.name}/jwt_secret"
}

resource "aws_secretsmanager_secret" "openrouter_api_key" {
  name = "${local.name}/openrouter_api_key"
}

resource "aws_secretsmanager_secret" "ytdlp_cookies" {
  name = "${local.name}/ytdlp_cookies"
}

resource "aws_secretsmanager_secret" "ytdlp_proxy_url" {
  name = "${local.name}/ytdlp_proxy_url"
}

############################
# Networking (minimal VPC)
############################

resource "aws_vpc" "main" {
  cidr_block           = "10.40.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = local.name }
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name}-igw" }
}

resource "aws_subnet" "public" {
  for_each                = { for idx, az in local.azs : idx => az }
  vpc_id                  = aws_vpc.main.id
  availability_zone       = each.value
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, each.key)
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name}-public-${each.value}" }
}

resource "aws_subnet" "private" {
  for_each          = { for idx, az in local.azs : idx => az }
  vpc_id            = aws_vpc.main.id
  availability_zone = each.value
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, each.key + 100)
  tags              = { Name = "${local.name}-private-${each.value}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${local.name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  for_each = aws_subnet.public
  domain   = "vpc"
  tags     = { Name = "${local.name}-nat-eip-${each.key}" }
}

resource "aws_nat_gateway" "nat" {
  for_each      = aws_subnet.public
  allocation_id = aws_eip.nat[each.key].id
  subnet_id     = each.value.id
  depends_on    = [aws_internet_gateway.igw]
  tags          = { Name = "${local.name}-nat-${each.key}" }
}

resource "aws_route_table" "private" {
  for_each = aws_subnet.private
  vpc_id   = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[each.key].id
  }
  tags = { Name = "${local.name}-private-rt-${each.key}" }
}

resource "aws_route_table_association" "private" {
  for_each       = aws_subnet.private
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private[each.key].id
}

# Gateway endpoints reduce NAT usage for S3/DynamoDB.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = concat([aws_route_table.public.id], [for rt in aws_route_table.private : rt.id])
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.dynamodb"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = concat([aws_route_table.public.id], [for rt in aws_route_table.private : rt.id])
}

############################
# ALB (HTTP only) + API Gateway (HTTPS)
############################

resource "random_password" "cloudfront_origin_secret" {
  length  = 32
  special = false
}

resource "aws_security_group" "alb" {
  name = "${local.name}-alb-sg"
  # Note: changing a security group description forces replacement; keep stable.
  description = "ALB security group (origin behind CloudFront)."
  vpc_id      = aws_vpc.main.id
}

resource "aws_security_group_rule" "alb_in_http" {
  type              = "ingress"
  security_group_id = aws_security_group.alb.id
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "alb_egress" {
  type              = "egress"
  security_group_id = aws_security_group.alb.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_lb" "api" {
  name               = replace("${local.name}-api", "_", "-")
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = [for s in aws_subnet.public : s.id]
}

resource "aws_lb_target_group" "api" {
  name        = replace("${local.name}-tg", "_", "-")
  port        = 80
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled = true
    path    = "/healthz"
    matcher = "200-399"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Forbidden"
      status_code  = "403"
    }
  }
}

resource "aws_lb_listener_rule" "forward_api_with_origin_header" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    http_header {
      http_header_name = local.origin_auth_header_name
      values           = [random_password.cloudfront_origin_secret.result]
    }
  }
}

resource "aws_apigatewayv2_api" "api" {
  name          = "${local.name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "alb_proxy" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  integration_uri        = "http://${aws_lb.api.dns_name}"
  payload_format_version = "1.0"
  timeout_milliseconds   = 30000

  # Prevent direct clients from hitting the ALB by requiring a secret header at the listener rule.
  request_parameters = {
    "overwrite:header.${local.origin_auth_header_name}" = random_password.cloudfront_origin_secret.result
    "overwrite:path"                                   = "$request.path"
  }
}

resource "aws_apigatewayv2_route" "root" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "ANY /"
  target    = "integrations/${aws_apigatewayv2_integration.alb_proxy.id}"
}

resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.alb_proxy.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
}

############################
# ECS (API service)
############################

resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/batch/${local.name}/worker"
  retention_in_days = 14
}

resource "aws_security_group" "api" {
  name        = "${local.name}-api-sg"
  description = "ECS service security group."
  vpc_id      = aws_vpc.main.id
}

resource "aws_security_group_rule" "api_ingress_from_alb" {
  type                     = "ingress"
  security_group_id        = aws_security_group.api.id
  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.alb.id
}

resource "aws_security_group_rule" "api_egress" {
  type              = "egress"
  security_group_id = aws_security_group.api.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.name}-ecs-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_exec" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "ecs_task_execution_secrets" {
  name = "${local.name}-ecs-exec-secrets"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.jwt_secret.arn,
          aws_secretsmanager_secret.openrouter_api_key.arn,
          aws_secretsmanager_secret.ytdlp_cookies.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_exec_secrets" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = aws_iam_policy.ecs_task_execution_secrets.arn
}

resource "aws_iam_role" "api_task" {
  name = "${local.name}-api-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "api_task" {
  name = "${local.name}-api-task-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query"
        ]
        Resource = [
          aws_dynamodb_table.jobs.arn,
          "${aws_dynamodb_table.jobs.arn}/index/*",
          aws_dynamodb_table.devices.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:AbortMultipartUpload",
          "s3:ListBucket",
          "s3:DeleteObject",
          "s3:PutObjectTagging"
        ]
        Resource = [
          aws_s3_bucket.uploads.arn,
          "${aws_s3_bucket.uploads.arn}/*",
          aws_s3_bucket.outputs.arn,
          "${aws_s3_bucket.outputs.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "batch:SubmitJob"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:CreatePlatformEndpoint",
          "sns:DeleteEndpoint",
          "sns:Publish"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.jwt_secret.arn,
          aws_secretsmanager_secret.ytdlp_cookies.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "api_task" {
  role       = aws_iam_role.api_task.name
  policy_arn = aws_iam_policy.api_task.arn
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.api_task.arn

  container_definitions = jsonencode([
    {
      name         = "api"
      image        = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      essential    = true
      portMappings = [{ containerPort = 8000, hostPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "REELCLAW_AWS_MODE", value = "1" },
        { name = "REELCLAW_DEV_AUTH", value = "0" },
        { name = "REELCLAW_AWS_REGION", value = var.aws_region },
        { name = "REELCLAW_UPLOADS_BUCKET", value = aws_s3_bucket.uploads.bucket },
        { name = "REELCLAW_OUTPUTS_BUCKET", value = aws_s3_bucket.outputs.bucket },
        { name = "REELCLAW_JOBS_TABLE", value = aws_dynamodb_table.jobs.name },
        { name = "REELCLAW_DEVICES_TABLE", value = aws_dynamodb_table.devices.name },
        { name = "REELCLAW_BATCH_JOB_QUEUE", value = aws_batch_job_queue.queue.name },
        { name = "REELCLAW_BATCH_JOB_DEFINITION", value = aws_batch_job_definition.worker.name },
        { name = "REELCLAW_ENABLE_APNS", value = tostring(var.enable_apns) },
        { name = "REELCLAW_SNS_PLATFORM_APPLICATION_ARN", value = var.apns_platform_application_arn },
        { name = "REELCLAW_REFERENCE_ANALYSIS_MAX_SECONDS", value = "25" },
        { name = "REELCLAW_APPLE_AUDIENCE", value = var.apple_audience },
        { name = "REELCLAW_YTDLP_COOKIES_SECRET_ID", value = aws_secretsmanager_secret.ytdlp_cookies.name }
      ]
      secrets = [
        { name = "REELCLAW_JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt_secret.arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "api"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [for s in aws_subnet.private : s.id]
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  lifecycle {
    ignore_changes = [task_definition]
  }

  depends_on = [aws_lb_listener.http]
}

############################
# AWS Batch (worker)
############################

resource "aws_security_group" "batch" {
  name        = "${local.name}-batch-sg"
  description = "Batch compute security group."
  vpc_id      = aws_vpc.main.id
}

resource "aws_security_group_rule" "batch_egress" {
  type              = "egress"
  security_group_id = aws_security_group.batch.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_iam_role" "batch_service" {
  name = "${local.name}-batch-service"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "batch_instance" {
  name = "${local.name}-batch-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch" {
  name = "${local.name}-batch-instance-profile"
  role = aws_iam_role.batch_instance.name
}

resource "aws_iam_role" "worker_task" {
  name = "${local.name}-worker-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "worker_task" {
  name = "${local.name}-worker-task-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query"
        ]
        Resource = [
          aws_dynamodb_table.jobs.arn,
          "${aws_dynamodb_table.jobs.arn}/index/*",
          aws_dynamodb_table.devices.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.uploads.arn,
          "${aws_s3_bucket.uploads.arn}/*",
          aws_s3_bucket.outputs.arn,
          "${aws_s3_bucket.outputs.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.openrouter_api_key.arn,
          aws_secretsmanager_secret.ytdlp_cookies.arn,
          aws_secretsmanager_secret.ytdlp_proxy_url.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "worker_task" {
  role       = aws_iam_role.worker_task.name
  policy_arn = aws_iam_policy.worker_task.arn
}

resource "aws_batch_compute_environment" "ce" {
  name         = "${local.name}-ce"
  type         = "MANAGED"
  service_role = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = 256
    desired_vcpus       = 0
    instance_type       = ["c6i.4xlarge"]
    subnets             = [for s in aws_subnet.private : s.id]
    security_group_ids  = [aws_security_group.batch.id]
    instance_role       = aws_iam_instance_profile.batch.arn
  }
}

resource "aws_batch_job_queue" "queue" {
  name     = "${local.name}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    compute_environment = aws_batch_compute_environment.ce.arn
    order               = 1
  }
}

resource "aws_batch_job_definition" "worker" {
  name = "${local.name}-worker"
  type = "container"

  container_properties = jsonencode({
    image            = "${aws_ecr_repository.worker.repository_url}:${var.worker_image_tag}"
    vcpus            = 8
    memory           = 30000
    jobRoleArn       = aws_iam_role.worker_task.arn
    executionRoleArn = aws_iam_role.ecs_task_execution.arn

    environment = [
      { name = "REELCLAW_AWS_REGION", value = var.aws_region },
      { name = "REELCLAW_UPLOADS_BUCKET", value = aws_s3_bucket.uploads.bucket },
      { name = "REELCLAW_OUTPUTS_BUCKET", value = aws_s3_bucket.outputs.bucket },
      { name = "REELCLAW_JOBS_TABLE", value = aws_dynamodb_table.jobs.name },
      { name = "REELCLAW_DEVICES_TABLE", value = aws_dynamodb_table.devices.name },
      { name = "REELCLAW_ENABLE_APNS", value = tostring(var.enable_apns) },
      { name = "REELCLAW_SNS_PLATFORM_APPLICATION_ARN", value = var.apns_platform_application_arn },
      { name = "REELCLAW_REFERENCE_ANALYSIS_MAX_SECONDS", value = "25" },
      { name = "REELCLAW_YTDLP_COOKIES_SECRET_ID", value = aws_secretsmanager_secret.ytdlp_cookies.name },
      { name = "REELCLAW_YTDLP_PROXY_SECRET_ID", value = aws_secretsmanager_secret.ytdlp_proxy_url.name },
      { name = "REASONING_EFFORT", value = "low" }
    ]

    secrets = [
      { name = "OPENROUTER_API_KEY", valueFrom = aws_secretsmanager_secret.openrouter_api_key.arn }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "worker"
      }
    }
  })

  timeout {
    attempt_duration_seconds = 3600
  }

  retry_strategy {
    attempts = 1
  }
}
