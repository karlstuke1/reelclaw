variable "aws_region" {
  type        = string
  description = "AWS region."
}

variable "env" {
  type        = string
  description = "Environment name (e.g., staging, prod)."
}

variable "api_image_tag" {
  type        = string
  description = "Docker tag for the API image in ECR."
  default     = "latest"
}

variable "worker_image_tag" {
  type        = string
  description = "Docker tag for the worker image in ECR."
  default     = "latest"
}

variable "enable_apns" {
  type        = bool
  description = "Whether to enable SNS->APNs device push."
  default     = false
}

variable "apns_platform_application_arn" {
  type        = string
  description = "SNS Platform Application ARN for APNs (optional)."
  default     = ""
}

variable "apple_audience" {
  type        = string
  description = "Apple Sign-In audience (iOS bundle id)."
  default     = ""
}

variable "github_repo" {
  type        = string
  description = "GitHub repo in owner/name form allowed to deploy via GitHub Actions OIDC. Leave empty to disable."
  default     = ""
}

variable "github_branch" {
  type        = string
  description = "GitHub branch name allowed to deploy via GitHub Actions OIDC."
  default     = "main"
}

variable "github_oidc_thumbprint" {
  type        = string
  description = "SHA1 thumbprint for GitHub Actions OIDC root CA."
  # This is GitHub's commonly documented thumbprint; override if AWS/GitHub rotate cert chains.
  default = "6938fd4d98bab03faadb97b34396831e3780aea1"
}
