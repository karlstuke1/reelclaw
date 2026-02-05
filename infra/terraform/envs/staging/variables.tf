variable "aws_region" {
  type        = string
  description = "AWS region to deploy into."
  default     = "us-east-1"
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
  description = "Whether to enable SNS->APNs device push. Requires a configured SNS Platform Application."
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
