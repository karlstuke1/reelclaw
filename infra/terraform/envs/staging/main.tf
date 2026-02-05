provider "aws" {
  region = var.aws_region
}

module "reelclaw" {
  source = "../../modules/reelclaw"

  aws_region = var.aws_region
  env        = "staging"

  api_image_tag    = var.api_image_tag
  worker_image_tag = var.worker_image_tag

  enable_apns                   = var.enable_apns
  apns_platform_application_arn = var.apns_platform_application_arn

  apple_audience = var.apple_audience
}

output "api_base_url" {
  value = module.reelclaw.api_base_url
}

output "alb_dns_name" {
  value = module.reelclaw.alb_dns_name
}

output "jobs_table_name" {
  value = module.reelclaw.jobs_table_name
}

output "devices_table_name" {
  value = module.reelclaw.devices_table_name
}

output "uploads_bucket_name" {
  value = module.reelclaw.uploads_bucket_name
}

output "outputs_bucket_name" {
  value = module.reelclaw.outputs_bucket_name
}

output "ecr_api_repo" {
  value = module.reelclaw.ecr_api_repo
}

output "ecr_worker_repo" {
  value = module.reelclaw.ecr_worker_repo
}
