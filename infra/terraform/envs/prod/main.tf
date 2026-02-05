provider "aws" {
  region = var.aws_region
}

module "reelclaw" {
  source = "../../modules/reelclaw"

  aws_region = var.aws_region
  env        = "prod"

  api_image_tag    = var.api_image_tag
  worker_image_tag = var.worker_image_tag

  enable_apns                   = var.enable_apns
  apns_platform_application_arn = var.apns_platform_application_arn

  apple_audience = var.apple_audience

  github_repo           = var.github_repo
  github_branch         = var.github_branch
  github_oidc_thumbprint = var.github_oidc_thumbprint
}

output "api_base_url" {
  value = module.reelclaw.api_base_url
}

output "alb_dns_name" {
  value = module.reelclaw.alb_dns_name
}

output "ecr_api_repo" {
  value = module.reelclaw.ecr_api_repo
}

output "ecr_worker_repo" {
  value = module.reelclaw.ecr_worker_repo
}

output "github_deploy_role_arn" {
  value = module.reelclaw.github_deploy_role_arn
}
