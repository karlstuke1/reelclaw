output "api_base_url" {
  value = aws_apigatewayv2_api.api.api_endpoint
}

output "alb_dns_name" {
  value = aws_lb.api.dns_name
}

output "jobs_table_name" {
  value = aws_dynamodb_table.jobs.name
}

output "devices_table_name" {
  value = aws_dynamodb_table.devices.name
}

output "uploads_bucket_name" {
  value = aws_s3_bucket.uploads.bucket
}

output "outputs_bucket_name" {
  value = aws_s3_bucket.outputs.bucket
}

output "ecr_api_repo" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_worker_repo" {
  value = aws_ecr_repository.worker.repository_url
}

output "github_deploy_role_arn" {
  value = try(aws_iam_role.github_deploy[0].arn, "")
}
