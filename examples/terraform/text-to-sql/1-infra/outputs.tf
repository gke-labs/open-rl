# Read by ../2-cluster through terraform_remote_state.

output "project_id" {
  value = var.project_id
}

output "cluster_endpoint" {
  value = google_container_cluster.openrl.endpoint
}

output "cluster_ca_certificate" {
  value = google_container_cluster.openrl.master_auth[0].cluster_ca_certificate
}

output "get_credentials" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.openrl.name} --zone ${var.zone} --project ${var.project_id}"
}
