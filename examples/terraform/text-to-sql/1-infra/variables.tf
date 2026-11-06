variable "project_id" {
  description = "GCP project with a billing account. Terraform enables the APIs it needs."
  type        = string
}

variable "zone" {
  description = "Zone with NVIDIA L4 (g2) capacity."
  type        = string
  default     = "us-central1-a"
}

variable "gpu_spot" {
  description = "Use a Spot VM for the GPU node: cheaper and often easier to get, but preemptible."
  type        = bool
  default     = false
}
