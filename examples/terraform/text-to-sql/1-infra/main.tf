provider "google" {
  project = var.project_id
}

# APIs the cluster and its Filestore share need.
resource "google_project_service" "services" {
  for_each           = toset(["compute.googleapis.com", "container.googleapis.com", "file.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

# Standard (not Autopilot) cluster on the default network. GKE's default
# version (1.36 today) supports DRA for GPUs, which OpenRL needs.
resource "google_container_cluster" "openrl" {
  name                = "open-rl-ttsql"
  location            = var.zone
  initial_node_count  = 0 # no default node pool; the GPU pool is below
  deletion_protection = false

  # Backs the bundle's shared volume (a standard-rwx Filestore PVC).
  addons_config {
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  depends_on = [google_project_service.services]
}

# One node with two L4s: one for the LoRA trainer, one for the vLLM sampler.
# The OpenRL control plane runs here too.
resource "google_container_node_pool" "gpu" {
  name       = "l4"
  cluster    = google_container_cluster.openrl.id
  node_count = 1

  node_config {
    machine_type = "g2-standard-24"
    disk_size_gb = 200 # the worker image is ~24GB, plus model weights
    spot         = var.gpu_spot

    # Workers get GPUs through DRA ResourceClaims, so GKE's driver install
    # and default device plugin are off; ../2-cluster installs the NVIDIA
    # driver and the NVIDIA DRA driver instead.
    guest_accelerator {
      type  = "nvidia-l4"
      count = 2
      gpu_driver_installation_config {
        gpu_driver_version = "INSTALLATION_DISABLED"
      }
    }

    labels = {
      "openrl.io/enabled"                       = "true" # opt in to the OpenRL scheduler
      "gke-no-default-nvidia-gpu-device-plugin" = "true"
      "nvidia.com/gpu.present"                  = "true"
    }
  }
}
