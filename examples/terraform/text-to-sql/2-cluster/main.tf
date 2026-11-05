# Only the providers outside the hashicorp/ namespace need declaring; Terraform
# finds google, helm and http on its own.
terraform {
  required_providers {
    kubectl       = { source = "alekc/kubectl" }
    kustomization = { source = "kbst/kustomization" } # third-party, see step 3
  }
}

# The cluster from ../1-infra. This is a separate root because the Helm and
# kubectl providers can only be configured once the cluster exists.
data "terraform_remote_state" "infra" {
  backend = "local"
  config  = { path = "${path.module}/../1-infra/terraform.tfstate" }
}

locals {
  infra = data.terraform_remote_state.infra.outputs
  kube = {
    host                   = "https://${local.infra.cluster_endpoint}"
    cluster_ca_certificate = base64decode(local.infra.cluster_ca_certificate)
    token                  = data.google_client_config.current.access_token # your gcloud identity
  }
}

provider "google" {
  project = local.infra.project_id
}

data "google_client_config" "current" {}

provider "helm" {
  kubernetes = local.kube
}

provider "kubectl" {
  host                   = local.kube.host
  cluster_ca_certificate = local.kube.cluster_ca_certificate
  token                  = local.kube.token
  load_config_file       = false
}

# 1. NVIDIA kernel driver + CUDA libraries, from Google's COS installer
#    DaemonSet (GKE's own driver install is off on the GPU pool).
data "http" "nvidia_driver_installer" {
  url = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml"
}

resource "kubectl_manifest" "nvidia_driver_installer" {
  yaml_body         = data.http.nvidia_driver_installer.response_body
  server_side_apply = true
}

# 2. NVIDIA DRA driver: publishes each GPU as a ResourceSlice, which the OpenRL
#    scheduler claims for every trainer and sampler worker. Installed in
#    kube-system, where GKE admits its node-critical pods without extra quota.
resource "helm_release" "nvidia_dra_driver" {
  name       = "nvidia-dra-driver-gpu"
  repository = "https://helm.ngc.nvidia.com/nvidia"
  chart      = "nvidia-dra-driver-gpu"
  version    = "25.8.0"
  namespace  = "kube-system"
  wait       = false # its kubelet plugin waits for the kernel driver

  values = [yamlencode({
    nvidiaDriverRoot            = "/home/kubernetes/bin/nvidia/"
    gpuResourcesEnabledOverride = true # required to enable GPU allocation in 25.8.0
  })]

  depends_on = [kubectl_manifest.nvidia_driver_installer]
}

# 3. OpenRL control plane (API server, scheduler, Redis, CRDs, shared volume).
#
# ############################################################################
# NOTE: THIRD-PARTY PROVIDER. kbst/kustomization
# (https://registry.terraform.io/providers/kbst/kustomization) is not a
# HashiCorp provider. It is used only to render the repo's own kustomize
# overlay, examples/text-to-sql (the LoRA bundle k8s/deploy/lora with
# Gemma 4 E2B as the default model), at plan time. That keeps the source of truth in
# the repo and in sync with the commit you have checked out: there is no
# rendered or vendored copy of the bundle to drift. As a result, any
# `git pull` that changes k8s/ shows up in your next `terraform plan`.
# ############################################################################
#
# The rendered bundle is written to charts/openrl/.tmp/ (gitignored), which
# the chart prints verbatim. The bundle creates its own namespace,
# openrl-system.
# Only renders here, so the provider needs no cluster credentials.
provider "kustomization" {
  kubeconfig_raw = ""
}

data "kustomization_build" "openrl" {
  path = "${path.module}/../../../../examples/text-to-sql"
}

locals {
  openrl_bundle = join("\n---\n", values(data.kustomization_build.openrl.manifests))
}

# Written under the chart (Helm can only read files inside it) and gitignored.
# "sensitive" only keeps the ~600 KB bundle out of the plan output.
resource "local_sensitive_file" "openrl_bundle" {
  filename = "${path.module}/charts/openrl/.tmp/openrl-lora.yaml"
  content  = local.openrl_bundle
}

resource "helm_release" "openrl" {
  name      = "openrl"
  chart     = "${path.module}/charts/openrl"
  namespace = "default"
  timeout   = 900

  # Unused by the chart. Helm only re-reads a local chart when its values or
  # version change, so this makes a changed bundle show up as a one-line diff.
  set = [{ name = "bundleSha256", value = sha256(local.openrl_bundle) }]

  depends_on = [local_sensitive_file.openrl_bundle]
}
