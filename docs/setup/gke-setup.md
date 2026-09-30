# GKE Setup Guide

This guide describes how to create a minimal GKE Standard cluster to run OpenRL workloads. It installs the OpenRL API server, the scheduler, Redis, and a shared Filestore PVC into the `openrl-system` namespace. Trainer and sampler workers are not deployed up front: the API server asks the scheduler for one when a client creates a model, and the scheduler places it on a GPU through a DRA `ResourceClaim`.

## Shape

| Component | Minimum used here | Why |
| --- | --- | --- |
| GKE version | `1.35` or newer | Required for [DRA for GPUs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/set-up-dra). |
| CPU node pool | `1 x e2-standard-4` | API server, scheduler, Redis, system pods. |
| GPU node pool | `1 x g2-standard-24` | Two NVIDIA L4 GPUs. LoRA trainer and sampler workers each need their own GPU. |
| Shared storage | `1Ti standard-rwx` Filestore PVC | Shared adapter snapshots, checkpoints, and Hugging Face cache. |
| Server images | one API server image, one worker image | Trainer and sampler workers share the worker image. |

Google references:

- GKE DRA for GPUs: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/set-up-dra
- G2 / NVIDIA L4 machine specs: https://docs.cloud.google.com/compute/docs/gpus#g2-vms
- Filestore CSI driver and `standard-rwx`: https://docs.cloud.google.com/filestore/docs/csi-driver

## 1. Set Variables

Choose a region and zone that have L4 capacity and quota.

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER="open-rl-ttsql"
```

Optional quota/capacity sanity check:

```bash
gcloud compute accelerator-types list \
  --filter="zone:(${ZONE}) AND name:nvidia-l4"
```

## 2. Create the Cluster

Enable the APIs used by this guide:

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  file.googleapis.com
```

Create the GKE Standard cluster with a small CPU node pool. DRA for GPUs needs GKE 1.35 or newer; add `--cluster-version` if the release channel default is older:

```bash
gcloud container clusters create "${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=100
```

Enable the managed Filestore CSI driver:

```bash
gcloud container clusters update "${CLUSTER}" \
  --location="${REGION}" \
  --update-addons=GcpFilestoreCsiDriver=ENABLED
```

> [!TIP]
> **Custom VPC Networks:** If your GCP project does not have a `default` VPC network, GKE's pre-provisioned Filestore StorageClasses will fail to provision. You will need to create a custom `StorageClass` that explicitly specifies your network (e.g., `network: your-vpc-name`) and update the PVC manifest to reference it.

Add a GPU node pool. Workers get their GPUs through DRA `ResourceClaim`s instead of the GKE device plugin, so the pool disables the default device plugin and the automatic driver install. The `openrl.io/enabled=true` label opts the nodes in to the OpenRL scheduler. A node with no `openrl.io/trainer` or `openrl.io/sampler` label accepts both roles, so both workers can land on this node, each on its own GPU.

```bash
gcloud container node-pools create open-rl-l4 \
  --cluster="${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=g2-standard-24 \
  --accelerator=type=nvidia-l4,count=2,gpu-driver-version=disabled \
  --node-labels=openrl.io/enabled=true,gke-no-default-nvidia-gpu-device-plugin=true,nvidia.com/gpu.present=true \
  --image-type=COS_CONTAINERD \
  --num-nodes=1 \
  --disk-size=200
```

Connect `kubectl`:

```bash
gcloud container clusters get-credentials "${CLUSTER}" --location="${REGION}"
```

Install the NVIDIA GPU driver, then the NVIDIA DRA driver (needs Helm v3), which publishes the GPUs as ResourceSlices for the scheduler to place against:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia/"
```

Check that both GPUs are published:

```bash
kubectl get resourceslices
```

## 3. Deploy OpenRL

Apply **one** release bundle. Each release publishes two:

| Bundle | Workers |
| --- | --- |
| `openrl-lora.yaml` | LoRA |
| `openrl-fft.yaml` | LoRA and full fine-tuning (FFT), plus the accelerator time-slicer and llm-d snapshot agent |

```bash
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

Replace `latest/download` with `download/<tag>` to pin a specific version. Server-side apply is required because the Workload CRD exceeds the client-side apply annotation limit. See [OpenRL on an existing DRA cluster](lora-dra.md) for what each bundle installs, and for using storage other than Filestore.

To track unreleased code on `main` instead, render the overlay from a checkout:

```bash
make render OVERLAY=k8s/deploy/lora VERSION=latest | kubectl apply --server-side -f -
```

Wait for the shared storage (PVC) to be bound:

```bash
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
```

Wait for the deployments to become ready:

```bash
kubectl -n openrl-system rollout status deploy/redis-store
kubectl -n openrl-system rollout status deploy/open-rl-scheduler
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

Useful logs:

```bash
kubectl -n openrl-system logs deploy/open-rl-api-server -f
kubectl -n openrl-system logs deploy/open-rl-scheduler -f
```

Worker pods appear once a client creates a model. Inspect them and their GPU allocations with:

```bash
kubectl -n openrl-system get workloads,resourceclaims,pods
```

## 4. Port-Forward the API server

To access the API server from your local machine:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

Smoke test:

```bash
curl http://127.0.0.1:9003/api/v1/healthz
curl http://127.0.0.1:9003/api/v1/get_server_capabilities
```

The OpenRL server is now available at `http://127.0.0.1:9003`.

## 5. Clean Up

Delete the cluster:

```bash
gcloud container clusters delete "${CLUSTER}" --location="${REGION}"
```
