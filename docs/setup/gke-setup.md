# GKE Setup Guide

This guide describes how to create a minimal GKE Standard cluster to run OpenRL workloads. It sets up the OpenRL gateway, one vLLM worker, one trainer worker, Redis, and a shared Filestore PVC.

This guide is based on the [Text-to-SQL recipe](../../examples/text-to-sql/README.md) requirements.

## Shape

| Component | Minimum used here | Why |
| --- | --- | --- |
| CPU node pool | `1 x e2-standard-4` | Gateway, Redis, system pods. |
| GPU node pool | `1 x g2-standard-24` | Two NVIDIA L4 GPUs, one for vLLM and one for the trainer. |
| GPU VRAM | `2 x 24 GB` | Expected separate 24 GB-class GPUs. |
| Shared storage | `100Gi standard-rwx` Filestore PVC | Shared adapter snapshots, checkpoints, and Hugging Face cache. |
| Server images | one gateway image, one worker image | vLLM and trainer share the worker image. |

Google references:

- GKE Standard GPU node pools: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpus
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

Create the GKE Standard cluster with a small CPU node pool:

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

Add a GPU node pool. You should name it something that identifies it (e.g., `open-rl-l4`), and ensure your recipe's Kustomize overlay selects this node pool by name. GKE exposes each GPU to a pod as `nvidia.com/gpu: 1`, so the vLLM and trainer pods can land on the same node but use separate GPUs.

```bash
gcloud container node-pools create open-rl-l4 \
  --cluster="${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=g2-standard-24 \
  --accelerator=type=nvidia-l4,count=2,gpu-driver-version=default \
  --image-type=COS_CONTAINERD \
  --num-nodes=1 \
  --disk-size=200 \
  --node-taints=nvidia.com/gpu=present:NoSchedule
```

Connect `kubectl`:

```bash
gcloud container clusters get-credentials "${CLUSTER}" --location="${REGION}"
```

## 3. Deploy OpenRL

Deploy the manifests using the Kustomize overlay. You should apply **only one** of the following, depending on your needs:

*   **Option A: Generic Base Setup** (without recipe-specific configurations):
    ```bash
    kubectl apply -k k8s/deploy/distributed-shared
    ```

*   **Option B: Recipe-Specific Setup** (e.g., for Text-to-SQL):
    The recipe overlay automatically includes the base setup and applies specific customizations. You do not need to apply the base setup separately.
    ```bash
    kubectl apply -k examples/text-to-sql
    ```

Wait for the shared storage (PVC) to be bound:

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
```

Wait for the deployments to become ready:

```bash
kubectl rollout status deploy/redis-store
kubectl rollout status deploy/open-rl-gateway
kubectl rollout status deploy/vllm-worker
kubectl rollout status deploy/open-rl-trainer-worker
```

Useful logs:

```bash
kubectl logs deploy/vllm-worker -f
kubectl logs deploy/open-rl-trainer-worker -f
kubectl logs deploy/open-rl-gateway -f
```

## 4. Port-Forward the Gateway

To access the gateway from your local machine:

```bash
kubectl port-forward svc/open-rl-gateway-service 9003:8000
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
