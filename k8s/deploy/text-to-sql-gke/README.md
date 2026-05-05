# Text-to-SQL GKE Guide

This guide creates a minimal GKE Standard cluster that can run the Gemma 4
Text-to-SQL SFT+RL recipe with the Open-RL gateway, one vLLM worker, one trainer
worker, Redis, and a shared Filestore PVC.

The cluster runs the backend only. You run the recipe client from your laptop or
Cloud Shell through a `kubectl port-forward`, which keeps the training script
exactly the same as local development.

## Shape

| Component | Minimum used here | Why |
| --- | --- | --- |
| CPU node pool | `1 x e2-standard-4` | Gateway, Redis, system pods. |
| GPU node pool | `1 x g2-standard-24` | Two NVIDIA L4 GPUs, one for vLLM and one for the trainer. |
| GPU VRAM | `2 x 24 GB` | The Text-to-SQL recipe expects separate 24 GB-class GPUs. |
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

Add one two-GPU L4 node pool named `ttsql-l4`. The overlay selects this node
pool by name. GKE exposes each GPU to a pod as
`nvidia.com/gpu: 1`, so the vLLM and trainer pods can land on the same node but
use separate GPUs.

```bash
gcloud container node-pools create ttsql-l4 \
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

## 3. Deploy Open-RL

```bash
kubectl apply -k k8s/deploy/text-to-sql-gke
```

Watch the PVC and pods:

```bash
kubectl get pvc open-rl-shared-pvc -w
```

```bash
kubectl get pods -l app.kubernetes.io/part-of=text-to-sql -w
```

Wait for the deployments:

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

In a separate terminal:

```bash
kubectl port-forward svc/open-rl-gateway-service 9003:8000
```

Smoke test:

```bash
curl http://127.0.0.1:9003/api/v1/healthz
curl http://127.0.0.1:9003/api/v1/get_server_capabilities
```

## 5. Run the Text-to-SQL Recipe

From your local checkout:

```bash
cd examples/rl/text-to-sql
TINKER_BASE_URL=http://127.0.0.1:9003 \
TINKER_API_KEY=tml-dummy \
uv run python texttosql_sft_grpo.py gemma4_e2b_rl_recipe
```

## 6. Visualizing Metrics

The training script logs metrics locally into the
`examples/rl/text-to-sql/artifacts/` directory.

Use the Text-to-SQL plotting utility to render the standard 4-panel recipe
figure:

```bash
cd examples/rl/text-to-sql
uv run python -m utils.plot \
  artifacts/texttosql_sft_grpo_gemma4_e2b_rl_recipe_full/metrics.jsonl
```

## 7. Clean Up

Delete the Open-RL Kubernetes resources:

```bash
kubectl delete -k k8s/deploy/text-to-sql-gke
```

Delete the cluster:

```bash
gcloud container clusters delete "${CLUSTER}" --location="${REGION}"
```
