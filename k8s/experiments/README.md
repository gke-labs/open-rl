# Kubernetes Dynamic Resource Allocation (DRA) Experiments

This directory contains experimental smoke tests and verification manifests for validating Kubernetes Dynamic Resource Allocation (DRA) oversubscription. 

In particular, `dra-shared-gpu-smoke.yaml` verifies that two independent Pods can reference a single shared `ResourceClaim` (`shared-gpu`) to get colocated onto the exact same physical NVIDIA GPU allocation without serializing CUDA execution or relying on standard time-sharing device plugins.

---

## Prerequisites & Architectural Notes

- **GKE Standard Cluster (v1.35+)**: GKE Autopilot clusters are **not supported** for these experiments. Autopilot's Warden admission webhook rejects custom Pod node selectors (e.g. `group.timeslice.io/trainers`), and Google manages kubelet runtime binaries, preventing custom third-party DRA kubelet plugin deployment.
- **Helm v3**: Required to deploy the experimental NVIDIA DRA driver chart.
- **Quota**: Ensure your GCP target project has available GCE quota for L4 GPUs (`nvidia-l4`) in the selected zone.

---

## Step-by-Step Cluster Setup & Execution

### 1. Create Base GKE Standard Cluster
Create a minimal GKE Standard cluster with a small standard CPU node pool for system components:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export CLUSTER="open-rl-dra"

gcloud container clusters create "${CLUSTER}" \
  --location="${REGION}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=100
```

### 2. Connect `kubectl` & Add Dedicated DRA GPU Node Pool
Connect credentials to your local shell:
```bash
gcloud container clusters get-credentials "${CLUSTER}" --location="${REGION}"
```

Add a dedicated L4 GPU node pool. 
- **Driver Disabled**: We explicitly pass `gpu-driver-version=disabled` to prevent GKE from installing the default device plugin, allowing the DRA driver to manage GPU device discovery.
- **Single Zone Target**: When attaching a GPU pool to a regional cluster, explicitly pass `--node-locations` targeting a specific zone with L4 capacity (e.g. `us-central1-b`).

```bash
gcloud container node-pools create gpu-dra \
  --cluster="${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="us-central1-b" \
  --machine-type=g2-standard-12 \
  --accelerator="type=nvidia-l4,count=1,gpu-driver-version=disabled" \
  --node-labels="group.timeslice.io/trainers=true,group.timeslice.io/samplers=true,gke-no-default-nvidia-gpu-device-plugin=true,nvidia.com/gpu.present=true" \
  --num-nodes=2
```

### 3. Install NVIDIA Runtime Driver & DRA Kubelet Plugin
Install the preloaded Container-Optimized OS (COS) NVIDIA driver DaemonSet:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

Install the experimental NVIDIA DRA Driver Helm chart (`nvidia-dra-driver-gpu`).
> [!IMPORTANT]
> **Gotcha - GPU DeviceClass Guard**: By default, the Helm chart disables publishing `DeviceClass: gpu.nvidia.com`. You **must** pass `--set resources.gpus.enabled=true --set gpuResourcesEnabledOverride=true` to force the kubelet plugin to register physical GPU resource slices.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia/" \
  --set resources.gpus.enabled=true \
  --set gpuResourcesEnabledOverride=true
```

> [!TIP]
> **Gotcha - ResourceQuota Scope Limitations**: In tenant or sandbox GCP projects (e.g. Anthos/Config Controller test environments), standard `ResourceQuota` policies often forbid Pods from requesting `system-node-critical` priority classes. If the DRA kubelet plugin fails to spawn, strip the priority class request:
> ```bash
> kubectl patch ds -n nvidia-dra-driver-gpu nvidia-dra-driver-gpu-kubelet-plugin \
>   --type=json -p='[{"op": "remove", "path": "/spec/template/spec/priorityClassName"}]'
> ```

Verify that physical GPU resource slices and device classes are published:
```bash
kubectl get deviceclasses
kubectl get resourceslices
```

---

## 4. Run the DRA Smoke Test

Apply the smoke test experiment:
```bash
kubectl apply -f k8s/experiments/dra-shared-gpu-smoke.yaml
```

### Verification Benchmarks (Pass Criteria)

1. **Colocated Node Placement**: Verify both Pods reach `Running` on the exact same GPU node:
   ```bash
   kubectl get pods -n dra-smoke -o wide
   ```
2. **Hardware VRAM Sharing**: Verify both isolated container logs output the exact same GPU UUID:
   ```bash
   kubectl logs -n dra-smoke smoke-a
   kubectl logs -n dra-smoke smoke-b
   ```
3. **ResourceClaim Consumer Binding**: Verify the shared claim reserves one single physical device (`gpu-0`) simultaneously for two distinct workload PIDs:
   ```bash
   kubectl describe resourceclaim -n dra-smoke shared-gpu
   ```

---

## 5. Teardown & Cleanup

```bash
kubectl delete ns dra-smoke
gcloud container clusters delete "${CLUSTER}" --location="${REGION}" --quiet
```
