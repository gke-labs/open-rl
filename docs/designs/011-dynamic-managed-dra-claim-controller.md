# Design Doc 011: Dynamic Managed DRA Claim Controller & GPU Worker Packing Proxy

## 1. Executive Summary

This document proposes a dynamic, Gateway-managed Dynamic Resource Allocation (DRA) claim controller for Open-RL. By leveraging node-pool-level hardware labels (`open-rl.io/memory-tier`, `open-rl.io/workload-type`, `open-rl.io/role`), the Open-RL Gateway acts as an intelligent DRA claim proxy that dynamically provisions, packs (`maxWorkersPerClaim`), and garbage-collects managed Kubernetes `ResourceClaim` objects (`resource.k8s.io/v1`).

---

## 2. Motivation & Problem Statement

Currently, Open-RL relies on static, pre-declared DRA `ResourceClaim` manifests (`06b-lora-gpu-resourceclaim.yaml`, `08b-lora-sampler-resourceclaim.yaml`). 

### Limitations:
1. **Manual Manifest Pre-allocation**: Static claims must be applied via `kubectl apply` prior to running jobs.
2. **Fixed Capacity Limits**: Static claim names (`open-rl-lora-trainer-gpu-1`) restrict the cluster from dynamically scaling claim handles as new jobs arrive.
3. **Lack of Automated Garbage Collection**: Unused claims remain allocated in the API server even when no active worker pods reference them.

---

## 3. Architecture Overview

### 3.1 Node Pool Level Hardware Labeling
GKE Node Pools are provisioned with permanent infrastructure labels:

| Node Pool | GPU Hardware | Applied Node Pool Labels |
| :--- | :--- | :--- |
| `np-lora-trainer` | NVIDIA L4 (24GB) | `open-rl.io/workload-type=lora`, `open-rl.io/memory-tier=24gb`, `open-rl.io/role=trainer` |
| `np-lora-sampler` | NVIDIA L4 (24GB) | `open-rl.io/workload-type=lora`, `open-rl.io/memory-tier=24gb`, `open-rl.io/role=sampler` |
| `np-fft-trainer` | NVIDIA H100 (80GB) | `open-rl.io/workload-type=full`, `open-rl.io/memory-tier=80gb`, `open-rl.io/role=trainer` |
| `np-fft-sampler` | NVIDIA H100 (80GB) | `open-rl.io/workload-type=full`, `open-rl.io/memory-tier=80gb`, `open-rl.io/role=sampler` |

---

### 3.2 Dynamic Claim Resolution & Worker Packing

When a new SFT or RL training job is submitted:

1. **Pre-flight VRAM Estimation**: Gateway estimates peak VRAM requirements (e.g. 24GB tier for LoRA vs 80GB tier for FFT).
2. **Worker Packing Check**: Gateway inspects active managed claims matching `memory-tier` and `role`. If an existing claim has active workers less than `maxWorkersPerClaim` (e.g., 2 workers per L4 GPU), the new worker pod is packed onto the existing claim.
3. **Dynamic Claim Creation**: If all active claims are at capacity but node pool capacity exists, Gateway dynamically creates a new managed `ResourceClaim` (`open-rl-managed-<role>-<hash>`) using the Kubernetes CustomObjects API.

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: open-rl-managed-lora-trainer-a1b2c3
  labels:
    open-rl.io/managed-by: "open-rl-gateway"
    open-rl.io/workload-type: "lora"
    open-rl.io/role: "trainer"
    open-rl.io/memory-tier: "24gb"
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
        selectors:
        - cel:
            expression: device.attributes['gpu.nvidia.com'].productName == 'NVIDIA L4'
```

---

### 3.3 Dynamic Garbage Collection

A background reconciliation thread inside `K8sWorkerManager` periodically checks all claims labeled `open-rl.io/managed-by=open-rl-gateway`:

- If a managed claim has `0` active worker pods referencing it, Gateway automatically deletes the claim via `custom_api.delete_namespaced_custom_object()`.
- Deallocating the claim releases physical GPU resources back to the node pool.

---

## 4. End-to-End Execution Sequence

```
Client               Gateway               K8s API Server           DRA Plugin / Node
  │                     │                         │                         │
  │── Submit Job ──────>│                         │                         │
  │                     │── Estimate VRAM ───────>│                         │
  │                     │── Check Active Claims ─>│                         │
  │                     │                         │                         │
  │                     │── (If Capacity Exists) ─│                         │
  │                     │   Create Managed Claim >│                         │
  │                     │                         │                         │
  │                     │── Spawn Worker Pod ────>│                         │
  │                     │   (with claimName &     │── Bind Physical GPU ───>│
  │                     │    nodeSelector)        │   from Labeled Pool     │
  │                     │                         │                         │
  │<─ Training Active ──│                         │                         │
  │                     │                         │                         │
  │── Job Complete ────>│                         │                         │
  │                     │── Delete Worker Pod ───>│                         │
  │                     │── Reconcile Claims ────>│                         │
  │                     │   (Delete 0-worker claim)                         │
```

---

## 5. Key Benefits

1. **Zero Manual Claim Declarations**: Eliminates static Kustomize claim manifests.
2. **GPU Packing Efficiency**: Maximizes hardware density via configurable `maxWorkersPerClaim`.
3. **Automated Resource Hygiene**: Dynamically cleans up claims upon worker pod termination, preventing claim leaks.
4. **Hardware Tier Guarantee**: CEL selectors and node pool selectors guarantee LoRA workloads stay on L4 nodes and FFT workloads stay on H100 nodes.
