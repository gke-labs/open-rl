# Design Doc 011: Open-RL Workload Scheduler — Dynamic Placement of RL Workloads on Heterogeneous Accelerator Fleets via Kubernetes DRA

## 1. Executive Summary

This document proposes the **Open-RL Workload Scheduler**, a core component of the Open-RL Control Plane responsible for the efficient, dynamic, and elastic scheduling of Reinforcement Learning (RL) workloads (Trainer and Sampler workers) across a heterogeneous accelerator fleet. Rather than attempting low-level GPU device scheduling or relying on static YAML manifests, the Open-RL Workload Scheduler focuses on **scheduling RL workloads** based on their high-level compute and memory characteristics. It leverages Kubernetes Dynamic Resource Allocation (DRA / `resource.k8s.io/v1`) as a foundational infrastructure primitive behind the scenes. The scheduler automatically estimates workload memory requirements, auto-scans cluster accelerator topology (`ResourceSlice` CRDs), provisions dynamic `ResourceClaim` objects using CEL device selectors, packs workers onto shared claim handles based on workload type, and drives GKE Cluster Autoscaler / Node Auto-Provisioning (NAP) for elastic on-demand cloud capacity expansion.

---

## 2. Motivation & Architectural Background

Multi-tenant Reinforcement Learning (RL) workloads—consisting of decoupled Trainer and Sampler workers—present unique placement constraints across heterogeneous accelerator fleets:

1. **RL Workload vs. Low-Level Device Scheduling**: Open-RL does not perform low-level GPU device scheduling. Instead, it **schedules RL workloads (Trainer and Sampler workers)** by translating high-level model requirements into native Kubernetes resource claims, leaving physical device binding to the Kubernetes scheduler and accelerator drivers.
2. **Heterogeneous Fleet Placement**: LoRA workloads on small base models (0.6B) operate cost-effectively on NVIDIA L4 GPUs (24GB VRAM), whereas Full Fine-Tuning (FFT) on 8B+ models requires high-memory NVIDIA H100 GPUs (80GB VRAM).
3. **Workload Isolation & State Management**: FFT workers swap full model and optimizer states via time-slicing daemons, requiring strict isolation from dynamic LoRA adapter workloads.
4. **Static Manifest Overhead & Resource Leakage**: Manual `ResourceClaim` pre-declaration prevents the cluster from scaling dynamically as concurrent RL jobs arrive, while static claims risk locking physical accelerators away from other workloads after job completion.

### 2.1 Why K8s DRA Pod Templates Fail for Shared RL Compute

Kubernetes DRA (`resource.k8s.io/v1`) natively offers `ResourceClaimTemplate` objects, where pods declare a template reference (`pod.spec.resourceClaims[].resourceClaimTemplateName`). Under this default model, the K8s control plane automatically creates a dedicated `ResourceClaim` owned strictly **1-to-1 by that individual pod lifecycle**.

**Why this 1-to-1 Pod Template model fails for Open-RL:**

1. **Cross-Pod GPU Sharing in FFT Workloads**: Open-RL uses accelerator time-slicing to allow multiple FFT training workers to share a single physical GPU by swapping model and optimizer states between training rounds. A pod-owned claim template creates an isolated 1-to-1 GPU binding, preventing secondary worker pods from attaching to the same underlying GPU resource.
2. **Base-Model Weight Sharing in LoRA Workloads**: Multiple LoRA workers sharing the same base model (e.g. `Qwen3-0.6B`) must be scheduled onto the **same claim handle** to share base model weights in host and GPU memory. Pod-owned claim templates force a brand-new, isolated claim instance for every pod, breaking shared-memory model loading.

By placing claim lifecycle management inside the **Open-RL Workload Scheduler**, `ResourceClaim` objects exist as **standalone, shared claim handles** (`open-rl-managed-<workload>-<role>-<hash>`). Multiple worker pods can reference the exact same claim name in `pod.spec.resourceClaims[].resourceClaimName`, unlocking GPU time-slicing, base-model weight reuse, and intelligent worker packing.

### 2.2 DRA as an Infrastructure Primitive — Zero User-Facing Complexity

A core design principle of Open-RL is that **Dynamic Resource Allocation (DRA) is treated strictly as an internal infrastructure primitive**.

#### User Experience (ML Engineer Perspective)
ML engineers and researchers interact with Open-RL exclusively through high-level Python SDKs or simple API payloads:

```python
# User submits job specifying only model intent and tuning strategy
client.create_model(
  base_model="Qwen/Qwen3-8B",
  fine_tuning_type="full",  # or "lora"
)
```

Users are **completely shielded** from:
- Writing or managing Kubernetes YAML manifests or `ResourceClaim` specs.
- Formulating CEL selector expressions or matching driver device classes.
- Specifying GPU node pool labels or tracking physical GPU device UIDs.

#### Automated Control Plane Translation
The Open-RL Control Plane translates high-level ML job intents (`base_model="Qwen/Qwen3-8B"`, `fine_tuning_type="full"`) into precise DRA claims, CEL hardware selectors (`device.attributes['gpu.nvidia.com'].productName == 'NVIDIA H100 80GB HBM3'`), and worker pod specifications behind the scenes. This decouples user productivity from underlying cluster infrastructure mechanics.

---

## 3. System Architecture

```
                               ┌─────────────────────────────────────────┐
                               │          Open-RL Control Plane          │
                               │        (RL Workload Scheduler)          │
                               │                                         │
                               │  1. Estimate VRAM Tier                  │
                               │  2. Scan ResourceSlices                 │
                               │  3. Resolve/Create Shared Claims        │
                               │  4. Reconcile Unused Claims             │
                               └────────────────────┬────────────────────┘
                                                    │
                                   Create Managed   │ List & Create
                                   ResourceClaim    │ Worker Pods
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │               Kubernetes API Server                     │
                       │           (resource.k8s.io/v1 API Group)                │
                       └───────────┬─────────────────────────────────┬───────────┘
                                   │                                 │
                   Discovers CEL   │                                 │ Triggers Node
                   Selectors       ▼                                 ▼ Auto-Provisioning
                       ┌───────────────────────┐         ┌───────────────────────┐
                       │   NVIDIA DRA Driver   │         │  Cluster Autoscaler / │
                       │    (ResourceSlices)   │         │  GKE Auto-Provisioning│
                       └───────────┬───────────┘         └───────────┬───────────┘
                                   │                                 │
                                   ▼                                 ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │             Physical GPU Nodes (L4 & H100)              │
                       └─────────────────────────────────────────────────────────┘
```

---

## 4. Key Technical Components

### 4.1 Workload VRAM Tier & Constraint Estimator

Before launching worker pods, the Workload Scheduler estimates peak memory requirements based on model parameter scale and fine-tuning strategy:

```python
def estimate_memory_tier(base_model: str, fine_tuning_type: str = "lora") -> str:
  """Map model parameters and fine-tuning mode to standardized VRAM tiers.

  - 24gb Tier: LoRA (0.6B to 8B) or small FFT models (<= 1.5B). Targets NVIDIA L4.
  - 80gb Tier: Full Fine-Tuning (7B/8B+) or large models (>= 14B). Targets NVIDIA H100 80GB.
  """
  model_lower = (base_model or "").lower()
  if fine_tuning_type == "full":
    if any(size in model_lower for size in ["7b", "8b", "14b", "32b", "70b"]):
      return "80gb"
    return "24gb"

  if any(size in model_lower for size in ["14b", "32b", "70b"]):
    return "80gb"
  return "24gb"
```

### 4.2 Dynamic `ResourceClaim` Provisioning via CEL Expressions

Instead of assigning static claims, the Workload Scheduler constructs a `ResourceClaim` (`resource.k8s.io/v1`) on-demand using Common Expression Language (CEL) selectors matching driver-published device attributes:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: open-rl-managed-full-trainer-57d0d9
  labels:
    open-rl.io/managed-by: open-rl-control-plane
    open-rl.io/workload-type: full
    open-rl.io/role: trainer
    open-rl.io/memory-tier: 80gb
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        count: 1
        deviceClassName: gpu.nvidia.com
        selectors:
        - cel:
            expression: device.attributes['gpu.nvidia.com'].productName == 'NVIDIA H100 80GB HBM3'
```

### 4.3 Workload Isolation & Worker Packing Control

The Workload Scheduler maintains claim packing capacity (`OPEN_RL_MAX_WORKERS_PER_CLAIM`, default `2`) while enforcing workload type boundaries:

1. **Workload-Type Filtering**: `_discover_eligible_claims()` queries claims matching `workload-type` (`lora` vs `full`), `role` (`trainer` vs `sampler`), and `memory-tier` (`24gb` vs `80gb`). LoRA and FFT workloads never share claims.
2. **Base-Model Affinity**: Multiple LoRA workers sharing the same base model reuse existing dynamic claims to enable local weight sharing.
3. **Capacity Boundaries**: Worker pods are added to an existing claim only if `active_workers < maxWorkersPerClaim`. When capacity is reached, a new dynamic claim is created.

**Example — Scaling Claim Capacity (`maxWorkersPerClaim = 3`):**
- If `OPEN_RL_MAX_WORKERS_PER_CLAIM=3`, the Workload Scheduler allows up to 3 workers per claim before allocating a new dynamic GPU claim.
- In a 3-job scenario (`job-a-8b`, `job-b-8b`, `job-c-8b`), all 3 8B jobs share **1 H100 trainer claim** (`3/3` workers) and **1 H100 sampler claim** (`3/3` workers), optimizing total cluster GPU consumption down to 2 H100 GPUs while maintaining capacity headroom.

### 4.4 Live Cluster Capacity & Topology Discovery

The Workload Scheduler auto-scans published `ResourceSlice` CRDs to discover available GPU products and memory capacities live from the cluster API:

- **Product Attribute**: `device.attributes['gpu.nvidia.com'].productName` (e.g. `'NVIDIA L4'`, `'NVIDIA H100 80GB HBM3'`)
- **Memory Capacity**: `device.capacity.memory.value` (e.g. `23034Mi` vs `81559Mi`)

This auto-discovery mechanism allows the control plane to validate that matching hardware exists or can be provisioned before issuing claim requests.

### 4.5 Dynamic On-Demand Scaling via Cluster Autoscaler / NAP

When cluster GPU capacity is exhausted:

1. Open-RL Workload Scheduler issues the managed `ResourceClaim` and launches the worker pod.
2. The worker pod enters `Pending` state.
3. Kubernetes Cluster Autoscaler (CA) and GKE Node Auto-Provisioning (NAP) inspect the `Pending` pod's `ResourceClaim` CEL selector (`device.attributes['gpu.nvidia.com'].productName == 'NVIDIA H100 80GB HBM3'`).
4. CA/NAP provisions a new GPU node pool on-demand.
5. The node joins the cluster, the NVIDIA DRA driver publishes a new `ResourceSlice`, and the pending claim is allocated immediately.

### 4.6 GKE Capacity Buffers Integration (`CapacityBuffer` CRD)

While dynamic Cluster Autoscaler / NAP scale-up provides on-demand hardware, spinning up cold GPU nodes introduces a 1–2 minute VM provisioning delay. To achieve near-zero pod startup latency for latency-sensitive RL training or inference bursts, Open-RL integrates natively with **GKE Capacity Buffers (`CapacityBuffer` CRD)**:

```yaml
apiVersion: buffer.gke.io/v1
kind: CapacityBuffer
metadata:
  name: open-rl-h100-gpu-buffer
spec:
  provisioningStrategy: buffer.gke.io/active-capacity  # or standby-capacity
  limits:
    resources:
      gpu: "2"
```

**Synergy with Open-RL Workload Scheduler:**
1. **Warm Buffer Headroom**: GKE maintains a pool of pre-provisioned active or standby GPU nodes running the NVIDIA DRA driver.
2. **Instant Claim Binding**: When the Workload Scheduler issues a dynamic `ResourceClaim`, the worker pod binds **instantly** to an available GPU in the capacity buffer without waiting for VM creation.
3. **Automatic Buffer Refill**: As Open-RL worker pods consume units from the buffer, Cluster Autoscaler automatically provisions replacement nodes in the background to maintain buffer headroom.

### 4.7 Multi-GPU Node Partitioning Example: 8x H100 Node (4x Trainer + 4x Sampler)

For large-scale RL workloads running on multi-GPU nodes (e.g. an 8x H100 node like `a3-highgpu-8g`), DRA allows the Workload Scheduler to request an exact count (`count: 4`) within a single claim:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: open-rl-managed-full-trainer-4x-h100
spec:
  devices:
    requests:
    - name: gpus
      exactly:
        count: 4
        deviceClassName: gpu.nvidia.com
        selectors:
        - cel:
            expression: device.attributes['gpu.nvidia.com'].productName == 'NVIDIA H100 80GB HBM3'
```

**Partitioning Flow:**
1. **Trainer Allocation (4x H100)**: Workload Scheduler issues a 4-GPU Trainer claim. The DRA driver allocates physical GPUs `gpu-0`, `gpu-1`, `gpu-2`, `gpu-3` on the node for 4-way Tensor/Pipeline Parallel training.
2. **Sampler Allocation (4x H100)**: Workload Scheduler issues a 4-GPU Sampler claim. The DRA driver allocates the remaining unreserved GPUs `gpu-4`, `gpu-5`, `gpu-6`, `gpu-7` on the same node for 4-way vLLM tensor-parallel inference.

Result: The 8x H100 node is partitioned into two equal 4-GPU sets with zero device collision, topology-aware isolation, and direct host/NVLink inter-process communication.

### 4.8 Automated Garbage Collection & Reconciliation

To prevent claim leaks, `reconcile_managed_claims()` runs periodically inside the control plane:

- Queries all claims labeled `open-rl.io/managed-by=open-rl-control-plane`.
- Cross-references active worker pods (`app in (open-rl-trainer-worker, open-rl-sampler-worker)`).
- If a managed claim has **0 active worker pods referencing it**, the control plane deletes the `ResourceClaim` custom object, releasing physical GPU resources back to the cloud pool.

---

## 5. End-to-End Sequence

```
Client             Control Plane             K8s API Server           DRA Plugin / CA            Node Pool
  │              (Workload Scheduler)               │                        │                       │
  │── Submit Job ──────>│                           │                        │                       │
  │                     │── Estimate VRAM Tier ────>│                        │                       │
  │                     │── Scan ResourceSlices ───>│                        │                       │
  │                     │                           │                        │                       │
  │                     │── Create ResourceClaim ───│                        │                       │
  │                     │   (with CEL Selector)    >│                        │                       │
  │                     │                           │                        │                       │
  │                     │── Spawn Worker Pod ──────>│                        │                       │
  │                     │   (Pod Pending)           │── Evaluate Claim ─────>│                       │
  │                     │                           │   (If No Free GPU)     │── Provision Node ────>│
  │                     │                           │                        │   via Autoscaler/NAP  │
  │                     │                           │                        │                       │
  │                     │                           │<── Publish Slice ──────│<── Node Ready ────────│
  │                     │                           │    (DRA Allocated)     │                       │
  │                     │                           │                        │                       │
  │<─ Training Active ──│                           │                        │                       │
  │                     │                           │                        │                       │
  │── Job Complete ────>│                           │                        │                       │
  │                     │── Delete Worker Pod ─────>│                        │                       │
  │                     │── Reconcile Claims ──────>│                        │                       │
  │                     │   (Delete 0-worker claim)                          │                       │
```

---

## 6. Implementation & In-Cluster Verification

The Open-RL Workload Scheduler was implemented in `src/server/k8s_worker_manager.py` and `src/server/worker_manager.py`, deployed via container image tag `0.6.12`, and verified in-cluster:

1. **Zero Pre-allocated Claims**: Verified `kubectl get resourceclaims` returns zero claims prior to job execution.
2. **Heterogeneous Placement Verification (`fft-gsm8k-rl-x3-hetero-8b-0.6b`)**:
   - `Qwen3-8B` FFT jobs were assigned `80gb` claims (`open-rl-managed-full-trainer-c4ed9c`) and bound to **NVIDIA H100** nodes.
   - `Qwen3-0.6B` FFT jobs were assigned `24gb` claims (`open-rl-managed-full-trainer-70923a`) and bound to **NVIDIA L4** nodes.
3. **Automated Reconciliation**: Upon job completion, `reconcile_managed_claims()` successfully deleted all dynamic claims, returning cluster claim count to zero.
