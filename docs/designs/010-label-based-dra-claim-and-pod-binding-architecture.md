# Design Doc 010: Label-Based DRA Claim & Pod Binding Architecture

**Author:** Open-RL Engineering Team  
**Status:** Proposed Design (`v1.0.0`)  
**Target Component:** `KubernetesWorkerManager` (`src/server/k8s_worker_manager.py`), Kubernetes DRA Claims (`k8s/deploy/distributed-fft-timeslice/`)  
**Related Documents:** [Design Doc 007](file:///usr/local/google/home/sunilarora/open-rl/docs/designs/007-configurable-dra-claim-pools-and-worker-scheduler.md), [Design Doc 009](file:///usr/local/google/home/sunilarora/open-rl/docs/designs/009-sampling-request-dispatch-and-worker-selection.md)  

---

## 1. Executive Summary

As Open-RL scales to multi-tenant Reinforcement Learning (RL) across heterogeneous Kubernetes GPU clusters (e.g., NVIDIA L4 for LoRA and NVIDIA H100 for Full Fine-Tuning), hardcoding Dynamic Resource Allocation (DRA) claim names into pod templates creates severe scheduling bottlenecks.

While static ConfigMaps (`open-rl-dra-pools`) or regex-based model matching tables can decouple claim names from code, they introduce brittle configuration registries that require manual updates whenever models or GPU nodes change. Furthermore, they fail to address **LoRA Base-Model Mutual Exclusion** and **Least-Privilege Kubernetes RBAC Security**.

This design document specifies a **Label-Based DRA Claim & Pod Binding Architecture** that uses Kubernetes-native label discovery and active Pod metadata as the single source of truth for GPU scheduling.

### Key Highlights
1. **Declarative Label Discovery**: Platform engineers create K8s `ResourceClaim` objects labeled with immutable hardware capabilities (`open-rl.io/workload-type`, `open-rl.io/role`, `open-rl.io/memory-tier`). The worker manager discovers eligible claims dynamically via K8s API label selectors.
2. **LoRA Base-Model Mutual Exclusion**: Enforcing that multiple LoRA jobs for the same base model share an assigned claim/pod, while jobs for different base models are strictly prevented from sharing the same physical GPU claim.
3. **Least-Privilege RBAC Architecture**: The Gateway ServiceAccount (`open-rl-sa`) only requires **read-only (`get`, `list`, `watch`)** access to DRA `ResourceClaim` objects. Bound model leases are tracked on **active Pod labels** (`open-rl.io/assigned-claim`, `open-rl.io/bound-base-model`), eliminating the need for elevated write/patch permissions on cluster DRA resources.
4. **Memory-Aware Routing & Auto-Promotion**: Automatically routing workloads across GPU memory tiers (`24gb` L4 vs `80gb` H100) based on `min_vram_gb` hints or SDK-computed footprint estimates.
5. **Comprehensive Trade-off Analysis**: Evaluating label discovery vs. ConfigMaps, pod metadata registries vs. claim mutation, memory tier labels vs. `ResourceSlice` inspection, and bin-packing vs. spread scheduling.

---

## 2. Motivation & Problem Statement

### 2.1 Limitations of Existing Approaches
1. **Hardcoded Manifests (`v0.6.5`)**: `KubernetesWorkerManager` currently hardcodes `"open-rl-lora-trainer-gpu-1"` for LoRA and `"open-rl-trainer-gpu-1"` for FFT. This prevents running multiple base models concurrently or scaling beyond a single physical GPU per tier.
2. **Brittle ConfigMap / Regex Mapping**: Mapping `base_model` regex patterns (`*70B*`, `*8B*`) to physical claims is arbitrary and maintenance-heavy. Every new model family requires updating regex tables.
3. **LoRA VRAM Collisions**: In Open-RL, a LoRA worker pod loads **one Base Model into GPU VRAM** and swaps lightweight PEFT adapters over it. If two jobs with **different** base models (e.g., `Qwen3-0.6B` and `Qwen3-8B`) are scheduled onto the same physical claim, they will immediately collide and crash with `CUDA out of memory`.
4. **RBAC Security Risks**: Granting a workload ServiceAccount `update` or `patch` permissions on `resourceclaims.resource.k8s.io` to mark claims as "in-use" violates least-privilege cluster security.

### 2.2 Architectural Goals
- **100% Kubernetes-Native Discovery**: Use K8s API label selectors as the dynamic claim pool registry.
- **Zero Pod `nodeSelector` Duplication**: Rely on native DRA CEL selectors (`device.capacity['memory']` or `DeviceClass`) inside `ResourceClaim` specs to handle 100% of node placement.
- **Strict Base-Model Mutual Exclusion**: Prevent VRAM collisions by locking a LoRA claim to a single base model until its active pods terminate.
- **Zero RBAC Overhead**: Operate with read-only access to DRA claims and CRUD access to Pods.

---

## 3. High-Level Architecture & Label Schema

```text
       ┌──────────────────────────────────────────────────────────────┐
       │                 1. Platform Admins Create                    │
       │                   Labeled DRA Claims                         │
       │  "lora-l4-gpu-1": [workload-type=lora, memory-tier=24gb]    │
       │  "lora-l4-gpu-2": [workload-type=lora, memory-tier=24gb]    │
       │  "fft-h100-gpu-1": [workload-type=full, memory-tier=80gb]   │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │            2. Worker Manager Discovers Claims                │
       │     k8s.list_resource_claims(label_selector="...lora...")     │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │          3. Inspect Active Pods (The Lock Registry)          │
       │     k8s.list_pods(label_selector="app=open-rl-trainer-worker")│
       └──────────────┬───────────────────────────────┬───────────────┘
                      │                               │
       Option A: Existing Pod Found    Option B: All Existing Pods
       for "Qwen3-0.6B" on Claim 1     on Claim 1 Run Different Models
                      │                               │
                      ▼                               ▼
       ┌────────────────────────────┐  ┌────────────────────────────┐
       │   REUSE CLAIM 1 / POD 1    │  │  SELECT IDLE CLAIM 2       │
       │  (Share Adapter VRAM Pool) │  │  (Stamp New Pod for 8B)    │
       └────────────────────────────┘  └────────────────────────────┘
```

### 3.1 Immutable DRA Claim Labels (Platform Admin Schema)
When cluster administrators provision K8s `ResourceClaim` objects, they apply immutable metadata tags defining the hardware capability:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: lora-l4-trainer-gpu-1
  namespace: default
  labels:
    open-rl.io/workload-type: "lora"       # "lora" | "full"
    open-rl.io/role: "trainer"             # "trainer" | "sampler"
    open-rl.io/memory-tier: "24gb"         # "24gb" | "40gb" | "80gb"
    open-rl.io/pool: "lora-default"        # Logical grouping name
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
      selectors:
      - cel:
          # Native DRA constraint: guarantees K8s schedules to an L4 node
          expression: "device.attributes['gpu.nvidia.com'].productName == 'NVIDIA L4'"
```

### 3.2 Dynamic Pod Labels (Workload Binding Registry)
When `KubernetesWorkerManager` launches a worker Pod, it labels the Pod with the assigned claim and loaded base model:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: open-rl-trainer-qwen3-0-6b-1
  namespace: default
  labels:
    app: "open-rl-trainer-worker"
    open-rl.io/workload-type: "lora"
    open-rl.io/role: "trainer"
    open-rl.io/assigned-claim: "lora-l4-trainer-gpu-1"   # <-- Bound Claim Handle
    open-rl.io/bound-base-model: "qwen-qwen3-0-6b"       # <-- VRAM Lock Identity
spec:
  containers:
  - name: trainer-worker
    ...
  resourceClaims:
  - name: lora-l4-trainer-gpu-1                         # <-- No nodeSelector needed!
```

---

## 4. Claim Discovery & Selection Algorithm

When a training job arrives for `(role="trainer", base_model="Qwen/Qwen3-8B", ft_type="lora", min_vram_gb=24)`, the worker manager executes the following scheduling pipeline:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     Step 1: Discover Candidate Claims                  │
│  Query K8s API: list_namespaced_custom_object("resourceclaims")        │
│  Filter by label: workload-type=lora, role=trainer, memory-tier>=24gb  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Step 2: Inspect Active Pod Locks                   │
│  Query K8s API: list_namespaced_pod("app=open-rl-trainer-worker")      │
│  Build mapping: claim_name -> list of running Pods referencing it.     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Step 3: Base-Model Affinity Check (LoRA)                │
│  Does any candidate claim have a running Pod where:                    │
│    `pod.labels["open-rl.io/bound-base-model"] == "qwen-qwen3-8b"`?     │
│  => If YES: REUSE THIS CLAIM & POD. Return immediately.                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Step 4: Mutual Exclusion Filter (LoRA)                  │
│  Filter OUT any candidate claim currently hosting a Pod where:         │
│    `pod.labels["open-rl.io/bound-base-model"] != "qwen-qwen3-8b"`.     │
│  Remaining list = IDLE / ELIGIBLE CLAIMS (0 running pods).             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Step 5: Deterministic Claim Selection                    │
│  Sort eligible idle claims alphabetically by metadata.name.            │
│  Select index 0 (`lora-l4-trainer-gpu-2`).                             │
│  Stamp new worker Pod with assigned claim and base-model labels.       │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Full Fine-Tuning (FFT) Time-Slicing Concurrency
For FFT workloads (`open-rl.io/workload-type: "full"`), each job runs in its own dedicated pod pair. In Step 3 & Step 4:
- Base-model mutual exclusion is **not required** (since time-slicing and snapshot DRAM offloading swap weights across jobs).
- Instead, the scheduler checks pod count capacity: `if len(claim_pods) < maxWorkersPerClaim` (e.g., `< 2` pods per H100 claim), the claim is eligible for time-sliced co-location.

---

## 5. Memory-Aware Routing & Auto-Promotion

To prevent CUDA OOM crashes and maximize hardware efficiency, the worker manager automatically routes jobs across GPU memory tiers based on `min_vram_gb` (supplied in recipe config or computed by the SDK):

$$\text{VRAM}_{\text{total}} \approx \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{optimizer}} + \text{VRAM}_{\text{KV-cache}}(\text{batch\_size} \times \text{max\_tokens})$$

```text
       min_vram_gb <= 24 GB         24 GB < min_vram_gb <= 80 GB        min_vram_gb > 80 GB
     ┌───────────────────────┐      ┌──────────────────────────┐     ┌───────────────────────┐
     │   LoRA Default Pool   │      │   Auto-Promote to 80GB   │     │  Multi-GPU FSDP Pool  │
     │  memory-tier: "24gb"  │      │   memory-tier: "80gb"    │     │  memory-tier: "80gb+" │
     │  (NVIDIA L4 24GB)     │      │   (NVIDIA H100 / A100)   │     │  (4x H100 Sharded)    │
     └───────────────────────┘      └──────────────────────────┘     └───────────────────────┘
```

1. **Standard LoRA (`min_vram_gb <= 24`)**: Queries `open-rl.io/memory-tier=24gb`, running on cost-effective L4 cards.
2. **High-Memory LoRA or FFT (`24 < min_vram_gb <= 80`)**: Automatically promotes the job to query `open-rl.io/memory-tier=80gb`, placing 70B models or long-context 8B runs onto H100 cards without researcher intervention.

---

## 6. Comprehensive Trade-off Analysis

### 6.1 Trade-off 1: Label-Based Discovery vs. Static ConfigMaps (`open-rl-dra-pools`)
- **Option A (Label-Based Discovery - Recommended)**:
  - *Pros*: Zero ConfigMap configuration files to maintain; real-time dynamic pool membership; adding a new claim instantly makes it available to the scheduler; 100% Kubernetes idiomatic.
  - *Cons*: Requires listing `ResourceClaim` objects from K8s API at schedule time; potential API server list-cache latency under extreme burst rates.
  - *Mitigation*: The worker manager can maintain a lightweight Redis atomic lock or short-lived in-memory cache (e.g., 5-second TTL) around claim discovery.
- **Option B (Static ConfigMap `open-rl-dra-pools`)**:
  - *Pros*: All pools and claims are explicitly enumerated in a single YAML file (`pools.yaml`); no need to list claims via K8s API.
  - *Cons*: Requires editing ConfigMaps whenever GPU nodes are added or decommissioned; risks configuration drift between the ConfigMap and actual K8s `ResourceClaim` inventory.

### 6.2 Trade-off 2: Pod Metadata as Registry vs. Mutating `ResourceClaim` Labels
- **Option A (Active Pod Labels as the Lock Registry - Recommended)**:
  - *Pros*: Fully adheres to **Least-Privilege RBAC Security** (`open-rl-sa` only requires read-only `get/list/watch` on `resourceclaims` and standard CRUD on `pods`); zero risk of a compromised container mutating cluster-scoped DRA resources; automatic lock cleanup when the Pod terminates.
  - *Cons*: Determining if a claim is busy requires listing pods and inspecting their `open-rl.io/assigned-claim` labels rather than reading a single status field on the claim.
  - *Mitigation*: In Kubernetes, pod listings filtered by label (`app=open-rl-trainer-worker`) are extremely fast and served directly from the API server's watch cache.
- **Option B (Writing `bound-base-model` Label to `ResourceClaim.metadata.labels`)**:
  - *Pros*: You can inspect the lock state directly on the `ResourceClaim` object (`kubectl get resourceclaims --show-labels`).
  - *Cons*: Requires granting `update` and `patch` RBAC permissions on `resourceclaims` to workload service accounts; requires explicit cleanup logic to strip labels when jobs complete (risk of orphaned locks if a pod crashes).

### 6.3 Trade-off 3: Explicit Memory-Tier Labels vs. Runtime `ResourceSlice` Inspection
- **Option A (Explicit Memory-Tier Labels `open-rl.io/memory-tier` - Recommended)**:
  - *Pros*: Pending (`WaitForFirstConsumer`) claims can be evaluated immediately without waiting for allocation; no need to query cluster-wide Node `ResourceSlice` objects; clear operational visibility (`24gb` vs `80gb`).
  - *Cons*: Platform engineers must apply the correct `memory-tier` label when creating `ResourceClaim` manifests.
- **Option B (Inspecting `ResourceSlice.capacity.memory` at Runtime)**:
  - *Pros*: Zero memory labels required on claims; reads ground-truth memory size directly from the NVIDIA DRA Driver's advertised hardware slice.
  - *Cons*: Cannot inspect memory size on unallocated pending claims; querying `ResourceSlice` objects across all nodes adds API complexity.

### 6.4 Trade-off 4: Alphabetical Bin-Packing vs. Spread Scheduling
- **Option A (Alphabetical / Lowest-Index First - Recommended)**:
  - *Pros*: Predictable bin-packing. Workloads always fill `claim-1` and `claim-2` first, leaving higher-indexed claims (`claim-5`, `claim-6`) completely idle and free for node down-scaling or maintenance.
  - *Cons*: Lowest-indexed GPUs experience continuous thermal and memory utilization while higher-indexed GPUs remain idle.
- **Option B (Deterministic Model Hashing / Spread)**:
  - *Pros*: Distributes distinct base models evenly across all available claims (`hash(base_model) % len(claims)`), balancing thermal load across nodes.
  - *Cons*: Reduces opportunities for cluster scale-down since workloads are scattered across all GPU nodes.

---

## 7. Implementation & Migration Roadmap

```text
Phase 1 (Immediate v0.6.6):
  ├── Add Label Discovery to `KubernetesWorkerManager` (Mode A)
  ├── Enforce LoRA Base-Model Mutual Exclusion via Pod Label queries
  └── Add read-only `resourceclaims` rule to `open-rl-sa` RBAC (`03-rbac.yaml`)

Phase 2 (Long-Term v0.7.0 / Design Doc 007):
  ├── Introduce `OpenRLWorkerPool` and `OpenRLWorker` CRDs
  ├── Migrate claim selection loop into standalone K8s CRD Controller
  └── Enable multi-GPU sharded claim pools (`count: 4` for FSDP/Megatron)
```

### 7.1 Verification Plan
1. **Unit Tests (`tests/test_dra_claim_scheduler.py`)**:
   - Verify label selector query generation (`workload-type=lora,role=trainer,memory-tier=24gb`).
   - Verify **LoRA Base-Model Affinity**: submitting a second job for `Qwen3-0.6B` reuses the claim assigned to the first job.
   - Verify **LoRA Mutual Exclusion**: submitting a job for `Qwen3-8B` skips a claim actively used by `Qwen3-0.6B` and assigns a free claim.
   - Verify **Auto-Promotion**: a job with `min_vram_gb=40` queries `memory-tier=80gb` instead of `24gb`.
2. **Kubernetes E2E Campaign (`make cluster-e2e`)**:
   - Deploy 2x L4 LoRA claims (`lora-gpu-1`, `lora-gpu-2`) and 2x H100 FFT claims (`fft-gpu-1`, `fft-gpu-2`).
   - Launch concurrent `lora-fft-gsm8k-rl-x4` benchmark across mixed base models and verify zero VRAM collisions or RBAC permission errors.

### 7.2 Future Work – Automated Worker & Job Garbage Collection
While active Pod labels (`open-rl.io/assigned-claim`) automatically unlock a claim when the Pod terminates, completed or failed FFT pods (`phase: Succeeded | Failed`) and idle LoRA worker pods remain in Kubernetes until explicitly deleted.

In a follow-up release, an automated Garbage Collection controller (or a Kubernetes `ttlSecondsAfterFinished` / idle-TTL eviction loop in `KubernetesWorkerManager`) will be introduced to:
1. Automatically delete terminal FFT worker pods after a configurable retention window (`e.g. 3600s`).
2. Evict and scale down shared LoRA worker pods that have remained idle (`0` active tenant jobs in Redis) for longer than an idle threshold (`e.g. 1800s`), freeing GPU claims for other base models.
