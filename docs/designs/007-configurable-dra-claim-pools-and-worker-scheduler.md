# Design Doc 007: Configurable DRA Claim Pools & Worker Placement Scheduler

**Author:** Open-RL Engineering Team  
**Status:** Proposed Design (`v1.0.0`)  
**Target Component:** Multi-Tenant Gateway, `KubernetesFFTWorkerManager` (`k8s_worker_manager.py`), K8s CRD Controller (`OpenRLWorkerPool`, `OpenRLWorker`)  
**Target Manifests:** `k8s/deploy/distributed-fft-timeslice/`, `k8s/crd/`  

---

## 1. Executive Summary

As Open-RL scales to multi-tenant Reinforcement Learning (RL) and Full Fine-Tuning (FFT) workloads across heterogeneous Kubernetes GPU clusters, hardcoding Dynamic Resource Allocation (DRA) claims creates severe operational bottlenecks.

Currently, worker pod templates hardcode a single physical GPU claim (`resourceClaimName: open-rl-trainer-gpu-1`). This limits cluster utilization, prevents co-locating multiple workloads across distinct GPU nodes, and precludes user-driven placement of trainer and sampler workers.

This design document specifies a **Configurable DRA Claim Pool and Worker Placement Scheduler** architecture for Open-RL.

### Key Highlights
1. **Configurable Claim Pools**: Abstracting physical GPU allocations into `OpenRLWorkerPool` resources that support both **existing pre-created K8s `ResourceClaim` objects** and **template-generated claims** derived from K8s native `ResourceClaimTemplate` specs.
2. **Worker Capacity Capping**: Capping the maximum number of co-located worker pods (e.g., max 2 trainer pods or 4 sampler pods) that can be packed onto a single physical DRA claim lock.
3. **Label-Based Placement Hints**: Matching workloads to appropriate claim pools via label selectors and placement hints (`poolRef`, `claimSelector`, or explicit `claimRef`).
4. **Declarative Kubernetes Controller Architecture**: Evolving `k8s_worker_manager` into a declarative K8s Custom Resource Definition (CRD) controller (`OpenRLWorkerPool` & `OpenRLWorker`), providing race-free optimistic concurrency scheduling, native `kubectl` observability, and automatic reconciliation.

---

## 2. Motivation & Problem Statement

### 2.1 Limitations of Current Hardcoded Claims
1. **Single-GPU Pinning**: `k8s_worker_manager.py` stamps worker pods using a static ConfigMap template ([05-worker-pod-template.yaml](file:///usr/local/google/home/sunilarora/open-rl/k8s/deploy/distributed-fft-timeslice/05-worker-pod-template.yaml)) referencing `open-rl-trainer-gpu-1`. All trainer workers end up pinned to the exact same physical GPU regardless of cluster size.
2. **No Capacity Capping**: There is no mechanism to limit how many trainer/sampler worker pods can be scheduled onto a single DRA claim, leading to memory pressure or unexpected OOM crashes when too many models attempt to co-locate.
3. **Lack of User Placement Control**: End users cannot direct specific workloads (e.g., a large 8B model vs a small 0.5B model) to appropriate hardware claims or dedicated node pools.

### 2.2 Core Objectives
- **Decouple Hardware from Templates**: Move away from hardcoded claim names in pod templates.
- **Support Existing & Managed Claims**: Allow cluster operators to either register existing pre-created K8s `ResourceClaim`s or let Open-RL auto-provision claims from K8s `ResourceClaimTemplate` specs.
- **Strict Worker Packing & Capping**: Enforce configurable worker limits per claim with `binpack` or `spread` allocation policies.
- **Declarative Scheduling**: Provide CRD-based scheduling (`OpenRLWorkerPool` and `OpenRLWorker`) to prevent scheduling race conditions across replicated Gateway instances.

---

## 3. High-Level Architecture

```text
                                  ┌───────────────────────────────────────────────┐
                                  │              Open-RL Gateway                  │
                                  │           (Job API / Metadata)                │
                                  └──────────────────────┬────────────────────────┘
                                                         │
                                        1. Submit Job with Placement Hints
                                                         │
                                                         ▼
                                  ┌───────────────────────────────────────────────┐
                                  │          OpenRLWorker CRD / Controller        │
                                  └──────────────────────┬────────────────────────┘
                                                         │
                                        2. Match Claim Pool & Capacity Cap
                                                         │
                                                         ▼
                                  ┌───────────────────────────────────────────────┐
                                  │               OpenRLWorkerPool                │
                                  │      (Capacity: 2 Workers / Claim Cap)       │
                                  └───────┬───────────────────────────────┬───────┘
                                          │                               │
                      Option A: Match     │ Available                     │ Full (2/2)
                      Existing Claims     ▼                               ▼
                       ┌────────────────────────┐      ┌────────────────────────┐
                       │  Claim: trainer-gpu-1  │      │  Claim: trainer-gpu-2  │
                       │  Active: 1/2 workers   │      │  Active: 2/2 workers   │
                       │  => [ASSIGN & LAUNCH]  │      │  => [SKIP / FULL]      │
                       └────────────────────────┘      └────────────────────────┘
                                          │
                      Option B: Provision │ Auto-Create
                      from Template       ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ K8s Native ResourceClaimTemplate (e.g. h100-template)   │
                       └────────────────────────────────────────────────────────┘
```

---

## 4. Custom Resource Definitions (CRDs) Specification

### 4.1 `OpenRLWorkerPool` CRD (Cluster Admin Resource)

The `OpenRLWorkerPool` custom resource defines a logical pool of physical GPU allocations, its target worker role, capacity caps, and placement policies.

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorkerPool
metadata:
  name: h100-trainer-pool
  namespace: default
spec:
  # Target worker role: "trainer", "sampler", or "shared"
  role: trainer

  # Maximum number of co-located worker pods packed onto a single DRA claim
  maxWorkersPerClaim: 2

  # Claim Selection & Packing Strategy: "binpack" (fill first) or "spread" (load balance)
  selectionPolicy: binpack

  # Behavior when all claims in pool hit max capacity: "queue" or "fail_fast"
  overflowPolicy: queue

  # -------------------------------------------------------------------------
  # CLAIM SOURCE SCHEME (Polymorphic: choose Mode A, B, C, or D)
  # -------------------------------------------------------------------------

  # Mode A: Discover existing K8s ResourceClaims via Label Selector
  claimSelector:
    matchLabels:
      open-rl.io/pool: h100-trainer-pool
      cloud.google.com/gke-accelerator: nvidia-h100-80gb

  # Mode B: Explicit list of pre-existing ResourceClaim names
  # staticClaims:
  #   - open-rl-trainer-gpu-1
  #   - open-rl-trainer-gpu-2

  # Mode C: Auto-manage claim lifecycles via K8s native ResourceClaimTemplate
  # templateRef:
  #   name: h100-80gb-claim-template
  # claimCount: 4

  # Mode D: Passthrough inline native K8s ResourceClaimSpec
  # claimSpec:
  #   devices:
  #     requests:
  #     - name: gpu
  #       count: 1
  #       exactly:
  #         deviceClassName: gpu.nvidia.com

status:
  totalClaims: 4
  activeWorkers: 3
  availableCapacity: 5
  claims:
    - name: open-rl-trainer-gpu-1
      activeWorkers: 2
      status: FULL
    - name: open-rl-trainer-gpu-2
      activeWorkers: 1
      status: AVAILABLE
```

#### Field Definitions:
- `role` *(string, required)*: `"trainer"`, `"sampler"`, or `"shared"`.
- `maxWorkersPerClaim` *(int, required)*: Maximum worker pods assigned to one claim lock simultaneously.
- `selectionPolicy` *(string, optional, default: `"binpack"`)*:
  - `"binpack"`: Pack active workers onto already-assigned claims before utilizing empty claims (maximizes consolidation).
  - `"spread"`: Distribute workers evenly across available claims to balance thermal and memory headroom.
- `overflowPolicy` *(string, optional, default: `"queue"`)*:
  - `"queue"`: Defer launching worker pod until an existing worker terminates.
  - `"fail_fast"`: Immediately reject worker launch with an `ErrPoolExhausted` status.
- `claimSelector` *(LabelSelector, optional)*: K8s label selector matching pre-created `ResourceClaim` objects.
- `staticClaims` *(list[string], optional)*: Array of explicit K8s `ResourceClaim` names.
- `templateRef` *(LocalObjectReference, optional)*: Reference to a K8s native `ResourceClaimTemplate`.
- `claimCount` *(int, optional)*: Target number of managed claims to instantiate when using `templateRef` or `claimSpec`.

---

### 4.2 `OpenRLWorker` CRD (Workload Request Resource)

When Open-RL registers a model or launches a worker, an `OpenRLWorker` resource is created representing the desired worker instance and its placement hints.

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata:
  name: open-rl-trainer-qwen8b-job-01
  namespace: default
spec:
  modelId: "qwen8b-job-01"
  role: "trainer"
  
  # Workload Placement Hints (Choose poolRef, claimSelector, or direct claimRef)
  poolRef: "h100-trainer-pool"

  # Optional direct override targeting an explicit existing claim
  # claimRef: "open-rl-trainer-gpu-1"

  # Container overrides (image, args, env)
  image: "gcr.io/cdrollouts-sunilarora/open-rl-server:0.3.74"

status:
  phase: "Scheduled" # Pending | Scheduled | Running | Failed | Terminated
  assignedClaim: "open-rl-trainer-gpu-1"
  podName: "open-rl-trainer-qwen8b-job-01"
  assignedNode: "gke-gpu-node-01"
  message: "Successfully bound to claim open-rl-trainer-gpu-1 (Active: 2/2)"
```

---

## 5. Claim Scheduling & Capacity Packing Algorithm

When scheduling an `OpenRLWorker`, the Controller / Scheduler executes the following selection pipeline:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Placement Resolution                         │
│  Check if worker has explicit `claimRef`. If set, use directly.       │
│  Otherwise, resolve target pool via `poolRef` or default role pool.   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. Candidate Discovery                          │
│  Fetch all ResourceClaims associated with the resolved pool.           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        3. Capacity Evaluation                          │
│  Count active non-terminal pods bound to each claim:                   │
│  `active_count = count(pods referencing claimName where phase != Term)`│
│  Filter out claims where `active_count >= pool.maxWorkersPerClaim`.    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        4. Selection Policy                             │
│  If `selectionPolicy == binpack`: Sort claims by descending active     │
│    count (fill partially loaded claims first).                         │
│  If `selectionPolicy == spread`: Sort claims by ascending active count  │
│    (prefer empty/least-loaded claims).                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    5. Binding & Pod Manifest Stamping                  │
│  Select top candidate claim.                                           │
│  Stamp `spec.resourceClaims[0].resourceClaimName = selected_claim`    │
│  Create worker Pod manifest in Kubernetes API.                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Strategy & Migration Plan

To ensure backwards compatibility while transitioning away from hardcoded claims, implementation is divided into two phases:

### Phase 1: In-Process Claim Pool Scheduler & ConfigMap Registry (Stepping Stone)
Before deploying full K8s CRD Controllers, upgrade `k8s_worker_manager.py` with an in-process pool scheduler:
1. **ConfigMap Registry**: Support `open-rl-dra-pools` ConfigMap defining static pools and existing claim selectors.
2. **Dynamic Claim Injection in `k8s_worker_manager.py`**:
   - Remove hardcoded `resourceClaimName` from [05-worker-pod-template.yaml](file:///usr/local/google/home/sunilarora/open-rl/k8s/deploy/distributed-fft-timeslice/05-worker-pod-template.yaml).
   - In `KubernetesFFTWorkerManager.render_pod()`, dynamically resolve candidate claims, count active worker pods via K8s CoreV1Api, enforce `maxWorkersPerClaim`, and inject the selected claim into `pod["spec"]["resourceClaims"][0]["resourceClaimName"]`.

### Phase 2: Full Kubernetes CRD Controller Architecture
1. **Define CRD Manifests**: Add `OpenRLWorkerPool` and `OpenRLWorker` OpenAPI v3 schemas under `k8s/crd/`.
2. **Build Controller Loop**: Implement controller reconciliation loop (using Go `controller-runtime` or Python `kopf`) running alongside the Gateway.
3. **Stateless Gateway Integration**: Update Gateway worker launch routes to create/watch `OpenRLWorker` custom resources instead of calling `CoreV1Api` directly.

---

## 7. Verification & Testability

1. **Unit Tests (`tests/test_dra_claim_scheduler.py`)**:
   - Test pool discovery with mock K8s CoreV1Api.
   - Test `maxWorkersPerClaim` capacity capping and overflow behavior (`queue` vs `fail_fast`).
   - Test `binpack` vs `spread` claim selection ordering.
2. **Cluster Integration Validation**:
   - Deploy `k8s/deploy/distributed-fft-timeslice/` with 2 pre-created DRA claims and `maxWorkersPerClaim: 2`.
   - Launch 3 concurrent FFT jobs and verify via `kubectl get pods` that Job 1 & 2 share Claim 1, and Job 3 is packed onto Claim 2.
