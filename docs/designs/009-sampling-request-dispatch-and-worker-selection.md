# Design Doc 009: Standardized Sampling & Training Request Dispatch and Shared LoRA Worker Selection

**Status**: Implemented & Verified (Phases 1-5 Complete - All Phases Verified on Kubernetes)  
**Author**: Open-RL Engineering  
**Date**: 2026-08-04  
**Target Branch**: `main`  
**Latest Version**: `0.6.5`  

---

## 1. Executive Summary

This design document outlines key architectural improvements to sampling and training request dispatch, queue contracts, worker selection, and resource sharing across `src/server/gateway.py`, `src/server/worker_manager.py`, `src/server/training_requests_processor.py`, `src/server/k8s_worker_manager.py`, and `src/server/store.py`.

Previously, sampling and training request handling exhibited four primary operational issues:

1. **Local Worker Selection Mismatch**: `WorkerManager` (the local process launcher) unconditionally spawned `server.vllm_sampler` for sampling workers and spawned separate `server.training_requests_processor` subprocesses per adapter `model_id`.
2. **Unnecessary Per-Model LoRA Worker Duplication**: A new PyTorch Trainer worker and vLLM Sampler worker were launched for every individual `model_id`. For LoRA fine-tuning jobs, vLLM's `enable_lora=True` and PyTorch's in-process `PeftModel.add_adapter` allow a single base model engine to serve dynamic LoRA adapters. Launching duplicate workers for every LoRA `model_id` wasted VRAM (3.8+ GB vs 1.9 GB) and host CPU memory.
3. **Heuristic Filesystem & Environment Variable Fallbacks**: Code previously fell back to `os.getenv("BASE_MODEL")` or inspected disk for `adapter_config.json`, risking incorrect model targeting under concurrency.
4. **Queue Contract Discrepancy Across Backends**: `InMemoryStore` threw `RuntimeError("Sampling queues require REDIS_URL")` for `put_sampling_request`, causing inconsistency between Redis and in-memory test harnesses.

---

## 2. Core Architectural Principles & Goals

### 1. `model_meta` as the Strict Single Source of Truth for `base_model`
- **Zero Environment Variable Fallbacks**: `open_rl:model_meta:<model_id>` (persisted during model creation in `store`) is the canonical source of truth for `base_model` and `fine_tuning_type`.
- Components (Gateway, Worker Managers, Sampler Workers) must retrieve `meta = store.get_model_metadata(model_id)` and read `meta.base_model`.
- There will be **no fallback on environment variables** (`BASE_MODEL`, `OPEN_RL_BASE_MODEL`, `VLLM_MODEL`) when resolving a model's base model. If `base_model` is missing in `model_meta`, an explicit exception is raised rather than falling back to global environment variables.

### 2. Base-Model Shared LoRA Sampler Workers & Queues
- **LoRA Mode (`fine_tuning_type == "lora"`)**:
  - Sampling requests are queued by `base_model` (queue key: `open_rl:sampler_queue:<base_model>`).
  - A single `server.lora_sampler` worker process/pod is launched per `base_model` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).
  - Subsequent LoRA jobs (different `model_id`s) sharing the same `base_model` reuse the active `lora_sampler` worker without spawning redundant processes or pods.
  - Individual requests specify `lora_id` (the specific LoRA adapter `model_id`) and `lora_path` (the PEFT directory), which the `lora_sampler` worker attaches dynamically via `vllm.lora.request.LoRARequest`.
- **FFT Mode (`fine_tuning_type == "full"`)**:
  - Sampling requests are queued by `model_id` (queue key: `open_rl:sampler_queue:<model_id>`).
  - A dedicated `server.vllm_sampler` worker process/pod is launched per `model_id` to manage time-sliced GPU access and in-place weight reloading.

### 3. Dynamic Local Sampler Worker Launching (`src/server/worker_manager.py`)
`FFTWorkerManager.launch_sampler()` will inspect the model's `fine_tuning_type` stored in `open_rl:model_meta:<model_id>`:
- If `fine_tuning_type == "lora"`: Launches `server.lora_sampler` keying by `base_model`.
- If `fine_tuning_type == "full"`: Launches `server.vllm_sampler` keying by `model_id`.

### 4. Metadata-Driven Gateway Dispatch (`src/server/gateway.py`)
`gateway.py` (`asample` and `create_sampling_session`) will eliminate `os.path.exists(...)` disk checks and environment variable fallbacks. The gateway strictly respects `fine_tuning_type` and `base_model` from `store.get_model_metadata(model_id)`:
- If `fine_tuning_type == "lora"`: Set `queue_id = meta["base_model"]`, `weights_path = None`, `lora_id = model_id`, `lora_path = peft_dir`. Enqueue to `open_rl:sampler_queue:<queue_id>`.
- If `fine_tuning_type == "full"`: Set `queue_id = model_id`, `weights_path = resolve_sampler_weights_path(model_id)`, `lora_id = None`, `lora_path = None`. Enqueue to `open_rl:sampler_queue:<queue_id>`.

### 5. Unified Sampling Queue Contract in `RequestStore` (`src/server/store.py`)
`InMemoryStore` will implement `sampling_queues: dict[str, asyncio.Queue]` for `put_sampling_request` and `get_sampling_requests_for_model`. This ensures that all backends (both Redis-backed and in-memory single-process mode) adhere to the standard `store.put_sampling_request(...)` interface.

### 6. LoRA Active Tenant Set Indexing & 1:1 Worker Mapping (`open_rl:active_tenants_set:<base_model>-<idx>`)
- **Worker-Scoped Rotation Sets**: To prevent cross-model queue stealing and support horizontal worker scaling, LoRA active tenant rotation sets in Redis are indexed by `base_model` and replica index (`idx`), e.g., `open_rl:active_tenants_set:Qwen/Qwen3-0.6B-1`.
- **1:1 Worker Mapping**: Each LoRA worker pod (e.g., `open-rl-trainer-qwen-qwen3-0-6b-1`) maps 1:1 to its indexed active tenant rotation set (`Qwen/Qwen3-0.6B-1`). A worker only polls and round-robins tenant sessions assigned to its specific `active_tenants_set` index.
- **Gateway Tenant Assignment**: When a LoRA model is created (`create_model`), the Gateway assigns the tenant UUID to the active tenant set for that base model. Initially, there is one active set per base model (`idx=1`); as workloads scale out to multiple replicas (`idx=1, 2, ...`), the Gateway balances tenant session assignments across the indexed active sets.

### 7. Multi-Tenant LoRA Queue Draining Before Cycling (`v0.6.4`)
- **Queue Draining Contract**: In `RedisStore.get_requests()` and `InMemoryStore.get_requests()`, when a LoRA trainer worker inspects the head tenant in the active rotation list (`lindex active_list 0`), it drains all pending requests from that tenant's queue (`open_rl:queue:{model_id}`) without rotating the tenant to the tail of the active list until the queue is empty (`llen == 0`).
- **Elimination of Adapter Thrashing**: Previously, unconditionally calling `brpoplpush(active_list, active_list)` rotated a tenant on every single request. In multi-tenant RL training where a single training step consists of multiple sequential RPC calls (`forward_backward` microbatches followed by `optim_step`), this caused the GPU to swap LoRA adapters back and forth on every request. Keeping the tenant at index 0 until all its pending requests are drained ensures that a tenant's entire training turn executes consecutively on GPU without adapter thrashing.

---

## 3. Detailed Specification & Implementation Flow

### 3.1 Gateway Dispatch and Session Creation (`src/server/gateway.py`)

#### `create_sampling_session()`
```python
model_meta = await store.get_model_metadata(model_id)
if not model_meta or not model_meta.get("base_model"):
  raise ValueError(f"Model metadata or base_model missing for model_id: {model_id}")

fine_tuning_type = model_meta.get("fine_tuning_type", "lora")
queue_id = model_meta["base_model"] if fine_tuning_type == "lora" else model_id

if get_sampler_backend() == "vllm" and queue_id:
  if is_fft_enabled():
    await ensure_sampler_launched(queue_id)
  # Wait for open_rl:sampler_ready:<queue_id> signal in Redis
```

#### `asample()`
```python
model_meta = await store.get_model_metadata(model_id)
if not model_meta or not model_meta.get("base_model"):
  raise ValueError(f"Model metadata or base_model missing for model_id: {model_id}")

base_model = model_meta["base_model"]
fine_tuning_type = model_meta.get("fine_tuning_type", "lora")

if fine_tuning_type == "lora":
  queue_id = base_model
  weights_path = None
  lora_id = model_id
  peft_dir = os.path.join(TMP_DIR, "peft", model_id, model_id)
  lora_path = peft_dir if os.path.exists(peft_dir) else None
else:
  queue_id = model_id
  resolved_path = resolve_sampler_weights_path(model_id) if is_sampler_weights_ref(model_id) or is_fft_enabled() else None
  weights_path = resolved_path
  lora_id = None
  lora_path = None

sampling_req = {
  "request_id": req_id,
  "prompt_token_ids": prompt,
  "max_tokens": max_tokens,
  "temperature": temperature,
  "stop": stop,
  "top_p": top_p,
  "top_k": top_k,
  "num_samples": num_samples,
  "lora_id": lora_id,
  "lora_path": lora_path,
  "weights_path": weights_path,
  "include_prompt_logprobs": include_prompt_logprobs,
  "model_id": queue_id,  # Target queue key for store.put_sampling_request
  "trace_context": carrier,
}

await store.put_sampling_request(sampling_req)
```

---

### 3.2 Worker Manager Renaming & Local Worker Manager (`src/server/worker_manager.py`)

- **Clean Class Rename**: `FFTWorkerManager` is renamed to `LocalWorkerManager`.
- **Factory Update**: `create_worker_manager()` returns `KubernetesWorkerManager` when `OPEN_RL_WORKER_MANAGER=kubernetes` and `LocalWorkerManager` otherwise.
- **Base-Model Target Resolution**: For LoRA mode (`fine_tuning_type == "lora"`), `launch_trainer()` and `launch_sampler()` resolve `target_id = base_model` (`Qwen/Qwen3-0.6B`), reusing active Trainer (GPU 0) and Sampler (GPU 1) processes.

#### `launch_sampler(target_id: str)`:
```python
meta = _fetch_metadata_from_store(model_id)
ft_type = meta.fine_tuning_type if meta else None
is_fft = os.getenv("OPEN_RL_ENABLE_FFT", "").lower() in ("true", "1")
is_lora = (ft_type == "lora") if ft_type is not None else (not is_fft)

base_model = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL")
target_id = (base_model or model_id) if is_lora else model_id

proc = self.sampler_processes.get(target_id)
if proc is not None and proc.poll() is None:
  return  # Reuse existing running sampler worker!

sampler_module = "server.lora_sampler" if is_lora else "server.vllm_sampler"

env = {**os.environ, "OPEN_RL_ENABLE_FFT": "false" if is_lora else "true"}
if base_model:
  env["BASE_MODEL"] = base_model

self.sampler_processes[target_id] = subprocess.Popen(
  _py_cmd(["gpu", "vllm"], sampler_module, target_id),
  cwd=self.project_dir,
  env=env,
  start_new_session=True,
)
```

---

### 3.3 Kubernetes Worker Manager (`src/server/k8s_worker_manager.py`)

- **Clean Class Rename**: `KubernetesFFTWorkerManager` is renamed to `KubernetesWorkerManager`.
- **Pod Naming with Suffix Indexing**:
  - Pod naming format: `open-rl-trainer-<sanitized_target_id>-<instance_id>` and `open-rl-sampler-<sanitized_target_id>-<instance_id>`.
  - Default `instance_id = 1`.
  - Default tenant capacity per worker instance = **unlimited** (all tenant jobs for the same base model share instance `1`).
- **Modular Pod YAML Rendering**:
  - `render_pod()` dispatches to dedicated rendering methods: `render_lora_pod()` for LoRA mode and `render_fft_pod()` for Full Fine-Tuning mode.
  - Eliminates nested conditionals and cleanly isolates LoRA standalone YAML construction from FFT time-sliced YAML construction.
- **LoRA Mode Execution (`render_lora_pod`)**:
  - Target ID: `target_id = base_model` (`Qwen/Qwen3-0.6B`).
  - Pod names: `open-rl-trainer-<sanitized_base_model>-1` and `open-rl-sampler-<sanitized_base_model>-1`.
  - Pod reuse: `read_pod()` checks for running instance `1`. If active, reuses it for subsequent adapter jobs (`job-a`, `job-b`).
  - Standalone execution: Pod sets `"accel-timeslicer": "false"`, strips timeslicer labels/env vars (`OPEN_RL_TIME_SLICE_*`), sets `OPEN_RL_ENABLE_FFT="false"`, `OPEN_RL_FINE_TUNING_TYPE="lora"`, and launches `server.lora_sampler` for inference.
  - DRA claims: Uses existing cluster DRA claims from manifest without time-slicer coordination.
- **FFT Mode Execution (`render_fft_pod`)**:
  - Target ID: `target_id = model_id`.
  - Pod names: `open-rl-trainer-<sanitized_model_id>-1` and `open-rl-sampler-<sanitized_model_id>-1`.
  - Time-slicing: Retains `"accel-timeslicer": "true"`, `timeslice.io/*` labels, `OPEN_RL_TIME_SLICE_*` env vars, and shared DRA claim.
  - Environment: Sets `OPEN_RL_ENABLE_FFT="true"`, `OPEN_RL_FINE_TUNING_TYPE="full"`, and launches `server.vllm_sampler` for inference.

---

### 3.4 Store Queue Standardization (`src/server/store.py`)

#### `InMemoryStore` Implementation:
```python
class InMemoryStore(RequestStore):
  def __init__(self):
    ...
    self.sampling_queues: dict[str, asyncio.Queue] = {}

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    if model_id not in self.sampling_queues:
      self.sampling_queues[model_id] = asyncio.Queue()
    await self.sampling_queues[model_id].put(req_data)

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    if model_id not in self.sampling_queues:
      return []
    queue = self.sampling_queues[model_id]
    if queue.empty():
      return []
    batch = [queue.get_nowait()]
    while not queue.empty():
      batch.append(queue.get_nowait())
    return batch
```

### 3.5 LoRA Worker Launch Contract & Active Set Routing (`--active-tenant-set-id`)

- **Worker Command Line Invocation**: When `KubernetesWorkerManager` launches a LoRA trainer pod (`open-rl-trainer-<sanitized_base_model>-<idx>`), it passes `--active-tenant-set-id "open_rl:active_tenants:Qwen/Qwen3-0.6B-1"` (or derives the active set key directly from the worker's assigned base model index).
- **Tenant Assignment on Gateway (`create_model`)**:
  - The Gateway maps each newly created LoRA tenant (`model_id`) to the active tenant rotation set for its `base_model` index:
    ```python
    active_set_id = f"{base_model}-1"  # Initial deployment: 1 active set per base model
    await redis.sadd(f"open_rl:active_tenants_set:{active_set_id}", model_id)
    await redis.rpush(f"open_rl:active_tenants:{active_set_id}", model_id)
    ```
- **Isolated Round-Robin Polling**:
  - The LoRA worker pod calls `get_requests(active_set_id=self.active_tenant_set_id)`.
  - It rotates via `BRPOPLPUSH` strictly over its assigned `active_set_id`.
  - This prevents any LoRA worker from observing, popping, or interfering with requests queued for other base models or other replica indices.

---

## 4. Streamlined 3-Phase Execution Strategy

Merging Gateway dispatch changes with Local Worker Manager updates ensures that queue keys (`open_rl:sampler_queue:<queue_id>`) match between queue producers (Gateway) and queue consumers (Sampler Workers) in a single coherent step.

Furthermore, local/dev host (`l4`) testing focuses on LoRA fine-tuning workflows, while Full Fine-Tuning (FFT) validation (which relies on `accel-timeslicer` daemonset and snapshot synchronization) is deferred to the final Kubernetes phase.

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Store Queue Standardization                  │
│ File: src/server/store.py                              │
│ Goal: In-memory sampling queue support                 │
│ Test: Local unit tests (`make test`)                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Local Sampler Selection, Shared LoRA Worker   │
│ Launching & Gateway Metadata Dispatch                  │
│ Files: src/server/worker_manager.py                   │
│        src/server/gateway.py                           │
│ Goal:  1. Worker manager launches lora_sampler         │
│           keyed by base_model for LoRA jobs.           │
│        2. Gateway routes requests to base_model        │
│           queue using model_meta, removing disk/env    │
│           checks.                                      │
│ Test: Local unit tests + LoRA dev host `l4` testing    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Kubernetes Worker Manager & FFT Alignment     │
│ File: src/server/k8s_worker_manager.py                 │
│ Goal: Shared LoRA pod management & in-cluster FFT with │
│       accel-timeslicer daemonset                       │
│ Test: `make cluster-e2e`                               │
└──────────────────────────┘
```

### **Phase 1: Store Queue Standardization [COMPLETED & VERIFIED]**
- **Target File**: `src/server/store.py`
- **Status**: **Completed & Verified** (unit tests passing).
- **Validation**: Added `sampling_queues` in `InMemoryStore` to support `put_sampling_request` and `get_sampling_requests_for_model` without requiring Redis.

### **Phase 2: Local Sampler & Trainer Selection, Shared Base-Model Workers & Gateway Metadata Dispatch [COMPLETED & VERIFIED]**
- **Target Files**: `src/server/worker_manager.py`, `src/server/gateway.py`, `src/server/training_requests_processor.py`
- **Status**: **Completed & Verified** on remote host `l4`.
- **Implementation**:
  - `WorkerManager.launch_sampler()` & `launch_trainer()` resolve `target_id = base_model` for LoRA mode (`Qwen/Qwen3-0.6B`).
  - Launches **1 PyTorch Trainer worker** on GPU 0 and **1 vLLM Sampler worker** on GPU 1 per base model.
  - Subsequent LoRA jobs (`job-a`, `job-b`) reuse the active workers without spawning redundant processes.
  - `LoraTrainingRequestsProcessor.run_once()` drains all queued requests for each adapter in sequence ("in one go").
  - Adapter weights are auto-saved to disk after every `optim_step`, with pointer updates (`sampling_session_id`) on `create_sampling_client` / `save_weights_for_sampler`.
- **Validation**:
  - Unit tests verified (97/97 passing).
  - Executed dual-job RL training benchmark `lora-gsm8k-rl-x2` (`Qwen/Qwen3-0.6B`, 5 steps) on remote GPU host `l4`.

### **Phase 3: Kubernetes Worker Manager Alignment & Per-Base-Model Active Tenant Rotation Sets [COMPLETED & VERIFIED]**
- **Target Files**: `src/server/k8s_worker_manager.py`, `src/server/gateway.py`, `src/server/store.py`
- **Status**: **Completed & Verified** on Kubernetes (`v0.6.3`).
- **Implementation**:
  - Aligned `KubernetesFFTWorkerManager` so LoRA trainer and sampler pods are named `open-rl-trainer-<sanitized_base_model>-<idx>` and `open-rl-sampler-<sanitized_base_model>-<idx>` and reused across LoRA sessions sharing the same base model.
  - Scoped active tenant rotation lists (`open_rl:active_tenants:<base_model>-<idx>`) and sets (`open_rl:active_tenants_set:<base_model>-<idx>`) by base model replica index (`idx=1`).
  - Verified 10-step single-tenant LoRA RL GSM8K benchmark (`lora-gsm8k-rl`) on Kubernetes.

### **Phase 4: Multi-Tenant LoRA Queue Draining & Full Fine-Tuning (FFT) Validation on Kubernetes [COMPLETED & VERIFIED]**
- **Target Files**: `src/server/store.py`, `k8s/deploy/distributed-fft-timeslice/04-gateway.yaml`
- **Status**: **Completed & Verified** on Kubernetes (`v0.6.4`).
- **Implementation**:
  - Updated `RedisStore.get_requests()` and `InMemoryStore.get_requests()` so that when a LoRA trainer worker inspects the head tenant in the active rotation list (`lindex active_list 0`), it drains all pending requests from that tenant's queue without rotating the tenant to the tail of the active list until the queue is empty (`llen == 0`).
  - Configured `OPEN_RL_ENABLE_FFT: "true"` on Gateway (`04-gateway.yaml`).
- **Validation**:
  - Executed 10-step concurrent dual LoRA RL GSM8K benchmark (`lora-gsm8k-rl-x2`) on `Qwen/Qwen3-0.6B`. Both `job-a` and `job-b` completed cleanly with zero adapter thrashing and ~21.9s average step time.
  - Executed 10-step Full Fine-Tuning RL GSM8K benchmark (`fft-gsm8k-rl`) on `Qwen/Qwen3-0.6B`. Accuracy improved from **18.75%** at Step 0 to **87.50%** at Step 8 (**81.25%** at Step 9) with ~17.7s average step time.

### **Phase 5: Concurrent L4 LoRA & H100 FFT Scheduling via Dedicated DRA ResourceClaims [COMPLETED & VERIFIED]**
- **Target Files**: `src/server/k8s_worker_manager.py`, `k8s/deploy/distributed-fft-timeslice/06b-lora-gpu-resourceclaim.yaml`, `08b-lora-sampler-resourceclaim.yaml`
- **Status**: **Completed & Verified** on Kubernetes (`v0.6.5`).
- **Implementation**:
  - Created dedicated DRA `ResourceClaims` for LoRA workloads (`open-rl-lora-trainer-gpu-1` & `open-rl-lora-sampler-gpu-1`), preventing DRA claim conflicts with H100 FFT pods (`open-rl-trainer-gpu-1` & `open-rl-sampler-gpu-1`).
  - Updated `KubernetesWorkerManager.render_lora_pod()` and `render_fft_pod()` to dynamically inject `nodeSelector.cloud.google.com/gke-accelerator: "nvidia-l4"` (with LoRA claims) for LoRA pods and `"nvidia-h100-80gb"` (with FFT claims) for FFT pods.
  - Enables running 4 concurrent RL jobs (2 LoRA on L4 + 2 FFT on H100) simultaneously on a GKE cluster.
- **Validation**:
  - Executed 10-step 4x concurrent heterogeneous RL benchmark (`lora-fft-gsm8k-rl-x4`) on `Qwen/Qwen3-0.6B` (`group_size=4, groups_per_batch=4`).
  - Both L4 LoRA jobs (`lora-a` & `lora-b`) completed all 10 steps sharing a single L4 trainer/sampler pod pair (~35.5s/step), with peak accuracy reaching **87.50%** (`lora-a`) and **68.75%** (`lora-b`).
  - Both H100 FFT jobs (`fft-a` & `fft-b`) completed all 10 steps on dedicated H100 trainer/sampler pods (~24.2s/step), reaching **62.50%** (`fft-a`) and **68.75%** (`fft-b`) peak accuracy.

---

## 5. Scope & Target Files

- **Source Files**:
  - `src/server/store.py` (Phase 1 & Phase 4 - Complete)
  - `src/server/worker_manager.py`, `src/server/gateway.py`, `src/server/training_requests_processor.py` (Phase 2 - Complete)
  - `src/server/k8s_worker_manager.py` (Phase 3 & Phase 5 - Complete)
- **Kubernetes Manifests**:
  - `k8s/deploy/distributed-fft-timeslice/04-gateway.yaml`, `05-worker-pod-template.yaml`, `09-sampler-pod-template.yaml` (Phase 3 & Phase 4 - Complete)
  - `k8s/deploy/distributed-fft-timeslice/06b-lora-gpu-resourceclaim.yaml`, `08b-lora-sampler-resourceclaim.yaml` (Phase 5 - Complete)
- **Test Files**:
  - `tests/test_redis_store.py` (Phase 1 - Complete)
  - `tests/test_worker_manager.py` & `tests/test_gateway_paths.py` (Phase 2 - Complete)
  - `tests/test_k8s_worker_manager.py` (Phase 3 & Phase 5 - Complete)

---

## 6. Verification & Testing Commands

1. **Local Unit & Linter/Formatter Testing**:
   ```bash
   export PATH=$PATH:$HOME/.local/bin && make fmt && make lint && make test
   ```
2. **Kubernetes Cluster E2E Benchmarking (`make cluster-e2e`)**:
   ```bash
   # Multi-Tenant LoRA RL GSM8K x2 (Concurrent Dual Jobs)
   make cluster-e2e IMAGE_TAG=0.6.4 E2E_SCENARIO=lora-gsm8k-rl-x2 E2E_ARGS="base_model=Qwen/Qwen3-0.6B steps=10 group_size=4 groups_per_batch=4 max_tokens=512"

   # Full Fine-Tuning RL GSM8K (Single-Tenant FFT)
   make cluster-e2e IMAGE_TAG=0.6.4 E2E_SCENARIO=fft-gsm8k-rl E2E_ARGS="base_model=Qwen/Qwen3-0.6B steps=10 group_size=4 groups_per_batch=4 max_tokens=512"
   ```

---

## 7. Verified Benchmark Results on Kubernetes Cluster (v0.6.3 & v0.6.4)

**Campaign Date**: 2026-08-04  
**Target Environment**: Kubernetes GPU Cluster (`distributed-fft-timeslice`)  
**Base Model**: `Qwen/Qwen3-0.6B`  
**Release Versions**: `v0.6.3` (Phase 3) & `v0.6.4` (Phase 4)  

### 1. Single-Tenant LoRA RL GSM8K (`lora-gsm8k-rl`, v0.6.3)
- **Worker Allocation**: Mapped 1:1 to active tenant set `Qwen/Qwen3-0.6B-1` (`open-rl-trainer-qwen-qwen3-0-6b-1` & `open-rl-sampler-qwen-qwen3-0-6b-1`).
- **Accuracy Progression**: **6.25%** (Step 0) → **75.00%** (Peak, Step 4) → **56.25%** (Step 9).
- **Stability**: Zero tensor shape mismatches (`3072 vs 4864` bug resolved via active tenant set indexing).

### 2. Concurrent Dual LoRA RL GSM8K (`lora-gsm8k-rl-x2`, v0.6.4)
- **Multi-Tenant Queue Draining**: Both `job-a` (`becd02b8-...`) and `job-b` (`1ba00139-...`) shared active set `Qwen/Qwen3-0.6B-1`. Draining each tenant's queue before cycling eliminated GPU adapter thrashing.
- **Accuracy Progression**:
  - **`job-a`**: **25.00%** (Step 0) → **75.00%** (Peak, Step 4) → **62.50%** (Step 9). Average step time: **~21.9s**.
  - **`job-b`**: **25.00%** (Step 0) → **68.75%** (Step 8) → **75.00%** (Step 9). Average step time: **~22.0s**.

### 3. Full Fine-Tuning RL GSM8K (`fft-gsm8k-rl`, v0.6.4)
- **Worker Allocation**: Dedicated FFT trainer (`open-rl-trainer-26e549d0-...`) and sampler (`open-rl-sampler-26e549d0-...`) worker pods.
- **Accuracy Progression**: **18.75%** (Step 0) → **87.50%** (Peak, Step 8) → **81.25%** (Step 9).
- **Step Timing & Delta Sync**: Average step time: **~17.7s total** (`train_step`: ~7.8s, `save_checkpoint` delta sync: ~0.9s, `sampling`: ~8.8s). Sparse delta in-place synchronization completed in under **~0.7s** per step.


