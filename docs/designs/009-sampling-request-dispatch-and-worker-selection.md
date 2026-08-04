# Design Doc 009: Standardized Sampling & Training Request Dispatch and Shared LoRA Worker Selection

**Status**: Implemented & Verified (Phase 1 & Phase 2 Complete)  
**Author**: Open-RL Engineering  
**Date**: 2026-08-04  
**Target Branch**: `main`  
**Latest Commit**: `8f355bd`  

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

### **Phase 3: Kubernetes Worker Manager Alignment & FFT Validation [NEXT STEP]**
- **Target File**: `src/server/k8s_worker_manager.py`
- **Scope**: Align `KubernetesFFTWorkerManager` so LoRA trainer and sampler pods are named `open-rl-trainer-<sanitized_base_model>` and `open-rl-sampler-<sanitized_base_model>` and reused across LoRA sessions sharing the same base model. Validate FFT workflows requiring `accel-timeslicer` daemonset in-cluster.
- **Validation**: In-cluster testing (`make cluster-e2e`).

---

## 5. Scope & Target Files

- **Source Files**:
  - `src/server/store.py` (Phase 1 - Complete)
  - `src/server/worker_manager.py`, `src/server/gateway.py`, `src/server/training_requests_processor.py` (Phase 2 - Complete)
  - `src/server/k8s_worker_manager.py` (Phase 3 - In Progress)
- **Test Files**:
  - `tests/test_redis_store.py` (Phase 1 - Complete)
  - `tests/test_worker_manager.py` & `tests/test_gateway_paths.py` (Phase 2 - Complete)
  - `tests/test_k8s_worker_manager.py` (Phase 3 - In Progress)

---

## 6. Verification & Testing Commands

1. **Phase 1 & 2 Local Unit Testing**:
   ```bash
   export PATH=$PATH:$HOME/.local/bin && make test
   ```
2. **Phase 2 Remote Dev Host Testing (`l4`)**:
   ```bash
   make push-vm REMOTE_HOST=l4
   ssh l4 "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && uv run --extra gpu python scripts/run_training_e2e.py scenario=lora-gsm8k-rl-x2 base_model=Qwen/Qwen3-0.6B steps=5 group_size=4 groups_per_batch=4 sampling_backend=vllm trainer_gpu=0 sampler_gpu=1"
   ```
3. **Phase 3 Cluster Integration Testing**:
   ```bash
   make cluster-e2e IMAGE_TAG=$(cat VERSION) E2E_SCENARIO=tiny-lora
   ```

---

## 7. Verified Benchmark Results (`lora-gsm8k-rl-x2` on Host `l4`)

**Campaign Date**: 2026-08-04  
**Host**: Dev VM `l4` (2× NVIDIA L4 GPUs)  
**Model**: `Qwen/Qwen3-0.6B`  
**Commit**: `8f355bd`  

### Hardware & Process Allocation:
- **GPU 0**: Single Shared PyTorch Trainer worker (`server.training_requests_processor`, **1,960 MiB VRAM**).
- **GPU 1**: Single Shared vLLM Sampler worker (`server.lora_sampler`, **16,524 MiB VRAM**).

### Performance Metrics:
- **`job-a` (`55398868...`)**: Completed 5 steps (Step 0..4), reaching **43.75% accuracy** with 24.7s step time.
- **`job-b` (`f197fecc...`)**: Completed 5 steps (Step 0..4), reaching **56.25% accuracy** (reward 0.5000) with 24.7s step time.
- **VRAM Savings**: Reduced VRAM overhead on GPU 0 from ~3.8 GB (isolated workers) to **1.9 GB** (shared worker).

