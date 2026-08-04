# Design Doc 009: Standardized Sampling Request Dispatch and Shared LoRA Worker Selection

**Status**: Proposed  
**Author**: Open-RL Engineering  
**Date**: 2026-08-04  
**Target Branch**: `main`  

---

## 1. Executive Summary

This design document outlines key architectural improvements to sampling request dispatch, queue contracts, sampler worker launching, and resource sharing across `src/server/gateway.py`, `src/server/worker_manager.py`, `src/server/k8s_worker_manager.py`, and `src/server/store.py`.

Currently, sampling request handling exhibits four primary operational issues:

1. **Local Sampler Selection Mismatch**: `FFTWorkerManager` (the local process launcher) unconditionally spawns `server.vllm_sampler` when launching a sampler worker for any model. `vllm_sampler` is designed specifically for Full Fine-Tuning (FFT) with time-sliced sleep/wake cycles and in-place weight reloading. Consequently, running LoRA fine-tuning jobs locally fails because the local worker manager does not launch `server.lora_sampler` (unlike `KubernetesFFTWorkerManager`, which inspects `meta.fine_tuning_type`).
2. **Unnecessary Per-Model LoRA Worker Duplication**: Currently, a new sampler worker is launched for every individual `model_id`. For LoRA fine-tuning jobs, vLLM's `enable_lora=True` feature allows a single base model engine to serve dynamic LoRA adapters via `LoRARequest`. Launching duplicate sampler workers for every LoRA `model_id` wastes VRAM and CPU host RAM.
3. **Heuristic Filesystem & Environment Variable Fallbacks**: In `src/server/gateway.py` (`asample`) and sampler workers, code currently falls back to `os.getenv("BASE_MODEL")` or inspects disk for `adapter_config.json`. Fallbacks to environment variables risk using the wrong base model if environment defaults change or if multiple models run concurrently.
4. **Queue Contract Discrepancy Across Backends**: `InMemoryStore` currently throws `RuntimeError("Sampling queues require REDIS_URL")` for `put_sampling_request` and `get_sampling_requests_for_model`, forcing `torch` single-process sampling mode to route through `enqueue()` (`open_rl:queue:<model_id>`) with a different payload schema than the `vllm` sampling queue (`open_rl:sampler_queue:<model_id>`).

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

### 3.2 Local Worker Manager (`src/server/worker_manager.py`)

#### `launch_sampler(target_id: str)`:
```python
proc = self.sampler_processes.get(target_id)
if proc is not None and proc.poll() is None:
  return  # Reuse existing running sampler worker!

meta = _fetch_metadata_from_store(target_id)
if meta and meta.base_model:
  base_model = meta.base_model
else:
  base_model = target_id

ft_type = meta.fine_tuning_type if meta else None
is_lora = (ft_type == "lora") if ft_type is not None else not is_fft_enabled()

sampler_module = "server.lora_sampler" if is_lora else "server.vllm_sampler"

env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
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

For Kubernetes, pods are named `open-rl-sampler-<sanitized_target_id>`. When `launch_sampler(target_id)` is called:
- `meta = _fetch_metadata_from_store(target_id)` strictly supplies `meta.base_model` to container environment variables.
- `read_pod("open-rl-sampler-" + sanitized_target_id)` checks if a pod for `target_id` (the `base_model` in LoRA mode or `model_id` in FFT mode) is active. If running, it skips pod creation and reuses the existing pod.

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

### **Phase 1: Store Queue Standardization**
- **Target File**: `src/server/store.py`
- **Scope**: Implement `sampling_queues` in `InMemoryStore`.
- **Validation**: Local unit tests (`make test`). Ensures non-Redis / local dev mode can process sampling queues cleanly.

### **Phase 2: Merged Local Sampler Selection & Gateway Metadata Dispatch (LoRA Focus)**
- **Target Files**: `src/server/worker_manager.py` & `src/server/gateway.py`
- **Scope**:
  - Update `FFTWorkerManager.launch_sampler()` to inspect `meta.fine_tuning_type`.
  - For LoRA: set `target_id = meta.base_model`, launch `server.lora_sampler` if not already running.
  - For FFT: set `target_id = model_id`, launch `server.vllm_sampler` if not already running.
  - Update `asample()` and `create_sampling_session()` in `gateway.py` to use `model_meta["base_model"]` as `queue_id` for LoRA mode.
  - Remove `os.path.exists(...)` checks for `adapter_config.json` and zero fallback to environment variables.
- **Validation**:
  - Unit tests in `tests/test_worker_manager.py` and `tests/test_gateway_paths.py`.
  - Push code to GPU dev host `l4` (`make push-vm REMOTE_HOST=l4`) and verify local process spawning and end-to-end LoRA recipe execution (`tiny_rl` or `lora-textsql`).

### **Phase 3: Kubernetes Worker Manager Alignment & FFT Validation**
- **Target File**: `src/server/k8s_worker_manager.py`
- **Scope**: Align `KubernetesFFTWorkerManager.launch_sampler()` so LoRA sampler pods are named `open-rl-sampler-<sanitized_base_model>` and reused across LoRA sessions sharing the same base model. Validate FFT workflows requiring `accel-timeslicer` daemonset in-cluster.
- **Validation**: In-cluster testing (`make cluster-e2e`).

---

## 5. Scope & Target Files

- **Source Files**:
  - `src/server/store.py` (Phase 1)
  - `src/server/worker_manager.py` & `src/server/gateway.py` (Phase 2 - Merged LoRA Dev Focus)
  - `src/server/k8s_worker_manager.py` (Phase 3 - Kubernetes & FFT)
- **Test Files**:
  - `tests/test_redis_store.py` (Phase 1)
  - `tests/test_worker_manager.py` & `tests/test_gateway_paths.py` (Phase 2)
  - `tests/test_k8s_worker_manager.py` (Phase 3)

---

## 6. Verification & Testing Commands

1. **Phase 1 & 2 Local Unit Testing**:
   ```bash
   export PATH=$PATH:$HOME/.local/bin && make test
   ```
2. **Phase 2 Remote Dev Host Testing (`l4`)**:
   ```bash
   make push-vm REMOTE_HOST=l4
   ssh l4 "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && make test"
   ```
3. **Phase 3 Cluster Integration Testing**:
   ```bash
   make cluster-e2e IMAGE_TAG=$(cat VERSION) E2E_SCENARIO=tiny-lora
   ```
