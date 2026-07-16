# Design Doc 002: Gateway Model Configuration & Metadata Refactor

**Status**: Proposed (Updated with Testing Strategy)  
**Author**: Open-RL Engineering  
**Date**: 2026-07-16  
**Target Branch**: `main`  

---

## 1. Executive Summary

This design document specifies the refactoring of model configuration retrieval, metadata persistence, and worker launching across `src/server/gateway.py`, `src/server/worker_manager.py`, and `src/server/k8s_worker_manager.py`.

Currently, the model creation and configuration flow suffers from four primary issues:
1. **Destructive Metadata Overwriting**: `create_model()` serializes a structured `TrainingModelMetadata` dataclass to Redis (`open_rl:model_meta:<id>`), but immediately upon calling `launch_worker_and_enqueue()`, that key is overwritten with an untyped dictionary, erasing fields like `created_at` and `training_kind`.
2. **Dual-Key Storage Redundancy**: Every model creation writes to both `open_rl:model_meta:<id>` and `open_rl:model_base:<id>`. Since `TrainingModelMetadata` is the standard format and contains `base_model`, `open_rl:model_base:<id>` is entirely redundant.
3. **Scattered & Header-Driven Configuration Extraction**: Because request body payloads from the SDK and clients strictly allow `base_model` today, additional training options and synchronization behaviors must be supplied cleanly via HTTP request headers. Precedence checks and default handling across `weight_sync_strategy` and `training_kind` ("full" vs "lora") are currently scattered across endpoint handlers.
4. **Leaky Parameter Passing Across Components**: Currently, `base_model` and `weight_sync_strategy` are repeatedly extracted from payloads or Redis in `launch_worker_and_enqueue()` and `ensure_sampler_launched()`, only to be passed as explicit arguments down to `WorkerManager.launch_trainer(model_id, base_model, weight_sync_strategy)` and `WorkerManager.launch_sampler(model_id, base_model, weight_sync_strategy)`.

### Core Architectural Principle: `model_id` as the Single Key for Metadata Lookup
To eliminate leaky parameter passing, this refactor establishes **`open_rl:model_meta:<model_id>` as the single, canonical source of truth for all model configuration details**.
Instead of passing `base_model`, `weight_sync_strategy`, and `training_kind` through function signatures across gateway helpers and worker managers:
1. `_extract_and_persist_model_metadata()` extracts configuration once upon creation/restoration and writes `TrainingModelMetadata` to Redis (`open_rl:model_meta:<model_id>`).
2. Component boundaries and helper signatures (`launch_worker_and_enqueue()`, `ensure_sampler_launched()`, `WorkerManager.launch_trainer()`, and `WorkerManager.launch_sampler()`) are simplified to accept solely **`model_id`**.
3. When a component (e.g., `WorkerManager` during subprocess or pod creation) requires `base_model` or `weight_sync_strategy` to configure its environment, it retrieves the `TrainingModelMetadata` directly from Redis using `model_id`.

---

## 2. Scope & Target Files

- **Primary Server Files**:
  - `src/server/gateway.py`
  - `src/server/worker_manager.py`
  - `src/server/k8s_worker_manager.py`
- **Primary Test Files**:
  - `tests/test_worker_manager.py`
  - `tests/test_k8s_worker_manager.py`
  - `tests/test_gateway_paths.py`
- **Classes & Functions Modified**:
  - `TrainingModelMetadata` (Dataclass: updated `base_model` to `str | None`)
  - `create_model()` (Endpoint: `POST /api/v1/create_model`)
  - `create_model_from_state()` (Endpoint: `POST /api/v1/create_model_from_state`)
  - `launch_worker_and_enqueue()` (Helper: signature and implementation simplified)
  - `ensure_sampler_launched()` (Helper: signature and implementation simplified)
  - `delete_model()` (Endpoint: `POST /api/v1/delete_model`)
  - `WorkerManager` Protocol & Implementations (`FFTWorkerManager`, `K8sWorkerManager`)
- **Functions Added**:
  - `_extract_and_persist_model_metadata()` (Helper in `gateway.py`)
  - `_fetch_metadata_from_store()` (Helper in `worker_manager.py` / `k8s_worker_manager.py`)

---

## 3. Detailed Architectural Specification

### Opportunity 1: Centralize Extraction & Persistence (`_extract_and_persist_model_metadata`)

We update the `TrainingModelMetadata` dataclass to allow `base_model: str | None = None`. This ensures that even when restoring a model from state where `base_model` may be omitted from the request payload, `open_rl:model_meta:<model_id>` is **always** written to Redis.

#### Updated Dataclass & New Helper Implementation (`src/server/gateway.py`)
```python
@dataclass
class TrainingModelMetadata:
  base_model: str | None
  created_at: float
  training_kind: str
  weight_sync_strategy: str | None = None


async def _extract_and_persist_model_metadata(
  model_id: str,
  req: dict[str, Any],
  request: Request | None = None,
  default_training_kind: str = "full",
) -> tuple[str | None, dict[str, Any], str]:
  """Extract and normalize model configuration from headers and payload, persisting TrainingModelMetadata exactly once."""
  base_model = req.get("base_model")
  if not base_model and default_training_kind != "restored":
    raise ValueError("base_model is required in request payload")

  full_config = dict(req.get("full_config") or {})
  user_meta = dict(req.get("user_metadata") or {})

  header_strategy = None
  training_kind = default_training_kind
  if request and hasattr(request, "headers"):
    header_strategy = request.headers.get("x-open-rl-weight-sync-strategy")
    if "x-open-rl-training-kind" in request.headers:
      training_kind = request.headers.get("x-open-rl-training-kind", default_training_kind)

  weight_sync_strategy = (
    header_strategy
    or req.get("weight_sync_strategy")
    or full_config.get("weight_sync_strategy")
    or user_meta.get("weight_sync_strategy")
  )
  if weight_sync_strategy in ("full", "delta"):
    full_config["weight_sync_strategy"] = weight_sync_strategy

  # Unconditionally persist metadata so model_meta is the canonical lookup for any model_id
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    training_kind=training_kind,
    weight_sync_strategy=weight_sync_strategy,
  )
  s = get_store()
  await s.set_value(f"open_rl:model_meta:{model_id}", json.dumps(asdict(meta_obj)))

  return base_model, full_config, training_kind
```

#### Refactored `create_model` (`src/server/gateway.py`)
```python
@app.post("/api/v1/create_model")
async def create_model(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_lora_training_client_async()"""
  model_id = str(uuid.uuid4())
  try:
    base_model, full_config, training_kind = await _extract_and_persist_model_metadata(
      model_id, req, request, default_training_kind="full"
    )
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model",
    model_id,
    {
      "base_model": base_model,
      "lora_config": req.get("lora_config") or {},
      "full_config": full_config,
      "training_kind": training_kind,
    },
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}
```

#### Refactored `create_model_from_state` (`src/server/gateway.py`)
```python
@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.get("state_path")
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "state_path is required"})

  resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  model_id = str(uuid.uuid4())
  try:
    base_model, full_config, training_kind = await _extract_and_persist_model_metadata(
      model_id, req, request, default_training_kind="restored"
    )
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model_from_state",
    model_id,
    {
      "base_model": base_model,
      "full_config": full_config,
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("restore_optimizer", False)),
      "training_kind": training_kind,
    },
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}
```

---

### Opportunity 2: Simplify Component Interfaces (`model_id`-driven Worker & Sampler Launchers)

By delegating metadata lookup to the component that actually needs it (`WorkerManager`), we remove leaky parameters (`base_model`, `weight_sync_strategy`) across `launch_worker_and_enqueue()` and `ensure_sampler_launched()`.

#### Refactored `launch_worker_and_enqueue` (`src/server/gateway.py`)
```diff
 async def launch_worker_and_enqueue(request: dict) -> str:
   """Ensure the model's dedicated trainer worker exists, then enqueue onto its queue.
 
   The launcher is idempotent per model_id, and Kubernetes (or the local process
   table) owns the worker's lifecycle from here; there is no separate launch
   queue. Launch failures resolve the future immediately so clients don't long-poll
   a request that can never be served.
   """
   assert fft_worker_manager is not None, "FFT worker manager is initialized by the app lifespan when FFT is enabled"
   request_id = request["request_id"]
-  base_model = request.get("payload", {}).get("base_model")
-  weight_sync_strategy = request.get("payload", {}).get("full_config", {}).get("weight_sync_strategy")
   await store.set_future(request_id, {"status": "pending"})
-  if base_model:
-    await store.set_value(
-      f"open_rl:model_meta:{request['model_id']}",
-      json.dumps({"base_model": base_model, "weight_sync_strategy": weight_sync_strategy}),
-    )
-    await store.set_value(f"open_rl:model_base:{request['model_id']}", base_model)
   try:
-    await asyncio.to_thread(
-      fft_worker_manager.launch_trainer,
-      request["model_id"],
-      base_model,
-      weight_sync_strategy=weight_sync_strategy,
-    )
+    await asyncio.to_thread(fft_worker_manager.launch_trainer, request["model_id"])
   except Exception as exc:
     traceback.print_exc()
     await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
     return request_id
   return await enqueue(request)
```

#### Refactored `ensure_sampler_launched` (`src/server/gateway.py`)
```diff
-async def ensure_sampler_launched(model_id: str, base_model: str | None = None) -> None:
+async def ensure_sampler_launched(model_id: str) -> None:
   if is_fft_enabled() and fft_worker_manager is not None and get_sampler_backend() == "vllm":
-    weight_sync_strategy = None
-    s = get_store()
-    val = await s.get_value(f"open_rl:model_meta:{model_id}") or await s.get_value(f"open_rl:model_base:{model_id}")
-    if val:
-      try:
-        meta = json.loads(val) if isinstance(val, str) else val
-        if isinstance(meta, dict):
-          if not base_model:
-            base_model = meta.get("base_model")
-          weight_sync_strategy = meta.get("weight_sync_strategy")
-      except Exception:
-        if not base_model and isinstance(val, str):
-          base_model = val
     try:
-      await asyncio.to_thread(
-        fft_worker_manager.launch_sampler,
-        model_id,
-        base_model,
-        weight_sync_strategy=weight_sync_strategy,
-      )
+      await asyncio.to_thread(fft_worker_manager.launch_sampler, model_id)
     except Exception:
       traceback.print_exc()
```

---

### Opportunity 3: WorkerManager Protocol & Metadata Lookup (`src/server/worker_manager.py` & `k8s_worker_manager.py`)

We update `WorkerManager.launch_trainer(model_id: str)` and `WorkerManager.launch_sampler(model_id: str)` to take only `model_id`. Inside the manager, a helper method `_fetch_metadata_from_store(model_id)` queries Redis (via `get_store()`) to retrieve `base_model` and `weight_sync_strategy` before configuring process environment variables or rendering K8s pod manifests.

#### Refactored `WorkerManager` Signatures (`src/server/worker_manager.py`)
```python
def _fetch_metadata_from_store(model_id: str) -> tuple[str | None, str | None]:
  """Retrieve base_model and weight_sync_strategy from canonical open_rl:model_meta:<model_id>."""
  from server.store import get_store
  s = get_store()
  try:
    val = s.get_value_sync(f"open_rl:model_meta:{model_id}")
    if val:
      meta = json.loads(val) if isinstance(val, str) else val
      if isinstance(meta, dict):
        return meta.get("base_model"), meta.get("weight_sync_strategy")
  except Exception:
    pass
  return None, None


class WorkerManager(Protocol):
  def launch(self, model_id: str) -> None: ...
  def launch_trainer(self, model_id: str) -> None: ...
  def launch_sampler(self, model_id: str) -> None: ...
  def shutdown(self, model_id: str) -> None: ...
  def shutdown_all(self) -> None: ...
```

#### Refactored `FFTWorkerManager` (`src/server/worker_manager.py`)
```python
  def launch_trainer(self, model_id: str) -> None:
    proc = self.train_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    base_model, weight_sync_strategy = _fetch_metadata_from_store(model_id)
    env = {
      **os.environ,
      "OPEN_RL_ENABLE_FFT": "true",
      "OPEN_RL_TIME_SLICE_JOB_ID": workload_job_id("trainer", model_id),
      "OPEN_RL_TIME_SLICE_GROUP": TRAINER_TIME_SLICE_GROUP,
    }
    if base_model:
      env["BASE_MODEL"] = base_model
    if weight_sync_strategy:
      env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync_strategy
    self.train_processes[model_id] = subprocess.Popen(
      _py_cmd(["gpu"], "server.training_requests_processor", model_id),
      cwd=self.project_dir,
      env=env,
      start_new_session=True,
    )

  def launch_sampler(self, model_id: str) -> None:
    proc = self.sampler_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    base_model, weight_sync_strategy = _fetch_metadata_from_store(model_id)
    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    if base_model:
      env["BASE_MODEL"] = base_model
    sampling_backend = os.getenv("SAMPLING_BACKEND", "vllm").lower()
    if sampling_backend == "vllm":
      sampler_env = env.copy()
      sampler_env["OPEN_RL_MODEL_ID"] = model_id
      sampler_env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id("sampler", model_id)
      sampler_env["OPEN_RL_TIME_SLICE_GROUP"] = SAMPLER_TIME_SLICE_GROUP
      if weight_sync_strategy:
        sampler_env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync_strategy
      sampler_gpu = os.getenv("SAMPLER_CUDA_VISIBLE_DEVICES")
      if sampler_gpu:
        sampler_env["CUDA_VISIBLE_DEVICES"] = sampler_gpu

      self.sampler_processes[model_id] = subprocess.Popen(
        _py_cmd(["gpu", "vllm"], "server.vllm_sampler", model_id),
        cwd=self.project_dir,
        env=sampler_env,
        start_new_session=True,
      )
```
*(Note: `K8sWorkerManager` in `src/server/k8s_worker_manager.py` follows the exact same pattern: `launch_trainer(self, model_id: str)` and `launch_sampler(self, model_id: str)` fetch `(base_model, weight_sync_strategy) = _fetch_metadata_from_store(model_id)` before calling `self._launch_pod(model_id, role="...", base_model=base_model, weight_sync_strategy=weight_sync_strategy)`).*

---

### Opportunity 4: Eliminate Dual-Key Storage in `delete_model` (`src/server/gateway.py`)

```diff
 @app.post("/api/v1/delete_model")
 async def delete_model(req: dict):
   model_id = req.get("model_id")
   if not model_id:
     return JSONResponse(status_code=400, content={"error": "model_id is required"})
   if is_fft_enabled():
     print(f"[GATEWAY] Requesting shutdown of workers for model {model_id}...")
     await store.put_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id, "op": "shutdown_workers"})
     await store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
-  await store.delete_values(f"open_rl:model_meta:{model_id}", f"open_rl:model_base:{model_id}")
+  await store.delete_values(f"open_rl:model_meta:{model_id}")
   return {"status": "ok"}
```

---

## 4. Summary of Changes & Impact

| Function / Component | Location | Nature of Change | Net Lines Impact |
| :--- | :--- | :--- | :---: |
| `TrainingModelMetadata` | `src/server/gateway.py` | **Refactored** — Made `base_model` optional (`str | None`) so metadata is always written upon creation/restoration. | `0` |
| `_extract_and_persist_model_metadata` | `src/server/gateway.py` | **New Function** — Centralizes validation, header parsing (`x-open-rl-weight-sync-strategy` / `x-open-rl-training-kind`), and `open_rl:model_meta:<model_id>` storage. | `+35` |
| `create_model` | `src/server/gateway.py` | **Refactored** — Replaced inline metadata extraction/storage and dual-key writing with helper call. | `-26` |
| `create_model_from_state` | `src/server/gateway.py` | **Refactored / Fix** — Replaced inline metadata extraction with helper call and included `base_model` / `full_config` in `make_training_request` payload. | `-6` |
| `launch_worker_and_enqueue` | `src/server/gateway.py` | **Refactored / Fix** — Removed destructive `model_meta` overwriting and `model_base` writing; simplified to take/pass only `model_id`. | `-14` |
| `ensure_sampler_launched` | `src/server/gateway.py` | **Refactored** — Removed legacy fallback reads for `open_rl:model_base:<id>`, string type checks, and extra arguments; simplified to take `model_id`. | `-18` |
| `delete_model` | `src/server/gateway.py` | **Refactored** — Removed deletion of `open_rl:model_base:<id>`. | `0` |
| `WorkerManager` / `FFTWorkerManager` / `K8sWorkerManager` | `src/server/worker_manager.py` / `k8s_worker_manager.py` | **Refactored** — Simplified `launch_trainer(model_id)` and `launch_sampler(model_id)` signatures; added `_fetch_metadata_from_store(model_id)` canonical lookup. | `-5` |
| `StoreStub` & Unit Tests | `tests/test_worker_manager.py` / `test_k8s_worker_manager.py` | **Test Fixes & Expansions** — Fixed `StoreStub.get_value_sync`, verified canonical lookup across local and Kubernetes worker managers. | `+40` |
| **Total** | Across server and test files | **Net Code Reduction & Architectural Clean-up** | `-24` |

---

## 5. Testing & Verification Plan

### 5.1 Critical Analysis of Existing Test Suite & Gaps Against `model_id` Canonical Lookup

A critical review of our existing unit tests (`tests/`) revealed four major blind spots and defects with respect to this architectural shift:
1. **Disconnected Stubs & Mocking Blind Spots (`test_worker_manager.py`)**: `GatewayInlineWorkerLaunchTest` mocks out `fft_worker_manager` with `WorkerManagerStub` (which records `launched_model_ids` without checking Redis metadata). Meanwhile, `FFTWorkerManagerTest` calls `manager.launch("Model_A.1")` directly without calling `create_model` first. Because of this separation, **the contract between the gateway writing `open_rl:model_meta:<id>` and the worker manager retrieving it from Redis is never tested end-to-end.**
2. **Out-of-Sync Production Code vs. Aspirational Tests (`test_launch_fetches_metadata_from_store`)**: `test_worker_manager.py:L153` tests `manager.launch_trainer("Model_A.1")` reading `open_rl:model_meta` from `InMemoryStore`. However, in our current `main` branch, `worker_manager.py` and `k8s_worker_manager.py` still take `(self, model_id, base_model=None, weight_sync_strategy=None)` and **do not query `open_rl:model_meta:<id>` at all** when invoked with `model_id` alone.
3. **Broken Test Stub (`StoreStub.get_value_sync`)**: Running `tests/test_worker_manager.py` crashes in `GatewayInlineWorkerLaunchTest.test_create_model_launches_worker_then_enqueues` with `AttributeError: 'StoreStub' object has no attribute 'get_value_sync'`. That is because `test_worker_manager.py:L81` calls `self.store.get_value_sync(...)`, but `StoreStub` only implemented `async def get_value(...)`.
4. **Missing Coverage for `create_model_from_state` Metadata & Payload Propagation**: `test_worker_manager.py:L95` (`test_create_model_from_state_launches_worker_then_enqueues`) only asserts that `launched_model_ids == [model_id]` and that `forwarded_requests[0]["op"] == "create_model_from_state"`. It **does not check whether `open_rl:model_meta:<model_id>` was written**, nor does it verify if `base_model` and `full_config` are forwarded in `command["payload"]`.
5. **Zero Metadata Lookup Coverage for Kubernetes Worker Manager (`test_k8s_worker_manager.py`)**: Every pod launch test in `test_k8s_worker_manager.py` calls `manager.launch("Model_A.1")` directly without mocking or checking `open_rl:model_meta:<model_id>`. There is no test verifying that `KubernetesFFTWorkerManager` sets `BASE_MODEL` and `OPEN_RL_WEIGHT_SYNC_STRATEGY` from Redis when creating K8s Pod specifications.

---

### 5.2 Required Unit Test Modifications & New Test Specs

To rigorously validate this design during implementation, the following concrete modifications and new test cases must be added:

#### 1. Fix `StoreStub` in `tests/test_worker_manager.py`
Add `get_value_sync` to `StoreStub` to prevent inline gateway tests from crashing with `AttributeError`:
```python
class StoreStub:
  ...
  def get_value_sync(self, key: str) -> str | None:
    return self.kv_store.get(key)
```

#### 2. Expand `test_create_model_from_state_launches_worker_then_enqueues` (`tests/test_worker_manager.py`)
Verify that `create_model_from_state` unconditionally writes `open_rl:model_meta:<model_id>`, does **not** write `open_rl:model_base:<id>`, and forwards `base_model` / `full_config` inside `command["payload"]`:
```python
  async def test_create_model_from_state_launches_worker_then_enqueues(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model_from_state({
        "state_path": "/tmp/checkpoint",
        "base_model": "restored-base",
        "weight_sync_strategy": "delta"
      })

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    
    # Assert canonical metadata persistence:
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "restored-base")
    self.assertEqual(meta["training_kind"], "restored")
    self.assertEqual(meta["weight_sync_strategy"], "delta")
    
    # Assert no dual-key writing:
    self.assertIsNone(self.store.get_value_sync(f"open_rl:model_base:{model_id}"))
    
    # Assert forwarded request payload contains configuration:
    req_payload = self.store.forwarded_requests[0]["payload"]
    self.assertEqual(req_payload["base_model"], "restored-base")
    self.assertEqual(req_payload["full_config"]["weight_sync_strategy"], "delta")
```

#### 3. Add Canonical Metadata Lookup Tests for `KubernetesFFTWorkerManager` (`tests/test_k8s_worker_manager.py`)
Add a new test asserting that when `K8sWorkerManager.launch("Model_A.1")` or `launch_sampler("Model_A.1")` runs, it queries `open_rl:model_meta:Model_A.1` and populates the K8s pod container's environment variables:
```python
  def test_launch_queries_model_metadata_for_pod_env(self) -> None:
    import json
    from server.store import InMemoryStore
    
    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps({
      "base_model": "gemma-4-k8s",
      "weight_sync_strategy": "full",
      "training_kind": "full"
    })
    api = _FakeCoreApi()
    
    with patch("server.store.get_store", return_value=s):
      self._manager(api).launch("Model_A.1")
      self._manager(api).launch_sampler("Model_A.1")

    # Verify Trainer Pod Env:
    trainer_container = api.created[0][1]["spec"]["containers"][0]
    trainer_env = {item["name"]: item["value"] for item in trainer_container["env"] if "value" in item}
    self.assertEqual(trainer_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(trainer_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

    # Verify Sampler Pod Env:
    sampler_container = api.created[1][1]["spec"]["containers"][0]
    sampler_env = {item["name"]: item["value"] for item in sampler_container["env"] if "value" in item}
    self.assertEqual(sampler_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(sampler_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")
```

#### 4. Add `ensure_sampler_launched(model_id)` Canonical Lookup Test (`tests/test_worker_manager.py`)
Assert that calling `ensure_sampler_launched(model_id)` passes solely `model_id` to `fft_worker_manager.launch_sampler(model_id)` and that the sampler process launcher retrieves the model configuration directly from Redis:
```python
  async def test_ensure_sampler_launched_delegates_to_worker_manager_with_model_id(self) -> None:
    import json
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}):
      self.store.kv_store["open_rl:model_meta:model-x"] = json.dumps({
        "base_model": "base-vllm",
        "weight_sync_strategy": "delta",
        "training_kind": "full"
      })
      await gateway.ensure_sampler_launched("model-x")

    self.assertEqual(self.worker_manager.launched_model_ids, ["model-x"])
```

---

### 5.3 End-to-End & Cluster Verification Workflows

After all unit tests pass under `make test`, execute the following E2E integration benchmarks to verify dynamic worker spawning on real or port-forwarded Kubernetes clusters:
1. **Local Port-Forward / Fast E2E Smoke Test**:
   ```bash
   make test e2e tiny-rl
   make test e2e fft-gsm8k TRAINING_TEST_ARGS='steps=5'
   ```
   *Verify in the gateway logs that trainer and vLLM sampler dynamic workers boot using `open_rl:model_meta:<id>` and that no `open_rl:model_base:<id>` keys are created in Redis.*
2. **Distributed Kubernetes E2E Benchmark**:
   ```bash
   make cluster-e2e IMAGE_TAG=$(cat VERSION 2>/dev/null || echo latest) \
     E2E_SCENARIO=fft-gsm8k-rl \
     E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=5"
   ```
   *Inspect the live progression table via `metrics.jsonl` inside the gateway pod to confirm clean step execution and dynamic pod orchestration.*
