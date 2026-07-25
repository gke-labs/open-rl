# Design Doc 001: Delta Weight Sync Cleanup & Streamlined Checkpointing Architecture

**Status**: Proposed  
**Author**: Open-RL Engineering  
**Date**: 2026-07-15  
**Target Branch**: `feat/delta-snapshot-weight-transfer-engine`  

---

## 1. Executive Summary

This design document outlines an architectural simplification and correctness fix for the **Delta Weight Sync strategy** in `FFTTrainingWorker`. Specifically, it addresses three critical aspects of the interaction between `prepare_model_for_training()`, `optim_step()`, and `save_state_delta()`:
1. **Destructive Reads & Non-Idempotent Checkpointing**: `save_state_delta()` currently clears `self._latest_delta_tensors` (`{}`) after reading, causing consecutive checkpoint calls within the same training step (`save_weights_for_sampler` followed by `save_state`) to write out corrupted `0%` delta files on the second save.
2. **Duplicate/Redundant Diffing Loops**: `save_state_delta()` contains a fallback `else:` diffing loop that duplicates the GPU-to-CPU sparse diff computation, shadow buffer mutations (`_update_prev_cpu_weight`), and offload checks from `optim_step()`.
3. **Repeated Model Parameter Scanning**: Both `optim_step()` and `save_state_delta()` repeatedly sum element counts (`p.numel()`) and iterate over `named_parameters()`, which breaks during CPU offloading when GPU tensors have `0-numel`.

By caching static model metadata (`self.model_layer_names` and `self.total_model_elements`) at model load time, designating `optim_step()` as the **exclusive single source of truth** for computing sparse weight diffs, and treating `save_state_delta()` as a **pure read-only consumer**, we eliminate ~35 lines of complex duplicate diffing logic, ensure idempotent multi-destination checkpointing, and achieve $O(1)$ instant serialization when no training step has occurred (e.g., Step 0 initialization).

---

## 2. Current Architecture & Pain Points

### A. The Destructive Read Bug (`_latest_delta_tensors.clear()`)
In `FFTTrainingWorker`, during a standard distributed fine-tuning step under `weight_sync_strategy="delta"`, `optim_step()` runs first:
```python
# In optim_step():
self._latest_delta_tensors = {
  "names": layer_names_list,
  "indices_list": indices_list,
  "values_list": values_list,
  "layer_lengths_list": layer_lengths_list,
}
```

Immediately following `optim_step()`, the gateway and worker controller routinely issue two consecutive save requests:
1. **`save_weights_for_sampler`**: Writes sparse delta weights to `/tmp/sampler_full/...` and publishes a Redis notification to trigger sampler synchronization.
2. **`save_state`**: Writes the persistent checkpoint for the training step to shared storage / NFS (`/mnt/shared/.../step_N`).

Under the current implementation of `save_state_delta()`:
```python
if self._latest_delta_tensors and "names" in self._latest_delta_tensors:
  layer_names_list = self._latest_delta_tensors["names"]
  # ... read tensors ...
  self._latest_delta_tensors = {}  # <-- DESTRUCTIVE READ (Bug)
else:
  # Fallback diffing loop ...
```

#### Failure Scenario:
1. `optim_step(step=10)` runs and computes a `2.5%` sparse weight diff against `_param_shadow`. It stores this diff in `_latest_delta_tensors` and updates `_param_shadow` with the new weights.
2. `save_weights_for_sampler` calls `save_state_delta()`. It consumes `_latest_delta_tensors` (`2.5%` diff), saves the file for samplers, and **clears** `_latest_delta_tensors`.
3. Milliseconds later, `save_state` calls `save_state_delta()` to save the Step 10 training checkpoint to shared NFS. Because `_latest_delta_tensors` was just cleared, execution falls through to the `else:` diffing loop.
4. Because `_param_shadow` was already updated in Step 1 to reflect the Step 10 weights, the fallback diffing loop compares `cur_t` against `_param_shadow` and finds **`0` changed elements**.
5. **Result**: The persistent Step 10 training checkpoint on disk is written as a corrupted `0%` delta file (`0` changed tensors) instead of the actual `2.5%` step delta.

---

### B. Duplicate Diffing Loop & O(N) Overhead at Step 0
When `save_state_delta()` is invoked when `_latest_delta_tensors` is empty (e.g., at Step 0 before any `optim_step()` has run, or during fallback saves), it executes a ~35-line `else:` diffing loop over all `named_parameters()`:
```python
else:
  for name, param in self.model.named_parameters():
    if not param.requires_grad:
      continue
    if param in self._param_shadow and self._param_shadow[param][1].numel() > 0 and param.numel() == 0:
      cur_t = self._param_shadow[param][1]
    else:
      cur_t = param.data
    prev_tensor = self._get_prev_cpu_weight(name, param)
    # ... allocate buffers, compare cur_t.ne(prev_gpu), extract nonzero indices, update shadow ...
```

#### Problems with this loop:
* **Duplicate Responsibility**: It duplicates the exact same diffing, CPU buffer allocation (`torch.empty(..., pin_memory=...)`), and shadow-updating logic already managed inside `optim_step()`.
* **Redundant O(N) Computation at Step 0**: When `prepare_model_for_training()` initializes the model under `weight_sync_strategy="delta"`, `self._param_shadow` is already populated with the base model weights. If `save_state_delta()` is called right at Step 0 before `optim_step()`, **zero weights have changed**. Running a full GPU-vs-CPU comparison loop across every parameter in the model just to confirm `0` changed elements wastes CPU/GPU cycles and PCIE bandwidth.
* **Offloading Vulnerability**: Computing `sum(p.numel() ...)` dynamically inside save operations fails when `cpu_offload=True` and the worker is asleep (`self._is_offloaded == True`), since GPU tensors have `0-numel`.

---

## 3. Proposed Architecture & Design

### A. Static Model Metadata Caching at Load Time
Because an LLM's architecture (`AutoModelForCausalLM`) and parameter count are static invariants throughout a fine-tuning job, we compute and cache `self.model_layer_names` and `self.total_model_elements` once when the model is prepared for training. This ensures $O(1)$ access across all subsequent training steps and eliminates zero-numel anomalies during CPU offloading.

### B. Single Source of Truth Principle (`optim_step` as Exclusive Writer)
In Open-RL, `optim_step()` is the single source of truth for model weight updates. Model parameters only change when an optimizer step executes.

Under the proposed design:
1. **`optim_step()` is the sole writer of `_latest_delta_tensors`**: Whenever `optim_step()` runs under `weight_sync_strategy="delta"`, it clears and re-computes `_latest_delta_tensors` from the current step diff and updates `_param_shadow`. `_latest_delta_tensors` remains immutable until the next `optim_step()` runs.
2. **`save_state_delta()` is a pure read-only consumer**: `save_state_delta()` never clears or mutates `_latest_delta_tensors`. Any number of save calls (`save_weights_for_sampler`, `save_state`, external queries) during Step $N$ will safely and idempotently package the exact same Step $N$ delta.

### C. O(1) Fast-Path for Unpopulated / Step 0 Deltas
If `save_state_delta()` is called when `_latest_delta_tensors` is empty (`{}`)—which only occurs before the first `optim_step()` has executed at Step 0—we know with absolute certainty that **no optimizer step has run and no weights have changed since initialization**.

Instead of running an $O(N)$ parameter-scanning diff loop, the `else:` branch reads directly from the cached `self.model_layer_names` and `self.total_model_elements` to construct and return a **valid 1D flat-packed empty delta** in true $O(1)$ time without touching PyTorch parameter tensors.

---

## 4. Detailed Implementation Specification

### 1. Instance Attribute Initialization (`FFTTrainingWorker.__init__`)
Following Open-RL class initialization rules (`AGENTS.md`), all attributes are explicitly initialized in `__init__`:
```python
self.model_layer_names: list[str] = []
self.total_model_elements: int = 0
```

### 2. Metadata Caching at Load Time (`FFTTrainingWorker.prepare_model_for_training`)
```python
def prepare_model_for_training(self) -> None:
  assert self.model is not None, "Model is not loaded. Call load_base_model first."

  for param in self.model.parameters():
    param.requires_grad_(True)
  self.trainable_params = trainable_model_parameters(self.model)

  # Cache static invariant model metadata once at initialization:
  self.model_layer_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
  self.total_model_elements = sum(p.numel() for p in self.model.parameters())

  if self.weight_sync_strategy == "delta":
    for param in self.model.parameters():
      if param.requires_grad and param not in self._param_shadow:
        cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
        cpu_buf.copy_(param.data, non_blocking=True)
        self._param_shadow[param] = (param.device, cpu_buf)
  # ...
```

### 3. Refactored `save_state_delta` (`FFTTrainingWorker.save_state_delta`)
```python
def save_state_delta(
  self,
  model_id: str,
  state_path: str,
  kind: str = "sampler",
) -> dict[str, Any]:
  assert self.model is not None, "Model must be loaded first."
  if self.cpu_offload and not self._is_offloaded:
    raise RuntimeError(
      "Cannot save state delta while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
      "GPU time-slicer lock is not held during save operations."
    )

  os.makedirs(state_path, exist_ok=True)

  # Pure read from the most recent optim_step() source of truth
  if self._latest_delta_tensors and "names" in self._latest_delta_tensors:
    layer_names_list = self._latest_delta_tensors["names"]
    indices_list = self._latest_delta_tensors["indices_list"]
    values_list = self._latest_delta_tensors["values_list"]
    layer_lengths_list = self._latest_delta_tensors["layer_lengths_list"]
    total_changed = self._latest_total_changed
    total_elements = self._latest_total_elements
  else:
    # Step 0 baseline / no optim_step() occurred yet -> emit valid empty delta in true O(1) time
    layer_names_list = self.model_layer_names
    layer_lengths_list = [0] * len(layer_names_list)
    total_changed = 0
    total_elements = self.total_model_elements
    indices_list = []
    values_list = []

  if indices_list:
    indices_flat = torch.cat(indices_list).to(torch.int32).contiguous()
    values_flat = torch.cat(values_list).contiguous()
  else:
    # Use model dtype for values_flat to ensure strict type consistency in empty checkpoints
    fallback_dtype = next(self.model.parameters()).dtype if self.model else torch.float32
    indices_flat = torch.empty(0, dtype=torch.int32, device="cpu")
    values_flat = torch.empty(0, dtype=fallback_dtype, device="cpu")

  layer_lengths_tensor = torch.tensor(layer_lengths_list, dtype=torch.int64, device="cpu")
  packed_delta = {
    "delta.indices_flat": indices_flat,
    "delta.values_flat": values_flat,
    "delta.layer_lengths": layer_lengths_tensor,
  }

  import safetensors.torch

  delta_path = os.path.join(state_path, "delta.safetensors")
  safetensors.torch.save_file(
    packed_delta,
    delta_path,
    metadata={"layer_names": json.dumps(layer_names_list)},
  )

  metadata = {
    "base_model": self.base_model_name,
    "created_at": datetime.now().isoformat(),
    "format": "sparse_delta",
    "kind": kind,
    "model_id": model_id,
    "changed_elements": total_changed,
    "total_elements": total_elements,
    "layer_names": layer_names_list,
    "density_pct": round(100.0 * total_changed / max(1, total_elements), 3),
    "timestamp": time.time(),
  }
  with open(os.path.join(state_path, "metadata.json"), "w") as f:
    json.dump(metadata, f)

  print(f"Saved sparse delta ({metadata['density_pct']}% changed elements, {total_changed}/{total_elements}) to {state_path}")
  return {"path": state_path, "density_pct": metadata["density_pct"]}
```

---

## 5. Lifecycle & State Transition Matrix

| Execution Scenario | Previous Behavior | Proposed Behavior |
| :--- | :--- | :--- |
| **Model Initialization**<br>(`prepare_model_for_training()`) | Did not cache model metadata. Every save/diff step rescanned model layers and dynamically summed `p.numel()`. | Explicitly initializes `self.model_layer_names` and `self.total_model_elements` once in $O(N)$ time. |
| **Step 0 Initialization**<br>(`save_state_delta()` before first `optim_step`) | Runs full $O(N)$ parameter loop across all layers comparing GPU vs CPU tensors (`0` diffs found). | Instantly reads cached metadata (`model_layer_names`, `total_model_elements`) and emits valid $0\%$ 1D flat-packed empty delta in true $O(1)$ time. |
| **Consecutive Saves in Step $N$**<br>(`save_weights_for_sampler` $\rightarrow$ `save_state`) | First save writes correct Step $N$ delta and **clears** `_latest_delta_tensors`. Second save runs fallback loop and writes **corrupted $0\%$ delta**. | Both saves purely read from `_latest_delta_tensors` and write **100% identical, correct Step $N$ deltas**. |
| **Next Step Transition**<br>(`optim_step(step=N+1)`) | Re-computes diff and overwrites `_latest_delta_tensors`. | `optim_step()` clears previous `_latest_delta_tensors` at step start, computes new diff against `_param_shadow`, and populates `_latest_delta_tensors` for Step $N+1$. |
| **CPU Offloading (`cpu_offload=True`)** | Required special guard `param.numel() == 0` inside fallback loop during save. | Offloading happens post-`optim_step()`. `save_state_delta()` either packages `_latest_delta_tensors` or reads cached instance attributes without touching 0-numel GPU tensors. |

---

## 6. Verification & Test Strategy

1. **Idempotency & Multi-Save Verification**:
   * Add a unit test in `tests/test_delta_weight_sync.py` that invokes `worker.optim_step(adam_params)` once, then calls `worker.save_state_delta(..., kind="sampler")` followed immediately by `worker.save_state_delta(..., kind="state")` to distinct directories.
   * Assert that both saved `delta.safetensors` and `metadata.json` files contain identical `changed_elements`, `density_pct`, `indices_flat`, and `values_flat`.
2. **Step 0 / Empty Delta Fallback Verification**:
   * Verify that calling `worker.save_state_delta()` immediately after `worker.create_model(...)` without calling `optim_step()` writes a valid Safetensors file containing `0` elements with `values_flat.dtype` matching the base model parameter dtype (`bfloat16` or `float32`) and `total_elements` accurately matching `self.total_model_elements`.
3. **E2E Integration & Cluster Regression Suite**:
   * Run the standard test suite: `make test`
   * Run E2E distributed timeslice scenarios (`fft-gsm8k-rl` and `fft-gsm8k-rl-x2`) to verify that gateway training requests, sampler synchronization via Redis, and checkpoint recovery operate cleanly under the streamlined sync architecture.

---

## 7. Summary of Benefits

* **Bug Elimination**: Guarantees checkpoint integrity across all storage destinations (Redis/local/NFS/Cloud) by removing destructive reads and eliminating 0-delta corruption on back-to-back saves.
* **Code Simplicity & Maintainability**: Removes ~35 lines of duplicate, complex diffing and shadow-updating code from `save_state_delta()`.
* **Performance Optimization**: Eliminates redundant $O(N)$ parameter scanning and GPU/CPU comparisons at Step 0 and on consecutive checkpoints, achieving true $O(1)$ serialization when no training step has occurred.
* **Adherence to Best Practices**: Strictly follows Open-RL class attribute rules by explicitly initializing `self.model_layer_names` and `self.total_model_elements` inside `__init__` and populating them once during model setup.
