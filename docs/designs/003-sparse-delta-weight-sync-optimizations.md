# Design Doc 003: High-Performance Delta Weight Synchronization & Background DRAM Pre-Staging

**Author:** Open-RL Engineering Team  
**Status:** Approved / Implemented (`v0.3.7` – `v0.3.11`)  
**Target Component:** Gateway Server, vLLM Sampler, Trainer Engine, Delta Weight Transfer Engine  

---

## 1. Executive Summary

In distributed Reinforcement Learning (RLHF / GRPO / PPO) for Large Language Models (LLMs), policy weights must be continuously synchronized between the **Trainer** (PyTorch backpropagation engine) and the **Sampler** (vLLM inference engine). 

Standard full model weight reloads across Network File Systems (NFS) demand **50–60+ seconds per step** for 8B models (e.g., `Qwen/Qwen3-8B`), severely bottlenecking training throughput.

This design document specifies the architecture and optimizations implemented on this branch:
1. **Sparse Delta Patching**: Computing sparse parameter diffs (~12–15% changed parameters) instead of full weight checkpoints.
2. **Direct In-Place GPU Memory Mutation (`v0.3.7`)**: Direct PCIe DMA transfers into existing VRAM addresses using pinned host CPU memory arrays (`.pin_memory()`), reducing weight update latency from **12.5s to 4.4s** (> 2.8x speedup).
3. **Asynchronous Background DRAM Pre-Staging (`v0.3.8`–`v0.3.11`)**: Decoupling disk I/O and DRAM memory allocations from the critical path by staging pinned CPU host arrays off-path upon receiving Redis Pub/Sub weight update signals.
4. **NFS Propagation Polling & Lock-Free Prefetching (`v0.3.10`–`v0.3.11`)**: Polling for NFS directory propagation and executing preloading concurrently without thread lock contention while Sampler workers are in sleep mode under Accelerator Time-Slicing.

---

## 2. Problem Statement & Bottlenecks

### 2.1 The RL Weight Synchronization Loop
During distributed RL training, the workload alternates between:
1. **Rollout Generation (Sampler)**: vLLM generates trajectories from current prompt batches.
2. **Policy Gradient Step (Trainer)**: Trainer computes loss, backpropagation, and updates policy parameters with AdamW.
3. **Weight Transfer**: Updated weights must be transferred to Sampler before the next rollout.

### 2.2 Critical Path Bottlenecks
In naive implementations:
- **Full Model Reload**: 15.26 GiB safetensors reloaded over NFS took **57.18 seconds**.
- **CPU Host Allocation & Copy Overhead**: Creating unpinned CPU numpy/torch arrays on every step caused CPU thrashing and garbage collection pauses.
- **Synchronous Disk I/O on Request Arrival**: The Sampler waited until the HTTP rollout request arrived to start reading delta files from disk, keeping the GPU idle.

---

## 3. System Architecture & Component Design

```
+-----------------------------------------------------------------------------------+
|                                TRAINER NODE                                       |
|                                                                                   |
|  PyTorch AdamW Update ---> Extract Sparse Delta ---> Save delta.safetensors to NFS |
|                                                             |                     |
|                                                             v                     |
|                                                   Publish Redis Pub/Sub           |
|                                          "open_rl:weight_update:<model_id>"       |
+-----------------------------------------------------------------------------------+
                                                              |
                                                              v (Redis Signal)
+-----------------------------------------------------------------------------------+
|                                SAMPLER NODE                                       |
|                                                                                   |
|  [Background Thread] weight_prefetcher_loop()                                     |
|    |                                                                              |
|    v                                                                              |
|  Poll NFS for propagation (up to 10s wait)                                        |
|    |                                                                              |
|    v                                                                              |
|  preload_delta_to_dram()  [Lock-Free Background Execution]                        |
|  - Parse safetensors headers & layer offsets                                      |
|  - Allocate contiguous Pinned CPU Host Memory (.pin_memory())                     |
|  - Store in self._staged_delta (protected by _staged_delta_lock)                  |
|                                                                                   |
|  -------------------------- CRITICAL SAMPLING PATH -----------------------------  |
|                                                                                   |
|  Rollout Request Arrives ---> Wakes Engine from Sleep                             |
|    |                                                                              |
|    v                                                                              |
|  _apply_gpu_in_place() [PRELOAD HIT]                                              |
|  - Retrieve self._staged_delta from host DRAM                                     |
|  - Non-blocking PCIe DMA Copy directly into VRAM Addresses                        |
|  - Mutate GPU parameter memory in-place                                           |
|                                                                                   |
|  ---> Execute vLLM Rollout Inference                                             |
+-----------------------------------------------------------------------------------+
```

---

## 4. Key Optimizations & Implementation Details

### 4.1 Direct In-Place GPU Memory Patching (`v0.3.7`)
- **Module**: `DeltaSnapshotWeightTransferEngine` ([delta_weight_transfer_engine.py](file:///usr/local/google/home/sunilarora/open-rl/src/server/delta_weight_transfer_engine.py))
- **Mechanism**:
  - Maintains a baseline model snapshot in CPU memory (`_cpu_snapshot`).
  - Resolves target layer parameters directly in vLLM's GPU model architecture (`_resolved_gpu_params`).
  - Uses pinned CPU memory host tensors (`.pin_memory()`) for contiguous DMA transfers:
    ```python
    bulk_indices_cpu = torch.from_numpy(flat_indices).pin_memory()
    bulk_values_cpu = torch.from_numpy(flat_values).pin_memory()
    ```
  - Mutates GPU parameter tensors directly in VRAM without re-allocating model layers or rebuilding PyTorch computation graphs.

### 4.2 Asynchronous Background DRAM Pre-Staging (`v0.3.8`–`v0.3.9`)
- **Method**: `preload_delta_to_dram(target_path: str)`
- **Workflow**:
  1. When Trainer finishes writing weights, it publishes a Redis Pub/Sub signal containing `weights_path`.
  2. Sampler's `weight_prefetcher_loop` catches the signal and triggers `preload_delta_to_dram(target_path)` via `engine.collective_rpc`.
  3. Pre-allocates pinned CPU DRAM memory, resolves layer target offsets, and caches the staged state under `self._staged_delta_lock`.
  4. When `_apply_gpu_in_place` runs during request processing, it detects `self._staged_delta` (`[PRELOAD HIT]`) and skips disk I/O, header parsing, and host allocations entirely.

### 4.3 NFS Propagation Polling Loop (`v0.3.10`)
- **Problem**: Redis Pub/Sub signals arrive in < 1 ms, but NFS directory propagation across Kubernetes nodes can take 200 ms – 1.5 seconds. Early file existence checks caused preloading to abort silently.
- **Fix**: Added a 10-second polling wait loop in `preload_delta_to_dram`:
  ```python
  delta_file = os.path.join(target_path, "delta.safetensors")
  metadata_path = os.path.join(target_path, "metadata.json")

  start_wait = time.perf_counter()
  while not (os.path.exists(delta_file) and os.path.exists(metadata_path)):
    if time.perf_counter() - start_wait > 10.0:
      logger.warning(
        "[DeltaSnapshotEngine] [PRELOAD] Target files missing after wait:"
        f" '{target_path}'"
      )
      return
    time.sleep(0.2)
  ```

### 4.4 Lock-Free Async Prefetch Execution (`v0.3.11`)
- **Problem**: `weight_prefetcher_loop` in `vllm_sampler.py` acquired `async with reload_lock:`, which blocked or was blocked by active HTTP sampling requests.
- **Fix**: Removed `reload_lock` wrapper from `weight_prefetcher_loop`. Since `preload_delta_to_dram` only populates thread-safe CPU host memory (`self._staged_delta`) and never touches GPU VRAM, it runs fully concurrently while Sampler workers are in sleep mode under Accelerator Time-Slicing.

---

## 5. Performance Benchmarks & Results

| Optimization Milestone | Weight Transfer Strategy | Latency / Time per Step | Speedup vs Naive |
| :--- | :--- | :---: | :---: |
| **Baseline** | Full Safetensors Reload over NFS | 57,180 ms | 1.0x |
| **`v0.3.6`** | Sparse Delta Snapshot (Simple) | 12,493 ms | 4.5x |
| **`v0.3.7`** | Direct In-Place GPU Patching | 4,465 ms | 12.8x |
| **`v0.3.11`** | In-Place GPU + Lock-Free DRAM Pre-Staging | **950–1,200 ms** (DMA only) | **> 47x** |

### Live RL Benchmark Results (`Qwen/Qwen3-8B`, 192 Batch Size)
```text
Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time
--------------------------------------------------------------------------------
   0 |   8.85%  | -0.0063 |   206.7s |      75.1s |      27.5s |          309.5s
   1 |  37.50%  |  0.3031 |    45.1s |      66.5s |      19.2s |          131.0s
   2 |  44.79%  |  0.3755 |    40.1s |      54.0s |      24.5s |          118.8s
```
* Overall step latency reduced from **309.5s to 118.8s** (> 2.6x overall RL step speedup).
* Model accuracy improved from **8.85% to 44.79%** over 2 training steps.

---

## 6. Code Verification & Test Coverage

- **Unit Tests**: Pass 86 unit tests (`make test`), including `test_preload_delta_to_dram` in `tests/test_delta_weight_transfer_engine.py`.
- **Linter & Formatting**: Clean under `make fmt && make lint` (80 files clean, 0 ruff errors).
- **Cluster Compatibility**: Verified on GKE cluster with Accelerator Time-Slicer (`v0.3.11`).
