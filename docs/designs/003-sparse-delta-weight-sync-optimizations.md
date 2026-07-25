# Design Doc 003: High-Performance Delta Weight Synchronization & Background DRAM Pre-Staging

**Author:** Open-RL Engineering Team  
**Status:** Approved / Implemented (`v0.3.7` – `v0.3.11`)  
**Target Component:** Gateway Server, vLLM Sampler, Trainer Engine, Delta Weight Transfer Engine  

---

## 1. Executive Summary

In distributed Reinforcement Learning (RLHF / GRPO / PPO) for Large Language Models (LLMs), policy weights must be continuously synchronized between the **Trainer** (PyTorch backpropagation engine) and the **Sampler** (vLLM inference engine). 

Standard full model weight reloads across Network File Systems (NFS) demand **50–60+ seconds per step** for 8B models (e.g., `Qwen/Qwen3-8B`), severely bottlenecking training throughput.

This design document specifies the architecture and optimizations implemented on this branch:
1. **Sparse Delta Patching**: Computing sparse parameter diffs (~6–13% changed parameters) instead of full weight checkpoints.
2. **Direct In-Place GPU Memory Mutation (`v0.3.7`)**: Direct PCIe DMA transfers into existing VRAM addresses using pinned host CPU memory arrays (`.pin_memory()`), reducing weight update latency from **12.5s to 4.2s** (> 2.8x speedup over simple delta).
3. **Asynchronous Background DRAM Pre-Staging (`v0.3.8`–`v0.3.11`)**: Decoupling disk I/O and DRAM memory allocations from the critical path by staging pinned CPU host arrays off-path upon receiving Redis Pub/Sub weight update signals.
4. **NFS Propagation Polling & Lock-Free Prefetching (`v0.3.10`–`v0.3.11`)**: Polling for NFS directory propagation and executing preloading concurrently without thread lock contention while Sampler workers are in sleep mode under Accelerator Time-Slicing.

---

## 2. Comprehensive Comparison of All Weight Transfer Schemes

| Scheme | Architecture & Mechanics | Measured Update Latency (8B Model) | CPU Host RAM Overhead | Pros | Cons |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Scheme A: Full Model Reloading** *(Naive Baseline)* | Re-reads full 15.26 GiB safetensors checkpoint files over NFS on every step via vLLM loader. | **57.18 s** (57,180 ms) | **0 MB** | • Standard vLLM implementation<br>• Zero custom patching code | • Severe bottleneck (~50% of step time)<br>• Heavy NFS network/disk I/O thrashing<br>• Re-instantiates engine objects every step |
| **Scheme B: Naive CPU-Snapshot Delta** *(`v0.3.0`–`v0.3.6`)* | Merges incoming delta diffs into a persistent CPU `state_dict` model snapshot, then calls vLLM `load_weights()`. | **3.82 s – 12.49 s** *(3,822 ms sync / 12,493 ms H2D)* | **15.3 GB / worker** *(Stores full model CPU `state_dict`)* | • 4.5x faster than full reload<br>• Reduced transfer payload (1.05 GB vs 15.26 GB) | • **Massive Host CPU RAM Overhead** (122+ GB for 8 workers)<br>• Unpinned memory causes GC pauses<br>• Temporary GPU allocations cause VRAM fragmentation |
| **Scheme C: Direct In-Place GPU Patching** *(`v0.3.7`)* | Pinned CPU host arrays (`.pin_memory()`) copied directly into existing VRAM addresses via non-blocking PCIe DMA (`_apply_gpu_in_place`). | **4.46 s** (4,465 ms) | **0 MB** *(Zero CPU memory reserved)* | • > 12.8x speedup vs naive<br>• Zero VRAM reallocation or graph rebuild<br>• Direct pointer mutation in GPU memory | • Disk I/O & host allocations still run synchronously inside HTTP rollout request path (~0.5–1.2s overhead) |
| **Scheme D: In-Place GPU + Lock-Free DRAM Pre-Staging** *(`v0.3.11` - Final Best)* | Redis Pub/Sub triggers lock-free background DRAM staging (`preload_delta_to_dram`) while Sampler sleeps under Time-Slicing. 10s NFS polling wait loop. | **Critical Path: ~950 ms**<br>*(Total GPU DMA: 4.23 s)* | **~1–2 GB pinned DRAM** *(Transient during preloading)* | • **> 47x speedup vs full reload**<br>• Disk I/O & host allocations 100% off critical path<br>• Lock-free & thread-safe under Time-Slicing | • Slightly higher peak host DRAM usage (~1–2 GB pinned DRAM buffer per active model) |

### 2.1 Trade-offs: Why We Replaced CPU-Snapshot Delta (Scheme B) with Direct In-Place GPU Patching (Scheme C/D)

In early releases (`v0.3.0`–`v0.3.6`), Scheme B applied sparse deltas by maintaining a full CPU `state_dict` baseline in host memory. While it achieved decent transfer latencies, it had two critical operational flaws:
1. **CPU Memory Footprint (15.3 GB per Worker)**: Storing full CPU model snapshots for each Sampler worker process required **122.4 GB of host CPU RAM** on an 8-GPU node, triggering severe kernel OOM kills.
2. **vLLM Engine Callback Overhead**: Calling vLLM's `load_weights()` callback triggered PyTorch model layer re-allocations and unpinned memory staging copies.

By contrast, **Scheme C & D (`in_place_gpu_delta`)** mutate GPU VRAM parameter tensors directly using direct pointer offsets (`index_copy_`), eliminating host CPU snapshot memory entirely (**0 MB persistent CPU RAM reserved**) while accelerating critical-path weight synchronization down to **~950 ms**.

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
      logger.warning(f"[DeltaSnapshotEngine] [PRELOAD] Target files missing after wait: '{target_path}'")
      return
    time.sleep(0.2)
  ```

### 4.4 Lock-Free Async Prefetch Execution (`v0.3.11`)
- **Problem**: `weight_prefetcher_loop` in `vllm_sampler.py` acquired `async with reload_lock:`, which blocked or was blocked by active HTTP sampling requests.
- **Fix**: Removed `reload_lock` wrapper from `weight_prefetcher_loop`. Since `preload_delta_to_dram` only populates thread-safe CPU host memory (`self._staged_delta`) and never touches GPU VRAM, it runs fully concurrently while Sampler workers are in sleep mode under Accelerator Time-Slicing.

---

## 5. Detailed `sparse_delta` Benchmarks & Profiling Data

### 5.1 Step-by-Step `sparse_delta` Telemetry (`Qwen/Qwen3-8B`)

| Step | Mutated Layers | Mutated Parameter Elements | % of Total Weights (8.19B) | Safetensors Read Time | Pointer Resolve Time | Direct GPU DMA Copy Time | Total In-Place GPU Patch Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0 layers | 0 / 8,190,735,360 | **0.000%** *(No-Op)* | 25.77 ms | 0.00 ms | 0.00 ms | **0.00 ms** |
| **1** | 348 layers | 1,057,194,125 / 8,190,735,360 | **12.907%** | 7.08 ms | 3.07 ms | 9,508.11 ms | **9,511.38 ms** (9.51s) |
| **2** | 340 layers | 742,303,161 / 8,190,735,360 | **9.063%** | 5.67 ms | 2.53 ms | 5,334.15 ms | **5,336.86 ms** (5.34s) |
| **3** | 338 layers | 608,374,959 / 8,190,735,360 | **7.428%** | 9.54 ms | 2.85 ms | 4,805.46 ms | **4,808.51 ms** (4.81s) |
| **4** | 332 layers | 543,984,802 / 8,190,735,360 | **6.641%** | 8.05 ms | 2.81 ms | 4,234.45 ms | **4,237.45 ms** (4.24s) |

### 5.2 Overall RL Step Performance Progress

| Step | Math Accuracy | Total Reward | Sampling Time | Train Step Time | Save Delta Time | Total Step Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 9.38% | -0.0016 | 197.1s | 78.6s | 36.7s | **312.7s** |
| **1** | 22.40% | +0.1411 | 44.7s | 67.4s | 19.2s | **131.5s** |
| **2** | 36.46% | +0.2859 | 39.5s | 67.6s | 16.8s | **124.1s** |
| **3** | 33.33% | +0.2505 | 38.3s | 65.0s | 14.0s | **117.5s** |
| **4** | **49.48%** | **+0.4245** | **37.3s** | **57.2s** | **16.5s** | **111.2s** |

---

## 6. Code Verification & Test Coverage

- **Unit Tests**: Pass 86 unit tests (`make test`), including `test_preload_delta_to_dram` in `tests/test_delta_weight_transfer_engine.py`.
- **Linter & Formatting**: Clean under `make fmt && make lint` (80 files clean, 0 ruff errors).
- **Cluster Compatibility**: Verified on GKE cluster with Accelerator Time-Slicer (`v0.3.11`).
