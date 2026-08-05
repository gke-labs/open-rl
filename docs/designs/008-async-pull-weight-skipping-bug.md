# Design Doc 008: Prefetch Removal & Session-ID Binding for Delta Weight Transfer Engine

**Author:** Open-RL Engineering Team & Jetski Pair Pilot  
**Status:** Architecture Finalized / Ready for Implementation  
**Target Component:** vLLM Pull-Based Sparse Delta Weight Transfer Engine  
**Target Branch:** `fft-fixes`  

---

## 1. Executive Summary

During concurrent 30-step Reinforcement Learning (RL) campaigns, we identified a critical architectural issue causing the vLLM Sampler to evaluate stale weights, leading to skipped intermediate step increments. Initially diagnosed as an asynchronous race condition in the Time-Slicer VRAM Sleep/Wake loop, a deep dive into the local codebase revealed exactly two core root causes:

1. **The Prefetch Pipeline Flaw**: The background `weight_prefetcher_loop` introduced massive thread-safety, Pub/Sub listener queueing, and staging corner cases, while the Trainer was actually missing direct async step-level notification signals entirely.
2. **Session-ID Target Path Mismatch**: The Gateway's sampling endpoint blind-constructed the request `weights_path` as exactly the base session directory `sampler_full/{model_id}`, completely omitting the step-specific `sampler_weights/sampler-X` suffix. This caused a total mismatch, failing the reload check and resulting in silent weight skipping.

To eliminate every single corner case, session-id tracking complexity, and stale weight evaluations, we are **completely removing the Prefetching pipeline** and enforcing strict **Session-ID to Target Path Matching** in the synchronous inference execution path.

---

## 2. Root Cause Analysis

### 2.1 The Prefetch Dead-End & Missing Notifications
Under the original Design Doc 006 design, the Sampler spun up a background `weight_prefetcher_loop` listening to the `open_rl:weight_update:{model_id}` Redis pub/sub channel. 
However, in continuous intermediate RL steps, the Trainer writes its sparse deltas to disk but **never actually publishes directly to that channel**. The only component doing so was the Client-triggered `save_weights_for_sampler` API. Because no notifications arrived during optimization loops, intermediate background staging never triggered, leaving the engine to fetch deltas synchronously or fallback to skipped states entirely. Furthermore, relying on background DRAM preloading while the vLLM engine is asleep or busy serving parallel requests introduced severe thread-safety and pipeline stalling bugs.

### 2.2 The Session-ID / Target Path Mismatch
When the Client creates a sampling client and sends generational inference batches to the Gateway, the `/api/v1/asample` endpoint constructs the request Queue payload. Currently, for Full Fine-Tuning, it resolves exactly:
```python
rel_path = model_id[len("tinker://") :] if model_id.startswith("tinker://") else model_id.lstrip("/")
local_path = os.path.join(TMP_DIR, "sampler_full", rel_path)
weights_path = local_path
```
Because the Trainer strictly increments and writes to step-specific NFS subdirectories (`.../sampler_weights/sampler-1`, `sampler-2`, etc.), the Request's `weights_path` failed the strict equality check in `vllm_sampler.py`. The Sampler therefore missed the delta updates, resulting in the NO-OP execution and skipped intermediate accuracy progression tracking.

---

## 3. The New Architecture: Prefetch Removal & Synchronous Enforcement

Following direct specification and architectural alignment, we are entirely abandoning background prefetching and sliding-window event queues in favor of strict, sequential, synchronous correctness.

### 3.1 Total Removal of Prefetching
We will completely eliminate the prefetching pipeline to guarantee that no asynchronous race conditions can exist between rollout execution and weight updates:
- **Delete Daemon**: Remove `weight_prefetcher_loop` and the Pub/Sub listener entirely from `vllm_sampler.py`.
- **Simplify Engine Structure**: Delete `preload_delta_to_dram` and the intermediate `StagedDeltaSnapshot` staging structures from `delta_weight_transfer_engine.py`.
- **Pure Synchronous Mainline Path**: All Host-to-Device GPU mutation logic will execute strictly within exactly the engine's main execution flow when a batch pops from Redis, relying fully on the already brilliant synchronous in-place patching logic natively present in `_apply_gpu_in_place()`.

### 3.2 Strict Session-ID to Target Path Matching
We will guarantee sequential evaluation integrity by ensuring the Request's target path binds strictly and flawlessly to exactly the intended intermediate step delta:
- **Path Resolution Rewrite**: Refactor `gateway.py` to ensure that generation requests mapping to Full Fine-Tuning sampling sessions carry exactly the fully-qualified, step-specific target path (`sampler_weights/sampler-X`).
- **Synchronous Sequence Barrier**: In `process_sampling_request()`, exactly when a request pops from Redis, the engine acquires the strict `reload_lock`. If the fully-mapped `weights_path` differs from the currently loaded state, it sequentially executes `receive_weights()`, directly reading the sparse delta from NFS, constructing the bulk pinned tensors, and applying the exact GPU VRAM mutation strictly before beginning the generation batch. 

---

## 4. Impact & Performance Assessment

By removing the background prefetch staging, reading from NFS will now occur synchronously in the request processing path. For large 8B models with sparse deltas, doing so adds exactly 10-15 seconds of Host memory construction right before inference begins. However, given that RL rollout generational generation itself commands ~30+ seconds of strict GPU compute, appending a highly deterministic ~10s synchronous patch load is fully acceptable. It completely eradicates pipeline stalling, guarantees absolute mathematical convergence validity, and delivers flawless architectural simplicity.

---

## 5. Next Steps / Implementation Patch
1. **Remove Prefetch API & Flag**: Completely deprecate the `enable_prefetching` field from `WeightSyncConfig`, removing environment variable parsing and HTTP header propagation across exactly the Server, API, and Client Side (`tinker_utils.py`).
2. **Apply local diffs** to `src/server/vllm_sampler.py` (Prefetch removal & `asample` lock simplification).
3. **Apply local diffs** to `src/server/delta_weight_transfer_engine.py` (Staged snapshot cleanup).
4. **Refactor `gateway.py`** to enforce Session-ID strict matching for step-specific delta mapping.
