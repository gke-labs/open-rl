# Open-RL TODOs & Future Improvements

## 1. Do Not Cache Worker Pod Templates in Memory
- **Current Behavior:** `KubernetesFFTWorkerManager.__init__` reads and parses pod templates (`/etc/open-rl/trainer/trainer-worker-pod.yaml`) once at startup and stores them in memory (`self.trainer_template`).
- **Improvement:** Read and parse the template file dynamically from disk on every `render_pod()` invocation so live ConfigMap updates take effect immediately without requiring a rolling restart of the gateway deployment (`kubectl rollout restart deployment open-rl-gateway`).

## 2. Implement Reliable Queue Acknowledgment for Training Steps
- **Current Behavior:** The trainer requests processor pops training request items immediately from Redis upon picking up a step.
- **Improvement:** Ensure that during a trainer worker crash or OOM kill, the queue item is dequeued / acknowledged only **after** the full completion of the training step (specifically, post saving the updated model checkpoint weights back to shared persistent storage). If interrupted earlier, the item should remain unacknowledged so another pod instance can safely retry the training step.

## 3. [COMPLETED] High-Throughput Batched Async DMA PCIe Offloading (3-Phase Architecture)
- **Status:** Implemented and verified in `src/training/fft_trainer_worker.py` (`FFTTrainingWorker.sleep()` and `wake_up()`). Added `self._opt_shadow` persistent dictionary to store pinned host buffers for AdamW optimizer states across training steps.
- **Current Behavior:** `FFTTrainingWorker.sleep()` iterates serially over model parameters, gradients, and AdamW optimizer states using `param.data.to("cpu", non_blocking=False).pin_memory()` followed immediately by `param.data = torch.empty(0, ...)`. This synchronous loop blocks the CPU per-tensor and allocates OS page-locked memory on the fly, yielding ~1.34 GB/s on H100 PCIe Gen5 (~11.9s to offload ~16 GiB total). In contrast, reloading (`wake_up()`) from pinned memory over async PCIe DMA achieves ~14 GB/s (~1.15s).
- **Improvement:** Eliminate the CPU offload bottleneck and prevent GPU memory corruption race conditions by restructuring `sleep()` into three distinct phases:
  1. **Phase 1 (Launch Batched Async DMA):** Keep persistent pinned host CPU buffers (`_param_shadow`, `_grad_shadow`, `_opt_shadow`) alive across training steps so `.pin_memory()` is never called after Step 1. Launch 100% asynchronous DMA transfers using `cpu_buf.copy_(gpu_tensor, non_blocking=True)` across all tensors *without* deleting or modifying GPU VRAM buffers.
  2. **Phase 2 (Single Barrier Synchronization):** Execute a single barrier (`torch.cuda.synchronize()`) at the end of the loop to wait once for all 16 GiB of concurrent DMA transfers to complete.
  3. **Phase 3 (Safe VRAM Deallocation):** Only after `synchronize()` returns and verifies that every byte is safely residing in host RAM, deallocate GPU tensors (`param.data = torch.empty(0, ...)`) and release CUDA allocator cache. This targets >14 GB/s and ~1-second CPU offloads without race conditions.

## 4. Automated Spot VM Preemption Resiliency in Worker Provisioner
- **Current Behavior:** `KubernetesFFTWorkerManager` launches trainer and sampler workloads as bare Kubernetes `Pod` objects (`open-rl-trainer-...` and `open-rl-sampler-...`). When GCP Spot VM preemption reclaims an underlying GPU spot node, Kubernetes terminates the bare worker pod without respawning a replacement, causing the distributed E2E training job to hang indefinitely waiting for rollouts or gradient updates.
- **Improvement:** Equip the worker provisioner with automated fault tolerance against spot preemption by either deploying worker workloads as lightweight Kubernetes `Deployment` or `ReplicaSet` controllers (with `replicas: 1`), or by implementing active pod lifecycle monitoring and automated recreation within `KubernetesFFTWorkerManager`. Upon spot preemption, Kubernetes or the provisioner will automatically schedule a replacement worker pod onto a fresh spot node. When the replacement pod boots, it will re-establish Redis tenant queue connections, re-acquire its time-slice registration lock, and seamlessly resume training directly from the latest serialized safetensors checkpoint preserved on shared persistent storage (`open-rl-shared-pvc` on 10 TiB Parallel Filestore).

## 5. [COMPLETED] Proactive Checkpoint Staging & Atomic Rename in Request Processor
- **Improvement:** In `optim_step`, immediately after the AdamW parameter update finishes—while the worker still holds the GPU time-slice lock and the updated weights are live in CUDA VRAM—the checkpoint is proactively serialized to a canonical staging folder on our shared 10 TiB filesystem (`/mnt/shared/open-rl/sampler_full/<model_id>_staging`). When the client subsequently calls `save_weights_for_sampler`, the processor executes an atomic OS rename (`os.rename(<staging_dir>, <requested_path>)`) in less than 1 millisecond. This completely eliminates redundant GPU memory reloading and snapshot agent context switching without requiring client-side request pipelining.

## 6. Leverage Existing Tiered Memory & Storage Offloading Abstractions
- **Current Behavior:** Open-RL currently implements custom, manual memory hierarchy management across GPU VRAM, pinned CPU DRAM, and persistent storage:
  - In `fft_trainer_worker.py`, we manually iterate across model parameter dictionaries and AdamW momentum states (`exp_avg`, `exp_avg_sq`), moving them back and forth between VRAM and host RAM (`v.to("cpu").pin_memory()`) during cooperative `sleep()` and `wake_up()` handoffs.
  - Checkpoint serialization is implemented as custom file I/O loops reading directly from host DRAM (`_param_shadow`) and serializing `.safetensors` shards to NFS.
  - The vLLM Sampler worker relies on custom `sleep(level=2)` commands to discard prefix caches and back up model parameters.
- **Improvement:** Adopt production-grade tiered storage libraries and abstractions to eliminate custom boilerplate, improve multi-threaded async I/O performance, and support cloud storage backends out-of-the-box:
  1. **PyTorch FSDP & Distributed Checkpoint (`DCP`):**
     - *AsyncCheckpointer:* Replace our custom `_param_shadow` disk serialization loop with native `torch.distributed.checkpoint.async_save`. DCP provides multi-threaded storage staging, zero-copy pinning, and backend-agnostic abstraction supporting NFS, AWS S3, and GCS out of the box without blocking PyTorch compute threads.
     - *FSDP Native CPUOffload:* Instead of manually iterating across parameter momentum dictionaries in `sleep()` and `wake_up()`, leverage FSDP's native `CPUOffload(offload_params=False, offload_optimizer_states=True)`. This delegates momentum state residence to PyTorch's internal CUDA allocator, enabling async stream transfers without custom application-level loop iteration.
  2. **vLLM / BlockSpaceManager Swapping Engine:**
     - Tap into vLLM's existing C++/CUDA paging engine (`BlockSpaceManager`), which currently manages KV cache swapping (`swap_out` / `swap_in`). Unify model parameter offloading with this engine so that when the time-slicer sends `RELEASE`, vLLM executes a native asynchronous CUDA stream swap of model weights into its pinned host memory pool, avoiding full disk reloads on wakeup.
  3. **TensorStore / DeepSpeed ZeRO Abstractions:**
     - Evaluate TensorStore for transactional, multi-dimensional array streaming across network filesystems, or adopt ZeRO-Infinity style async NVMe memory-mapping (`aio`) if workloads scale beyond host DRAM capacity.
