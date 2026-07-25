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
- **Improvement:** Equip the worker provisioner with automated fault tolerance against spot preemption by either deploying worker workloads as lightweight Kubernetes `Deployment` or `ReplicaSet` controllers (with `replicas: 1`), or by implementing active pod lifecycle monitoring and automated recreation within `KubernetesFFTWorkerManager`. Upon spot preemption, Kubernetes or the provisioner will automatically schedule a replacement worker pod onto a fresh spot node. When the replacement pod boots, it will automatically resume from the latest saved model checkpoint on shared persistent storage (`/mnt/shared/open-rl/...`), ensuring uninterrupted distributed fine-tuning.

## 5. HuggingFace Hub Token (`HF_TOKEN`) Injection via Kubernetes Secret / ConfigMap
- **Current Behavior:** Dynamically spawned `Trainer` and `Sampler` worker containers run without authentication (`user_id=public`), causing strict HuggingFace Hub CDN throttling (`403 Forbidden` / lengthy retry loops on `xet_client` during first-time model caching) and failing on gated models (`e.g., Gemma 2, Gemma 4, Llama 3`).
- **Improvement:** Update `k8s_worker_manager.py` (`render_pod()`) and static Kubernetes manifests to inject an optional `open-rl-hf-secret` (`or ConfigMap via envFrom / set_env`) into all worker containers, ensuring authenticated, high-bandwidth model downloads without rate-limiting across multi-pod cluster setups.

## 6. Pin `uv` Image to Specific SHA Digest in Dockerfiles
- **Current Behavior:** `src/server/Dockerfile` and `src/server/Dockerfile.gateway` copy the `uv` binary from `ghcr.io/astral-sh/uv:latest`. Because the `:latest` tag is mutable and updated frequently upstream, any new `uv` release invalidates the Docker BuildKit cache at that layer, forcing a complete rebuild from scratch (e.g. re-downloading Python, dependency trees, and recompiling custom vLLM C++ extensions).
- **Improvement:** Pin the `uv` image to a specific static version or SHA digest (e.g. `ghcr.io/astral-sh/uv:0.2.27` or via `@sha256:...`) across all `Dockerfile`s to guarantee reproducible, blazing-fast cached builds unless dependency files (`pyproject.toml`, `uv.lock`) actually change.
