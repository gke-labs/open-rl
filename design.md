# Kube-RL Server MVP: System Architecture & Design

This document summarizes the final architecture of the Kube-RL API backend after refactoring it to a multi-tenant, batched "Clock Cycle" engine. The server is designed to emulate the behavior of high-throughput RL infrastructure while minimizing VRAM footprint via LoRA hot-swapping.

## High-Level Architecture

The API backend consists of two primary layers:
1. **The Asynchronous Gateway (FastAPI)**: Handles incoming HTTP requests from the Tinker SDK client, issues immediate future tracking IDs, and pushes workloads to a central asynchronous queue.
2. **The Clock Cycle Engine (PyTorch/PEFT)**: A continuous background engine that drains the global request queue, batches operations by model tenant (`model_id`), manages PyTorch hardware resources lock-step, and executes actual tensor math.

![API Backend Architecture](design_arch.svg)

## Key Components

### 1. Asynchronous Request Queue & Polling
- **Problem**: Serving large LLMs synchronously via REST API (`asyncio.to_thread` directly inside HTTP handlers) causes catastrophic concurrency failures, OOM errors, and race conditions when multiple users hit endpoints simultaneously.
- **Solution**: The server utilizes an `asyncio.Queue()`. HTTP handlers simply append a payload to the queue and instantly return a `req_id`.
- **Latency & Polling**: The client SDK leverages a `retrieve_future` polling mechanism. If the server has not completed processing the `req_id` (via the background engine), it returns a `{status: "pending"}` structure that natively triggers the client to `try_again`.

### 2. Multi-Tenant LoRA Architecture
- **Problem**: Loading a multi-billion parameter base model (e.g., Qwen 3) consumes ~10-20GB+ of VRAM. Hosting multiple specialized models concurrently is impossible on a standard GPU.
- **Solution (`peft`)**: The `TrainerEngine` initializes and statically anchors exactly **one** Base Model in VRAM. When a client calls `/api/v1/create_model`, the engine downloads an initial low-rank (e.g., Rank 16) adaptation layer (LoRA), mapping it uniquely to that client's `model_id` via `model.add_adapter()`.
- **GPU Residency & Hot-Swapping**: All adapter weights (typically 10-50MB) reside on the GPU in VRAM at all times alongside the massive base model. When switching between tenants, `engine.set_active_adapter()` does *not* move weights across the PCIe bus; it merely flips a logical pointer within PyTorch to route math through that specific tenant's dictionary of LoRA matrices.
- **Thread-safe Initialization**: A `threading.Lock()` securely forces parallel clients to wait while the Base Model physically loads into GPU memory. Subsequent simultaneous client joiners acquire the lock, recognize the base model is warm, and inject only their tiny LoRA layers.

### 3. The Clock Cycle Engine
- The engine operates an infinite `while True` loop (`clock_cycle_loop`) deployed as a background task. 
- It rests until it detects at least one item in the queue. Upon waking, it briefly sleeps (`0.05s`) to deliberately "pipeline" and vacuum up concurrently arriving network requests.
- **Batched Execution**: It separates mixed incoming network requests by their originating `model_id`.
- **Single-Worker Race Condition Prevention**: Because there is only one worker pulling from the queue (the `clock_cycle_loop`), execution is perfectly sequential. This enforces a strict, isolated hardware timeline: `set_active_adapter` is invoked, and then `model.forward()`, `loss.backward()`, and `optimizer.step()` are executed atomically. If multiple workers were used, Tenant B could swap the active adapter in the middle of Tenant A's backward pass, poisoning the gradients.
- **Hardware Hot-Swapping Overhead**: It executes sequentially over each tenant group. Because it only executes `engine.set_active_adapter(model_id)` once per tenant batch, it drastically cuts down on the sluggish `peft` adapter switching overhead that occurs when interleaving single math operations.

### 4. Stateful Tensor Workloads
The `TrainerEngine` isolates math execution strictly by `model_id` to prevent gradient poisoning:
- **`model.train()` Semantics**: Rather than triggering a training loop, `.train()` merely flips the PyTorch execution graph into a training state, ensuring that dropout layers activate and gradient history is actively tracked in memory during forward passes.
- **`optimizers` dict**: Each tenant maintains its very own `torch.optim.AdamW` instance stored in a dictionary. Because optimizers like AdamW have "momentum" (remembering statistics about previous gradients), sharing an optimizer would cause Tenant A to mathematically poison Tenant B. The dictionary ensures Tenant A's math only updates Tenant A's adapter.
- **Sanitization & Clamping**: Explicit float-handling prevents `NaN` or `Infinity` from collapsing the JSON serialization. Variables like `.gather()` lengths, `-inf` logprobs, and `torch.exp()` overflow differences are precisely clamped to preserve mathematical stability.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_` explicitly checks for gradient explosions (`grad_norm=NaN`) before triggering `optimizer.step()`.

### 5. Unified Inference & Training sync
- By directing `/api/v1/asample` generation requests *through* the core clock cycle queue instead of resolving them immediately in the HTTP handlers, the server completely side-steps race conditions where inference adapter hot-swapping could occur in the middle of an in-flight backpropagation pass.

## v2: Multi-GPU Architecture (Proposed)

While the single-GPU MVP successfully emulates high-throughput production systems, Reinforcement Learning (e.g., GRPO) remains bound by generation speed. To scale throughput, we must physically separate training and inference across multiple GPUs using a dedicated inference engine like vLLM.

### 1. Split-Service Architecture
The API Gateway will act as a unified router:
- **GPU 0 (Training)**: Runs the existing PyTorch Clock Cycle Engine. Continues to handle `forward_backward` and `optim_step` batching.
- **GPU 1 (Inference)**: Runs a dedicated `vLLM` AsyncLLMEngine instance with `enable_lora=True` to handle high-speed `/api/v1/asample` generation requests.

### 2. LoRA Weight Synchronization
Separating training and inference introduces the challenge of weight synchronization. In an RL loop, the optimizer updates the adapter on GPU 0. Before the next epoch, GPU 1 must receive these updated weights to generate the next batch of rollouts.
- **Disk-Based Sync**: When `save_weights_and_get_sampling_client` is called, the PyTorch engine flushes the updated PEFT adapter to a shared disk location (e.g., `/tmp/kube-rl/peft/tenant_id/epoch`). 
- **Dynamic Loading**: The API Gateway then instructs the vLLM engine to dynamically load this new adapter path via vLLM's `LoRARequest`.

### 3. Queue Management & Sync Barriers
Uncoupling inference from the central Clock Cycle queue re-introduces race conditions. A generation request could hit the Inference GPU *before* the synchronization step finishes updating vLLM's LoRA adapters.
- **The Solution (Sync Barrier)**: The gateway will maintain state locks per `model_id`. When a client triggers an adapter save, the gateway locks that tenant's inference endpoint. Generation requests for that tenant remain queued or pending until the new LoRA weights are fully committed to disk and successfully loaded by the vLLM engine.
