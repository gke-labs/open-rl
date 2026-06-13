# Full Fine-Tuning (FFT) Dynamic Sampler & Weight Swapping

This document describes the design, architecture, and implementation of the dynamic weight-reloading sampler loop for Full Fine-Tuning (FFT) Reinforcement Learning (RL) in the Open-RL framework.

---

## 1. Context & Key Challenge
In Reinforcement Learning, the training loop alternates continuously between:
1. **Sampling**: Generating completions (rollouts) using the current model weights.
2. **Training**: Updating the model weights using the collected rollouts.

For **LoRA**, vLLM natively supports dynamic adapter loading at runtime (via Multi-LoRA). 
For **Full Fine-Tuning (FFT)**, the entire base model weights are modified. vLLM does not natively support changing the base model architecture/weights dynamically during active inference. 

To run both PyTorch FSDP training and high-throughput vLLM inference on resource-constrained hardware (e.g. sharing a single GPU, or running on separate GPUs), we utilize **vLLM Sleep Mode** (Level 2 Sleep) and **Redis Queue-Based Pull Sampling** to perform fast, in-place base weights reloading without restarting the sampler engine.

---

## 2. Architecture: Queue-Based Pull Model

Open-RL uses a **decoupled, queue-based pull architecture** to coordinate gateway requests, PyTorch training, and vLLM sampling. 

### Sequence Diagram:
```mermaid
sequenceDiagram
    autonumber
    participant Client as Tinker Client
    participant GW as Gateway Server
    participant Redis as Redis Queue & Futures
    participant TS as Trainer Process (PyTorch)
    participant VS as Sampler Process (vLLM)

    Note over Client, VS: 1. Initialization (Dynamic Worker Launch)
    Client->>GW: create_model(model_id)
    GW->>Redis: enqueues launch request
    Note over GW: Worker Launch Processor drains launch queue
    GW->>TS: Dynamically Spawns training request processor (--model-id)
    GW->>VS: Dynamically Spawns vLLM sampler worker (--model-id)
    VS->>VS: Initializes vLLM Engine (Sleep Mode enabled)
    VS->>Redis: Set open_rl:sampler_ready:{model_id} = 1

    Note over Client, VS: 2. Iterative RL Training / Rollout Loop
    Client->>GW: create_sampling_session(model_path)
    Note over GW: Blocks until open_rl:sampler_ready key is 1
    GW-->>Client: Returns session ID

    Client->>GW: sample_async(prompt, model_id)
    GW->>Redis: Pushes request (prompt, weights_path) to sampler_queue:<model_id>
    
    Note over VS: Sampler pops request from Redis
    alt weights_path != current_loaded_weights
        VS->>VS: await engine.sleep(level=2) (frees VRAM)
        VS->>VS: await engine.wake_up(tags=["weights"])
        VS->>VS: await engine.collective_rpc("reload_weights", weights_path)
        VS->>VS: await engine.wake_up(tags=["kv_cache"])
    end
    VS->>VS: await engine.generate(prompts)
    VS->>Redis: Resolves future with completions
    GW-->>Client: Returns completions
```

---

## 3. Key Components

### 1. Redis Request Store (`src/server/store.py`)
Provides list-based queueing for sampling requests to decouple the API gateway from sampler processes:
- `put_sampling_request(req_data)`: Pushes a serialization of sampling prompts, constraints, and target `weights_path` onto `open_rl:sampler_queue:<model_id>`.
- `get_sampling_requests_for_model(model_id)`: Drains the queue in batches and passes them to the sampler worker.

### 2. API Gateway (`src/server/gateway.py`)
- `/api/v1/create_sampling_session`: Resolves the active model ID and blocks requests until the dynamically spawned sampler worker registers itself as ready in Redis.
- `/api/v1/asample`: Resolves Tinker sequence/session IDs (e.g. `tinker://model-id/sampler_weights/sampler-seq`) to absolute local directories under `/tmp/open-rl/sampler_full/`. It packages the target directory as `weights_path` and enqueues the request to Redis.

### 3. Worker Launcher & Compatibility (`src/server/worker_launch_processor.py` & `scripts/run_training_e2e.py`)
- **FFT Mode**: Trainer and sampler processes are launched dynamically when a new model is initialized (`create_model`):
  - Spawns Trainer: `python -m server.training_requests_processor --model-id <model_id>`
  - Spawns Sampler: `python -m server.vllm_sampler --model-id <model_id>` (overriding `CUDA_VISIBLE_DEVICES` using `SAMPLER_CUDA_VISIBLE_DEVICES`).
- **LoRA Mode**: The sampler worker is launched statically on startup with `--model-id <base_model_name>` and drains the corresponding queue directly.
- **Readiness Checks**: The launcher uses a raw socket Redis client wrapper (`redis_key_ready`) to verify when a statically launched sampler has completed startup/compilation.

### 4. Dynamic Sampler Worker (`src/server/vllm_sampler.py`)
Runs a headless pull-mode loop:
- Initializes the AsyncLLMEngine.
- Blocks until requests are pushed to `open_rl:sampler_queue:<model_id>`.
- Drains the queue in batches and executes them concurrently using `asyncio.gather(*tasks)`. This allows vLLM to internally batch concurrent rollout requests for maximum hardware utilization.
- Uses a local `reload_lock = asyncio.Lock()` to handle weights reloading safely:
  - If multiple concurrent requests are popped, they check if `weights_path` matches the loaded model.
  - The first task to acquire the lock will perform the sleep-wake-reload loop if a weights change is detected:
    1. Calls `engine.sleep(level=2)` to discard active weights and KV caches, releasing ~85% of GPU memory.
    2. Calls `engine.wake_up(tags=["weights"])` to allocate memory for the new checkpoint.
    3. Calls `engine.collective_rpc("reload_weights")` to load safetensors in-place.
    4. Calls `engine.wake_up(tags=["kv_cache"])` to initialize a clean KV cache pool.
  - Subsequent concurrent tasks with matching paths bypass reloading immediately once the lock is released.
- Feeds token ids into `engine.generate()` and pushes completions to the future's Redis channel.

---

## 4. Key Performance Characteristics

Measurements captured during end-to-end training runs using `Qwen2.5-0.5B` on NVIDIA L4 GPUs:
- **Sleep Memory Release**: Freeing `27+ GiB` of VRAM is nearly instantaneous.
- **Weights Allocation**: `~0.05 seconds`.
- **In-place weights reload (disk to VRAM)**: `~0.80 seconds` for a `0.92 GiB` model checkpoint.
- **KV Cache Allocation**: `~0.67 seconds`.
- **Total Weights Swap Latency**: **`~1.5 seconds`** in the active RL training loop.

---

## 5. Sharing vs. Dedicated Sampler Workers (Multi-Job Design)

When running multiple concurrent RL jobs ($N > 2$) sharing a GPU or node resources, two deployment configurations can be selected based on model sizes:

### Option A: Single Shared vLLM Instance
A single `vllm-worker` process runs on the GPU. It pulls requests sequentially from Redis and checks `weights_path`. If Job A and Job B both submit requests, it swaps weights back and forth:
- **Pros**:
  - **Memory (RAM) Efficiency**: The model weights are loaded into CPU memory only once. Excellent for large models (13B+) where running multiple instances would exhaust host RAM.
- **Cons**:
  - **I/O Latency**: Every swap requires reading safetensors from the network filesystem and compiling. Swapping takes **`~1.5 seconds`** on every step.

### Option B: Dedicated vLLM Instance Per Job
Each job launches its own dedicated `vllm_sampler` process (each listening to its own `sampler_queue:<model_id>`). When Job A is active, Job B's sampler process is put to Sleep Level 2.
- **Pros**:
  - **Fast Wake Up**: In sleep mode, weights remain in host CPU memory. Waking up only requires copying weights from CPU memory to GPU VRAM (no disk I/O), taking only **`~0.7 seconds`** (twice as fast as Option A).
- **Cons**:
  - **RAM Overhead**: Each inactive instance holds a full copy of the model weights in CPU RAM. High risk of system OOMs on large models.

### Recommendation Grid:
- **Small/Medium Models (up to ~8B)**: Use **Option B (Dedicated Instances)** to gain a 2x latency benefit during step switches.
- **Large Models (13B+)**: Use **Option A (Shared Instance)** to maintain host memory stability.
- **Heterogeneous Architectures (e.g. Qwen + Gemma)**: **Must** use **Option B** (separate processes), as a single instance cannot reload between different configuration shapes.

---

## 6. Out-of-Order / Parallel Sampling of Different Weight Versions

If a client (or multiple clients sharing an instance) submits concurrent sampling requests for **different** weights versions (e.g. Request A for `sampler-1` and Request B for `sampler-2` simultaneously):

1. **Serialized Swapping (Thrashing)**: If they are processed sequentially, the worker will successfully process both but will thrash back and forth, triggering a full `sleep -> wake -> reload -> wake` cycle on every step transition, which adds `~1.5 seconds` of latency overhead per swap.
2. **Cancellation on Active Generations**: If Request B (triggering a reload to `sampler-2`) starts executing *concurrently* (via `asyncio.gather`) while Request A (using `sampler-1`) is still generating tokens inside vLLM:
   - Request B will acquire the reload lock and call `engine.sleep(level=2)`.
   - vLLM's `sleep` call immediately cancels all active generations and frees memory.
   - Request A's running generation will be interrupted and fail with a `RequestFailedResponse` (e.g., EngineDeadError or cancellation exception).
   - Request B's generation will succeed.
   
*Note: In normal GRPO/PPO RL training, the training loop is strictly synchronous (all rollouts for the current policy weights are collected and completed before the trainer updates weights for the next step). Therefore, parallel execution of different weight versions does not occur within a single job context. For multi-job contexts, **Option B (Dedicated Instances)** should be used to isolate environments.*

---

## 7. Configuration & Environment Variables

To configure and run disaggregated GPU training/sampling:

| Environment Variable | Description |
| :--- | :--- |
| `OPEN_RL_ENABLE_FFT=true` | Configures the framework, gateway, and vLLM sampler to run in Full Fine-Tuning mode. |
| `SAMPLING_BACKEND=vllm` | Sets the sampling engine to vLLM (defaults to PyTorch/Torch if unset). |
| `CUDA_VISIBLE_DEVICES=0` | Sets the GPU visibility for the PyTorch trainer (bound to the gateway process). |
| `SAMPLER_CUDA_VISIBLE_DEVICES=1` | Sets the GPU visibility for the dynamic vLLM sampler subprocess. |
| `VLLM_GPU_MEMORY_UTILIZATION=0.70` | Allocates VRAM bounds for the sampler engine. |

### Running the End-to-End Suite:
```bash
make test e2e tiny-fft-rl TRAINING_TEST_ARGS="sampling_backend=vllm trainer_gpu=0 sampler_gpu=1 steps=10"
```
