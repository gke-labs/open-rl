# Open-RL: High-Throughput RL Training Infrastructure

Open-RL implements a training API for fine tuning LLMs using reinforcement learning. The APIs are inspired by Tinker API and are built using a minimalist PyTorch and PEFT (Parameter-Efficient Fine-Tuning) stack.

## Architecture

The API backend consists of two primary layers designed to minimize VRAM footprint and handle high-throughput workloads:
1. **The Asynchronous Gateway (FastAPI)**: Handles incoming HTTP requests from the Tinker SDK client, issues immediate future tracking IDs, and pushes workloads to a central asynchronous queue.
2. **The Clock Cycle Engine (PyTorch/PEFT)**: A continuous background engine that drains the global request queue, batches operations by model tenant (`model_id`), manages PyTorch hardware resources lock-step, and executes actual tensor math.

![API Backend Architecture](design_arch.svg)

### Key Architectural Components

- **Asynchronous Request Queue & Polling**: To prevent concurrency failures and OOM errors when serving large LLMs synchronously, the server utilizes an `asyncio.Queue()`. HTTP handlers append a payload to the queue and instantly return a `req_id`. The client SDK leverages a `retrieve_future` polling mechanism to track the execution state.
- **Multi-Tenant LoRA Architecture**: The engine initializes and statically anchors exactly **one** Base Model in VRAM. When a client provisions a model, the engine injects a low-rank (e.g., Rank 16) adaptation layer (LoRA) mapped uniquely to that client's `model_id`. A thread-safe lock secures the base model during initialization.
- **The Clock Cycle Engine**: Operating as an infinite background loop, it rests until it detects queue items. It briefly sleeps upon waking to deliberately "pipeline" concurrent requests. Because it batches execution by `model_id`, it executes `set_active_adapter` only once per tenant batch, drastically cutting down on sluggish adapter switching overhead.
- **Stateful Tensor Workloads**: Math execution is strictly isolated by `model_id` to prevent gradient poisoning. Each tenant maintains its own isolated `torch.optim.AdamW` instance. Explicit float-handling prevents serialization collapses, and gradient clipping protects against gradient explosions.
- **Unified Inference & Training Sync**: By directing generation requests through the core clock cycle queue instead of immediately resolving them in HTTP handlers, the server systematically prevents race conditions where inference adapter hot-swapping might disrupt an in-flight backpropagation pass.

## The 4 Key Training Primitives

To train a model against the Open-RL backend, you utilize the 4 fundamental SDK primitives: Model Creation, Forward-Backward Pass, Optimizer Step, and Sampling. 

Below is a basic python training loop showcasing these 4 primitives using Supervised Fine-Tuning (SFT) as an example:

```python
import asyncio
import tinker
from tinker import types

async def training_loop():
    # Connect to the local server
    service_client = tinker.ServiceClient(base_url="http://localhost:8000")

    # -------------------------------------------------------------
    # Primitive 1: Create Model for Training
    # Dynamically injects a Rank 16 LoRA adapter isolated to your scope
    # -------------------------------------------------------------
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507", 
        rank=16
    )

    # ... generate datums (tokens, target_tokens, weights) ...

    for epoch in range(10):
        # -------------------------------------------------------------
        # Primitive 2: Forward-Backward Pass
        # Dispatches datums to the server. Computes cross-entropy loss, 
        # accumulates gradients, and returns log-probability metrics.
        # -------------------------------------------------------------
        fwdbwd_result = await training_client.forward_backward_async(
            datums, 
            loss_fn="cross_entropy"
        )
        
        loss_metrics = fwdbwd_result.loss_fn_outputs
        
        # -------------------------------------------------------------
        # Primitive 3: Optimizer Step
        # Instructs the server to apply gradients (AdamW) with clipping
        # -------------------------------------------------------------
        optim_result = await training_client.optim_step_async(
            types.AdamParams(learning_rate=5e-4)
        )
        
        print(f"Epoch {epoch+1} complete")

    # -------------------------------------------------------------
    # Primitive 4: Sample (and Save)
    # Extracts the newly formed LoRA adapter weights and initializes 
    # a dedicated Inference client for text generation tests.
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="my_model_v1"
    )
    
    response = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=[32, 54, 12, ...]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=20, temperature=0.7)
    ).result()
    
    # Process sequence arrays from response.sequences
    
asyncio.run(training_loop())
```

### Example: Reinforcement Learning (RLVR) Loop

In a Reinforcement Learning loop like GRPO, the same 4 primitives are arranged into an active generate-and-reward cycle:

```python
import asyncio
import tinker
from tinker import types

# Placeholder Environment & Reward Functions
def generate_math_problem() -> str: ...
def compute_advantages(rewards: list[float]) -> list[float]: ...
def parse_and_score_response(text: str) -> float: ...

async def rlvr_loop():
    service_client = tinker.ServiceClient(base_url="http://localhost:8000")

    # 1. Create Model
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507", rank=16
    )

    for epoch in range(10):
        # 2A. Extract sampling client from current weights
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"rlvr_epoch_{epoch}"
        )
        
        prompt_text = generate_math_problem()
        
        # 2B. Sample multiple rollouts (e.g. N=8) from the prompt
        response = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=[...]),
            num_samples=8,
            sampling_params=types.SamplingParams(max_tokens=100, temperature=0.9)
        ).result()
        
        # 3. Score the rollouts using the environment
        rewards = []
        for seq in response.sequences:
            text = decode(seq.tokens)
            rewards.append(parse_and_score_response(text))
            
        advantages = compute_advantages(rewards)
        
        # ... package sequences, text, and advantages into datums ...

        # 4. Forward-Backward Pass (Importance Sampling)
        # We pass the advantages to RL objective function
        await training_client.forward_backward_async(
            datums, 
            loss_fn="importance_sampling",
            loss_fn_config={"clip_range": 0.2} 
        )
        
        # 5. Optimizer Step
        await training_client.optim_step_async(types.AdamParams(learning_rate=1e-5))

asyncio.run(rlvr_loop())
```

## Running the Examples

Ensure `uv` and `uvicorn` are installed, then launch the server and clients concurrently using the provided Makefile:

```bash
# Start the Backend Server (Term 1)
make run-server

# Run a Multi-Tenant SFT Simulation (Term 2)
make run-client-sft ARGS="--parallel --epochs 5"

# Run a Reinforcement Learning loop via importance_sampling (Term 2)
make run-client-rlvr
```

# CLI Tool Usage

The project includes a CLI tool for inspecting and interacting with trained adapters.

### 1. List Available Adapters
View all fine-tuned sessions, including their aliases and creation timestamps.

```bash
make run-cli-list
```

**Output Example:**
```
ID                                       | ALIAS                          | CREATED
----------------------------------------------------------------------------------------------------
882faa32-3cc3-4dc6-9269-5e7c8aa7c01f     | rlvr_concise_capital           | 2026-02-15 03:27:24
1b705680-aee0-49f5-bd70-914f4260ab50     | rlvr_concise_answer            | 2026-02-15 03:27:11
```

### 2. Chat with an Adapter
Interactively test a specific adapter model.

```bash
make run-cli-chat MODEL=<model_id>
```

**Optional: System Prompt Override**
```bash
make run-cli-chat MODEL=<model_id> PROMPT="You are a pirate."
```

### 3. Generic Usage
For other commands or arguments not covered by shortcuts:

```bash
make run-cli list
make run-cli chat --model <id> --temperature 0.9
```
