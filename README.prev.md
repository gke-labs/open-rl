# Open-RL: High-Throughput RL Training Infrastructure

Open-RL implements a training API for fine tuning LLMs using reinforcement learning. The APIs are inspired by Tinker API and are built using a minimalist PyTorch and PEFT (Parameter-Efficient Fine-Tuning) stack.

## Architecture

Open-RL is designed as a federated, multi-accelerator architecture that physically separates the PyTorch training loop from the vLLM inference engine. This decouples reinforcement learning generation speed from backpropagation, allowing high-throughput deployments without creating CPU bottlenecks or blocking the Python GIL.

![Distributed Architecture Flow](distributed_arch.svg)

### Key Architectural Components

- **The Asynchronous Gateway & Clock Cycle Engine (Training GPU)**: The HTTP Gateway handles requests from the client SDK (via long-polling), issuing future tracking IDs and pushing workloads to an async queue (or Redis). Concurrently, the Clock Cycle Engine (PyTorch) continuously drains this queue, batches operations by model tenant (`model_id`) to minimize sluggish adapter switching, manages hardware resources lock-step, and executes actual tensor math.
- **The vLLM Inference Worker (Inference GPU+)**: A dedicated inference worker that continuously hot-swaps LoRA weights and handles all high-speed generation requests concurrently. This engine permanently hosts the identical base model as the training worker.
- **LoRA Weight Synchronization**: To avoid slow PCIe/NVLink data movement via NCCL into vLLM's custom backend, the PyTorch engine saves the trained LoRA adapters (`.safetensors`) directly to a shared parallel file system (e.g. Google Cloud Managed Lustre). The vLLM worker receives the file path over HTTP and streams the weights directly into its PagedAttention VRAM blocks in milliseconds.
- **Multi-Tenant LoRA Architecture**: The engines initialize and statically anchor exactly **one** massive Base Model in VRAM. Clients uniquely inject low-rank (e.g., Rank 16) adaptation layers mapped to their `model_id`, vastly reducing memory overhead and allowing dozens of tenants to train simultaneously on the same GPU.
- **Stateful Tensor Workloads & Math Safety**: Execution is strictly isolated by `model_id` to prevent gradient poisoning. Each tenant maintains its own completely isolated `torch.optim.AdamW` instance to prevent cross-contamination of momentum. Explicit bounds checking and gradient clipping protect against explosion.
- **TITO (Tokens-In, Tokens-Out) Generation**: All inference operates natively on integer arrays. The API gateway routes `prompt_token_ids` to vLLM and receives generated `token_ids` back, ensuring perfect mathematical alignment with PyTorch for loss computation and preventing tokenization whitespace drift.
- **Sync Barriers**: When a client requests an adapter save, the Gateway enforces a state lock for that tenant. Generation requests remain queued until the new LoRA weights are fully committed to disk and successfully hot-swapped by the vLLM engine, preventing race conditions.

## Model Configuration & Critical Warnings

> [!IMPORTANT]
> **Synchronized Model Architecture Required**
> The server's inference engine (vLLM) is pre-loaded with a specific base model at startup (defined by `VLLM_MODEL`). When a client requests training via `create_model(base_model=...)`, the PyTorch trainer will load that specific model.
>
> **You MUST ensure `VLLM_MODEL` matches the `base_model` requested by the client.**
> - If they differ, the server will crash or error out when attempting to apply LoRA adapters (trained on the client's base model) to the vLLM engine's disparate base model.
> - **Local Development**: The `Makefile` defaults `VLLM_MODEL` to `Qwen/Qwen2.5-0.5B` for speed.
> - **Remote/Production**: You must explicitly set `VLLM_MODEL` in the environment or via `make run-server VLLM_MODEL=...` to match your intended training target (e.g. `Qwen/Qwen3-4B-Instruct-2507`).

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

## Kubernetes Deployment (GKE)

The open-rl server is designed to run on a Kubernetes cluster with NVIDIA GPUs (tested on 2x L4 GPUs). The deployment uses a remote VM as a Docker builder to speed up container construction.

### 1. Configure the Remote Builder
Because building environments with large tensor libraries is resource-intensive, we use a remote VM (`HOST=b3`) for compiling the Docker image.

```bash
# Set up Docker and GCR authentication on the remote builder VM
make remote-build-setup HOST=b3
```

### 2. Build and Push the Image
The image is built remotely using Docker BuildKit and pushed to Google Container Registry (GCR).

```bash
# Sync local code, build the image on the remote VM, and push it
make remote-build HOST=b3
make remote-push HOST=b3
```

### 3. Deploy to the Cluster
Before deploying the distributed architecture, ensure the Cloud Storage for Lustre API is enabled on your GCP project. This is required for the CSI driver to provision the high-performance parallel file system volume dynamically:

```bash
gcloud services enable lustre.googleapis.com
```

You must also configure **Private Services Access** for your VPC network. If your PVC remains in a `Pending` state with an error stating `the network has not been peered with Google managed services`, it means your cluster cannot securely reach the Managed Lustre backend. Run these one-time setup commands (assuming you are using the `default` network):

```bash
# 1. Enable the Service Networking API
gcloud services enable servicenetworking.googleapis.com

# 2. Allocate an IP range for Google managed services
gcloud compute addresses create google-managed-services-default \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=20 \
    --description="Peering for Managed Lustre" \
    --network=default

# 3. Create the private connection
gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=google-managed-services-default \
    --network=default
```

You must also enable the Managed Lustre CSI driver addon on your GKE cluster:

```bash
gcloud container clusters update au-rl-1 \
    --location=us-central1 \
    --update-addons=LustreCsiDriver=ENABLED
```

**Configuring OpenTelemetry (GKE Workload Identity)**:
To securely allow the GKE pods to export telemetry data (like Trainer GPU utilization) to Google Cloud Trace without granting node-level permissions, you must bind the `Cloud Trace Agent` role to the Kubernetes ServiceAccount (`open-rl-sa`) using Direct Workload Identity Federation:

```bash
PROJECT_ID="YOUR_PROJECT_ID"
PROJECT_NUM=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="principal://iam.googleapis.com/projects/${PROJECT_NUM}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/open-rl-sa" \
    --role="roles/cloudtrace.agent" \
    --condition=None
```

Apply the Kubernetes manifests. The deployment spins up a fully distributed, multi-node architecture utilizing Google Cloud Managed Lustre for high-performance adapter synchronization:
1. **`open-rl-gateway`**: The PyTorch Training Gateway Deployment (Allocated to its own dedicated L4 GPU node)
2. **`vllm-worker`**: The vLLM Inference Worker Deployment (Allocated to its own dedicated L4 GPU node, horizontally scalable)
3. **`redis-broker`**: The Async Workload State Broker Deployment
4. **`open-rl-lustre-pvc`**: A 1.2TB Managed Lustre `ReadWriteMany` network share mounted universally at `/mnt/lustre/open-rl`.

```bash
kubectl apply -f server/kubernetes/distributed-lustre/

# Watch the distinct pods transition to Running status
kubectl get pods -l 'app in (open-rl-gateway, vllm, redis)' -w
```

### 4. Connect to the Server
The service is exposed internally as a `ClusterIP`. To connect your local SDK client to the GKE deployment, set up a secure port-forward to the PyTorch Gateway service:

```bash
kubectl port-forward svc/open-rl-gateway-service 8000:8000
```
Your SDK clients (e.g. `ServiceClient(base_url="http://localhost:8000")`) will now route traffic directly to the distributed GKE cluster.

> [!TIP]
> If a local process gets stuck on port 8000 from an old port-forward or server run, you can instantly terminate it with `make kill-server`.



## Operating the Architecture

Open-RL multiplexes workloads via a central `StateStore`, which defaults to in-memory queues for local dev but scales horizontally via Redis.

To run the full stack:

### Step 1: Start Redis (Optional but Recommended)
For true distributed execution, ensure a Redis server is running.
```bash
# On Debian/Ubuntu Linux
make install-redis
make start-redis
```

### Step 2: Start the PyTorch Training Gateway
This process handles HTTP validation, training queues, LoRA weight saving, and executing the PyTorch `TrainerEngine` backpropagation loop.

```bash
# Terminal 1
# Optionally set REDIS_URL to shift the workload queues to Redis
REDIS_URL="redis://localhost:6379" make run-server
```

### Step 3: Start the vLLM Inference Worker
This independent process pre-loads the Base Model into VRAM and handles all high-throughput inference requests, hot-swapping the LoRA weights saved by the PyTorch gateway.

```bash
# Terminal 2
# Optionally override the model with VLLM_MODEL=...
make run-vllm
```
*Note: The PyTorch Gateway automatically routes inference requests to `http://127.0.0.1:8001/generate`. You can override this by setting `VLLM_URL` on the PyTorch Gateway.*

### Step 4: Execute Workloads
With the backend topology running, you can now launch parallel algorithmic clients:

```bash
# Terminal 3
# Run a Multi-Tenant SFT Simulation
make run-sft-parallel

# Run a Parallel Reinforcement Learning loop (GRPO/Importance Sampling)
make run-rlvr-parallel
```

## Reproduce FunctionGemma Fine-Tuning

The repo includes a FunctionGemma SFT reproduction script based on the Google guide:
`client/functiongemma_sft.py`.
It loads data from Hugging Face dataset `bebechien/SimpleToolCalling`
with local fallback at `client/data/functiongemma_simple_tool_calling.json`.

```bash
# 1) Start server with matching base model
make run-function-gemma-server

# 2) Run FunctionGemma training + eval
make run-function-gemma-sft
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
