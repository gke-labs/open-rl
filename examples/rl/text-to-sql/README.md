# Text-to-SQL RL Recipe

## Overview

This recipe provides a complete guide to fine-tuning a base LLM model to generate correct SQL queries from natural language questions, optimizing specifically for execution correctness (i.e., the generated SQL returns the correct result when executed).

- **Base Model**: [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)
- **Dataset**: [philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql)

The **goal** is to demonstrate how to use the Open-RL infrastructure to run training locally on a single machine with multiple GPUs. This provides a baseline and understanding before scaling to a distributed Kubernetes (K8s) cluster in later guides.

**What the core script does**: The core training script [texttosql_sft_grpo.py](texttosql_sft_grpo.py) orchestrates the training loop. It performs the following actions:
*   Calls our Open-RL server (gateway) to request samples from vLLM.
*   Executes the generated SQL queries in a local SQLite database to compute rewards.
*   Sends these rewards back to the server to update the LoRA adapter weights via the trainer.

<details>
<summary><b>Sequence Diagram</b></summary>

This sequence diagram illustrates the interaction between the client script, the gateway/trainer, and the vLLM sampler during the RL training phase, noting the key functions called.

```mermaid
sequenceDiagram
    participant Client as Client Script<br/>(texttosql_sft_grpo.py)
    participant Gateway as Gateway / Trainer<br/>(GPU 1)
    participant Sampler as vLLM Sampler<br/>(GPU 0)

    Note over Client,Sampler: 1. Rollout Phase
    Client->>Gateway: 1.1 Save current weights for sampler<br/>(save_weights_and_get_sampling_client_async)
    Gateway-->>Client: Acknowledged
    Client->>Gateway: 1.2 Request N samples for prompt<br/>(sample_async)
    Gateway->>Sampler: 1.3 Forward generation request (API)
    Sampler-->>Gateway: 1.4 Return generated SQL completions
    Gateway-->>Client: Return samples to client

    Note over Client,Sampler: 2. Reward Phase
    Client->>Client: 2.1 Execute SQL in local SQLite<br/>(build_rollout -> score_eval_prediction)
    Client->>Client: 2.2 Calculate rewards & advantages

    Note over Client,Sampler: 3. Update Phase
    Client->>Gateway: 3.1 Send data for Forward/Backward (Grads)<br/>(forward_backward_async)
    Gateway->>Gateway: Compute loss & gradients on GPU 1
    Gateway-->>Client: Acknowledged
    Client->>Gateway: 3.2 Send request to step optimizer<br/>(optim_step_async)
    Gateway->>Gateway: Update weights on GPU 1
    Gateway-->>Client: Acknowledged
```
</details>



## Prerequisites

### 1. Provision GPU VM

First, you need a machine with GPUs. These requirements are based on the ~10.5 GB model size, running both the sampler and trainer in parallel, and the overhead from Torch Inductor autotuning (where PyTorch compiles and optimizes GPU kernels at runtime for your specific hardware, requiring additional memory).

*   **GPUs**: At least 2 GPUs are recommended (one for the vLLM sampler, one for the trainer). We run them on separate GPUs because both are memory-intensive and sharing a single GPU will likely cause Out-of-Memory errors.
*   **VRAM**: At least 23 GB of VRAM per GPU (e.g., NVIDIA L4 is sufficient).
*   **System RAM**: At least 32 GB of system RAM (for Gemma 4 E2B; larger models may require more).

<details>
<summary><b>`gcloud` command to provision GCE VM in GCP</b></summary>

If you are using Google Cloud Platform (GCP), you can create a suitable GCE VM using the following command. For more details, see [GCE GPU VM G-Series docs](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series).

```bash
gcloud compute instances create openrl-vm \
    --machine-type=g2-standard-24 \
    --accelerator=type=nvidia-l4-vws,count=2 \
    --zone=us-central1-a \
    --boot-disk-size=50GB \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accelerator-2404-amd64-with-nvidia-580 \
    --maintenance-policy=TERMINATE \
    --metadata=enable-osconfig=TRUE,enable-oslogin=true \
    --restart-on-failure
```
</details>

### 2. Access VM and Clone the Repository

Once you have accessed your VM, clone the repository and stay in the repository root for the following steps:

```bash
git clone https://github.com/gke-labs/open-rl.git
cd open-rl
```

### 3. System Packages

Ensure you have the required build tools, Python headers, and `uv` installed on the machine:

```bash
sudo apt update && sudo apt install -y build-essential python3.12-dev make
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 4. Sanity Check

Run the sanity check script to ensure your environment meets the requirements:

```bash
python3 examples/rl/text-to-sql/utils/sanity_check.py
```

## Deploying OpenRL

All commands below assume you are in the **repository root** directory.

### 1. Patch vLLM

Patch vLLM for Gemma 4 LoRA support. This is a temporary local patch for duplicate LoRA module registration, related to [vllm-project/vllm#39246](https://github.com/vllm-project/vllm/issues/39246).

```bash
(cd src/server && \
 uv run --extra vllm python scripts/patch_vllm_lora_dedup.py)
```

### 2. Start the vLLM Sampler

In your **first terminal session**, start the vLLM sampler on GPU 0. We set the required environment variables locally for this terminal.

```bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=google/gemma-4-e2b
export VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM

# Recommended to avoid Hugging Face rate limits
# export HF_TOKEN="your_huggingface_token"
make vllm
```

### 3. Start the Open-RL Server

In a **second terminal session**, start the Open-RL gateway and trainer on GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=1
export BASE_MODEL=google/gemma-4-e2b
export SAMPLING_BACKEND=vllm
make server
```

## Running the Training

Open a **third terminal session** to run the training script.

### 1. Common Environment Variables

You can copy and paste these into your training terminal before proceeding.

```bash
# Open-RL Gateway URL
export TINKER_BASE_URL=http://127.0.0.1:9003

# Dummy API key for local gateway
export TINKER_API_KEY=tml-dummy

# Recommended to avoid Hugging Face rate limits
# export HF_TOKEN="your_huggingface_token"
```

### 2. Execute Training

Set the `MODE` environment variable to control how the script runs.

Supported modes with known-good configurations:
*   **`full`**: Runs both SFT (`5` steps on `100` examples, learning rate `5e-5`) and RL (`80` steps, `8` prompts x `8` samples per step, learning rate `5e-6`).
*   **`rl_only`**: Skips SFT and runs only RL (`80` steps, `8` prompts x `8` samples per step, learning rate `5e-6`) from the base model.

```bash
# Option 1: Full SFT + RL (Default)
export MODE="full"

# Option 2: RL Only (from scratch)
# export MODE="rl_only"

# Run training
(cd examples/rl/text-to-sql && \
 uv run python texttosql_sft_grpo.py gemma4_e2b_rl_recipe phase=$MODE)
```

## Results

After training, you can plot the metrics. Run this from the repository root. We use the `$MODE` variable to point to the correct artifact directory.

```bash
(cd examples/rl/text-to-sql && \
 uv run python -m utils.plot \
   artifacts/texttosql_sft_grpo_gemma4_e2b_rl_recipe_$MODE/metrics.jsonl)
```

The plotter renders several curves to help you understand training progress:
*   **Execution Match**: The percentage of generated SQL queries that returned the correct result when executed. This is the primary metric for success.
*   **RL Reward EWMA**: The Exponentially Weighted Moving Average (smoothed average) of the RL reward.
*   **Compile Rate EWMA**: The smoothed rate at which generated SQL queries were successfully compiled by the database.
*   **SFT Loss**: The loss during the Supervised Fine-Tuning phase. It should decrease.

Here are the actual plots from known-good runs for each mode. Expand the sections below to see the plots and their interpretation.

<details>
<summary><b>RL Only Results</b></summary>

*   In this mode, training starts directly with RL without a prior SFT phase.
*   You should see the **Execution Match** curve start low (around 8%) and steadily increase as the model learns to generate correct SQL based on execution rewards.
*   The **Compile Rate** should also increase, showing the model learns valid SQL syntax.
*   The **RL Reward** should show an upward trend.

![Text-to-SQL curves RL Only](./results/texttosql-curves-rl-only.png)
</details>

<details>
<summary><b>Full SFT + RL Results</b></summary>

*   In this mode, the model first undergoes SFT for 5 steps, which typically raises the baseline execution accuracy from ~8% to ~12%.
*   After SFT, the RL phase begins. You should see a significant jump in **Execution Match** during the RL phase, reaching around 40% in known-good runs.
*   The **SFT Loss** curve (usually shown in a separate plot or as part of the logs) should show a decrease during the first 5 steps.
*   This plot demonstrates the combined effect of format learning (SFT) and correctness optimization (RL).

![Text-to-SQL curves SFT and RL](./results/texttosql-curves-sft-and-rl.png)
</details>

## Advanced: Customizing the Run

If you want to experiment further, you can override default configurations by appending them to the command line:

```bash
(cd examples/rl/text-to-sql && \
 uv run python texttosql_sft_grpo.py gemma4_e2b_rl_recipe sft.steps=10 rl.steps=100)
```

Common overrides:
*   `sft.steps=5`
*   `sft.learning_rate=5e-5`
*   `rl.steps=80`
*   `rl.learning_rate=5e-6`
*   `rl.prompts_per_step=8`
*   `rl.samples_per_prompt=8`

## Troubleshooting

*   **CUDA Out of Memory (OOM)**: If vLLM fails with OOM during startup, ensure you have enough VRAM and set a higher value for `VLLM_GPU_MEMORY_UTILIZATION` (e.g., 0.9).
*   **System Hangs during model loading**: Ensure you have at least 32GB of system RAM. Loading large models can overwhelm smaller RAM configurations.
