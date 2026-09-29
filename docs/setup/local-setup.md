# Local Setup Guide

This guide describes how to set up a local environment (or a single VM) to run OpenRL workloads.

## Prerequisites

### 1. Provision GPU VM

First, you need a machine with GPUs. These requirements are based on running both the sampler and trainer in parallel.

*   **GPUs**: At least 2 GPUs are recommended (one for the vLLM sampler, one for the trainer).
*   **VRAM**: At least 23 GB of VRAM per GPU (e.g., NVIDIA L4 is sufficient).
*   **System RAM**: At least 32 GB of system RAM.

<details>
<summary><b>`gcloud` command to provision GCE VM in GCP</b></summary>

If you are using Google Cloud Platform (GCP), you can create a suitable GCE VM using the following command.

```bash
gcloud compute instances create openrl-vm \
    --machine-type=g2-standard-24 \
    --accelerator=type=nvidia-l4,count=2 \
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

Ensure you have the required build tools, Python headers, Redis, and `uv` installed on the machine:

```bash
sudo apt update && sudo apt install -y build-essential python3.12-dev make redis-server
sudo systemctl enable --now redis-server
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

vLLM JIT-compiles LoRA kernels at startup and needs `nvcc`. The GPU VM images
ship only the driver, so install the CUDA toolkit as well:

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
```

### 4. Sanity Check

Run the sanity check script to ensure your environment meets the requirements:

```bash
python3 examples/text-to-sql/utils/sanity_check.py
```

## Deploying OpenRL

All commands below assume you are in the **repository root** directory.

### 1. Start the OpenRL Server

The API server now launches a dedicated trainer and vLLM sampler process per model
on demand (they share request queues and futures through Redis), so a single
terminal starts everything:

```bash
export REDIS_URL=redis://127.0.0.1:6379/0
export BASE_MODEL=google/gemma-4-e2b
export SAMPLING_BACKEND=vllm
export VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM
export TRAINER_CUDA_VISIBLE_DEVICES=0
export SAMPLER_CUDA_VISIBLE_DEVICES=1

# Required for gated models such as google/gemma-4-e2b (and recommended
# generally to avoid Hugging Face rate limits)
# export HF_TOKEN="your_huggingface_token"
make server
```

The OpenRL server is now available at `http://127.0.0.1:9003`. Worker logs are
written to `$OPEN_RL_TMP_DIR` (default `/tmp`) as `trainer_<model>.log` and
`sampler_<model>.log`.
