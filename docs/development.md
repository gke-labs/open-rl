# Development setup

OpenRL runs two ways:

- **Production** is Kubernetes. Follow [Getting started](getting-started.md).
- **Development** runs the API server from a checkout with `make` targets, on a
  laptop or a GPU VM. This page covers that.

All commands run from the repository root. They need [uv](https://docs.astral.sh/uv/)
on your `PATH`; `make help` lists the targets.

## On a laptop (CPU)

`make server` starts the API server with the trainer in the same process,
sampling with torch. No GPU, Redis, or vLLM is involved. The default model is
`google/gemma-4-e2b`; on a CPU a small model is quicker:

```bash
uv sync --extra cpu
uv --project examples sync
make server BASE_MODEL=Qwen/Qwen3-0.6B
```

In another terminal, run the tiny examples against it. Single-process mode
serves the one model it loaded, so pass the same model:

```bash
uv --project examples run python examples/tiny/tiny_sft.py base_model=Qwen/Qwen3-0.6B sample_after_train=true
uv --project examples run python examples/tiny/tiny_rl.py base_model=Qwen/Qwen3-0.6B steps=10 learning_rate=1e-4
```

CPU is fine for small models and for working on the API server; use a GPU VM
for anything larger.

## On a GPU VM

With `REDIS_URL` set, the API server runs as it does on Kubernetes: it launches
a trainer and a vLLM sampler as separate processes when a client creates a
model, and they share queues through Redis.

### Machine

The trainer and the sampler each need a GPU:

*   **GPUs**: two, with at least 23 GB each (an NVIDIA L4 is enough).
*   **System RAM**: at least 32 GB.

<details>
<summary><b><code>gcloud</code> command to create a suitable VM on GCP</b></summary>

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

### Packages

On the VM, clone the repository and install build tools, Python headers, Redis,
and uv:

```bash
git clone https://github.com/gke-labs/open-rl.git
cd open-rl
sudo apt update && sudo apt install -y build-essential python3.12-dev make redis-server
sudo systemctl enable --now redis-server
curl -LsSf https://astral.sh/uv/install.sh | sh
```

vLLM compiles LoRA kernels at startup and needs `nvcc`. The GPU VM images ship
only the driver, so install the CUDA toolkit too:

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
```

Check the environment:

```bash
python3 examples/text-to-sql/utils/sanity_check.py
```

### Start the server

```bash
export REDIS_URL=redis://127.0.0.1:6379/0
export VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM   # for the default model, google/gemma-4-e2b
export TRAINER_CUDA_VISIBLE_DEVICES=0
export SAMPLER_CUDA_VISIBLE_DEVICES=1
# export HF_TOKEN=...   # avoids Hugging Face rate limits; required for gated models
make server
```

The server listens on `http://127.0.0.1:9003`. Worker logs go to
`$OPEN_RL_TMP_DIR` (default `/tmp`) as `trainer_<model>.log` and
`sampler_<model>.log`. Pass `BASE_MODEL=<model>` to use another model, and drop
`VLLM_ARCHITECTURE_OVERRIDE` unless it is a Gemma 4 model.

To work from a laptop against the VM, sync your checkout with
`make push-vm REMOTE_HOST=<host>` and pull results back with `make pull-vm`.

## On a kind cluster

To work on the scheduler or the Kubernetes manifests against real GPUs without
GKE, run a kind cluster on a GPU VM: see [dev/kind](../dev/kind/README.md).

## Tests and contributions

`make test` runs the unit tests and `make lint` the formatting checks.
[CONTRIBUTING.md](../CONTRIBUTING.md) covers the end-to-end tests and the pull
request process, and [Configuration](configuration.md) lists every environment
variable.
