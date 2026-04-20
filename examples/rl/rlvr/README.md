# RLVR Demo

The RLVR (Reinforcement Learning with Verifiable Rewards) demo showcases training a model to answer questions in a specific format using a reward function that verifies the correctness and format of the answer.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   cd src && uv sync --extra cpu
   cd ../examples && uv sync
   ```

## Running the Training Server

Start the local single-process server:
```bash
make server
```

## Running the RL Script

Execute the training script:
```bash
cd examples/rl/rlvr
uv run python rlvr.py --jobs 1 --steps 5 --base-model "Qwen/Qwen3-4B-Instruct-2507"
```

## Contents

* `rlvr.py`: The main training script.
* `rlvr-job.yaml`: Kubernetes Job manifest for single job.
* `rlvr-job-parallel.yaml`: Kubernetes Job manifest for parallel array job.
* `README.md`: This documentation file.
