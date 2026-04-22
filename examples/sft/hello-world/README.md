# Hello World SFT Sandbox

A minimal introductory fine-tuning pipeline to familiarize yourself with basic Open-RL API interactions and training loops.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   cd src && uv sync --extra cpu
   cd ../examples && uv sync
   ```

## Running the Training Server

Start the local single-process Open-RL server from the repository root:
```bash
make server
```

## Running the SFT Script

Navigate to this recipe and execute the script:
```bash
cd examples/sft/hello-world
uv run python sft.py --base-model "Qwen/Qwen3-0.6B"
```

## Contents

* `sft.py`: The main training script.
* `README.md`: This documentation file.
