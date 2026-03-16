# FunctionGemma Supervised Fine-Tuning Guide

This guide shows how to run the local FunctionGemma SFT demo.

## Prerequisites

1. **Install dependencies**:
   ```bash
   cd server && uv sync --extra cpu
   cd ../client && uv sync
   ```
2. **Accept the model terms**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
3. **Authenticate with Hugging Face**:
   ```bash
   uv run hf auth login
   ```

## Running the Training Server

```bash
cd server
OPEN_RL_SINGLE_PROCESS=1 \
SAMPLER_BACKEND=engine \
OPEN_RL_BASE_MODEL="google/functiongemma-270m-it" \
uv run --extra cpu uvicorn src.main:app --host 127.0.0.1 --port 9000
```

This starts a local server on port 9000, preloads `google/functiongemma-270m-it`, and runs the gateway plus engine loop in one process.

You can use `make run-function-gemma-server` as a wrapper around the same `uv run --extra cpu ...` flow.

## Running the SFT Script

In a second terminal, run:

```bash
cd client
uv run --python 3.12 functiongemma-demo
```

This runs `client/functiongemma_sft.py`. The script:
- loads `bebechien/SimpleToolCalling`
- splits the data into train and eval halves
- measures baseline tool-selection rate
- trains a LoRA adapter with cross-entropy
- measures tuned tool-selection rate
- writes `artifacts/functiongemma_sft_metrics.png`

## Common arguments

Pass arguments through the Make target with `ARGS="..."`. `chz` expects `key=value` arguments:

```bash
make run-function-gemma ARGS="epochs=5 assert_loss_drop=true"
```

Supported arguments in `functiongemma_sft.py`:
- `epochs=<int>`: Number of training epochs (default 10).
- `eval_limit=<int>`: Max number of evaluation examples (default 20).
- `base_model=<str>`: The model repository (default `google/functiongemma-270m-it`).
- `base_url=<str>`: The training server URL (default `http://127.0.0.1:9000`).
- `dataset=<str>`: The dataset to train on (default `bebechien/SimpleToolCalling`).
- `rank=<int>`: The LoRA rank to use (default 16).
- `plot_path=<str>`: Output path for the metrics chart.
- `assert_loss_drop=true`: Fails the script execution if the training loss does not improve.
- `min_loss_drop=<float>`: Required relative loss improvement if asserting (default 0.05).
- `ci=true`: Run in continuous integration mode.
