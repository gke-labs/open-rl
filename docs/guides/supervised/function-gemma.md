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
make server BASE_MODEL=google/functiongemma-270m-it
```

Or manually:

```bash
cd server
OPEN_RL_SINGLE_PROCESS=1 \
SAMPLER=torch \
BASE_MODEL="google/functiongemma-270m-it" \
uv run --extra cpu uvicorn src.gateway:app --host 127.0.0.1 --port 9003
```

## Running the SFT Script

In a second terminal, run:

```bash
cd client
uv run --python 3.12 python -u functiongemma_sft.py base_url="http://127.0.0.1:9003"
```

This runs `client/functiongemma_sft.py`. The script:
- loads `bebechien/SimpleToolCalling`
- splits the data into train and eval halves
- measures baseline tool-selection rate
- trains a LoRA adapter with cross-entropy
- measures tuned tool-selection rate
- writes `artifacts/functiongemma_sft_metrics.png`

## Common arguments

Pass `chz` arguments as `key=value`:

```bash
cd client
uv run --python 3.12 python -u functiongemma_sft.py epochs=5 assert_loss_drop=true
```

Supported arguments in `functiongemma_sft.py`:
- `epochs=<int>`: Number of training epochs (default 10).
- `eval_limit=<int>`: Max number of evaluation examples (default 20).
- `base_model=<str>`: The model repository (default `google/functiongemma-270m-it`).
- `base_url=<str>`: The training server URL (default `http://127.0.0.1:9003`).
- `dataset=<str>`: The dataset to train on (default `bebechien/SimpleToolCalling`).
- `rank=<int>`: The LoRA rank to use (default 16).
- `plot_path=<str>`: Output path for the metrics chart.
- `assert_loss_drop=true`: Fails the script execution if the training loss does not improve.
- `min_loss_drop=<float>`: Required relative loss improvement if asserting (default 0.05).
- `ci=true`: Run in continuous integration mode.
