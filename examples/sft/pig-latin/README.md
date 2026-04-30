# Pig Latin Supervised Fine-Tuning

This script demonstrates fine-tuning a model to translate English into Pig Latin using word-level translations. It supports configurable `chz` presets for both **Qwen** and **Gemma** models.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   cd src && uv sync --extra cpu
   cd ../examples && uv sync
   ```

## Running the Training Server

### Option 1: Qwen (Default)
Start the local single-process Open-RL server for Qwen (`BASE_MODEL` defaults to `Qwen/Qwen3-0.6B`):
```bash
make server
```

### Option 2: Gemma
Start the local single-process Open-RL server for Gemma (set `BASE_MODEL`):
```bash
make server BASE_MODEL=google/gemma-3-1b-it
```

## Running the SFT Script

### Option 1: Qwen
```bash
cd examples/sft/pig-latin
uv run python piglatin_sft.py qwen
```

### Option 2: Gemma
```bash
cd examples/sft/pig-latin
uv run python piglatin_sft.py gemma
```

## Contents

* `piglatin_sft.py`: The main training script.
* `piglatin_sft_notebook.ipynb`: Jupyter notebook version.
* `piglatin_data.json`: Dataset for training.
* `README.md`: This documentation file.
* `../../../tests/`: Unit tests for the example.
