# FunctionGemma Supervised Fine-Tuning Guide

This guide shows how to run the local FunctionGemma SFT demo.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   cd src && uv sync --extra cpu
   cd ../examples && uv sync
   ```
2. **Accept the model terms**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
3. **Authenticate with Hugging Face** (required for gated models):
   ```bash
   uv run hf auth login
   ```

## Running the Training Server

Start the local server preloaded with FunctionGemma:
```bash
make server BASE_MODEL=google/functiongemma-270m-it
```

## Running the SFT Script

Execute the training script:
```bash
cd examples/sft/function-gemma
uv run python functiongemma_sft.py
```

## Contents

* `functiongemma_sft.py`: The main training script.
* `README.md`: This documentation file.
