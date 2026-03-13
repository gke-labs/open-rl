# Open-RL Client

This directory contains the client-side scripts for interacting with the Open-RL API.

## FunctionGemma Demo

Script: `client/functiongemma_sft.py`

Prereqs:

- accept the FunctionGemma model terms: https://huggingface.co/google/functiongemma-270m-it
- set `HF_TOKEN` or run `uv run hf auth login`

From the repo root, start the local FunctionGemma server in one terminal:

```bash
export HF_TOKEN=...
make run-function-gemma-server
```

Then run the demo in a second terminal:

```bash
make run-function-gemma
```

What it does:

- starts a local Open-RL server on `http://127.0.0.1:9000`
- runs the gateway and engine loop in one process, so Redis and a separate worker are not required
- loads `google/functiongemma-270m-it`
- trains on Hugging Face `bebechien/SimpleToolCalling`
- runs pre/post evaluation
- saves `artifacts/functiongemma_sft_metrics.png`


![FunctionGemma Result](./artifacts/functiongemma_sft_metrics.png)

## Pig Latin SFT

Script: `client/piglatin_sft.py`

This script demonstrates fine-tuning a model to translate English into Pig Latin using word-level translations. It supports configurable `chz` presets for both **Qwen** and **Gemma** models.

### Option 1: Qwen (Default)

Start the local single-process Open-RL server for Qwen:

```bash
make run-pig-latin-server
```

Then run the training demo in a second terminal:

```bash
make run-pig-latin-sft
```

What it does:

- starts a local Open-RL server on `http://127.0.0.1:9001`
- loads `Qwen/Qwen3-0.6B`
- trains a LoRA adapter on word-level Pig Latin pairs
- runs pre/post translation evaluation and saves plots into `artifacts/`

![Pig Latin Qwen Result](./artifacts/piglatin_qwen_metrics.png)

### Option 2: Gemma

Start the local single-process Open-RL server for Gemma:

```bash
make run-pig-latin-gemma-server
```

Then run the training demo in a second terminal:

```bash
make run-pig-latin-gemma-sft
```

What it does:

- starts a local Open-RL server on `http://127.0.0.1:9002`
- loads `google/gemma-3-1b-it`
- trains a LoRA adapter on word-level Pig Latin pairs
- runs pre/post translation evaluation and saves plots into `artifacts/`

![Pig Latin Gemma Result](./artifacts/piglatin_gemma_metrics.png)


## RLVR Demo

The RLVR (Reinforcement Learning with Verifiable Rewards) demo showcases training a model to answer questions in a specific format using a reward function that verifies the correctness and format of the answer.

It supports parallel training jobs, allowing you to train multiple behaviors simultaneously (e.g., answering capital cities vs. just providing the answer).

![RLVR Result](./rlvr_result.png)
