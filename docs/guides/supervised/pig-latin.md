# Pig Latin Supervised Fine-Tuning

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

![Pig Latin Qwen Result](../../../client/artifacts/piglatin_qwen_metrics.png)

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

![Pig Latin Gemma Result](../../../client/artifacts/piglatin_gemma_metrics.png)
