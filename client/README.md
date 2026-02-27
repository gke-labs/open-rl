# Open-RL Client

This directory contains the client-side scripts for interacting with the Open-RL API.

## RLVR Demo

The RLVR (Reinforcement Learning with Verifiable Rewards) demo showcases training a model to answer questions in a specific format using a reward function that verifies the correctness and format of the answer.

It supports parallel training jobs, allowing you to train multiple behaviors simultaneously (e.g., answering capital cities vs. just providing the answer).

![RLVR Result](./rlvr_result.png)

## FunctionGemma SFT

Use `functiongemma_sft.py` to reproduce tool-calling SFT:

```bash
uv run --python 3.12 python functiongemma_sft.py
```

To download the functiongemma model you must agree to terms and conditions located here: https://huggingface.co/google/functiongemma-270m-it

After agreeing, log in with Hugging Face:

```bash
uv run hf auth login
```

Dataset source:
- Primary: Hugging Face `bebechien/SimpleToolCalling`


