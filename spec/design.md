# Cute-RL API Design Spec

## Overview
This project builds an API for post-training LLMs, inspired heavily by the [Tinker API](https://tinker-docs.thinkingmachines.ai/). 
The core goal is to provide a backend for SFT (Supervised Fine-Tuning) initially, with the architecture built to support RL (e.g. GRPO, PPO) down the line.

The system is composed of:
1. **Server (`server/`)**: Runs on a remote Linux GPU machine (Ubuntu). Exposes the Tinker-compatible API. Uses Huggingface TRL for training operations and vLLM for inference operations.
2. **Client**: Runs locally (e.g., on Mac), using the official Tinker Python client library, connecting to the `server`.

## Tech Stack
- **Language**: Python 3.10+
- **Dependency Management**: `uv`
- **Web Framework**: FastAPI (ideal for async Python APIs matching Tinker's async nature)
- **Training Engine**: Huggingface `trl`, `peft` (LoRA), `transformers`
- **Inference Engine**: `vllm`

## MVP Scope (Single User, Single Model, SFT)
To match the Tinker API experience for SFT, the server will implement the following core workflows:
1. **Service Client Operations**
   - Extrac model capabilities
   - `create_lora_training_client(base_model, rank)`
2. **Training Client Operations**
   - `forward_backward(data, loss_fn)`: Accumulate gradients via TRL.
   - `optim_step(optimizer_params)`: Apply accumulated gradients.
3. **Sampling Client Operations**
   - `save_weights_and_get_sampling_client(name)`: Pass LoRA adapters to the inference engine.
   - `sample(prompt, sampling_params)`: Generate text using vLLM.

## Architecture Guidelines
### Process Model
- **Monolithic API Server**: For the MVP, the FastAPI server will host the PyTorch training state in memory. 
- **Inference Integration**: vLLM can be initialized alongside the training state or as a separate subprocess managed by the server. LoRA weights from the training step will be hot-swapped into the vLLM engine.
- **State Management**: Since it's a single-user MVP, the optimizer and model weights will be maintained in a global or singleton session state on the server. Callers to `forward_backward` will mutate this global state.

### Code Organization
```text
.
├── server/
│   ├── pyproject.toml
│   └── src/
│       ├── main.py          # FastAPI application
│       ├── api/             # Routes (tinker compatible)
│       ├── trainer/         # TRL & PEFT wrappers
│       └── inference/       # Inference wrappers (vLLM, later)
└── spec/
    └── design.md
```

## MVP Technical Decisions

Based on your feedback, we have finalized the following constraints for the MVP:
1. **Client Library**: We will strictly use the official `tinker` Python package to interface with our server. Our server will implement the exact Tinker REST APIs and JSON payload schemas (e.g., `/api/v1/get_server_capabilities`, `/api/v1/create_model`, `/api/v1/forward_backward`, etc.). Let's make sure things like `types.Datum` and Tensor responses are exactly shaped as expected.
2. **Under the Hood (vLLM & TRL)**: For the MVP, we will assume we are only using LoRA, and only going to train/sample a single model. We will keep everything within a simple PyTorch environment initially and extend out to `vllm` later.
3. **Data Modality**: The MVP will focus purely on Text.
4. **LoRA Specifics**: We will use PEFT LoRA under the hood natively.
5. **Async / Future Paradigm**: The Tinker API expects an asynchronous interface where calls return an `UntypedAPIFuture` (which is just a dictionary `{ "request_id": "<uuid>" }`), and the client polls `/api/v1/retrieve_future` until the state is no longer a `TryAgainResponse`. The server will use FastAPI's background tasks and an in-memory execution queue dictionary to manage training loop tasks and return responses when polled by the client.

## Phase 2: Reinforcement Learning (REINFORCE)
The next MVP phase involves extending the `TrainerEngine` to support RL.
1. Implement the `"importance_sampling"` loss function in `engine.forward_backward`.
   - Tinker expects `loss_fn_inputs` for this to include `target_tokens`, `logprobs` (the reference/sampling logprobs), and `advantages`.
   - The engine will compute the log-probabilities of the target tokens using the current policy (LoRA model), then compute the importance weight `rho = exp(current_logprobs - sampling_logprobs)`.
   - The loss will be `L = -rho * advantages`.
2. Add support for gathering the generated tokens and computing their log-probabilities during the `asample` endpoint to serve as the reference `logprobs`.
3. Create a new client test script `test_rl_workflow.py` that generates rollouts, applies a simple reward function (e.g., character length or specific word presence), computes advantages, and submits the `importance_sampling` training step.
