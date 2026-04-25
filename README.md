# Open-RL: self-hosted API for your RL Infrastructure

Open-RL implements [Tinker](https://tinker-docs.thinkingmachines.ai/) compatible API for fine-tuning language models that you can run on your own infrastructure (machine or a kubernetes cluster). You can use the Tinker SDK to orchestrate RL training loops by writing imperative Python code directly from your local machine.

# Why Tinker

We love Tinker. Tinker simplifies LLM post-training for developers and researchers. The Tinker API provides a smarter abstraction that decouples the underlying infrastructure from the RL training loop. This gives AI researchers complete control over their training algorithms, data loops, and loss functions and platform engineers the ability to scale the infrastructure independently.

**Bonus**: you can use [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) that has awesome tutorials/recipes and utilities!

## Quick Start

Follow the [Pig Latin notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) or [Text-to-SQL notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb) to see supervised fine-tuning in action. Follow the [RLVR example](examples/rl/rlvr/README.md) or [Text-to-SQL RL recipe](examples/rl/text-to-sql/README.md) to see reinforcement learning in action.

Snippet below shows a sample Reinforcement Learning loop like GRPO, where the 4 API primitives are used to create a generate-and-reward-train loop:

```python
import asyncio
import tinker
from tinker import types

# Placeholder Environment & Reward Functions
def generate_math_problem() -> str: ...
def compute_advantages(rewards: list[float]) -> list[float]: ...
def parse_and_score_response(text: str) -> float: ...

async def rlvr_loop():
    service_client = tinker.ServiceClient(base_url="http://localhost:8000")

    # 1. Create Model
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507", rank=16
    )

    for epoch in range(10):
        # 2A. Extract sampling client from current weights
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"rlvr_epoch_{epoch}"
        )
        
        prompt_text = generate_math_problem()
        
        # 2B. Sample multiple rollouts (e.g. N=8) from the prompt
        response = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=[...]),
            num_samples=8,
            sampling_params=types.SamplingParams(max_tokens=100, temperature=0.9)
        ).result()
        
        # 3. Score the rollouts using the environment
        rewards = []
        for seq in response.sequences:
            text = decode(seq.tokens)
            rewards.append(parse_and_score_response(text))
            
        advantages = compute_advantages(rewards)
        
        # ... package sequences, text, and advantages into datums ...

        # 4. Forward-Backward Pass (Importance Sampling)
        # We pass the advantages to RL objective function
        await training_client.forward_backward_async(
            datums, 
            loss_fn="importance_sampling",
            loss_fn_config={"clip_range": 0.2} 
        )
        
        # 5. Optimizer Step
        await training_client.optim_step_async(types.AdamParams(learning_rate=1e-5))

asyncio.run(rlvr_loop())
```

## Documentation & Guides

Detailed guides and runnable examples are structured under `docs/` and `examples/`:

- 🎓 **Guides:**
  - [Pig Latin SFT Notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) | [script guide](docs/guides/supervised/pig-latin.md)
  - [Text-to-SQL SFT Notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb) | [Text-to-SQL RL Recipe](examples/rl/text-to-sql/README.md)
  - [RLVR (Verifiable Rewards) Demo](examples/rl/rlvr/README.md)
- 📖 **[Architecture](docs/architecture.md)**
- 🚀 **[Kubernetes Deployment Guide (GKE)](docs/deployment.md)**

## Roadmap

- [ ] Full Finetuning support
- [ ] Model Checkpoints API
- [ ] Use advance k8s primitives such as gang scheduling, kueue for capacity/quota management

## Contributing

This project is licensed under the [Apache 2.0 License](LICENSE).

We welcome contributions! Please see [docs/contributing.md](docs/contributing.md) for more information.

We follow [Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Disclaimer

This is not an officially supported Google product.

This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.
