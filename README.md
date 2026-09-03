# OpenRL: self-hosted API for your RL Infrastructure

[![Release](https://img.shields.io/github/v/release/gke-labs/open-rl?label=release)](https://github.com/gke-labs/open-rl/releases/latest)

> **Research preview.** OpenRL is an early-stage project from GKE Labs. Expect the API surface
> and architecture to keep evolving.

OpenRL implements [Tinker](https://tinker-docs.thinkingmachines.ai/) compatible API for fine-tuning language models that you can run on your own infrastructure (machine or a kubernetes cluster). You can use the Tinker SDK to orchestrate RL training loops by writing imperative Python code directly from your local machine.

📖 For the full story behind why we built OpenRL, read our introductory blog post:
[Introducing OpenRL: A self-hosted post-training API for fine-tuning LLMs](https://opensource.googleblog.com/2026/06/introducing-openrl-a-self-hosted-post-training-api-for-fine-tuning-llms.html)
(Google Open Source Blog, June 2026).

## Why OpenRL

Agentic RL on LLMs carries a lot of systems complexity. Running a single RL loop means
coordinating dataset selection and cleaning, choosing RL environments, debugging the training
loop, managing reward signals, handling inference mismatches, allocating hardware, and
operating the infrastructure underneath all of it.

Our view is that AI research and infrastructure concerns are too tightly coupled in today's
tooling. Separating them lets infrastructure engineers and AI researchers move independently —
much the way Kubernetes separated infrastructure concerns from application development.

That is why we built on Tinker. Tinker simplifies LLM post-training for developers and
researchers by hiding post-training infrastructure behind four API primitives. Researchers keep
complete control over their training algorithms, data loops, and loss functions; platform
engineers scale, orchestrate, and operate the infrastructure independently. OpenRL lets you run
those same APIs on hardware you control.

**Bonus**: you can use [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) that has awesome tutorials/recipes and utilities!

### Share GPUs across jobs

A traditional RL loop runs sequentially: trainers wait on samplers, and samplers wait on
environments to score rewards — work that is often CPU- or network-bound. Expensive GPUs sit
idle for much of the loop. Because the API decouples the loop from the hardware, OpenRL can run
multiple RL jobs concurrently and pack their training and sampling steps together, which lifts
overall GPU utilization.

### Prototype locally, scale remotely

Putting infrastructure behind an API also means researchers stop wrestling with heavy Python
and CUDA dependency stacks. During R&D you can run your RL loop on a Mac while pointing it at
training APIs hosted on a Kubernetes cluster or a pool of VMs, then scale up without rewriting
the loop.

### Autoresearch

We expect frontier AI research to become increasingly automated, and abstracting away the
infrastructure is groundwork for that. The [autoresearch recipes](examples/autoresearch/README.md),
adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), run parallel
experiments for parameter sweeps and reward-signal improvement against a shared OpenRL gateway.

## What OpenRL is not

- **It is not a managed service.** OpenRL is self-hosted. The goal is that it is easy to deploy
  and operate on your own Kubernetes cluster.
- **It is not an RL framework.** You keep full control over your RL loop; OpenRL only provides
  the training and sampling APIs underneath it.

## Quick Start

 - Follow the [Pig Latin notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) or [Text-to-SQL notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb) to see supervised fine-tuning in action.
 - Follow the [Text-to-SQL RL recipe](examples/text-to-sql/README.md) to see reinforcement learning in action.

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
  training_client = await service_client.create_lora_training_client_async(base_model="Qwen/Qwen3-4B-Instruct-2507", rank=16)

  for epoch in range(10):
    # 2A. Extract sampling client from current weights
    sampling_client = training_client.save_weights_and_get_sampling_client(name=f"rlvr_epoch_{epoch}")

    prompt_text = generate_math_problem()

    # 2B. Sample multiple rollouts (e.g. N=8) from the prompt
    response = sampling_client.sample(
      prompt=types.ModelInput.from_ints(tokens=[...]), num_samples=8, sampling_params=types.SamplingParams(max_tokens=100, temperature=0.9)
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
    await training_client.forward_backward_async(datums, loss_fn="importance_sampling", loss_fn_config={"clip_range": 0.2})

    # 5. Optimizer Step
    await training_client.optim_step_async(types.AdamParams(learning_rate=1e-5))


asyncio.run(rlvr_loop())
```

## Documentation & Guides

Detailed guides and runnable examples are structured under `docs/` and `examples/`:

- **Guides:**
  - Supervised finetuning:
    - [Pig Latin SFT Notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) & [script guide](examples/sft/pig-latin/README.md)
    - [Text-to-SQL SFT Notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb)
  - Reinforcement Learning:
    - [Text-to-SQL RL Recipe](examples/text-to-sql/README.md)
- **Technical Documentation**:
  - [Architecture](docs/architecture.md)
  - [Tinker Client Compatibility](docs/tinker-client-compatibility.md)
- **Deployment**:
  - [Kubernetes Deployment Guide (GKE)](docs/setup/gke-setup.md)

## Roadmap

The FY 2026 roadmap lives in [ROADMAP.md](ROADMAP.md). It covers reliability and production
readiness, multi-GPU support, TPUs, more models, observability, recipes, and community. To
propose a change, open an issue with the `roadmap` label.

## Acknowledgements

We are grateful to the open source AI communities whose work inspired OpenRL, in particular
[Thinking Machines](https://thinkingmachines.ai/), [vLLM](https://github.com/vllm-project/vllm),
[PyTorch](https://pytorch.org/), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl),
[verl](https://github.com/volcengine/verl), [SkyRL](https://github.com/NovaSky-AI/SkyRL), and
[llm-d](https://github.com/llm-d/llm-d).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a
development environment, find an issue to work on, and open a pull request.

Participation in this project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Community

<!-- TODO: Add a developer mailing list, chat channel, and public developer meeting once
     these are set up. -->

- **Questions and discussion**: [GitHub Issues](https://github.com/gke-labs/open-rl/issues)
- **Reporting a vulnerability**: see our [Security Policy](SECURITY.md) — please do not open a
  public issue for security reports
- **How the project is run**: [GOVERNANCE.md](GOVERNANCE.md)
- **Who maintains it**: [MAINTAINERS.md](MAINTAINERS.md)

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Disclaimer

This is not an officially supported Google product.

This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.
