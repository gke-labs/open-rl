---
title: "OpenRL: Full Fine-Tuning, Multi-GPU, Agentic RL and More"
date: 2026-10-14T12:00:00Z
authors: ["Sunil Arora (on behalf of OpenRL Team)"]
summary: "A look at what we built in OpenRL over the last two months: full fine-tuning with delta weight sync, multi-GPU training with context parallelism, hero runs on dedicated GPUs, custom trainer images, the Harvey LAB recipe, and agent-friendly skills and observability."
tags: ["release"]
draft: false
---

Over the last two months, most of our work on OpenRL has come from running long context and longer-horizon RL jobs on real clusters, and fixing the bottlenecks we hit along the way.

Each of the topics below deserves its own deep dive, and we will publish dedicated posts on them over the next month. For now, here is an overview of what is available in our latest release.

## Summary

- **Full fine-tuning and delta weight sync.** Alongside LoRA, OpenRL now supports full-parameter fine-tuning, using GPU time-slicing and in-place delta weight sync so only the weights that changed in a step move from trainer to sampler.
- **Multi-GPU training.** A trainer/sampler can now span multiple GPUs. This enables medium scale RL post training e.g. long context training for Gemma 4 4B /Qwen3.5-9B, Gemma 4 31B with medium context.
- **Hero runs on dedicated GPUs.** A single flag gives a long-running hero run its own trainer and sampler GPUs with no sharing or time-slicing.
- **Custom trainer images.** A job can specify its own container image for the trainer worker when you need a custom training stack or different accelerator libraries.
- **Harvey LAB recipe.** An end-to-end agentic RL recipe for Harvey's Legal Agent Benchmark, using isolated sandbox, is now included in the repository.
- **Agent skill and run observability.** We now ship an OpenRL skill for your favorite coding agent, a new status API to observe and debug active RL runs.

All of these are configured through job settings passed via `TINKER_TAGS` or `user_metadata`, so you can switch between them without editing your training script.

## Full fine-tuning and delta weight sync

OpenRL started with LoRA, where multiple jobs on the same base model share a single resident trainer and sampler, and each RL step only writes a small adapter file. We now also support full-parameter fine-tuning:

```bash
TINKER_TAGS=openrl.fine_tuning_type=full python train.py
```

Full fine-tuning introduces two systems challenges that LoRA avoids: GPU memory contention and weight-sync overhead between steps.

First, because a full fine-tuning job modifies the base weights, each job gets its own trainer and sampler workers. To run on smaller clusters without requiring dedicated GPUs for every worker, OpenRL coordinates the trainer and sampler through an accelerator time-slicer built with [`llm-d`](https://llm-d.ai/blog/increase-researcher-velocity-rl-llm-d-time-slicing): the trainer and sampler take turns on the GPU, offloading state when parked so a full fine-tuning RL loop can run even on a single GPU. (See our [case study on `llm-d` blog](https://llm-d.ai/blog/increase-researcher-velocity-rl-llm-d-time-slicing) for how this increases researcher velocity.)

Second, at the end of every RL step, updated weights have to move from the trainer to the sampler. Writing and loading a full checkpoint after every step quickly dominates step time. OpenRL now support delta weight sync: instead of writing a full checkpoint, the trainer computes only the tensor elements that changed during the step, and the sampler patches them directly in memory.

In our measurements on an 8B model, only about 12% of weights changed on step 1, dropping to under 3% by step 25. Sending only the delta shrank the transfer from a 16 GB checkpoint to ~1.2 GB and cut sampler weight-update time from ~35 seconds to ~5 seconds.

## Multi-GPU training and context parallelism

Originally, an OpenRL trainer was a single process on a single GPU. On an Nvidia L4, that meant models topped out around 8B parameters, and long contexts ran out of memory for activations well before the weights filled the card.

We added a multi-GPU trainer backend built on [NVIDIA NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) that shards a job across multiple GPUs using FSDP2:

- `openrl.trainer_gpus=N` runs one trainer across `N` GPUs (up to 8), sharding model state and batches across them.
- `openrl.trainer_cp=C` enables context parallelism across `C` of those GPUs (`C` must divide `N`), splitting individual long sequences across devices so activation memory doesn't bottleneck on a single card.

```bash
TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4 python train.py
```

In practice, this lets us run models like Gemma 4 4B at very long context lengths, and 30B-class models like Gemma 4 31B or Gemma 3 27B at medium to long context.

On the sampling side, you can scale out multiple vLLM sampler workers on separate GPUs, all pulling from the same request queue:

```bash
TINKER_TAGS=openrl.sampler_replicas=2 python train.py
```

For example, our Harvey LAB recipe runs 32k-token agent turns inside a 192k-token context window. A 192k-token sequence does not fit on one GPU, so the recipe uses the Automodel trainer across four GPUs with four-way context parallelism.

Today, the Automodel backend supports LoRA on Kubernetes and can resume both adapter weights and optimizer state from checkpoints.

## Hero runs on dedicated GPUs

By default, OpenRL shares GPUs across jobs so small experiments don't leave hardware idle: LoRA jobs on the same base model share a single trainer and sampler, and full fine-tuning jobs time-slice GPUs.

That works well during a sweep of short experiments, but a hero run is different—it runs for hours and shouldn't wait behind another job's turn on the GPU. Setting `openrl.exclusive=true` gives the job its own trainer and sampler pods on dedicated GPUs with time-slicing disabled:

```bash
TINKER_TAGS=openrl.exclusive=true python train.py
```

You can run your sweep on shared GPUs, pick the best configuration, and launch the hero run with the same script plus `openrl.exclusive=true`.

## Bring your own trainer image

OpenRL includes two built-in trainer backends (`pytorch` and `automodel`), but you can now point a job at your own trainer container image instead. Pass the image reference as the trainer backend and the scheduler will launch that image for the job's trainer worker:

```bash
TINKER_TAGS=openrl.trainer_backend=ghcr.io/your-org/trainer:tag python train.py
```

This makes it straightforward to test custom training loops, different framework versions, or hardware-specific trainer builds without redeploying the cluster. Support for custom sampler images will follow the same pattern.

## Agentic RL example: Harvey LAB

Harvey's Legal Agent Benchmark (LAB) is a set of 1,251 long-horizon legal tasks across different practice areas. In each task, the model works inside a sandboxed environment containing source documents (contracts, board minutes, financial filings) and uses tools to inspect files and draft deliverables such as memos, letters, spreadsheets, and timelines, which are then graded against a rubric by an LLM judge.

The [Harvey LAB recipe](https://github.com/gke-labs/open-rl/tree/main/examples/harvey_labs) in our repository is a complete GRPO training loop built on `tinker-cookbook`. It uses [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) to isolate the per-task workspaces and includes a Kubernetes Job manifest to run the training loop inside the cluster. Over 30 steps, held-out reward improves from 0.57 to 0.72–0.74. We use this recipe to test OpenRL on realistic multi-turn workloads, and it serves as a working reference if you are building your own tool-using RL environments.

## Configuring jobs through settings

All of the options above are passed as `openrl.*` settings—either in `user_metadata` when creating a session or model in Python, or through `TINKER_TAGS` in the environment:

```python
client = tinker.ServiceClient(
  base_url=OPENRL,
  user_metadata={"openrl.fine_tuning_type": "full", "openrl.exclusive": "true"},
)
```

```bash
TINKER_TAGS=openrl.fine_tuning_type=full,openrl.weight_sync=delta python train.py
```

Here is the current list of settings:

| Setting | What it does |
|---|---|
| `openrl.fine_tuning_type` | `lora` (default) or `full` |
| `openrl.exclusive` | give the job its own dedicated GPUs (`true` / `false`) |
| `openrl.sampler_replicas` | number of vLLM samplers, `1` to `8` |
| `openrl.trainer_backend` | `pytorch` (default), `automodel`, or a custom container image |
| `openrl.trainer_gpus` | GPUs for the trainer, `1` to `8` (Automodel) |
| `openrl.trainer_cp` | GPUs that share one sequence for context parallelism (Automodel) |
| `openrl.weight_sync` | `delta` (default) or `full` (full fine-tuning only) |
| `openrl.fft_seed`, `openrl.fft_cpu_offload` | random seed and CPU offload (full fine-tuning only) |

## Agent-friendly OpenRL: skills, observability, and discovery

More of our users now rely on coding agents to launch, monitor, and debug their RL runs. We made a set of improvements so both developers and coding agents can work with an OpenRL cluster directly through the HTTP endpoint, without needing `kubectl` access:

- **An OpenRL skill for your favorite coding agent.** Every release now ships with an installable agent skill (plus a Claude Code plugin in our repository). It works alongside the [`tinker-cookbook` skills](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/skills) and gives your coding agent the context it needs to connect to OpenRL, pick the right settings, size GPUs, and troubleshoot runs.
- **A new API to observe and debug RL runs.** A new model status endpoint makes it easy to see what a job is doing at any moment—whether its trainer and sampler workers are waiting for GPU placement, loading weights, or actively serving requests—so a cold start never looks like a silent hang.
- **Easier capability and settings discovery.** The server now advertises its version, runtime limits, and full settings schema over HTTP, and a new `openrl-doctor` CLI lets you verify a deployment from your laptop in one command.

## What we are working on next

Next up on our roadmap:
- **Multi-GPU full fine-tuning** on the Automodel trainer backend, followed by GPU time-slicing for multi-GPU workloads.
- **Custom sampler images and prefix-aware routing** across sampler replicas.

We will share deeper posts on each of the capabilities above over the coming weeks. In the meantime, if you want to try them out, see the [getting-started guide](https://github.com/gke-labs/open-rl/blob/main/docs/getting-started.md) or browse the [v0.0.6](https://github.com/gke-labs/open-rl/releases/tag/v0.0.6) and [v0.0.7](https://github.com/gke-labs/open-rl/releases/tag/v0.0.7) release notes on GitHub. If you hit any issues, please [let us know](https://github.com/gke-labs/open-rl/issues).

## Acknowledgements

Huge thanks to [Shuby](https://github.com/ShubyM), [Hoda](https://github.com/hodaaaaaaaaaa) and [Aishu](https://github.com/aishukamal) for making these releases happen.

---

**One more thing: OpenRL is joining the CNCF!** We are thrilled to share that OpenRL has been accepted into the Cloud Native Computing Foundation (CNCF) Sandbox ([cncf/sandbox#518](https://github.com/cncf/sandbox/issues/518)), and the project donation is now in progress. We will share more on what this means for the project and community in a separate post soon.
