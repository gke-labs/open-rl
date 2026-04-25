# Open-RL Examples

This directory contains examples, demos, and helper scripts for using the Open-RL framework. These are not part of the core library but serve as recipes for training and evaluation.

## Prerequisites

* **Install [uv](https://docs.astral.sh/uv/):** Follow the official installation guide to install the fast Python package manager.
* **Synchronize Dependencies:** Run the following command to set up the environment:
  ```bash
  cd examples
  uv sync
  ```

---

## Examples Overview

### Supervised Fine-Tuning (SFT)
* **[Hello World SFT Sandbox](sft/hello-world):** A minimal fine-tuning pipeline that trains a model to output a specific, constant target answer (e.g., "foo") for a set of hardcoded questions, serving as an introductory "hello world" for basic API interactions.
* **[Pig Latin Translation](sft/pig-latin):** Teaches a model to perform specialized Pig Latin transformations, demonstrating custom token-level targets and loss masks.
* **[Text-to-SQL SFT](sft/text-to-sql):** Adapts Gemma 3 into a specialized database query assistant capable of generating SQL statements.
* **[FunctionGemma](sft/function-gemma):** A recipe specifically targeted at fine-tuning tool-use capabilities, enabling models to reliably select and invoke functions.

### Reinforcement Learning (RL)
* **[PPO Math Verification](rl):** Implements Proximal Policy Optimization (PPO) with advantages to verify step-by-step math reasoning paths.
* **[RLVR Demo](rl/rlvr):** Showcases Reinforcement Learning with Verifiable Rewards (RLVR) on geography tasks, using deterministic format verification as the primary reward signal.
* **[Text-to-SQL RL](rl/text-to-sql):** Runs the Gemma 4 SFT+RL recipe with SQL execution rewards and curve plotting.
* **[Tinker RL Basic K8s Jobs](rl/tinker-rl-basic):** Example Kubernetes Job manifests for deploying scalable, distributed RL workloads to a multi-tenant cluster.

---

