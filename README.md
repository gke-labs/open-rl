# Open-RL

## The Problem
Reinforcement Learning (RL) for language models is complex distributed system workload. A standard RL loop involves tight, cyclical data dependencies: the training phase requires generated samples, and the sampling phase requires fresh weights produced by the training phase. This is further complicated by:
- **Multi-Turn Trajectories:** Sampling often require multi-turn interactions and tool calls within external environments.
- **Complex Grading:** Evaluating samples may require calling external reward services or prompting secondary models (e.g., in student/teacher distillation).
- **Distribution Shifts:** Maintaining training stability requires meticulous management of data distribution shifts (e.g., tuning on-policy vs. off-policy dynamics).

Traditional Machine Learning frameworks—which were largely optimized for the pre-training era—address this by providing a monolithic architecture. They tightly couple the underlying generation and training infrastructure with the implementation of the core training loop itself. While this provides a simplified configuration experience, it completely removes the flexibility required by modern AI researchers. Furthermore, this monolithic execution typically locks up hardware resources synchronously, causing critical accelerator (GPU) underutilization while waiting on environment steps or grading computations.

## The Key Insight
Recent investigations (inspired by systems like Tinker) prove that it is possible to abstract the complex distributed infrastructure required for RL behind a vastly simplified interface.

By abstracting infrastructure operations into four fundamental primitives, Open-RL allows AI researchers to treat infrastructure as simple, modular building blocks. This decouples the infrastructure layer entirely from the core training loop. As a result, researchers gain the full flexibility to construct arbitrary RL algorithms—without having to fight the underlying framework.

Crucially, this abstraction provides a clean division of responsibilities: AI researchers own the mathematical logic of the training loop, while platform engineers own the underlying, independently scalable distributed infrastructure. 

Furthermore, abstracting these operations behind an asynchronous API means the individual training and sampling computations are completely decoupled from the client. These operations can now be flexibly dispatched to, and scaled across, a shared pool of accelerators. This architectural shift unlocks the **time-slicing** of accelerators among *multiple* concurrent RL jobs. Instead of a single RL job monopolizing a GPU while it waits for synchronous environment steps, the system dynamically interleaves workloads from multiple tenants. This drives hardware utilization significantly higher than what is possible under a single-job, monolithic paradigm.

## The Open-RL Architecture
To realize this vision, Open-RL implements a deeply decoupled, Kubernetes-based architecture designed to maximize both accelerator utilization and developer flexibility:

1. **API Gateway:** The central entry point exposing an asynchronous API for the four key primitives: `train-policy-batch`, `update-policy-weights`, `save-weights-for-sampling`, and `generate-samples`.
2. **Training Sub-system:** Dedicated, horizontally scalable GPU workers focused entirely on executing high-throughput forward/backward passes and optimizer steps.
3. **Sampler Sub-system:** Independently scalable inference workers (powered by state-of-the-art engines like vLLM) optimized specifically for high-speed text generation.
4. **Policy Weights Sub-system:** A robust synchronization layer that distributes policy weights between the trainer and sampler sub-systems.

Because training, inference, and storage are isolated sub-systems, they can be scaled completely independently—and even deployed on fundamentally different hardware architectures tailored to their specific data-flow requirements.

## Researcher UX: The Clean Training Loop
Because the complex infrastructure (VRAM management, LoRA hot-swapping, multi-node communication) is entirely handled by the Open-RL server endpoints, the AI researcher's User Experience (UX) is radically improved. An AI researcher can now orchestrate massive distributed RL jobs by writing clean, imperative Python code directly from their local machine (e.g., a Mac).

Below is pseudocode illustrating how simply a researcher can construct a complex RL loop utilizing Open-RL's building blocks:

```python
trainer = Client(host="http://<self-hosted-svc>")

def rl_loop(trainer, sampler, env, dataset, num_steps):
    # Resume from an existing checkpoint if any
    dataset = dataset.set_step(...)
    
    for step in range(num_steps):
        # 1. Ensure sampler is using fresh weights 
        # (Triggers implicit weight copying via the Policy Weights sub-system)
        sampler = trainer.save_weights_and_get_sampler()
        
        # 2. Fetch initial set of prompts
        prompts = dataset.get_batch()
        envs = build_envs(prompts)
        
        # 3. Distributed Sampling
        # Rollouts run in parallel. The environment can include complex multi-turn logic.
        samples = runRollouts(sampler, envs, batch)
        
        # 4. Grade the Results
        rewards = [grade_sample(sample) for sample in samples]
        
        # 5. Optimize
        training_batch = get_training_batch(prompts, samples, rewards)
        loss_metrics = trainer.forward_backward(training_batch)
        optim_metrics = trainer.update_weights()
        
        print(metrics)

        if step % CHECKPOINT_INTERVAL == 0:
            trainer.save_checkpoint()
```

## Core Constraints & Enablers
- **LoRA (Low-Rank Adaptation):** Open-RL assumes the use of LoRA fine-tuning, which efficiently retains the quality of full fine-tuning for most domains. By anchoring a frozen base model on shared GPUs, the system leverages LoRA to achieve near-instant memory context switching between multiple time-sliced RL jobs.
- **Soft Multi-Tenancy:** The initial design assumes that the datasets and weights for multiple RL jobs can safely reside on high-performance shared infrastructural storage layers.

## Setup Docs
- [Client README](client/README.md)

## Contributing

This project is licensed under the [Apache 2.0 License](LICENSE).

We welcome contributions! Please see [docs/contributing.md](docs/contributing.md) for more information.

We follow [Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Disclaimer

This is not an officially supported Google product.

This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.
