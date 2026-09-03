# OpenRL Roadmap (FY 2026)

OpenRL's mission is to provide a best in class, self-hostable, Kubernetes native platform for post training LLMs.

This roadmap covers FY 2026. It lists the areas we are investing in and the initiatives under each. The order within a section is a rough priority, and we will revisit it as we learn.

---

## Where we are

We think OpenRL is nearing a milestone where it has established a solid foundation. LoRA and full fine-tuning both run behind the same Tinker-compatible API, and delta weight sync, time-slicing and the DRA based scheduler are all on main and will be available in our next release. We have started using the platform ourselves on realistic fine-tuning workloads, and that experience shaped the priorities below.

---

## Key Focus Areas

### Reliability and production readiness

OpenRL is a multi-tenant platform and people will run long, expensive jobs on it. The platform has to make failures loud and recovery boring. In practice most of the time lost on a real workload goes to the environment rather than the model: graders that fail quietly, workers that get preempted, resumes that do not restore everything. We want those failure modes handled by the platform rather than by whoever is watching the run. Key initiatives:

* Production ready guides for installation, configuring and operating self hosted OpenRL covering day-0 and day-2 scenarios. 
* Resiliency to Spot and preemptible workers without operator intervention.
* Secrets handling for model hub and judge credentials through Kubernetes primitives.
* Per-tenant quotas and fairness that are visible and configurable.
* A versioned API and an upgrade path between releases.

### Multi-GPU and advanced topologies support

OpenRL's scheduler provisions trainer and sampler workers on a single GPU today. To support bigger scale RL workloads, we want the trainers and samplers to support multiple GPUs and advanced topologies within a node. We will start with samplers, since vLLM already handles tensor parallelism, then bring the trainer along, then add an engine for the topologies PyTorch alone does not give us. Please note that we are not targeting multi host scenarios in the short term. Key initiatives:

* Extend the scheduler and the Workload spec to provision multi-GPU trainers and samplers with DP/TP configuration.
* Multi-GPU samplers.
* Multi-GPU trainers for full fine-tuning of medium size dense models.
* Add support for the Megatron training engine, which also opens the door to MoE models.
* Publish a recipe for a medium size model such as Gemma 4 31B to demonstrate multi-GPU support.

### Expand accelerator support

OpenRL supports Nvidia GPUs today and we want to expand to other hardware accelerators. Our initial focus will be TPUs, because that forces us to prove the architecture scales to a genuinely different accelerator rather than a different GPU. The bet we are testing is that the four API operations plus a container image per worker are enough of a contract to swap the runtime underneath. Key initiatives:

* Implement a proof-of-concept for TPU support to investigate the changes needed in OpenRL's architecture and design, including the trainer and sampler runtimes, DRA, and what replaces GPU time-slicing.
* Implement MVP support for TPUs.
* Ensure the recipes and guides work well on both GPUs and TPUs, with switching the accelerator being a one line change for the user.

### Expand supported models

OpenRL supports smaller dense models today. We have tested it with dense Qwen and Gemma models out of the box up to the 10B range. We want to support MoE models next, as well as scale up to the 30B range. Some of this depends on the multi-GPU work above, and some of it is about making a new model family "just work" instead of needing changes in the platform. Key initiatives:

* Scale full fine-tuning and LoRA to dense models in the 30B range.
* Support MoE models for both training and sampling.

### Observability, management dashboard and CLI

OpenRL is a multi-tenant system and it is critical for platform engineers as well as AI researchers to be able to observe and manage the post training jobs (and other resources) in the system. It is also important to make this functionality available in a CLI and not just a web interface, because increasingly most of the AI research and system administration is being done by agentic tools and a CLI is critical to facilitate that. We will begin with read-only workflows. Key initiatives:

* CLI to observe system resources: jobs, tenants, workloads, queues and accelerator placement, with output agents can consume.
* Web interface (dashboard) to observe the same resources along with accelerator utilization.
* Per-step training metrics exported in a standard format so users can bring their own dashboards.
* A small set of management operations, starting with cancelling a job.

### Recipes, guides and docs

OpenRL being Tinker-compatible allows our users to use the awesome work the Tinker team has been doing in publishing tutorials and cookbooks. We specifically want to focus on more realistic use-cases such as fine-tuning in the legal and finance domains, and on guides that cover hosting the platform on Kubernetes with different configurations and providers. We also want to keep closing the gap in the Tinker API surface, since that compatibility is the promise everything else rests on. Key initiatives:

* Publish the LAB (Legal Agent Benchmark) recipe with Qwen3.5-9B, Gemma 4 E4B and Gemma 4 31B support.
* Add a recipe in a second domain such as finance.
* Reproduce existing AI research papers and publish them as recipes.
* Widen Tinker compatibility, starting with the checkpoints API.
* Deployment guides for GKE and for Kubernetes with DRA in general, kept in step with the manifests.
* Demo videos.

### Community

There is a lot to be built and we would like to build it with our users and partners in the Kubernetes/CNCF/AI community. We want to be proactive with the community outreach as well as organize the project so others can contribute easily without friction. Some of the initiatives:

* Improve system deployment guides to make it easy for our users and partners to install and configure the system.
* Improve the CI and presubmit tooling to improve the dev loop.
* Have a list of good-first issues and guides for external contributors and partners.
* A public project board that mirrors this roadmap, so status lives in one place.
* Organize regular office hours for users and developers.
* A regular release cadence with release notes.
* Evangelize the project by doing demos, talks and publishing in different avenues.

---

## How to change this

This file is the roadmap; the checklist in the README will be replaced with a link here. If you want something added or reordered, open an issue with the `roadmap` label and say which focus area it belongs to and what done would look like.
