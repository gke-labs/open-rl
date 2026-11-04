# Harvey LAB RL on GKE

This guide runs the [Harvey LAB recipe](../../examples/harvey_labs/README.md)
at full scale on GKE. The recipe's driver runs as a Kubernetes Job in the
OpenRL training cluster, episode sandboxes run on Agent Sandbox in a separate
cluster, and the judge runs behind an OpenAI-compatible endpoint that the
driver can reach. The sandboxes have no network egress, so only the driver
talks to the judge.

```
training cluster                                    sandbox cluster
  openrl-system: API server, scheduler, workers       Agent Sandbox warm pool
  harvey-labs Job (driver) ──claims, tool calls──────▶ LAB sandbox pods
        │
        └──grading──▶ judge (OpenAI-compatible, e.g. GLM-5.2 on vLLM)
```

## What you need

| Component | What the reference run used |
| --- | --- |
| OpenRL on GKE | [GKE setup](gke-setup.md) with an 8 × H100 80 GB GPU node pool. Four GPUs run the Automodel trainer with context parallelism across 4 GPUs, and four GPUs run vLLM samplers. |
| Sandbox cluster | [LAB sandboxes on Agent Sandbox](../../examples/harvey_labs/agent-sandbox/README.md) in a separate cluster on the same VPC. |
| Judge | GLM-5.2-FP8 on vLLM, served as `gpt-glm-5.2` and reachable from pods in the training cluster. |
| Shared volume | A `ReadWriteMany` PVC in the Job's namespace for logs and checkpoints. `cluster-job.yaml` uses `open-rl-shared-pvc`. |

## 1. Set the context window on the API server

Episodes in this recipe reach up to 196,608 tokens. Because the OpenRL API
server passes its token limits to worker pods only when creating them, set
these environment variables on the API server before starting the run:

```bash
kubectl -n openrl-system set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=196608 OPEN_RL_TRAIN_TOKEN_BUDGET=196608
```

If Workloads for the run already exist, delete them after updating the API
server so OpenRL recreates the worker pods with the new limits.

The Job requests its trainer and samplers through `TINKER_TAGS` with
`openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4,openrl.sampler_replicas=4`.
Without context parallelism, the trainer runs out of GPU memory. One 80 GB H100
fits about 80k tokens of Qwen3.5-9B, and data parallelism does not help because
each GPU still holds a full sequence.

## 2. Build the driver image

[Dockerfile](../../examples/harvey_labs/Dockerfile) packages the recipe client,
the LAB fork at a pinned revision, and LAB's grading environment. Build the
image from the repository root:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE=<registry>/harvey-labs-client,_DOCKERFILE=examples/harvey_labs/Dockerfile,_TAG=<tag> .
```

## 3. Launch the Job

Update [cluster-job.yaml](../../examples/harvey_labs/cluster-job.yaml) with
the driver image you built, `OPENAI_BASE_URL` for your judge, your PVC name,
and `RUN`, the run name used for the log directory. The manifest writes out the
recipe's default arguments explicitly so the Job spec alone reproduces the run.

Apply the Job manifest and stream the driver logs:

```bash
kubectl apply -f examples/harvey_labs/cluster-job.yaml
kubectl logs job/harvey-labs -f
```

The first `create_model` call starts the trainer, and the first sampling call
starts the samplers. Add `openrl.recipe=rl` to `TINKER_TAGS` to start the
samplers together with the trainer instead. Watch the Workloads and pods start:

```bash
kubectl -n openrl-system get workloads,pods -o wide
```

While the Job runs, inspect progress from a pod that runs the driver image and
mounts the shared volume. The driver image has the recipe installed as a module,
not as the `harvey-results` command:

```bash
/app/.venv/bin/python -m harvey_labs.results log_dir=/mnt/shared/harvey-labs/<RUN>
```

## What to expect

On the reference 8 × H100 setup, a full run progresses as follows:

- Once the node has cached the container image and the shared Hugging Face
  cache holds the model weights, the trainer starts within a few minutes.
  Samplers take longer on their first start while vLLM compiles Qwen3.5
  kernels.
- The step-0 evaluation of 50 held-out tasks runs first and takes the longest.
- A training step of 48 episodes took 18 to 35 minutes on 8 H100 GPUs. Sampling
  dominates step time. Routing all turns of an episode to the sampler that
  already holds its prefix cut step time in half. On one 8 × B200 node with the
  same `TINKER_TAGS`, a step after the first took about 29 minutes.
- The held-out pooled criterion pass rate rose from 0.54 at step 0 to 0.78 at
  step 35.

## When episodes score zero

Per-episode `stop/<reason>` metrics record how episodes ended.
`lab/no_output=1` means the episode wrote no deliverable, so there was nothing
to grade. `lab/failed_before_grading=1` means the episode ended before grading
started, for example on a parse error.

| Symptom | Likely cause |
| --- | --- |
| Every episode ends with `context_overflow` before writing deliverables | The context window is too small. A 32k or 64k context window is too small for LAB, and the default recipe uses 196,608 tokens. |
| Many turns hit the per-turn cap (`max_tokens_reached`) | The per-turn token limit is too small for the turn that writes the deliverable. Keep `max_tokens` at 16k or higher. |
| `lab/reward_error` is high | The judge endpoint is unreachable or failing. Check `grading.log` under `<lab_root>/results/<run-id>/` and verify firewall rules between the training cluster and the judge. |
| The trainer is OOM-killed on the first step | The context window exceeds trainer GPU memory. Use the Automodel trainer with `openrl.trainer_cp`, or lower `max_trajectory_tokens` and the API server's limits together. |
| Episodes fail before the first tool call | The sandbox claim failed. Check that the warm pool can hold a step's training episodes plus evaluation tasks, and verify that the driver pod can reach the sandbox pods. |

## Cleanup

Delete the driver Job and remove the run's trainer and sampler Workloads:

```bash
kubectl delete job harvey-labs
kubectl -n openrl-system get workloads   # delete the run's trainer and sampler Workloads
```

The driver releases its sandbox claims when it exits. Any claims that it could
not release expire after 15 minutes.
