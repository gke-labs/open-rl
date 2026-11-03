# Harvey LAB RL on GKE

How to run the [Harvey LAB recipe](../../examples/harvey_labs/README.md) at
its full size: the recipe's driver runs as a Kubernetes Job in the cluster
that runs OpenRL, the episodes' sandboxes run on Agent Sandbox in a cluster of
their own, and the judge is an OpenAI-compatible endpoint both can reach.

```
training cluster                                    sandbox cluster
  openrl-system: API server, scheduler, workers       Agent Sandbox warm pool
  harvey-labs Job (driver) ──claims, tool calls──────▶ LAB sandbox pods
        │
        └──grading──▶ judge (OpenAI-compatible, e.g. GLM-5.2 on vLLM)
```

## What you need

| Piece | What the reference run used |
| --- | --- |
| OpenRL on GKE | [GKE setup](gke-setup.md), with a GPU pool of 8 × H100 80 GB: 4 for the Automodel trainer (context parallelism over 4 GPUs) and 4 for vLLM samplers |
| Sandbox cluster | [LAB sandboxes on Agent Sandbox](../../examples/harvey_labs/agent-sandbox/README.md), on the same VPC |
| Judge | GLM-5.2-FP8 on vLLM, served as `gpt-glm-5.2`, reachable from the training cluster's pods |
| Shared volume | A `ReadWriteMany` PVC in the Job's namespace for logs and checkpoints (`open-rl-shared-pvc` in `cluster-job.yaml`) |

## 1. Let the workers fit the window

The recipe's episodes go up to 196,608 tokens. Worker pods take their limits
from the API server's environment when they are created, so set them before
the run starts, or delete the run's Workloads afterwards so they are recreated:

```bash
kubectl -n openrl-system set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=196608 OPEN_RL_TRAIN_TOKEN_BUDGET=196608
```

The Job asks for the trainer and samplers it needs through `TINKER_TAGS`:
`openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4,openrl.sampler_replicas=4`.
Without context parallelism the trainer runs out of memory: one 80 GB H100
fits about 80k tokens of Qwen3.5-9B, and data parallelism doesn't help because
each GPU still holds a whole sequence.

## 2. Build the driver image

[Dockerfile](../../examples/harvey_labs/Dockerfile) packages the client, the
LAB fork at a pinned revision, and LAB's grading environment. From the
repository root:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE=<registry>/harvey-labs-client,_DOCKERFILE=examples/harvey_labs/Dockerfile,_TAG=<tag> .
```

## 3. Launch the Job

Edit [cluster-job.yaml](../../examples/harvey_labs/cluster-job.yaml): the image
you built, `OPENAI_BASE_URL` for your judge, the PVC, and `RUN` (the run's
name, which becomes its log directory). The arguments are the recipe's
defaults written out, so a run is reproducible from the manifest alone.

```bash
kubectl apply -f examples/harvey_labs/cluster-job.yaml
kubectl logs job/harvey-labs -f
```

The first `create_model` starts the trainer; the first sampling call starts
the samplers. Watch them come up:

```bash
kubectl -n openrl-system get workloads,pods -o wide
```

Read results from the shared volume while the run goes, from any pod that
mounts it:

```bash
uv run harvey-results log_dir=/mnt/shared/harvey-labs/<RUN>
```

## What to expect

- The trainer is up in a few minutes once the image is on the node and the
  model is in the shared Hugging Face cache. Samplers take longer the first
  time, while vLLM compiles Qwen3.5's kernels.
- The step-0 eval of 50 tasks comes first and takes the longest.
- A training step of 48 episodes took 18 to 35 minutes on 8 H100s. Sampling
  dominates, and routing each episode's turns to the sampler that already
  holds its prefix halved the step time.
- Held-out pass rate went from 0.54 at step 0 to 0.78 at step 35.

## When episodes score zero

Per-episode `stop/<reason>` metrics say how episodes ended, and
`lab/failed_before_grading=1` means the grader never ran.

| Symptom | Likely cause |
| --- | --- |
| Every episode ends on `context_overflow` before writing | The window is too small. 32k or 64k is not enough for LAB; the recipe uses 196,608. |
| Episodes end on `max_tokens_reached` | The per-turn cap is too small for the turn that writes the deliverable. Keep `max_tokens` at 16k or more. |
| `lab/reward_error` is high | The judge is unreachable or failing. Check `grading.log` under `<lab_root>/results/<run-id>/` and the firewall between the cluster and the judge. |
| The trainer is OOM-killed on the first step | The window doesn't fit the trainer. Use the Automodel trainer with `openrl.trainer_cp`, or lower `max_trajectory_tokens` and the API server's limits together. |
| Episodes fail before their first tool call | The sandbox claim failed. Check that the warm pool has room for a step's episodes plus its eval tasks, and that the driver can reach the sandbox pods. |

## Cleanup

```bash
kubectl delete job harvey-labs
kubectl -n openrl-system get workloads   # delete the run's trainer and sampler Workloads
```

The driver releases its sandbox claims when it stops; claims it could not
release expire on their own after 15 minutes.
