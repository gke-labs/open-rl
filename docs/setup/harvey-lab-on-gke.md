# Harvey LAB RL on the GKE scheduler stack

How to run the Harvey LAB LoRA recipe against an OpenRL cluster deployed with
the scheduler (`OPEN_RL_WORKER_MANAGER=scheduler`, see
[FFT time slicing](../fft/time-slicing.md)), with one GPU for the trainer and
one for the sampler. Verified on 2026-09-07 with Qwen3.5-9B on two H100 nodes.

## What the cluster needs

The stack on `main` runs the recipe end to end only with the server fixes on
the `harvey/k8s-stack` branch (all in `src/server` and `src/training`):

| Fix | Why the recipe needs it |
| --- | --- |
| API server env passthrough to worker pods | Scheduler-placed samplers otherwise start at `max_model_len=8192`; LAB episodes need 64k or more. |
| Per-session adapter snapshots | The trainer rewrote one adapter directory in place while the sampler read it over the shared volume. |
| Adapter keys remapped to the hub layout | Qwen3.5 and Gemma 4 hubs wrap the text model; vLLM silently applies no adapter otherwise. |
| `train_unembed` ignored by default | The tinker client asks for an `lm_head` adapter; vLLM rejects the whole adapter for these model families. |
| Sampler host memory: 6 B/param, 3x limit | vLLM 0.25 peaks near 170 GiB of host memory for a minute while warming up Qwen3.5-9B. |
| Chunked target-logprob projection | The full-logits path needs ~60 GiB at 64k tokens; one H100 cannot hold it beside the model. |
| `VLLM_MAX_MODEL_LEN` capped at the model's limit | One API-server-wide value must not crash-loop smaller models. |
| LoRA targets found through PEFT wrappers | The LoRA runtime is shared per base model; the second model to arrive otherwise fails `create_model`. |

Build and roll them from that branch (Cloud Build must be enabled in the project):

```bash
make cloud-build-api-server cloud-build-server GCP_PROJECT=<project>
kubectl set image deployment/open-rl-api-server API server=gcr.io/<project>/open-rl-api-server:<tag>
kubectl set env deployment/open-rl-api-server OPEN_RL_WORKER_IMAGE=gcr.io/<project>/open-rl-server:<tag>
```

Then set the run knobs on the API server. They ride along into every worker pod:

```bash
kubectl set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=131072 OPEN_RL_TRAIN_TOKEN_BUDGET=81920 \
  VLLM_MAX_LORA_RANK=64 VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  ENABLE_GRADIENT_CHECKPOINTING=1 FLA_TILELANG=1
```

`FLA_TILELANG=1` is for Hopper (H100/H200); leave it unset elsewhere. Worker
pods read these when they are created, so change them before a run, or delete
the Workloads so the next run recreates the pods.

## One GPU per role

The scheduler's default `binpack` strategy seats the sampler on the trainer's
claim when a node accepts both roles. Give each role its own node with labels;
unlabeled nodes count as both roles, so disable the rest for the run:

```bash
kubectl label node <trainer-node> openrl.io/sampler=false --overwrite
kubectl label node <sampler-node> openrl.io/trainer=false --overwrite
kubectl label node <every-other-gpu-node> openrl.io/enabled=false --overwrite
```

Qwen3.5-9B needs 80 GB GPUs. The estimator sizes its LoRA trainer at ~21 GiB,
so an enabled L4 node would take it and OOM on the first batch.

## Reaching the API server

The API server service is ClusterIP. From a workstation:

```bash
nohup sh -c 'while true; do kubectl port-forward svc/open-rl-api-server-service 9003:8000 >/dev/null 2>&1; sleep 1; done' &
```

If the driver runs on another machine (it needs Podman and the LAB checkout),
forward the port on: `ssh -N -R 9003:127.0.0.1:9003 <driver-host>`.

`kubectl port-forward` keeps a dead session when the API server pod is replaced
(every `kubectl set env` or `set image` on the deployment does that). Kill and
restart the loop after an API server rollout, then check
`curl http://127.0.0.1:9003/api/v1/healthz` from the driver host; the tinker
client gives up after a few minutes of connection errors.

## Driver host

Follow the recipe [README](../../examples/harvey_labs/README.md): `uv sync`
in `examples/`, run `harvey_labs/setup_lab.sh`, and export the judge
credentials. The default judge is GLM (`judge_model=gpt-glm-5.2`) through an
OpenAI-compatible endpoint: `OPENAI_BASE_URL` and `OPENAI_API_KEY`. For Gemini,
pass `judge_model=gemini-3.5-flash` and export `GEMINI_API_KEY`.

## Run

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run harvey-train \
  base_url=http://127.0.0.1:9003 \
  model_name=Qwen/Qwen3.5-9B renderer_name=qwen3_5 \
  max_trajectory_tokens=81920 max_tool_result_tokens=4096 max_tokens=16384 \
  batch_size=1 rollouts_per_example=4 max_steps=2 task=<area>/<task> \
  log_path=artifacts/harvey-labs/k8s-smoke
```

`max_trajectory_tokens` must not exceed the API server's `VLLM_MAX_MODEL_LEN`.
Drop `task=` for the seeded 300/50 split. Context and generation budgets
decide whether episodes reach the judge at all:

| Setting | What happened |
| --- | --- |
| 32k or 64k context | Every episode ended by `stop/context_overflow` before writing a deliverable: reward floor, nothing graded. |
| `max_tokens=4096` per turn | The turn that writes the deliverable is long; hitting the cap ends the episode ungraded. Use 16k. |
| 96k and 128k context | All episodes completed and graded (pass fraction 0.54 to 0.74 untrained), then the trainer step OOMed on one 80 GB H100. Activation CPU offload does not help: the failing 11.7 GiB allocation is a per-layer transient of the eager GDN path, which the server image falls back to without causal-conv1d and the fla kernels. |
| 80k context | Works on one H100 as configured above: 3 of 4 episodes graded, mean pass fraction 0.52, reward 0.49, optimizer step done. |

Per-episode `stop/<reason>` metrics in the cookbook output say how episodes
ended; `lab/failed_before_grading=1` means the reward function never ran. The first `create_model` creates a
trainer Workload, the first sampling session a sampler Workload; watch them:

```bash
kubectl get workloads
kubectl get pods -l app=open-rl-worker -o wide
```

## What to expect

- Trainer up in 2-3 minutes (image already on the node, model in the shared HF cache).
- Sampler up in 8-10 minutes: vLLM's warmup of the Qwen3.5 GDN kernels takes
  about 8 minutes and peaks near 170 GiB of host memory on the way. The
  API server waits 300 s for sampler readiness on the first session; the client
  retries, so the first step is slow but does not fail.
- An 80k step with 4 rollouts of one task took 11 minutes end to end: 8.5
  sampling and grading, 2.8 training. The recipe defaults to 128k; on one
  H100 that needs the server image built with the `fastpath` extra
  (causal-conv1d plus the fla kernels) so Qwen3.5's GDN layers stop running
  the eager fallback, or a second trainer GPU.

## Cleanup

LoRA Workloads outlive the run (the runtime is shared per base model):

```bash
kubectl delete workload lora-<base-model>-0-trainer lora-<base-model>-0-sampler
```

Restore the node labels you changed (`openrl.io/enabled=true`, roles to `true`).
