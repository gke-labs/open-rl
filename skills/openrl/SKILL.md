---
name: openrl
description: Use a deployed OpenRL server (a self-hosted, Kubernetes-native Tinker API for LoRA and full fine-tuning) from the client side. Covers only what differs from the hosted Tinker service: connecting to the endpoint, the openrl.* settings, what the cluster is doing while a call blocks, worker lifecycle, OpenRL's own errors, and where the tinker-cookbook skills' hosted-service advice does not apply. Use when a task mentions OpenRL, open-rl, the OpenRL API server, "our Tinker endpoint", TINKER_BASE_URL pointing at a cluster, or debugging a run against one. For how to train (recipes, hyperparameters, evaluation, SDK debugging) use the tinker-cookbook skills; this skill says when to load them.
---

# OpenRL from the client side

OpenRL release: dev

OpenRL (github.com/gke-labs/open-rl) serves the Tinker API on a Kubernetes
cluster your platform team runs. Any `tinker` SDK client or `tinker-cookbook`
recipe works against it; your training loop runs on your machine and the
GPUs do the work on the cluster. Facts here were verified on v0.0.6 plus the
model view added by design 016, on 2026-10-12. The line above is replaced
with the release tag when the skill ships with a release.

## This skill and the tinker-cookbook skills

This skill covers the deployment: how to reach it, its settings, what it is
doing while you wait, and its own errors. How to train is the tinker-cookbook
skills' job, and they are the better reference:

| Question | Skill |
|---|---|
| How do I set up SFT, RL, DPO, distillation; which model, which hyperparameters, how to evaluate, how to read `metrics.jsonl` | tinker-cookbook `research` |
| Why is a step slow or a session hanging once the workers are up; renderer and tokenization problems; what an SDK error means | tinker-cookbook `debug` |
| Anything about connecting to OpenRL, `openrl.*` settings, workers starting, sessions expiring, a 400 that names a setting | this skill |

The cookbook skills live at github.com/thinking-machines-lab/tinker-cookbook
under `skills/`; `references/cookbook.md` says how to install them next to
this one. They assume the hosted Tinker service; the same file lists each
place that matters and what is true on OpenRL instead. Read it before
following cookbook advice about API keys, model choice, checkpoints or
"is Tinker down".

## Quick card

| | |
|---|---|
| Endpoint from a laptop | `kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000`, then `http://127.0.0.1:9003`. Run it in a loop; it drops on server restarts (see runbook). |
| Endpoint in the cluster | `http://open-rl-api-server-service.openrl-system.svc.cluster.local:8000` |
| Auth | None. The SDK insists on a key: `TINKER_API_KEY=tml-dummy-key`. |
| Is it up, which release | `GET /api/v1/healthz`; `GET /api/v1/get_server_capabilities` -> `openrl.version`, `openrl.worker_launcher`, `openrl.fft_enabled`, `openrl.settings_schema_url` |
| What is my run doing | `GET /api/v1/openrl/models/<model_id>`: the model's workers, each with a `state` (`waiting_for_placement`, `starting`, `ready`, `failed`, `stopped`) and a human-readable `detail`; whether its session is live; a one-line `summary`. The model id is `trainer.model_id`. Before you have one, or for the whole cluster, only `kubectl -n openrl-system get workloads` (needs cluster access). |
| One-command check | `uv --project examples run openrl-doctor` from a clone, or `python -m common.openrl_doctor` in the client image |
| Base models | Any Hugging Face model id. `supported_models` lists only the configured default. First use of a model downloads it on the cluster. |
| LoRA or full fine-tuning | `openrl.fine_tuning_type=lora` (default) or `full`; full needs `openrl.fft_enabled: true` in capabilities |
| Precision (required) | every model must set `openrl.trainer_dtype` and `openrl.sampler_dtype`, equal for now: `bfloat16`, or `float32` for full fine-tuning on the `pytorch` trainer. LoRA and `automodel` are `bfloat16` only. No default, no fallback: a GPU that cannot run the dtype fails the worker |
| A model too big for one GPU | first a bigger device with `openrl.exclusive=true`; then `openrl.trainer_backend=automodel` with `openrl.trainer_gpus=N` to shard the trainer across `N` GPUs on one node (LoRA only). The sampler always needs one device that fits the model. Sizing rules in `references/lifecycle.md` |
| Settings | the server's OpenAPI document, `components.schemas.OpenRLSettings` (`/openapi.json`, rendered at `/docs`); `references/settings.md` says how to read and pass them. Pass as `user_metadata` or `TINKER_TAGS=openrl.key=value` |
| Keep the run alive | Keep the client process and the port-forward up. Session TTL is 120 s; workers are released about 2 minutes after the last heartbeat. |
| Release workers now | `POST /api/v1/delete_model {"model_id": ...}` (not in the SDK) |

## Start here

1. **Check the deployment.** `openrl-doctor`, or curl `healthz` and `get_server_capabilities`.
   Read `openrl.version`; a server without an `openrl` section predates v0.0.7 and has no
   model view.
2. **Pick the settings.** Read them from the server: `curl -s $OPENRL/openapi.json | jq
   .components.schemas.OpenRLSettings`; `references/settings.md` says how to read and pass
   them. Every run needs `openrl.trainer_dtype` and `openrl.sampler_dtype`; most need nothing else. Full fine-tuning, exclusive GPUs, more sampler replicas and the
   Automodel trainer are settings, not code.
3. **Write or run the loop with the cookbook skills.** `examples/tiny/tiny_sft.py` and
   `tiny_rl.py` in the repository are complete minimal loops; cookbook recipes run unchanged
   with `base_url=` pointing at the server. `references/cookbook.md` says what to ignore in
   the cookbook's hosted-service advice.
4. **Know what waiting looks like.** `references/lifecycle.md`: which worker starts when,
   what each worker state means, how long each normally lasts, and when workers go away.
5. **When something is wrong**, start from the symptom in `references/runbook.md`. Every
   entry says what to check from the client, the fix, and what to tell the platform owner.
   Symptoms that are not about the deployment hand off to the cookbook `debug` skill.

## Rules that save runs

- Read `summary` from `/api/v1/openrl/models/<id>` before deciding a run is hung. Launch
  the job, then ask the server about the model while a call blocks; the examples print
  the model id and the exact `curl` after `create_model`. Branch on a worker's `state`;
  `detail` is for a human and its wording can change.
- A port-forward that is down for more than two minutes ends the run: the server releases
  the workers and the adapter state in them. Save state (`save_state`) if a run must survive
  that, and restart the forward before relaunching.
- Call `delete_model` when a script exits normally, so the next run does not wait for the reaper.
- Do not delete Workloads or pods under a live run; that strands the client (issue #246).
- Settings are strings on the wire: `"true"`, `"2"`.
- Before choosing a model above about 8B, do the sizing in `references/lifecycle.md`:
  a trainer that fits nowhere waits in `waiting_for_placement` with `NoCapacity`, it does
  not fail.

## Files

| Path | Contents |
|---|---|
| `references/cookbook.md` | which tinker-cookbook skill to use for what, and where their hosted-service advice differs on OpenRL; the supported Tinker surface |
| `references/settings.md` | where the settings are described (the server's OpenAPI document) and how to pass them; no copy of the list |
| `references/connect.md` | URLs, which server you are talking to, the model view and its fields |
| `references/lifecycle.md` | workers, their states, normal durations, release and cleanup |
| `references/runbook.md` | OpenRL symptoms: check from the client -> fix -> what to tell the platform owner |
