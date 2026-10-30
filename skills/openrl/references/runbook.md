# Runbook: symptom -> check from the client -> fix -> what to tell the platform owner

This runbook is for symptoms the deployment causes. Once the workers are `ready`, a slow
step, a hang, a renderer problem or an SDK error is the cookbook `debug` skill's territory; the row
below that matches says so. Check first, always: `GET /api/v1/openrl/models/<model_id>` and read `summary`. Compare
worker ages with the durations in `lifecycle.md` before calling anything hung. Without a
model id (the client is still inside `create_model`) the only view is
`kubectl -n openrl-system get workloads`, which needs cluster access; without that,
wait out the normal duration, then ask the platform owner.

| Symptom | Check | Fix | Tell the platform owner |
|---|---|---|---|
| No output for minutes after `create_model` or the first `sample` | the worker's `state`, `detail` and age in the model view, from another shell; the examples print the model id and the `curl` after `create_model` | wait while `waiting_for_placement` or `starting` and younger than the normal duration; a cold node needs 5 to 10 minutes more | nothing |
| Worker `waiting_for_placement`, `detail` `Pending (NoCapacity: ...)` | the reason names what is missing: a node accepting the role, or a device with enough memory | for a trainer: a bigger pool via `openrl.exclusive=true`, else `openrl.trainer_backend=automodel` with `openrl.trainer_gpus=N` to spread it (LoRA only); for a sampler: a bigger device or a smaller model, nothing spreads a sampler. Sizing in `lifecycle.md` | the reason text; which nodes should carry `openrl.io/enabled=true` and the role labels |
| Worker `waiting_for_placement`, `detail` `Pending (WaitingForCapacity)` | nothing more from the endpoint: the fleet is not visible to clients yet | wait; idle workers of other jobs are released about two minutes after their clients exit | how long you have waited; `kubectl -n openrl-system get workloads` shows them who holds the GPUs |
| `starting` for over 10 minutes with `detail` `Placing (PodCreated) on <node>` | `age_seconds`; the view cannot tell an image pull from a pod that cannot start | wait to 15 minutes; a cold node pulls a multi-GiB image and mounts the shared volume | the entry's `node` and `workload`; `kubectl -n openrl-system describe pod` on the worker pod shows `ImagePullBackOff` or a mount error |
| Worker `failed` | `detail`: the scheduler's reason; a sampler failing on a small GPU usually ran out of memory | smaller model, or a larger device via `openrl.exclusive` on a bigger node pool | the `workload` name; the owner reads the pod log |
| A worker `ready` but every request to it fails or hangs past the normal duration | the view cannot see a crash-looping container; it shows the pod's last phase | treat as failed: `delete_model` and start again on a smaller model or a bigger device | the `workload` name; `kubectl -n openrl-system get pods` shows the restart count |
| Client logs `Retrying request ... because of try_again` for many minutes; model view says `no live session and no workers; the model's workers were released` | `session_live: false` in the model view: the session expired once | the run is lost on servers up to v0.0.6. Stop the client, make sure the port-forward is up, start a new run; restore from the last `save_state` path if there is one | time of the gap, so they can see whether the server restarted |
| `Session heartbeat failed for N seconds` in the client log | `curl .../healthz` through the same URL the client uses | if `healthz` fails: the connection, restart the port-forward loop. Under 120 s the run continues; over 120 s see the row above. If `healthz` is fine, the SDK's own thread is blocked: the cookbook `debug` skill, GIL contention | nothing, unless `healthz` itself fails |
| HTTP 400 `The x-open-rl-fine-tuning-type header was removed; set openrl.fine_tuning_type ...` | the client sets a removed header or `full_config` | pass the setting named in the message as `user_metadata` or `TINKER_TAGS` | nothing |
| HTTP 400 `openrl.<name>: unknown setting` or `must be ...` | spelling and value against the schema: `curl -s $OPENRL/openapi.json \| jq .components.schemas.OpenRLSettings` | fix the value; values are strings | nothing |
| HTTP 400 `openrl.exclusive needs a server that launches workers per model` (or `sampler_replicas`, `trainer_backend`, `trainer_gpus`) | `openrl.worker_launcher` in capabilities is not `scheduler` | run against the Kubernetes deployment; these settings need it | nothing |
| HTTP 400 on `create_model` with `openrl.fine_tuning_type=full` saying FFT is disabled | `openrl.fft_enabled` in capabilities | use LoRA, or ask for the FFT release | "please deploy `openrl-fft.yaml`" |
| Samples are token soup or a different model's style | `get_info` for your model returns your base model as `tokenizer_id`; if it does, the renderer is the suspect: the cookbook `debug` skill | if `tokenizer_id` is wrong, report; this was a server bug fixed in v0.0.5 | OpenRL version and the model id |
| Steps are two to three times slower than yesterday | `shares_workers: true` in the model view means other LoRA jobs on the same base model share your trainer; several FFT jobs time-slice a GPU | expected sharing; stagger runs if wall-clock matters. If nothing shares your workers, profile the client: the cookbook `debug` skill | nothing |
| Everything slow or `healthz` failing | `healthz` latency | stop clients and wait | the `healthz` output and time |
| Any other SDK error, a hang with all workers `ready`, output that differs from vLLM | | the cookbook `debug` skill; its error decoder covers the Tinker-level 400s and the SDK's retry behaviour | |

## Where the server-side evidence is, for the platform owner

- API server: `kubectl -n openrl-system logs deploy/open-rl-api-server`. Worker release
  lines read `No live session uses <owner>; tearing its workers down`.
- Workers: pods named `orw-<workload>`, deleted on release, so a post-mortem needs the
  cluster's log store (Cloud Logging on GKE).
- Workloads: `kubectl -n openrl-system get workloads` shows the phases and reasons the
  model view's `detail` quotes, for every job at once.

## Cleanup hygiene

- Normal exit: call `POST /api/v1/delete_model` so the workers go now instead of in two minutes.
- A killed client: nothing to do; the reaper releases its workers.
- Never `kubectl delete workload` under a live run; the client is not told and polls forever
  (issue #246).
