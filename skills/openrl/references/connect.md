# Connecting

## URLs

| From | URL |
|---|---|
| A laptop | `kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000` in a loop, then `http://127.0.0.1:9003` |
| A pod in the cluster | `http://open-rl-api-server-service.openrl-system.svc.cluster.local:8000` |

A port-forward loop that survives server restarts:

```bash
while true; do kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000; sleep 1; done
```

The SDK reads `TINKER_BASE_URL`, or pass `base_url=` to `tinker.ServiceClient`. The server
does not check credentials; set `TINKER_API_KEY` to any string because the SDK refuses to
start without one.

## Which server you are talking to

```bash
curl -s http://127.0.0.1:9003/api/v1/get_server_capabilities | python3 -m json.tool
```

The `openrl` object holds `version` (the release tag, or `sha-<commit>` for a build from
main), `worker_launcher` (`scheduler` on Kubernetes, `process` or `single_process` for local
servers), `sampler_backend`, `fft_enabled`, `models`, `settings_schema_url` (the settings
are described once, as `components.schemas.OpenRLSettings` in `/openapi.json`), `limits` and, on releases, `skill_url` for the copy of this skill that
matches the server. A response without `openrl` is a server older than v0.0.7. The response names the
release but not the worker images; in particular it does not say which Automodel trainer
image a deployment runs, or whether one was set at all (see `lifecycle.md`).

## OpenRL-only endpoints

Not part of the Tinker API; plain HTTP GET/POST with JSON.

| Endpoint | Use |
|---|---|
| `GET /api/v1/openrl/models/{model_id}` | one model: `settings`, `session_live` (a session that created or restored this model is live; on shared workers other jobs' sessions do not count), `shares_workers`, `workers` (below), `workers_error` (set, with `workers` empty, when the server could not ask the cluster; the rest of the view is still valid), and `summary`, one line that says what to know first |
| `POST /api/v1/delete_model` | `{"model_id": "..."}`; releases the model's workers now (an FFT or exclusive job) or frees its adapter (a shared LoRA runtime) |
| `GET /openapi.json`, `/docs` | the full API |

A worker entry in the model view:

| Field | Meaning |
|---|---|
| `role` | `trainer` or `sampler` |
| `state` | one of `waiting_for_placement`, `starting`, `ready`, `failed`, `stopped`; the same five on every server, branch on this. `ready` means the worker has its model loaded and is serving; until then it is `starting` with `loading the model` in `detail` |
| `detail` | free text for a human: the scheduler's own words (Workload phase and reason, node); wording can change, do not parse it |
| `workload`, `node` | names to give the platform owner; `null` on a server that runs workers as processes |
| `age_seconds` | since the worker was requested |
| `summary` | `role state: detail, age` in one line |

The SDK's training client exposes the model id as `trainer.model_id`. The view is scoped
to one model: a model id is a random UUID the server minted for the client that created
it, so knowing it is the capability, as for every Tinker call. There is no list of models
or sessions and no fleet view; those wait for a tenancy design (design 016, deferred).
