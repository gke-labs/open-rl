# 018: Run diagnostics

**Status:** Proposed.

A person or an agent asks one question of a run: why is it not behaving? OpenRL
cannot answer it today from the server:

- Command records carry no step, and sample requests evict trainer ops. At the
  default cap of 2,000 records, a run with eight samplers keeps under two steps.
- Where a worker ran is rebuilt from pods, device slices and Workloads, not
  recorded. Samplers record nothing.
- A request lost between the Redis push and the trainer's pop looks the same
  as a client that sends no work.
- Nothing records which worker holds a request, so a request held by a dead
  worker looks like a slow one.

This design records where each process ran, which step each command belongs
to, and which requests are outstanding. A read-only API under
`/api/v1/inspect` joins those records into facts. The dashboard UI and agents
use the same routes.

## Records

Every key is in the API server's Redis. Each record expires
`OPEN_RL_OPS_TTL_DAYS` (default 7) after its last write; `0` keeps it.

| Record | Key | Writer | Holds |
|---|---|---|---|
| Command record | `open_rl:operations:<run>` for trainer ops, `open_rl:operations:<run>:samples` for samples | Trainer, one per op. Sampler, one per failed sample only | `operation`, `status`, `error_type`, `started_at`, `at`, `elapsed_seconds`, `request_id`, `role`, `runtime_id`, `pod`, `pod_uid`, `node`, finite `metrics`, the names of non-finite ones in `nonfinite`, and `step` on trainer ops. The newest `OPEN_RL_OPS_SAMPLE_LIMIT` (2,000) of each list |
| Sample count and time | `open_rl:sampled:<run>`, `open_rl:sampled_at:<run>` | Sampler and torch trainer, on every sample | The run's samples sent, and when the last one finished. A vLLM sampler keeps one command record per group of samples it served, not one per sample. A torch trainer's samples keep theirs |
| Step counter and step times | `open_rl:steps:<run>`, `open_rl:step_times:<run>` | Trainer, on each successful `optim_step` | The run's step count, and the times of its last 64 steps |
| Request counters | `open_rl:ledger:<run>`, one hash | API server (`enqueued`), trainer (`popped`, `answered`, `failed`) | Trainer requests that reached each point |
| Worker record | `open_rl:worker:<id>`, one per process | Every trainer and sampler, rewritten every 5 s by a thread | Pod, pod UID, node, role, runtime, GPU UUIDs (`null` when NVML is unreadable), phase, recent GPU turns, outstanding requests by op and run with the oldest's age, and the settings env with secrets left out. A record not rewritten for 30 s belongs to a gone process |
| Session record | `open_rl:session_record:<id>` | API server, on create, heartbeat and model attach | Created and last-seen times, the session's models, and its redacted metadata |
| SDK errors | `open_rl:sdk_errors:<session>` | API server, from the SDK's telemetry | The last 200 error events, without credentials |
| Placement | `open_rl:placements:<runtime>`, one record per Workload placement | API server, which lists the scheduler's Workloads every 30 s and before each release | Workload, role, pod, node, GPU UUIDs, placed and released times |
| Workload status | `status.gpuUUIDs`, `status.placedAt`, `status.releasedAt` | Scheduler, on bind and release | The source of placement records. The scheduler also exports `openrl_gpu_held{run_id, role, pod, node, UUID}` while a GPU is held and logs one `openrl.placement` line per change |

Each event a writer records also prints one JSON line, so Cloud Logging keeps a
history longer than the Redis TTL. The line carries the Cloud Logging labels
`openrl_source=telemetry` and `openrl_run_id=<run>` or `openrl_session_id=<id>`,
and its `message` is `telemetry <kind>`. A run's kinds are `enqueued`, `popped`,
`answered` and `op` (a command record), all written through
`telemetry.record(run_id, kind, **fields)`. Other kinds are `placement`,
`session_created`, `model_attached` and `sdk_event`. An answered vLLM sample
and a session heartbeat print no line. Fields are redacted before they print.
No route reads these lines.

A run's display name comes from the `name`, `run_name`, `recipe_name` and
`git_rev` keys of its `user_metadata`, kept in its model metadata.

`openrl_gpu_held` joins DCGM series to runs. A time-sliced GPU has one series
per holder, so reduce it per UUID before the join:

```promql
100 * DCGM_FI_PROF_GR_ENGINE_ACTIVE * on(UUID) group_left() max by (UUID) (openrl_gpu_held{run_id="$run", role="trainer"})
```

Utilization comes from `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, which averages the exporter's whole 10 s collection interval.
`DCGM_FI_DEV_GPU_UTIL` covers only about the last second of each interval, so a load with the same period aliases. On an
A100 running 5 s on and 5 s off, it read 88% while engine activity read 51%. The read API falls back to it where engine
activity is missing.

## Routes

`python -m server.dashboard` serves these routes. The API server serves them
too when `OPEN_RL_INSPECTION_API=true`. Every route is a `GET` under
`/api/v1/inspect`.

| Route | Answers |
|---|---|
| `/runs?ended=` | Which run to open: every live run in one line (lifecycle, steps, seconds since its last op and step, median step time, unanswered requests and how many a dead worker holds, samples queued, failures, relaunches), longest since its last op first. Built from the counters, the last sample time and the last record of each list, never the command history |
| `/runs/{id}/summary` | What is wrong with this run: the run facts below, and GPU duty by role over the run's span |
| `/runs/{id}/steps?last=&fields=` | What each step did: trainer ops grouped by step, with each op's count, failures, seconds and metric spread, and in `client` the metrics the client wrote for the step, kept by the `fields` prefixes. `client_directory` names the client directory, or is null when none matched |
| `/runs/{id}/commands?since=&until=&role=` | The raw command records, newest first. A vLLM sampler's samples appear as the groups it served, with a count, and one record per failed sample |
| `/sessions/{id}/summary` | What a client did: its session record, whether it is live, its SDK errors, and a line per model over the session's span |
| `/inventory` | What exists now: runs with their lifecycle (`ended`, `waiting`, `placed`, `active`), nodes with their GPUs, the GPUs held now, pods with problems, and one error per source that could not be read |
| `/activity?run_id=&node_id=&since=&until=` | Who ran where and when: one lane per placement with its GPU turns and recorded op intervals. A sampler lane's samples are the groups it served |
| `/accelerators?run_id=&node_id=&since=&until=` | What each GPU is doing: node, slot, model, every placement holding it, utilization now and over the window, memory used of total |
| `/query?name=&run_id=&node_id=&since=&until=&points=` | A named PromQL query (`gpu_util`, `gpu_util_now`, `gpu_busy`, `gpu_memory_mib`, `gpu_memory_total_mib`, `vllm_*`) over a run's or node's GPUs and pods, or the fleet's GPUs |
| `/logs?run_id=&pod=&q=&severity=&since=&until=&cursor=` | The run's lines, newest first. From Cloud Logging, its worker pods by the scheduler's owner label and the lines an in-cluster client labels `openrl_run_id=<run>` and `openrl_source=client`, without telemetry's own lines; a stderr line takes the level it prints. With `pod=client` and a client directory, its `logs.log`, whose lines carry no time, so the window does not apply and the cursor is a byte offset |

Bounds are ISO-8601 or a span such as `1h`, at most seven days, and default to
the last 30 minutes.

A run's client metrics and logs come from its client directory, the
tinker-cookbook's `log_path`, read where the server mounts it
(`OPEN_RL_CLIENT_RUNS_ROOT`, `/mnt/shared` by default). The client sends
nothing. `metrics.jsonl` holds one line per step with its `step`, read from the
offset the last read stopped at. `logs.log` is paged back from its end. A
directory is a run's when its `checkpoints.jsonl` names `tinker://<run>/`, or
before the first save when its `config.json` was written within 60 s of the
run's creation for the run's base model. When the match is ambiguous, the run
has no client directory. `docs/configuration.md` gives the rules.
`timing_spans.jsonl` in the same directory is a later source for trace export.
Some runs keep client metrics in Redis under `open_rl:run_metrics:<run>`;
`metrics.jsonl` wins a step both have.

The `/runs/{id}` routes, `/activity` and `/logs` also take a session id in place
of a run id. A session with one run answers for that run. A session with several
runs answers 409 and names them.

### Run facts

`/runs/{id}/summary` returns each block only when it has content. An absent
block means nothing of its kind was recorded.

| Block | Holds |
|---|---|
| `steps`, `step_seconds`, `since_last_step_s`, `last_op`, `last_metrics` | Optimizer steps, the median and last gap between them, and the newest trainer op |
| `since_last_op_s`, `trainer_idle_s` | For a live run, seconds since any op and since the trainer's last op |
| `goodput` | The share of the run's span spent training, the rest split into checkpoint, forward-only, rollout (a group of samples being served, or a recorded torch sample in flight) and client (nothing in flight) seconds |
| `requests` | Unanswered trainer requests with the holding pod and whether its worker still reports; the counters, with `lost = enqueued - popped - queued`; samples waiting in the sampler queue |
| `failures` | Failed trainer ops: the count, the 5 newest, and a log read around the newest |
| `relaunches` | Workers replaced in the same pod during the run, with a log read around the first. A clean exit, or a shared runtime's idle pod recreated under its name, is not one |
| `saves` | `save_state` calls, failures, the last success and the steps since |
| `optimizer` | Ops with non-finite metrics and the first of them, and the last 5 grad norms |
| `samplers` | Sends (the sample count), failures (the failed records), and the median time of the groups its samplers served |
| `config` | Workers whose echoed settings differ from what the run asked for, or `no echo` |
| `placement` | Placements recorded, and Workloads with no node yet |
| `incidents` | The 5 newest Kubernetes terminations and scheduling failures on the run's pods |
| `gpu_busy` | Mean GPU duty by role. A GPU that time-sliced roles shared counts under the roles joined, such as `sampler+trainer` |

A log read is a route and its arguments, such as
`{"read": "logs", "args": {"run_id": "…", "pod": "…", "since": "…", "until": "…"}}`,
so an agent follows a failure to its traceback in one call.

## Rules

- **Facts, not verdicts.** No route says whether a run is healthy. There are
  no thresholds; the reader draws the conclusion.
- **Telemetry never raises or blocks.** Every hook suppresses its own errors.
  Every store write runs in the background and is dropped after 5 s. An op
  returns or raises exactly as it would without telemetry, even when the store
  raises, hangs or is missing. `tests/test_telemetry_faults.py` checks this.
- **Reads are bounded and shared.** The fleet snapshot is cached for 5 s and a
  run's command records for 10 s. A stale entry is refreshed by the next
  caller, and every caller gets the last good result until the refresh lands,
  unless that result is older than both 60 s and six windows. A failed read is
  retried by the next caller. The 32 most recently used entries are kept.
- **Logs stay in their run.** A log read takes only pods with the run's owner
  label, so a run ID cannot open another run's logs.
- **The client API does not list the fleet.** The API server mounts the routes
  only with `OPEN_RL_INSPECTION_API=true`.

## Known gaps

- Workers return a NaN loss or grad norm as 0 and infinity as ±9999
  (`sanitize_float`), so those read as numbers in command records. Other
  non-finite metrics are named in `nonfinite`.
- A local stack has no Cloud Logging and no Workloads, so its runs have no
  logs or placement blocks.
- Runs recorded before step stamps, step times or the failure counter leave
  those fields null, and their steps are derived from the order of their ops.
- Sampler latency is the time a group of samples took, not each request's. Runs
  recorded before group records, which kept neither, have no sampler latency,
  rollout split or sample bars. vLLM's per-pod charts stay on `/query`.
- Client metrics and logs need the client's directory on a volume the server
  reads. A run whose `log_path` is not on the shared volume, and a client on a
  laptop, have none. Runs started together on one base model match no directory
  by time, so each has none until its first checkpoint.
- Facts are computed on request. Alerts would need a timer and thresholds.
- Not recorded yet: cgroup OOM kills and peak memory in the worker record, NVML
  utilization for GPU facts without Prometheus, and the request's holder
  incarnation in the counters.
