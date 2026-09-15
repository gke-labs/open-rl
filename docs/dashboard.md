# Cluster dashboard

The gateway serves `/dashboard`. The UI and agents read the same read-only JSON under `/api/v1/dashboard`:

- `GET /api/v1/dashboard/snapshot`: runs joined to their workers, current GPU placements, placement history, nodes with their DRA devices, pods, scheduler Workloads and claim ledgers, and per-source errors.
- `GET /api/v1/dashboard/runs/{run_id}`: one run with its pods, workloads and current placements.
- `GET /api/v1/dashboard/runs/{run_id}/metrics`: the worker's own operation records (outcome, duration, queue delay, numeric metrics such as loss) inside `since`/`until`.
- `GET /api/v1/dashboard/runs/{run_id}/logs`: Cloud Logging entries for every pod the run has owned, live or gone. Supports `q`, `pod`, `node`, `severity`, `since`, `until`, `limit`, `cursor`.
- `GET /api/v1/dashboard/pods/{pod}/logs`: current or previous container output from the kubelet for any pod in the namespace.
- `GET /api/v1/dashboard/allocations/{placement_id}/metrics`: DCGM utilization and memory for the allocation's GPUs, from Prometheus.
- `GET /api/v1/dashboard/experiments`: reward, correctness and optimizer curves read from each run's `metrics.jsonl` on the shared volume.

`/docs` describes the parameters. The API does not provide exec, filesystem access, secrets or cluster mutation.

## Where the data comes from

| Data | Source | Survives |
|---|---|---|
| Runs, steps, status | Redis run metadata written by the gateway | Redis |
| Operation metrics | Workers post one record per request; the gateway keeps the last `SAMPLE_LIMIT` per run in Redis | Redis |
| Current placements, nodes, pods, devices | Kubernetes API, read at most every 5 seconds | live |
| Placement history | The gateway records every live placement to Redis every 10 seconds (`open_rl:placement:*`, 7 days) | Redis |
| Run logs | Cloud Logging, scoped to the run's pod names and lifetimes | Cloud |
| GPU utilization and memory | DCGM through a Prometheus endpoint (`OPEN_RL_PROMETHEUS_URL`), matched by GPU UUID | Prometheus |
| Experiment curves | `metrics.jsonl` and `config.json` under `OPEN_RL_TMP_DIR/runs` | shared volume |

Nothing lives only in the gateway's memory. Replacing the gateway pod loses no history.

## Run and device identity

FFT workloads use the logical run ID. LoRA workloads use a shared base-model runtime, so several runs refer to one trainer and one sampler; the UI and API say so with `shared_runtime` and `runtime_run_ids`. Run membership uses Workload ownership, never a pod-name prefix. GPUs are identified by DRA device IDs from ResourceSlices and by their UUID for metrics; a GPU index is never inferred.

## GKE identity

The gateway detects its project, cluster and location from the metadata server (`OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, `OPEN_RL_GKE_LOCATION` override; `OPEN_RL_GKE_TELEMETRY=false` disables). It authenticates with Application Default Credentials. On GKE that means Workload Identity: the gateway's Kubernetes service account needs `roles/logging.viewer` on the project, bound to its `principal://` identity. A credential or scope refusal is remembered for ten minutes and reported as such.

Other clusters can keep the same pages by pointing the run-log source at a collector that answers the same shape; Kind installs without Cloud Logging still have pod logs from the kubelet through `pods/{pod}/logs`.

## GPU metrics

Set `OPEN_RL_PROMETHEUS_URL` to a Prometheus-compatible endpoint that scrapes the DCGM exporter. On GKE with Managed Prometheus that is a `prometheus-engine/frontend` deployment with `--query.project-id`, running as a service account with `roles/monitoring.viewer`. The exporter DaemonSet in `k8s/deploy/distributed-fft-timeslice/10-dcgm-monitoring.yaml` waits for the driver library in an init container; without that, an exporter scheduled onto a freshly created node before the driver installer finishes logs "NVML doesn't exist" and exports nothing until restarted.

## Pages

- **Overview**: every recorded run with status, kind, completed steps and elapsed time, with active runs first. Selecting a run opens its activity comparison.
- **Nodes**: one lane per node, one row per GPU, allocation bars over the selected window from placement history. Click a node or its compact allocation bars to expand a full-width Gantt of recorded runs on that node, with GPU utilization below. The expansion starts with all node GPUs, including jobs on separate cards. GPU buttons filter individual cards. Runs with no recorded activity in the window are omitted; loading and telemetry errors remain visible. Click a run name to open its run page and compare trainer/sampler activity there. Clicking an activity block only highlights that run in the node view; it preserves the selected GPU and time window. The run page inherits the time window and links back to the selected node and GPU. Pinch or Ctrl/⌘-scroll over a timeline to zoom around the pointer; drag empty timeline space to pan. Ordinary scrolling still moves the page. The picker, activity timeline and utilization chart share the same window, from one minute to 24 hours within the snapshot’s retained day. GPU queries wait until gestures finish; cached samples are redrawn while moving. Blocks provide operation context, not per-job utilization measurements or proof that a gap was idle.
- **Scheduler**: workloads waiting for placement with the scheduler's reason, and claim reservations.
- **Experiments**: recipe metrics per run, grouped by sweep directory. Select a run for its reward and correctness curves.
- **Health**: source errors, pod problems and unready nodes.
- **Run**: aligned trainer and sampler activity across nodes, with links to each node's GPU inspector. Expand Metrics for operation timings, worker metrics and the process table. Logs provides Cloud Logging with search, pod filter and paging.

Runs sharing a LoRA process retain distinct colors and links on Nodes. The UI fetches each run’s retained operation records through the snapshot time and clips their intervals locally, so zooming through an operation does not hide it when its completion falls outside the visible window. Their recorded operations determine the colored intervals; GPU utilization still comes from the physical allocation. Short sampler bursts may be narrower than the GPU telemetry's sampling interval.

The node expansion stays below its fleet row and identifies the physical node. It includes every reported GPU, and unmapped placements remain visible when all GPUs are selected. GPU telemetry reuses the allocation metrics endpoint: only enough physical allocations to cover the node's devices are queried, so shared LoRA runs do not duplicate metrics requests. Devices without telemetry retain an unavailable state; the all-GPU average is only shown when every selected physical device has a sample. Idle and CPU nodes can also be expanded. Run-wide process comparison belongs to the run page, where process links lead back to the corresponding nodes.

## Sharing an inspection

The address bar preserves the selected view. **Copy link** freezes its exact time window, including when the current view follows live data. A fresh tab restores the selection; it fetches the same underlying APIs and still requires access to the dashboard.

- `#run/{run_id}/activity` opens a run's process comparison; `/logs` opens logs and `/metrics` expands advanced metrics.
- `#nodes?node={node_name}&gpu={device_id}` opens that node. Omit `gpu` for all GPUs. An optional `placement` highlights the compact allocation that was clicked.
- Older placement links resolve to the placement's node; the former `panel`, `group` and `layout` options are ignored.
- `duration` is the window length in seconds (60–86400); `end` is its end as Unix seconds, including fractions. Without `end`, the window follows the current snapshot.
- Run links also preserve log search `q`, `pod`, an incident's `event` timestamp, and the originating node view in `back`.

Query parameters follow the hash route and use URL encoding. For an agent, convert `end - duration` and `end` to ISO timestamps for the API's `since` and `until` parameters. Short node labels use unique hostname suffixes; full node names remain in tooltips and JSON.

## Recorded preview

`make dashboard-capture` records gateway responses through the port-forward, and `make dashboard-fixture` serves the capture locally at `http://127.0.0.1:9017/dashboard`. The fixture server reads static assets from the working tree and marks responses with `recorded_at`; the UI labels the page as a recording and does not follow live logs.

Time and log filters work within the captured rows. Pagination covers only those rows, and missing run metrics remain explicitly unavailable. The capture defaults to three detailed runs; use `python3 dev/capture_dashboard_fixture.py --runs N` to include more. Captured JSON is never modified by replay.

## Front end

Plain ES modules, no build step. `app.js` routes and polls; `store.js` holds the snapshot and selection; `navigation.js` restores and shares inspection URLs; `cache.js` fetches anything else and re-renders when it lands. Every page is a function from state to markup, patched into the document by `morph` so a refresh keeps scroll, focus and open panels. Charts are markup too: an SVG stretched to its box with HTML axes, so nothing is measured.
