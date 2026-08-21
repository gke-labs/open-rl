# Operations dashboard

The gateway serves a cluster-operations dashboard at **`/dashboard`** (e.g. `http://localhost:9003/dashboard`
after `make server`). It is for operating and understanding the cluster — W&B owns training curves and
experiment analysis; nothing here duplicates metrics.

## Views

- **Cluster** — a pannable canvas of what is actually running: the gateway and its configured services
  (Redis, shared storage, vLLM) on the left, and real Kubernetes nodes grouped by GPU pool on the right,
  with the actual pods on each node. Edges are drawn only for connections the gateway is configured to
  make. Clicking a pod opens a full-height panel with its status and live logs (close with the ×, a click
  outside, or Escape). Drag to pan, scroll to move, ctrl+scroll to zoom.

  Clicking a GPU pool's header drills into a **pool screen** with a full-width duty-cycle chart:
  allocation duty — GPUs claimed by non-terminal pods (`nvidia.com/gpu` requests, or DRA resource
  claims) divided by pool capacity — sampled into an in-memory ring buffer each time the cluster is
  polled (throttled to one sample per 5s, last 120 kept for the gateway's lifetime). The chart is
  **stacked by job**: claims are attributed to runs via the `timeslice.io/job-id` pod labels (a run's
  trainer and sampler pods share one band; unlabeled pods fall back to their `app` label). Bands use a
  CVD-validated categorical palette with a legend naming each job, and hovering shows a per-job
  breakdown at that point in time; the pool's nodes are listed below. Back, Escape, or the Cluster tab
  returns to the canvas with pan/zoom intact. This is truthful scheduler state, not device utilization;
  DCGM owns that.
- **Runs** — minimal lifecycle control: name, run ID, base model, a W&B link when one is recorded, and a
  Stop button when there is something to stop (a worker process, queued work, or labeled pods).
- **Health** — current problems first, then a **Load** section of measured stats (active runs, queued
  requests per run, worker-launch backlog, Redis memory and clients, gateway RSS, disk free, pod and
  GPU totals), then gateway / storage / Kubernetes / visibility checks. Node `MemoryPressure` and
  `DiskPressure` conditions surface under Problems.

The page polls every 8 seconds and updates in place — canvas position, selection, and open log panels
survive refreshes. Manual refresh and a light/dark toggle are in the top bar.

## JSON API and ops CLI

The UI serves humans; the same primitives are exposed as JSON for agents and scripts:

| Primitive | Endpoint | CLI |
| --- | --- | --- |
| health | `GET /api/v1/dashboard/health` | `make ops health` |
| problems | `GET /api/v1/dashboard/problems` | `make ops problems` |
| inspect | `GET /api/v1/dashboard/cluster` | `make ops inspect` |
| runs | `GET /api/v1/dashboard/runs` | `make ops runs` |
| run detail | `GET /api/v1/dashboard/runs/{run_id}?logs=N` | `make ops run <run_id> --logs N` |
| logs | `GET /api/v1/dashboard/pods/{pod}/logs` | `make ops logs <pod>` |
| launch | `POST /api/v1/create_model` | `make ops launch --base-model <model>` |
| stop | `POST /api/v1/dashboard/runs/{run_id}/stop` | `make ops stop <run_id>` |

`dev/tools/ops.py` is stdlib-only and always prints JSON; point it at a remote gateway with
`BASE_URL=http://host:9003`.

## Data sources — everything degrades gracefully

- **Kubernetes** is optional: the `kubernetes` Python client is loaded lazily and the dashboard works
  without it (the Cluster view then shows only gateway-local components and says so). In-cluster
  credentials are tried first, then the local kubeconfig. Listing nodes requires cluster-scope RBAC and
  is skipped when denied; fetching pod logs from the gateway's service account requires the `pods/log`
  verb.
- **Runs** are discovered from Redis queues and model metadata keys, the shared filesystem
  (`$OPEN_RL_TMP_DIR/peft`, `$OPEN_RL_TMP_DIR/checkpoints`), and the gateway's own FFT worker processes.
  A W&B URL is shown when adapter `metadata.json` or model metadata records `wandb_url`.
- **Stop** does only what is truthfully stoppable: terminates the gateway-launched worker process,
  clears the run's Redis queues, and deletes pods labeled `timeslice.io/job-id` for the model. It
  reports exactly which actions it took.

## Demo mode

`OPEN_RL_DASHBOARD_DEMO=1` makes every endpoint return fictional data for developing or demoing the UI
without a cluster. Every payload carries `"demo": true` and the UI shows a banner stating the data is
fictional; demo stop performs no action.
