# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON API under
`/api/v1/inspect`. Agents use the same routes; `/docs` and `/openapi.json`
describe them, and [Design Doc 018](designs/018-run-diagnostics.md) explains
what each answers.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

`make dashboard` runs `python -m server.dashboard`. Point it at the API
server's Redis with `REDIS_URL`; without it the dashboard has its own empty
store. It reads Kubernetes in `OPEN_RL_WORKER_NAMESPACE`, else
`POD_NAMESPACE`, else `openrl-system`. Run `make dashboard-build` after changing
`src/server/dashboard/ui` and before `uv build`.

On GKE, `kubectl apply -k k8s/deploy/dashboard` deploys it into
`openrl-system` with the API server's image, which builds the UI in a Node
stage. To open it from a laptop:

```sh
kubectl -n openrl-system port-forward svc/open-rl-dashboard 8080:8000
```

Then open `http://127.0.0.1:8080/dashboard`.

The API lists every run in the fleet, so the API server does not mount it by
default. For a trusted local server, run `make dashboard-build`, then
`OPEN_RL_INSPECTION_API=true make server`, and open
`http://127.0.0.1:9003/dashboard`.

Logs need Cloud Logging, nodes and pods need Kubernetes, and GPU and sampler
charts need a metrics backend. A source that cannot be read shows as an
error, never as an empty cluster. [Run telemetry](configuration.md#run-telemetry)
lists the settings that choose the sources. Logs come from the project's
default buckets, usually kept 30 days. To keep them longer, route the
namespace's logs to a bucket and set `OPEN_RL_LOG_VIEWS` to its view, which
needs `roles/logging.viewAccessor`.

## Pages

- **Overview** lists runs with a status label, search, and status filters.
- **Run** shows status, settings, an activity timeline with GPU rows and duty,
  metrics by step, sampler charts, and logs. Click the timeline to open the
  logs around that time.
- **Nodes** shows which process held which GPU and when, each GPU's duty cycle,
  and GPU utilization and memory. Drag or press A/D to pan; use Ctrl/⌘ + wheel,
  pinch, or W/S to zoom.
- **Health** lists pod and node problems and sources that could not be read.

## Lifecycle

Each run in `/inventory` has a `lifecycle`, which says where it is, not whether
it is healthy:

- **ended**: the API server marked it finished, its workers stopped, or it has
  no workers and no live session.
- **waiting**: a worker is not placed yet, or the server has not created one
  for a live session.
- **placed**: every worker is placed.
- **active**: the scheduler cannot be read, so placement is unknown.

## Client metrics and logs

The Run page charts the metrics a tinker-cookbook client writes to
`metrics.jsonl` in its `log_path`, and its logs' `client` source pages the
client's `logs.log`. The client sends nothing extra. On GKE, set the run's
`log_path` on the shared volume, such as `/mnt/shared/<project>/<run>`, which
`k8s/deploy/dashboard` mounts read only. Locally, set
`OPEN_RL_CLIENT_RUNS_ROOT` to the directory above the client's `log_path`.
[Client metrics and logs](configuration.md#client-metrics-and-logs) says how a
directory is matched to a run.
