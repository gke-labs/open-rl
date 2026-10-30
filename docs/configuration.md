# Configuration

OpenRL is configured with environment variables. The examples below use plain
shell commands so they work even if `make` is not installed. The root
`Makefile` wraps the same commands for convenience.

## Run outside Kubernetes

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Start the API server and trainer in one process, sampling with torch:

```bash
BASE_MODEL=google/gemma-4-e2b \
uv run --extra cpu python -m uvicorn server.api_server:app --host 127.0.0.1 --port 9003
```

Because `REDIS_URL` is unset, the API server runs the trainer loop in its own
process and samples with torch. This works on a laptop without a GPU.

To train and sample on GPUs with vLLM, set `REDIS_URL`. The API server then
launches a trainer and a vLLM sampler as separate worker processes when a client
creates a model, and they share queues through Redis:

```bash
REDIS_URL=redis://127.0.0.1:6379/0 \
BASE_MODEL=google/gemma-4-e2b \
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
TRAINER_CUDA_VISIBLE_DEVICES=0 \
SAMPLER_CUDA_VISIBLE_DEVICES=1 \
uv run --extra gpu python -m uvicorn server.api_server:app --host 127.0.0.1 --port 9003
```

The equivalent Makefile shortcuts are:

```bash
make server
REDIS_URL=redis://127.0.0.1:6379/0 VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
  TRAINER_CUDA_VISIBLE_DEVICES=0 SAMPLER_CUDA_VISIBLE_DEVICES=1 make server
```

## Core variables

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | unset | Hugging Face model id loaded by the trainer and, when using vLLM, by the sampler. |
| `SAMPLING_BACKEND` | `torch` in one process, `vllm` when `REDIS_URL` is set | Sampling backend selector. `torch` samples in the training process. `vllm` samples through the sampling pools described below and needs `REDIS_URL`. |
| `REDIS_URL` | unset | Enables distributed mode by switching the request store to Redis. Leave unset for a single-machine run. |
| `OPEN_RL_FUTURE_TTL_S` | `300` | How long resolved request results stay readable by `retrieve_future` after a worker resolves them. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root directory for adapter snapshots under `peft/` and saved states under `checkpoints/`. Required with `OPEN_RL_WORKER_MANAGER=scheduler`, set to the shared volume every worker pod mounts. |
| `OPEN_RL_TRAIN_TOKEN_BUDGET` | `auto` | Maximum `batch_size * max_sequence_length` for padded trainer chunks inside one `forward_backward` request. `auto` uses the smaller of 16384 and the tokens whose activations fit in half of the free device memory, read at each request. A number is used as given; `0` runs one datum per pass. |
| `TRAINER_CUDA_VISIBLE_DEVICES`, `SAMPLER_CUDA_VISIBLE_DEVICES` | unset | GPUs given to the trainer and sampler worker processes the local worker manager launches. |

## Worker manager

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_WORKER_MANAGER` | `local` | How the API server starts trainer and sampler workers: `local` runs them as subprocesses, `scheduler` creates `Workload`s for the OpenRL scheduler on Kubernetes, and `none` leaves workers to something else. |
| `OPEN_RL_ACCEL_TIMESLICER_SOCKET` | `/tmp/open-rl/accel-timeslicer.sock` | Unix socket path for a local accelerator time-slicer. Used when `OPEN_RL_ACCEL_TIMESLICER_HOST` is unset. |
| `OPEN_RL_ACCEL_TIMESLICER_HOST` | unset | Node-local accelerator time-slicer host for Kubernetes workers. When set, the worker uses TCP instead of the Unix socket; Kubernetes sets this from `status.hostIP`. |
| `OPEN_RL_ACCEL_TIMESLICER_PORT` | `9753` | Node-local accelerator time-slicer TCP port for Kubernetes workers. |

For local FFT subprocess mode, start `python -m accel_timeslicer.serve` before the
workers run. The local launcher tags each worker with a time-slice job id and
starts it in its own process group so the CUDA checkpoint backend can discover
the active GPU PIDs. Kubernetes deploys the equivalent process with the
`open-rl-accel-timeslicer` DaemonSet, which layers on top of the llm-d snapshot
backend by default for physical checkpoint/restore.

## vLLM variables

| Env var | Default | What it does |
| --- | --- | --- |
| `VLLM_ARCHITECTURE_OVERRIDE` | unset | Optional architecture override passed to the in-repo vLLM worker. Gemma 4 examples use `Gemma4ForCausalLM`. |
| `VLLM_ENABLE_MULTIMODAL` | `0` | By default the samplers pass `limit_mm_per_prompt={"image": 0, "video": 0}`. Text checkpoints published as `*ForConditionalGeneration` otherwise make vLLM reserve a multi-GiB encoder cache during startup that no OpenRL code path can use, which can OOM engine init. Gemma 4 samplers also load the text-only `Gemma4ForCausalLM`. Set to `1` to restore stock vLLM behaviour and keep the full model. |

## Client variables

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` | `http://127.0.0.1:9003` | Base URL used by example clients and scripts. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the Tinker SDK. Local OpenRL does not enforce auth. |
| `HF_TOKEN` | unset | Required for gated Hugging Face models. `uv run hf auth login` is the easiest setup path. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout for debugging. |

## Kubernetes deployment

On Kubernetes the release bundles set these variables. The API server runs with
`REDIS_URL` and `OPEN_RL_WORKER_MANAGER=scheduler`; it creates a `Workload` for
each trainer and sampler, and the OpenRL scheduler starts their pods. The API
server writes each worker's environment into the `Workload`'s pod template, so
there are no worker deployments to configure by hand.

## Sampling pools

Sampling goes through sampling pools. A pool serves one FFT job, one exclusive
LoRA job, or every shared LoRA job on one base model. Pending requests stay in a
Redis stream until a result is saved, so the Redis in `k8s/deploy/base` keeps an
append-only file on a persistent volume and never evicts keys.

A pool that one job owns runs vLLM in the sampler's own process when it has one
replica. A pool with two or more replicas (`openrl.sampler_replicas`), and every
pool that shared LoRA jobs use, runs `vllm serve` Workloads behind one CPU
Deployment, named after the pool, that runs the dispatcher, Envoy and the llm-d
endpoint picker (configured by the `open-rl-sampler-router` ConfigMap in
`k8s/deploy/base`). A shared pool grows to the most replicas any of its jobs asks
for. On one machine every pool has one replica, in process. See
[the sampling pools design](designs/017-sampling-pools.md).

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_DISPATCH_CONCURRENCY` | `256` | Requests one pool keeps in flight. |
| `OPEN_RL_SAMPLE_DEADLINE_SECONDS` | `3600` | How long a sampling request waits for a result before it fails. Set it on the API server. Worker pods and pool dispatchers get the same value. |
| `OPEN_RL_DISPATCHER_IMAGE` | the worker image | Image for each pool's dispatcher container. The release sets it to the API-server image, which carries the dispatcher without the GPU stack and pulls in seconds. |

## Inspection API

The API server serves a read-only inventory of runs at
`/api/v1/inspect/inventory`. It reads runs and their live sessions from the
same Redis as the rest of the server.

| Env var | Default | What it does |
| --- | --- | --- |
| `REDIS_URL` | unset | The Redis that holds model records and live sessions. Unset, the inventory has no runs. |

### Metrics and logs on GKE

The `lora`, `fft` and `distributed-fft-timeslice` bundles ship an NVIDIA DCGM
exporter (`k8s/deploy/dcgm`). GKE's managed DCGM exporter skips DRA node pools,
so without it those nodes report no GPU metrics.

The steps below are for the inspection API's metrics and logs reads. It reads
Cloud Monitoring and Cloud Logging as the API server's Kubernetes
ServiceAccount, `open-rl-sa`, through Workload Identity. Enable Workload
Identity when you create the cluster (`--workload-pool=<project>.svc.id.goog`).
Enabling it on an existing cluster replaces its nodes.

```bash
PRINCIPAL=principal://iam.googleapis.com/projects/<number>/locations/global/workloadIdentityPools/<project>.svc.id.goog/subject/ns/openrl-system/sa/open-rl-sa
gcloud projects add-iam-policy-binding <project> --role=roles/monitoring.viewer --member="$PRINCIPAL"
gcloud projects add-iam-policy-binding <project> --role=roles/logging.viewer --member="$PRINCIPAL"
```

`<number>` is the project number. If you read logs through a log bucket view,
also grant `roles/logging.viewAccessor` on that view.

## Run telemetry

Trainers, samplers and the API server record each run in Redis.
`python -m server.dashboard` serves the records read only under
`/api/v1/inspect`; `/docs` describes each route. Design Doc 018 explains them.

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_OPS_SAMPLE_LIMIT` | `2000` | Newest command records kept per run, for trainer ops and samples separately. The scheduler passes it to worker pods. |
| `OPEN_RL_OPS_TTL_DAYS` | `7` | Days a record lives after its last write. `0` keeps records. The scheduler passes it to worker pods. |
| `OPEN_RL_INSPECTION_API` | unset | No longer read: since #388 the API server mounts `/api/v1/inspect` itself. It lists every run, so keep that server away from clients. On GKE it reads nodes cluster-wide, which `k8s/deploy/base` does not grant; apply `k8s/deploy/dashboard`, whose RBAC binds the same service account. |
| `OPEN_RL_PROMETHEUS_URL` | unset | Prometheus for GPU and vLLM metrics. Unset, the dashboard uses Managed Prometheus on GKE. |
| `OPEN_RL_GKE_PROJECT` / `OPEN_RL_GKE_CLUSTER` / `OPEN_RL_GKE_LOCATION` | from the metadata server | The cluster whose metrics and logs the dashboard reads. `OPEN_RL_GKE_TELEMETRY=false` turns GKE reads off. |
| `OPEN_RL_LOG_VIEWS` | the project | Comma-separated Cloud Logging views to read run logs from, such as a long-retention bucket's view. |
| `OPEN_RL_CLIENT_RUNS_ROOT` | `/mnt/shared` | Where the dashboard looks for each run's client directory, three levels deep. |

### Client metrics and logs

The dashboard reads a run's client metrics and logs from the directory the
tinker-cookbook client writes, its `log_path`. The client sends nothing extra.
On GKE, set each run's `log_path` on the shared volume, such as
`/mnt/shared/<project>/<run>`, which the API server also mounts at
`/mnt/shared`. A run whose directory the server cannot read has no client
metrics or logs.

A directory is a run's when its `checkpoints.jsonl` names `tinker://<run>/`.
Before the first checkpoint, it is a run's when its `config.json` was written
within 60 seconds of the run's creation and names the run's base model. When
several directories match a run and the time does not single one out, none
is used.
