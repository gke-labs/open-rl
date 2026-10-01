# Math-RL Autoresearch Recipe

This recipe is the OpenRL/Tinker analogue of
[vivekvkashyap/autoresearch-rl](https://github.com/vivekvkashyap/autoresearch-rl).
This recipe uses the same minimal recipe contract as text-SQL, with a different
TOML command. The agent edits one file, `config.toml`; `autoresearch.toml`
declares this recipe's fixed OpenRL/Tinker command.

```toml
command = "python -m recipes.math_rl.train config=recipes/math_rl/config.toml run_dir={run_dir} run_name={run_name} base_url=$TINKER_BASE_URL attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["recipes/math_rl/config.toml"]
metric = "accuracy"
```

`train.py` is this recipe's command target. A different recipe can use any
command declared in TOML as long as it writes the configured metric to
`metrics.jsonl`.

Unlike the original prime-rl setup, this recipe does not allocate two GPUs per
researcher. Researcher pods call a shared OpenRL API server via `TINKER_BASE_URL`;
the cluster-side model/trainer stack owns GPU placement. The composed GKE stack
sets the shared `BASE_MODEL` to `Qwen/Qwen2.5-0.5B-Instruct`, matching the
`autoresearch-rl` base model.

Mapping from `autoresearch-rl`:

- `config.toml`: the only editable agent file
- `train.py`: this recipe's fixed training script
- `autoresearch.toml`: command, editable files, and graphed metric
- `program.md`: human-owned instructions for the agent
- cluster setup: provided by the parent GKE guide and Kustomize overlays

Use the parent [autoresearch README](../../README.md) for the common cluster run
flow.

## Local Attempt Run

From `examples`, with an OpenRL API server reachable on a port:

```bash
export TINKER_BASE_URL=http://127.0.0.1:9003
uv run --no-sync --package open-rl-client python -m run_attempt \
  recipe=recipes/math_rl/autoresearch.toml \
  researcher=local-math \
  attempt_timeout_minutes=5 \
  name=default-config \
  log_root=artifacts/autoresearch/math_rl
```

Serve the UI for local artifacts:

```bash
uv run python -m ui.observer \
  log_root=artifacts/autoresearch/math_rl \
  port=8080 \
  serve=True
```

Clear local artifacts:

```bash
uv run python -m run_attempt \
  clean=True \
  log_root=artifacts/autoresearch/math_rl
```

## Kubernetes Run

Use the normal [GKE setup guide](../../../../docs/setup/gke-setup.md) to deploy
OpenRL, or reuse an existing backend. Then add the autoresearch researchers and
UI:

```bash
kubectl apply -k examples/autoresearch/recipes/math_rl
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The add-on deploys into the current namespace, not `openrl-system`: researchers
are clients of the API server. They keep their logs on their own
`open-rl-autoresearch-shared` volume.

For a single-command demo, use the convenience overlay that composes the normal
OpenRL backend with the autoresearch add-on:

```bash
kubectl apply --server-side -k examples/autoresearch/recipes/math_rl/gke
kubectl -n openrl-system wait --for=condition=available deployment/open-rl-scheduler --timeout=5m
kubectl -n openrl-system wait --for=condition=available deployment/open-rl-api-server --timeout=5m
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The OpenRL stack goes into `openrl-system` and the add-on into the current
namespace. The GKE cluster needs DRA GPU nodes, set up as in the GKE setup guide. The
researcher pods wait on `READY_URLS`, so attempts do not start until the API
server is healthy. Trainer and sampler workers are not running at that point:
the scheduler starts them when the first attempt creates a training client,
so that attempt also waits for the model to load.

The researcher pods reach the API server by its cross-namespace service name:

```text
TINKER_BASE_URL=http://open-rl-api-server-service.openrl-system:8000
```

## Overlay Settings

The math-RL overlay sets:

- `RECIPE=recipes/math_rl/autoresearch.toml`
- `LOG_ROOT=/mnt/shared/open-rl/autoresearch/math_rl`
- `ATTEMPT_TIMEOUT_MINUTES=5`
- `AGENT_TIMEOUT_MINUTES=10`
- `READY_URLS=http://open-rl-api-server-service.openrl-system:8000/api/v1/healthz`
- `TINKER_BASE_URL=http://open-rl-api-server-service.openrl-system:8000`
- `BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` in the composed GKE stack

The recipe `program.md` tells each researcher sandbox to tune `config.toml`, run
`RUN_ATTEMPT_COMMAND`, keep concise notes, and keep only improved commits.
