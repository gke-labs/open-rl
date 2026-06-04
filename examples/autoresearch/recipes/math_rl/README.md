# Math-RL Autoresearch Recipe

This recipe is the OpenRL/Tinker analogue of
[vivekvkashyap/autoresearch-rl](https://github.com/vivekvkashyap/autoresearch-rl).
This recipe uses the same minimal recipe contract as text-SQL, with a different
TOML command. The agent edits one file, `config.toml`; `autoresearch.toml`
declares this recipe's fixed OpenRL/Tinker command.

```toml
command = "python -m recipes.math_rl.train config=recipes/math_rl/config.toml run_dir={run_dir} attempt_name={attempt_name} base_url=$TINKER_BASE_URL attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["recipes/math_rl/config.toml"]
metric = "accuracy"
```

`train.py` is this recipe's command target. It discovers the served model from
the OpenRL/Tinker backend; `config.toml` intentionally contains only training
knobs. A different recipe can use any command declared in TOML as long as it
writes the configured metric to `metrics.jsonl`.

Unlike the original prime-rl setup, this recipe does not allocate two GPUs per
agent. Agent pods call a shared OpenRL gateway via `TINKER_BASE_URL`;
the cluster-side model/trainer stack owns the model and GPU placement.

Mapping from `autoresearch-rl`:

- `config.toml`: the only editable agent file
- `train.py`: this recipe's fixed training script
- `autoresearch.toml`: command, editable files, and graphed metric
- `program.md`: human-owned instructions for the agent
- cluster setup: provided by the parent GKE guide and Kustomize overlays

Use the parent [autoresearch README](../../README.md) for the common cluster run
flow.

## Local Attempt Run

From `examples/autoresearch`, with an OpenRL gateway reachable on a port:

```bash
export TINKER_BASE_URL=http://127.0.0.1:9003
uv run python -m harness.attempt \
  recipe=recipes/math_rl/autoresearch.toml \
  agent_id=local-math \
  attempt_timeout_minutes=5 \
  baseline=True \
  log_root=artifacts/autoresearch \
  run_name=math-rl
```

Serve the UI for local artifacts:

```bash
uv run python -m harness.serve \
  log_root=artifacts/autoresearch \
  run_name=math-rl \
  port=8080 \
  serve=True
```

Clear local artifacts:

```bash
rm -rf artifacts/autoresearch/math-rl
```

## Kubernetes Run

This assumes you followed the parent GKE setup path for the cluster, shared
storage, vLLM worker, trainer worker, and OpenRL gateway. Then add the
autoresearch agents and UI:

The parent autoresearch manifests require the Agent Sandbox CRD:
`agents.x-k8s.io/v1alpha1/Sandbox`.

```bash
cd examples/autoresearch
uv run python -m harness.cli recipes/math_rl session_name=alpha
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The CLI copies this recipe into a generated overlay under `.runs/` and runs
`kubectl apply -k` for you.

For a single-command demo, use the convenience overlay that composes the normal
OpenRL backend with the autoresearch add-on:

```bash
kubectl apply -k examples/autoresearch/recipes/math_rl/gke
kubectl wait --for=condition=available deployment/vllm-worker --timeout=20m
kubectl wait --for=condition=available deployment/open-rl-gateway --timeout=5m
kubectl wait --for=condition=available deployment/open-rl-trainer-worker --timeout=20m
cd examples/autoresearch
uv run python -m harness.cli recipes/math_rl session_name=alpha
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The shared base init container waits until vLLM, the trainer worker, and the
gateway health endpoints are reachable before the agent starts.

The agent pods use the in-cluster gateway URL:

```text
TINKER_BASE_URL=http://open-rl-gateway-service:8000
```

## Overlay Settings

The math-RL overlay sets:

- `RECIPE=recipes/math_rl/autoresearch.toml`
- `LOG_ROOT=/mnt/shared/open-rl/autoresearch`
- `RUN_NAME=math-rl-alpha`
- `ATTEMPT_TIMEOUT_MINUTES=5`
- `AGENT_TIMEOUT_MINUTES=10`
- `TINKER_BASE_URL=http://open-rl-gateway-service:8000`
- `BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` in the composed GKE stack, owned by
  the backend deployment rather than the recipe config

The recipe `program.md` tells each agent sandbox to tune `config.toml`, run
`RUN_ATTEMPT_COMMAND`, keep concise notes, and keep only improved commits.
