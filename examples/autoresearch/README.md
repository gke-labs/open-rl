# OpenRL Autoresearch Demo

This adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
to OpenRL: an agent repeatedly edits one allowed target, runs a bounded
measured attempt, keeps commits that improve the configured metric, and resets
the rest. The same recipe contract works locally or in Kubernetes; in a cluster,
each run can live in its own pod and act as a researcher while sharing the same
storage and OpenRL backend.

## Minimal Recipe Shape

An autoresearch task needs three recipe-owned things:

```text
<recipe>/
  program.md          # instructions for the agent
  autoresearch.toml   # command, editable files, and graphed metric
  thing_to_edit.py    # often also the command target declared in TOML
```

`program.md` tells the agent what to edit, what metric matters, what files are
off-limits, and how to decide keep/reset.

`autoresearch.toml` is the harness contract. It says how to run one attempt,
which files the agent may edit, and which metric decides whether the attempt
improved:

```toml
task = "my_task"
command = "uv run recipe.py run_dir={run_dir} attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["thing_to_edit.py"]
metric = "accuracy"
metric_label = "accuracy"
metric_mode = "max"
```

The command can be any runnable benchmark or training loop. It just needs to:

- accept the args used in `command`, usually at least `run_dir`
- write attempt artifacts under `run_dir`
- exit nonzero on failure
- log the configured metric to `run_dir/metrics.jsonl`

```python
ml_logger.log_metrics({"accuracy": 0.73}, step=1)
```

Use the cookbook `ml_logger` for this; the shared harness treats
`metrics.jsonl` as the only metric source. The command target does not need a
special filename; it can be the editable recipe file itself, a fixed runner, or
`bash run.sh`.

## Included Recipes

Both recipes use the same `program.md` + `autoresearch.toml` contract:

| Recipe | Command Target | Editable | Metric | Guide |
| --- | --- | --- | --- | --- |
| Text-SQL | `recipes.text_sql.train` | `train.py` | `accuracy` | [Text-SQL](recipes/text_sql/README.md) |
| Math-RL | `recipes.math_rl.train` | `config.toml` | `accuracy` | [Math-RL](recipes/math_rl/README.md) |

Use the recipe guides for local one-attempt runs, local UI serving, and
recipe-specific settings.

## Architecture

Autoresearch runs as a small Kubernetes add-on around the shared OpenRL
infrastructure. A recipe overlay starts the UI plus one researcher Sandbox per
researcher. Each Sandbox runs Gemini CLI, edits the recipe, launches attempts,
and calls the shared OpenRL/Tinker services.

![Autoresearch architecture](arch.png)

## Cluster Run

Create the API secret for agent-backed researcher pods:

```bash
kubectl create secret generic researcher-agent-secrets \
  --from-literal=GEMINI_API_KEY="${GEMINI_API_KEY}"
```

Choose one recipe overlay:

```bash
# Fast text-SQL, no model server.
kubectl apply -k examples/autoresearch/recipes/text_sql

# Math-RL add-on. First deploy OpenRL with docs/setup/gke-setup.md,
# or reuse an existing backend at http://open-rl-gateway-service:8000.
kubectl apply -k examples/autoresearch/recipes/math_rl

# Convenience one-shot Math-RL stack: OpenRL backend + autoresearch add-on.
kubectl apply -k examples/autoresearch/recipes/math_rl/gke
```

Each overlay starts one Sandbox that runs one Gemini CLI researcher. If that
process exits nonzero or the pod crashes, the run stops; Kubernetes does not
retry it. The intended recovery is to inspect the UI/logs and start a new run
explicitly.

Open the UI:

```bash
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

```text
http://localhost:8080/experiments.html
```

Use the normal [GKE setup guide](../../docs/setup/gke-setup.md) for cluster,
GPU, storage, and the OpenRL backend. These overlays add researcher sandboxes and
the UI on top of that shared backend.

Researcher pods wait for comma-separated `READY_URLS` before the agent starts, so
early pod startup does not race vLLM, the trainer worker, or the gateway. The
convenience Math-RL stack sets those URLs for vLLM, trainer, and gateway health.

## Shared Pieces

```text
run_research_agent.sh  # launches one timeout-bounded agent
run_attempt.py         # runs one measured attempt and records UI events
ui/observer.py         # read-only UI server over recorded events
k8s/base/              # reusable Sandbox/UI resources
```

`run_attempt.py` runs recipe code and writes artifacts. The UI reads only
`LOG_ROOT/*/ui_events.jsonl`; clearing `LOG_ROOT` resets attempts, live rows,
and per-attempt agent-log cursors.

The launcher passes the recipe-adjacent `program.md` to Gemini as the prompt.
That program tells the agent to edit only the declared target, commit the
attempt, run `eval "${RUN_ATTEMPT_COMMAND}"`, record the metric, and reset if
the metric did not improve.

## Adding A Recipe

Copy one existing recipe directory and update:

- `program.md`
- `autoresearch.toml`
- the command target, if you keep one
- the editable target
- `kustomization.yaml` settings: `RECIPE`, `LOG_ROOT`, and
  `ATTEMPT_TIMEOUT_MINUTES`
- optionally `AGENT_TIMEOUT_MINUTES`, if the researcher pod should stop before
  Kubernetes cleanup does
- optionally `READY_URLS`, if attempts need external services to be healthy
  before the agent starts

The shared wrapper handles logs, diffs, metrics, status, and UI events. Recipe
code should focus on running the benchmark or training loop and emitting the
metric.

## Timeouts And Cleanup

`ATTEMPT_TIMEOUT_MINUTES` caps one measured training/eval run. Every attempt gets
the same value, so scores are comparable.

`AGENT_TIMEOUT_MINUTES` caps the outer Gemini process. One agent can run several
attempts inside this window: run the default config, edit, commit, run attempt,
decide keep/reset, then repeat. Setup happens before this clock starts.

Clean up a session:

```bash
OVERLAY=examples/autoresearch/recipes/text_sql \
  examples/autoresearch/cleanup_research_session.sh
```

To also clear shared run data:

```bash
DELETE_ARTIFACTS=1 \
LOG_ROOT=/mnt/shared/open-rl/autoresearch/text_sql \
OVERLAY=examples/autoresearch/recipes/text_sql \
  examples/autoresearch/cleanup_research_session.sh
```
