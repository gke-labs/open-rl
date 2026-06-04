# Text-SQL Autoresearch Recipe

This is the fast deterministic recipe for checking the autoresearch loop. It
uses the minimal recipe contract: `program.md` tells the agent to edit
`train.py`, and `autoresearch.toml` declares this recipe's command and metric.

```toml
command = "python -m recipes.text_sql.train run_dir={run_dir} data_dir={run_root} attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["recipes/text_sql/train.py"]
metric = "accuracy"
```

This recipe samples the configured base model through the shared OpenRL gateway
for both the unmodified baseline attempt and later agent-edited attempts. `prepare.py` owns the fixed
dataset and scoring helpers; `train.py` is the editable runnable attempt.

Use the parent [autoresearch README](../../README.md) for the common cluster run
flow. This page only covers text-SQL-specific run behavior and settings.

`train.py` samples a fixed Text-SQL dataset slice through `prepare.py`, matching
the core GRPO script's bounded split: 5,000 train examples and 50 held-out
SQLite-executable scoring examples. The default training path is a small GRPO
Tinker loop over the train split, then eval samples the trained adapter and
scores its SQL completions by execution `accuracy`. The dataset filter keeps
examples with seed data and row-returning target queries, so scoring compares
returned rows.

## Local Run

From `examples/autoresearch`:

```bash
export TINKER_BASE_URL=http://127.0.0.1:9003

uv run python -m harness.attempt \
  recipe=recipes/text_sql/autoresearch.toml \
  agent_id=local-text-sql \
  baseline=True \
  log_root=artifacts/autoresearch \
  run_name=text-sql
```

Serve the UI for local artifacts:

```bash
uv run python -m harness.serve \
  log_root=artifacts/autoresearch \
  run_name=text-sql \
  port=8080 \
  serve=True
```

Open:

```text
http://localhost:8080/
```

Clear local text-SQL artifacts:

```bash
rm -rf artifacts/autoresearch/text-sql
```

## Kubernetes Run

This assumes you followed the parent GKE setup path for the cluster, shared
storage, vLLM worker, trainer worker, and OpenRL gateway. From the repo root:

The parent autoresearch manifests require the Agent Sandbox CRD:
`agents.x-k8s.io/v1alpha1/Sandbox`.

```bash
cd examples/autoresearch
uv run python -m harness.cli recipes/text_sql session_name=alpha
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The CLI copies this recipe into a generated overlay under `.runs/` and runs
`kubectl apply -k` for you. The recipe directory can still keep a checked-in
Kustomize overlay for defaults, but normal launches should use the CLI.

## Overlay Settings

The text-SQL overlay sets:

- `RECIPE=recipes/text_sql/autoresearch.toml`
- `LOG_ROOT=/mnt/shared/open-rl/autoresearch`
- `RUN_NAME=text-sql-alpha`
- `TINKER_BASE_URL=http://open-rl-gateway-service:8000`
- `ATTEMPT_TIMEOUT_MINUTES=30`
- `AGENT_TIMEOUT_MINUTES=10`

The recipe `program.md` tells each agent sandbox to edit `train.py`, run
`RUN_ATTEMPT_COMMAND`, keep concise notes, and keep only improved commits using
`accuracy`.
