# Text-SQL Autoresearch Recipe

This is the fast deterministic recipe for checking the autoresearch loop. It
uses the minimal recipe contract: `program.md` tells the agent to edit
`train.py`, and `autoresearch.toml` declares this recipe's command and metric.

```toml
command = "python -m recipes.text_sql.train run_dir={run_dir} data_dir={log_root} attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["recipes/text_sql/train.py"]
metric = "accuracy"
```

This recipe samples the configured base model through the shared OpenRL gateway
for both the unmodified default-config attempt and later agent-edited attempts. `prepare.py` owns the fixed
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

From `examples`:

```bash
export TINKER_BASE_URL=http://127.0.0.1:9003
export BASE_MODEL=google/gemma-4-e2b

uv run python -m run_attempt \
  recipe=recipes/text_sql/autoresearch.toml \
  researcher=local-text-sql \
  name=default-config \
  log_root=artifacts/autoresearch/text_sql
```

Serve the UI for local artifacts:

```bash
uv run python -m ui.observer \
  log_root=artifacts/autoresearch/text_sql \
  port=8080 \
  serve=True
```

Open:

```text
http://localhost:8080/experiments.html
```

Clear local text-SQL artifacts:

```bash
uv run python -m run_attempt \
  clean=True \
  log_root=artifacts/autoresearch/text_sql
```

## Kubernetes Run

This assumes your cluster and shared storage already exist. From the repo root:

```bash
kubectl apply -k examples/autoresearch/recipes/text_sql
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The recipe directory customizes the shared `k8s/base` resources through
Kustomize. There is no separate hand-written Kubernetes YAML for this recipe.

## Overlay Settings

The text-SQL overlay sets:

- `RECIPE=recipes/text_sql/autoresearch.toml`
- `LOG_ROOT=/mnt/shared/open-rl/autoresearch/text_sql`
- `TINKER_BASE_URL=http://open-rl-gateway-service:8000`
- `BASE_MODEL=google/gemma-4-e2b`
- `READY_URLS=http://open-rl-gateway-service:8000/api/v1/get_server_capabilities`
- `ATTEMPT_TIMEOUT_MINUTES=30`
- `AGENT_TIMEOUT_MINUTES=10`

The recipe `program.md` tells each researcher sandbox to edit `train.py`, run
`RUN_ATTEMPT_COMMAND`, keep concise notes, and keep only improved commits using
`accuracy`.
