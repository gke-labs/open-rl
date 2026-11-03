# Harvey LAB RL

Train a legal agent on Harvey's Legal Agent Benchmark (LAB) with
tinker-cookbook's GRPO loop on OpenRL. Each episode runs in its own sandbox:
the agent reads the task's source documents, runs commands, and writes
deliverables (`.docx`, `.md`, ...). LAB's judge grades them against the task's
full rubric, and the reward is the fraction of criteria passed.

With the defaults, Qwen3.5-9B went from 0.54 to 0.78 pooled criterion pass
rate on the 50 held-out tasks in 35 steps. A paired re-evaluation with 4
attempts per task measured +14.5 points of rubric quality on graded episodes
(95% CI 10.4 to 18.7), with 45 of 50 tasks improving.

## What you need

| Piece | Default recipe | Notes |
| --- | --- | --- |
| OpenRL server | A GKE deployment ([GKE setup](../../docs/setup/gke-setup.md)) | The 196,608-token window needs a trainer with context parallelism: Automodel on 4 × 80 GB GPUs with CP4, plus 4 vLLM samplers. Eight H100s cover it. |
| Judge | GLM-5.2 behind an OpenAI-compatible endpoint, reasoning off | Any endpoint that serves the model named by `judge_model`. Gemini also works (see [Setup](#setup)). |
| Sandboxes | Podman on the driver's machine | Use [Agent Sandbox](agent-sandbox/README.md) when the driver runs in a cluster. See [Sandbox backends](#sandbox-backends). |
| LAB | The fork that `setup_lab.sh` clones | Tasks, tools, sandbox image, and the grader. |

For a first try without that much hardware, see [Smoke test](#smoke-test).

## Setup

Commands run from `examples/`. From the repository root, add
`--project examples` to `uv run`.

```bash
cd examples
uv sync
harvey_labs/setup_lab.sh
```

`setup_lab.sh` clones the LAB fork into `harvey_labs/harvey-labs` (the
recipe's default `lab_root`) and runs LAB's own setup: uv, LAB's Python
dependencies, pandoc, Podman, and the sandbox image. Set `LAB_ROOT`, `LAB_REPO`
or `LAB_REF` to change where and what it clones.

Point the grader at the judge, in the environment or LAB's `.env`:

```bash
export OPENAI_BASE_URL=http://<judge-host>:8000/v1   # serves judge_model, default gpt-glm-5.2
export OPENAI_API_KEY=unused                          # any value for a self-hosted endpoint
```

For Gemini instead, pass `judge_model=gemini-3.5-flash` and export
`GEMINI_API_KEY`. LAB also asks Claude to match a deliverable to its file when
the file name doesn't make it obvious; that needs `ANTHROPIC_API_KEY`, and
without it those deliverables count as missing.

Check the OpenRL server before a long run:

```bash
uv run openrl-doctor --base-url http://127.0.0.1:9003
```

## Train

Start a port-forward to the OpenRL API server
(`kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000`),
then:

```bash
export TINKER_API_KEY=tml-dummy-key   # OpenRL ignores it; the tinker client requires one
export TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4,openrl.sampler_replicas=4
uv run harvey-train base_url=http://127.0.0.1:9003 log_path=artifacts/harvey-labs/my-run
```

The API server must allow the window: set its `VLLM_MAX_MODEL_LEN` and
`OPEN_RL_TRAIN_TOKEN_BUDGET` to at least `max_trajectory_tokens` (196608)
before the run creates its workers.

```bash
kubectl -n openrl-system set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=196608 OPEN_RL_TRAIN_TOKEN_BUDGET=196608
```

Arguments use `chz`'s `key=value` syntax; `uv run harvey-train --help` and
[config.py](config.py) list every option. The defaults are the recipe:

| | Default |
| --- | --- |
| Model | `Qwen/Qwen3.5-9B`, renderer chosen from the model |
| Batch | 8 tasks × 6 attempts per step (`batch_size`, `rollouts_per_example`), minibatches streamed while sampling continues (`stream_minibatches`) |
| Optimizer | LoRA rank 32, `learning_rate=2e-4` constant, no KL penalty |
| Loss and reward | Importance sampling with group-centered advantages; reward = criteria passed / total, 0 when ungraded |
| Episode limits | `max_turns=40`, `max_tokens=32768` per turn, `max_trajectory_tokens=196608`, `max_tool_result_tokens=16384` |
| Length | `max_steps=35` over 300 train tasks |
| Eval | 50 held-out tasks at step 0 and every 5 steps (`eval_every`), plus a final eval, 1 attempt per task (`eval_rollouts_per_task`) |
| Split | `task_split_seed=0` picks the held-out slice; `train_split_seed=242` orders the train pool. Eval scenario families never appear in training. |

Notes that matter in practice:

- An episode that stops at the per-turn cap or runs out of window is graded on
  what it has written. A malformed tool call ends the episode, scored 0.
- Tool results longer than `max_tool_result_tokens` are cut, with a hint to
  read on from an `offset` or use `grep`.
- 300 train tasks in batches of 8 leave a short last batch at step 38, which
  has hung streamed training before. Keep `max_steps` at 37 or below, or make
  `train_tasks` a multiple of `batch_size`.
- `task=<name>` trains on one task instead of the split.

## Evaluate

Evaluate a saved checkpoint by its `sampler_path` in `checkpoints.jsonl`, or
leave `checkpoint` out to evaluate the base model:

```bash
uv run harvey-eval \
  base_url=http://127.0.0.1:9003 \
  checkpoint=tinker://MODEL/sampler_weights/LABEL \
  log_path=artifacts/harvey-labs/eval
```

With one attempt per task, the same weights scored 5 to 7 points apart from
one eval to the next. To compare two checkpoints, evaluate both with
`eval_rollouts_per_task=4` and the same task split, judge, and episode limits.

## Results

```bash
uv run harvey-results log_dir=artifacts/harvey-labs/my-run
uv run harvey-results log_dir=artifacts/harvey-labs/my-run json=True plot=True
```

Training saves `results.json` and `run_plot.png` on exit, including partial
runs with metrics. `plot=True` refreshes both snapshots; omit it for a
read-only query. Add `metrics=True` with `json=True` for raw diagnostics.
For a custom plot, use `uv run harvey-plot log_dir=... title="My run" out=run.png`.
Extend [results.read_results](results.py) for new analyses instead of writing
another parser.

The plot shows rollout rewards, batch means, an EMA, and evaluation scores.
Steps count **completed training batches**: batch 0 finishes at step 1;
its pre-update eval is step 0. Cookbook's streaming logs retain zero-based
indices, so the last of eight minibatches is shown as `7/8`.

| Artifact | Source |
| --- | --- |
| `<lab_root>/results/<run-id>/scores.json`, `report.html` | LAB's evaluation CLI: rubric verdicts and report. Judge progress goes to `grading.log`. |
| Same directory: `metrics.json`, `config.json`, `tinker_history.jsonl` | Recipe records LAB tool counters, episode settings, and transcript. |
| `<log_path>/metrics.jsonl`, `iteration_*/*_rollout_summaries.jsonl` | Cookbook aggregates and rollout records, including recipe `lab/*` metrics and final eval. |
| `<log_path>/results.json`, `run_plot.png` | Derived summary and plot. |

Eval reports prefer the **pooled criterion pass rate**: total passed / total
criteria. Cookbook stores mean criterion counts; their ratio gives this rate.
For episodes scoring 1/2 and 2/8, pooled success is **30%**, while the mean
of episode percentages is **37.5%**. Legacy episode averages stay separate
in the report and plot.

Pass rate counts unfinished and ungraded episodes as 0, so it mixes "finished
more often" with "wrote better deliverables". To tell them apart, compare
`lab/graded` (the share of episodes the judge graded) and the pass fraction
on graded episodes, not just the pooled rate.

Training reward averages episode rewards and may include termination penalties.
Missing outputs and grading failures score zero. Terminal `lab/*` flags average
over recorded episodes; `parse_error`, `context_overflow`, and
`max_tokens_reached` average over transitions. Check cookbook's error diagnostics
for failures without a rollout record.

## Sandbox backends

| Backend | Use it when | How |
| --- | --- | --- |
| Podman | The driver runs on a machine with Podman | The default; `setup_lab.sh` builds the image |
| Agent Sandbox | The driver runs in a cluster | `sandbox_warmpool=<SandboxWarmPool>`; set up with [agent-sandbox/README.md](agent-sandbox/README.md) |
| Your own | Anything else | Inject a factory, below |

Inject a factory through the Python training or evaluation API:

```python
from harvey_labs.train import RunConfig, run
from my_backend import sandbox_factory

await run(RunConfig(base_url="http://127.0.0.1:9003"), sandbox_factory=sandbox_factory)
```

The factory is shared across train and eval. It receives a `SandboxRequest`
and returns a `LabSandbox` extending cookbook's `SandboxInterface`; see
[sandbox.py](sandbox.py) for the contract. Provide an isolated workspace,
LAB-compatible tools, and binary-safe output collection into the local grading
directory. Remote backends own leases/heartbeats; the judge runs locally.
Clean up failed startup in the factory; after return, the environment group
owns cleanup.

## Run in a cluster

[docs/setup/harvey-lab-on-gke.md](../../docs/setup/harvey-lab-on-gke.md)
runs the driver as a Kubernetes Job next to OpenRL, with sandboxes on Agent
Sandbox. [cluster-job.yaml](cluster-job.yaml) is the Job and
[Dockerfile](Dockerfile) builds its image.

## Smoke test

To check the wiring end to end on one 80 GB GPU per role, train on one task
with a small window:

```bash
uv run harvey-train base_url=http://127.0.0.1:9003 \
  task=<area>/<task> batch_size=1 rollouts_per_example=4 max_steps=2 \
  max_trajectory_tokens=81920 max_tokens=16384 max_tool_result_tokens=4096 \
  log_path=artifacts/harvey-labs/smoke
```

Set the API server's `VLLM_MAX_MODEL_LEN` and `OPEN_RL_TRAIN_TOKEN_BUDGET` to
81920 for this. A smaller window or per-turn cap than this ends most episodes
before they write anything, so a smoke test that scores 0 everywhere usually
means the limits are too tight, not that the setup is broken.

## Development

Use `chz` for recipe configuration, data objects, and CLI entrypoints.

```bash
uv run python -m harvey_labs.renderers.qwen35_renderer
uv run python -m harvey_labs.renderers.gemma4_renderer
```

Each renderer has an inline `unittest` that checks sampled tokens survive
across turns, using the real tokenizer and no model weights or GPU. Training
must see exactly the tokens that were sampled: a renderer that re-renders
earlier turns differently breaks prefix reuse and trains on text the model
never produced. Gemma uses Google's unmodified template and response schema
at a pinned HF revision, with Transformers parsing tool calls; malformed
arguments fail parsing. The first Gemma run downloads these assets into the
HF cache.

For API server configuration, see
[docs/configuration.md](../../docs/configuration.md).
