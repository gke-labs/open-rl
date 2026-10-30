# Harvey LAB RL

Train on Harvey's Legal Agent Benchmark using tinker-cookbook's GRPO loop.
The model completes tasks in a sandbox; LAB's judge scores the deliverables
against the full rubric. The default model is `Qwen/Qwen3.5-9B`.

## Setup

Start an Open-RL API server, then install the client and LAB environment:

```bash
cd examples
uv sync
harvey_labs/setup_lab.sh
```

Setup clones LAB and prepares Podman, pandoc, and the sandbox image. Configure
credentials for the judge and `ANTHROPIC_API_KEY` for LAB's deliverable
matcher, in the environment or LAB's `.env`. The default judge is GLM through
an OpenAI-compatible endpoint: export `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
For Gemini instead, set `judge_model=gemini-3.5-flash` and `GEMINI_API_KEY`.

Commands below run from `examples/`. From the repository root, add
`--project examples` to `uv run`. Arguments use `chz`'s `key=value` syntax;
use `--help` or [config.py](config.py) for all options.

## Train and evaluate

```bash
TINKER_API_KEY=tml-dummy-key uv run harvey-train \
  base_url=http://127.0.0.1:9003 \
  log_path=artifacts/harvey-labs/my-run
```

- The defaults are the Qwen3.5-9B recipe that raised held-out reward from 0.57 to 0.72-0.74 in 30 steps: 48 rollouts a step, 32k tokens a turn in a 192k window, LoRA at lr 2e-4. That window needs the gateway's `VLLM_MAX_MODEL_LEN` and `OPEN_RL_TRAIN_TOKEN_BUDGET` at 196608 and a trainer that fits it, such as `TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4`.
- `model_name` selects the model. The sampler's context window must support `max_trajectory_tokens`.
- `task=<name>` selects one training task. Otherwise, the seeded split defaults to 300 train / 50 eval tasks and excludes eval scenario families from training.
- Cookbook evaluates at step 0 when evaluation is enabled. `final_eval=True` is the default; `eval_rollouts_per_task` (default 1) sets repeats per eval task.
- `stream_minibatches=True` (the default) overlaps sampling and training. Gradient clipping, SDK request sizing, and console logging use upstream defaults.

Evaluate a saved sampler checkpoint using its `sampler_path` from `checkpoints.jsonl`:

```bash
uv run harvey-eval \
  base_url=http://127.0.0.1:9003 \
  checkpoint=tinker://MODEL/sampler_weights/LABEL \
  log_path=artifacts/harvey-labs/eval
```

Omit `checkpoint` to evaluate the base model. Keep the task pool, seed, judge,
and rollout count fixed when comparing runs.

## Results

Start here when analyzing a run, including from an agent:

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

Eval reports prefer **pooled criterion pass rate**: total passed / total
criteria. Cookbook stores mean criterion counts; their ratio gives this rate.
For episodes scoring 1/2 and 2/8, pooled success is **30%**, while the mean
of episode percentages is **37.5%**. Legacy episode averages stay separate
in the report and plot.

Training reward averages episode rewards and may include termination penalties.
Missing outputs and grading failures score zero. Terminal `lab/*` flags average
over recorded episodes; `parse_error`, `context_overflow`, and
`max_tokens_reached` average over transitions. Check cookbook's error diagnostics
for failures without a rollout record.

## Custom sandboxes

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
owns cleanup. Podman is the bundled default.

## Development

Use `chz` for recipe configuration, data objects, and CLI entrypoints.

```bash
uv run python -m harvey_labs.renderers.qwen35_renderer
uv run python -m harvey_labs.renderers.gemma4_renderer
```

Each renderer has an inline `unittest` for sampled-token preservation across
turns using the real tokenizer, without model weights or a GPU. Gemma uses
Google’s unmodified template and response schema at a pinned HF revision,
with Transformers parsing tool calls; malformed arguments fail parsing. The
first Gemma run downloads these assets into the HF cache. Keep tests focused here;
avoid a broad recipe suite or generated fake harness packages. For API server configuration,
see [docs/configuration.md](../../docs/configuration.md).
