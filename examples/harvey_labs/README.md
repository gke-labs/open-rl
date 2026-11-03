# Harvey LAB RL

Train a legal agent on Harvey's Legal Agent Benchmark (LAB) using the
`tinker-cookbook` GRPO loop on OpenRL. Each episode runs in an isolated
sandbox. The agent reads the task's source documents, runs commands, and writes
deliverables such as `.docx` and `.md` files. LAB's judge grades those
deliverables against the task's full rubric, and the reward is the fraction of
criteria passed.

With the default settings, Qwen3.5-9B improved from a 0.54 to a 0.78 pooled
criterion pass rate on the 50 held-out tasks in 35 steps. A paired
re-evaluation with 4 attempts per task measured a +14.5-point gain in rubric
quality on graded episodes, with a 95% CI of 10.4 to 18.7. Scores improved on
45 of the 50 tasks.

## What you need

| Component | Default recipe | Notes |
| --- | --- | --- |
| OpenRL deployment | A GKE deployment. See [GKE setup](../../docs/setup/gke-setup.md). | The 196,608-token context window requires a trainer with context parallelism. The default recipe uses Automodel on 4 × 80 GB GPUs with CP4, plus 4 vLLM samplers. Eight H100 GPUs fit the full setup. |
| Judge | GLM-5.2 behind an OpenAI-compatible endpoint with reasoning turned off | Any endpoint that serves the model named by `judge_model` works. Gemini also works. See [Setup](#setup). |
| Sandboxes | Podman on the driver machine, where the recipe client runs | When the driver runs in a cluster, use [Agent Sandbox](agent-sandbox/README.md). See [Sandbox backends](#sandbox-backends). |
| LAB | The fork cloned by `setup_lab.sh` | Provides the tasks, tools, sandbox image, and grader. |

To test the setup on smaller hardware first, see [Smoke test](#smoke-test).

## Setup

Run the commands below from `examples/`. If you run them from the repository
root, pass `--project examples` to `uv run`.

```bash
cd examples
uv sync
harvey_labs/setup_lab.sh
```

`setup_lab.sh` clones the LAB fork into `harvey_labs/harvey-labs`, which is the
recipe's default `lab_root`. The script then runs LAB's setup to install `uv`,
LAB's Python dependencies, `pandoc`, Podman, and the sandbox image. Set
`LAB_ROOT`, `LAB_REPO`, or `LAB_REF` to change the checkout path, repository,
or Git ref.

Configure the judge endpoint for the grader in your shell environment or in
LAB's `.env` file:

```bash
export OPENAI_BASE_URL=http://<judge-host>:8000/v1   # serves judge_model, default gpt-glm-5.2
export OPENAI_API_KEY=unused                          # any value for a self-hosted endpoint
```

To use Gemini instead, pass `judge_model=gemini-3.5-flash` and export
`GEMINI_API_KEY`. When a file name does not clearly identify which deliverable
it belongs to, LAB calls Claude to match the file. Set `ANTHROPIC_API_KEY` to
enable file matching. Without `ANTHROPIC_API_KEY`, LAB counts ambiguous
deliverables as missing.

Check the OpenRL API server before a long run:

```bash
uv run openrl-doctor --base-url http://127.0.0.1:9003
```

## Train

Forward local port 9003 to the OpenRL API server with
`kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000`,
then start training:

```bash
export TINKER_API_KEY=tml-dummy-key   # OpenRL ignores it; the tinker client requires one
export TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4,openrl.sampler_replicas=4
uv run harvey-train base_url=http://127.0.0.1:9003 log_path=artifacts/harvey-labs/my-run
```

Before `harvey-train` creates its workers, the API server must allow the
196,608-token context window. Set `VLLM_MAX_MODEL_LEN` and
`OPEN_RL_TRAIN_TOKEN_BUDGET` on the API server to at least
`max_trajectory_tokens`, which defaults to 196608:

```bash
kubectl -n openrl-system set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=196608 OPEN_RL_TRAIN_TOKEN_BUDGET=196608
```

Pass overrides using `chz` `key=value` arguments. Run `uv run harvey-train --help`
or check [config.py](config.py) for every option. Running without overrides
uses the reference recipe defaults:

| Setting | Default |
| --- | --- |
| Model | `Qwen/Qwen3.5-9B`, with the renderer chosen from the model name |
| Batch | 8 tasks per step via `batch_size` and 6 attempts per task via `rollouts_per_example`. `stream_minibatches` streams minibatches to the trainer while sampling continues. |
| Optimizer | LoRA rank 32, constant `learning_rate=2e-4`, and no KL penalty |
| Loss and reward | Importance sampling with group-centered advantages. Reward equals passed criteria divided by total criteria, or 0 when ungraded. |
| Episode limits | `max_turns=40`, `max_tokens=32768` per turn, `max_trajectory_tokens=196608`, and `max_tool_result_tokens=16384` |
| Length | `max_steps=35` across 300 training tasks |
| Evaluation | 50 held-out tasks at step 0, every 5 steps via `eval_every`, and after the final step, with 1 attempt per task via `eval_rollouts_per_task` |
| Task split | `task_split_seed=0` selects the held-out slice, and `train_split_seed=242` orders the training pool. Scenario families in the evaluation set never appear in training. |

Keep these runtime behaviors in mind when configuring a run:

- When an episode stops at the per-turn token limit or fills the context
  window, the grader scores whatever deliverables the agent already wrote. A
  malformed tool call ends the episode with a score of 0.
- The recipe truncates tool results longer than `max_tool_result_tokens` and
  appends a hint to continue reading from an `offset` or narrow the search with
  `grep`.
- Dividing 300 training tasks into batches of 8 leaves a short batch at step
  38. Short batches have hung streamed training in earlier runs, so keep
  `max_steps` at 37 or lower, or set `train_tasks` to a multiple of
  `batch_size`.
- Pass `task=<name>` to train on a single task instead of the seeded split.

## Evaluate

Pass a saved checkpoint's `sampler_path` from `checkpoints.jsonl` to evaluate
that checkpoint, or omit `checkpoint` to evaluate the base model:

```bash
uv run harvey-eval \
  base_url=http://127.0.0.1:9003 \
  checkpoint=tinker://MODEL/sampler_weights/LABEL \
  log_path=artifacts/harvey-labs/eval
```

At 1 attempt per task, repeated evaluations of the same weights varied by 5 to
7 points. To compare two checkpoints, evaluate both with
`eval_rollouts_per_task=4` and use the same task split, judge, and episode
limits.

## Results

Inspect a finished or active run with `harvey-results`:

```bash
uv run harvey-results log_dir=artifacts/harvey-labs/my-run
uv run harvey-results log_dir=artifacts/harvey-labs/my-run json=True plot=True
```

When training exits, it writes `results.json` and `run_plot.png`, even for
partial runs that recorded metrics. Pass `plot=True` to refresh both files, or
omit `plot=True` for a read-only summary. Pass `metrics=True` with `json=True`
to include raw diagnostic rows in the JSON output. To generate a custom plot,
run `uv run harvey-plot log_dir=... title="My run" out=run.png`. Extend
[results.read_results](results.py) for new analyses instead of writing a
separate parser.

The plot displays per-rollout rewards, batch means, an exponential moving
average (EMA), and evaluation scores. Step numbers count completed training
batches. Batch 0 finishes at step 1, and its pre-update evaluation appears at
step 0. Streaming logs from `tinker-cookbook` keep zero-based indices, so the
eighth of eight minibatches logs as `7/8`.

| Artifact | Source |
| --- | --- |
| `<lab_root>/results/<run-id>/scores.json`, `report.html` | LAB's evaluation CLI writes the rubric verdicts and HTML report here, and logs judge progress to `grading.log`. |
| `metrics.json`, `config.json`, `tinker_history.jsonl` in the same directory | The recipe records LAB tool counters, episode settings, and the message transcript here. |
| `<log_path>/metrics.jsonl`, `iteration_*/*_rollout_summaries.jsonl` | `tinker-cookbook` writes step aggregates, rollout summaries, the recipe's `lab/*` metrics, and the final evaluation here. |
| `<log_path>/results.json`, `run_plot.png` | The recipe writes the derived run summary and plot here. |

Evaluation reports default to the pooled criterion pass rate, calculated as
total passed criteria divided by total criteria across all episodes.
`tinker-cookbook` stores mean passed and mean total criterion counts per
episode, and their ratio gives the pooled rate. Across two episodes that pass
1/2 and 2/8 criteria, the pooled pass rate is 30%, whereas the mean of the
episode percentages is 37.5%. Reports and plots list legacy episode-average
scores separately.

Because the pooled pass rate counts unfinished and ungraded episodes as 0, an
increase can reflect either higher completion rates or stronger deliverables.
To separate those effects, compare `lab/graded`, the share of episodes that the
judge graded, alongside the pass fraction on graded episodes.

Training reward averages episode rewards and may include termination penalties.
Episodes with missing outputs or grading failures score 0. Terminal `lab/*`
flags average over recorded episodes, whereas `parse_error`,
`context_overflow`, and `max_tokens_reached` average over transitions. For
failures that leave no rollout record, check the `tinker-cookbook` error
diagnostics.

## Sandbox backends

| Backend | When to use it | How to configure it |
| --- | --- | --- |
| Podman | The driver runs on a machine with Podman | Enabled by default. `setup_lab.sh` builds the sandbox image. |
| Agent Sandbox | The driver runs in a cluster | Pass `sandbox_warmpool=<SandboxWarmPool>`. Follow [agent-sandbox/README.md](agent-sandbox/README.md) to set up the warm pool. |
| Custom backend | You use another sandbox runtime | Pass a custom factory in Python, as shown below. |

Pass a custom factory to the Python training or evaluation API:

```python
from harvey_labs.train import RunConfig, run
from my_backend import sandbox_factory

await run(RunConfig(base_url="http://127.0.0.1:9003"), sandbox_factory=sandbox_factory)
```

Training and evaluation share the same factory interface. Each call receives a
`SandboxRequest` and returns a `LabSandbox` that extends `tinker-cookbook`'s
`SandboxInterface`. See [sandbox.py](sandbox.py) for the full contract. Your
backend must provide an isolated workspace, LAB-compatible tools, and
binary-safe output collection into the local grading directory. Remote backends
manage their own leases and heartbeats, while the judge runs locally. If
startup fails before the factory returns, clean up inside the factory. After
the factory returns, the environment group owns cleanup.

## Run in a cluster

To run the driver as a Kubernetes Job alongside OpenRL with sandboxes on Agent
Sandbox, follow [docs/setup/harvey-lab-on-gke.md](../../docs/setup/harvey-lab-on-gke.md).
[cluster-job.yaml](cluster-job.yaml) defines the Job manifest, and
[Dockerfile](Dockerfile) builds the driver image.

## Smoke test

To verify the end-to-end setup on one 80 GB GPU per worker role, train on a
single task with a smaller context window:

```bash
uv run harvey-train base_url=http://127.0.0.1:9003 \
  task=<area>/<task> batch_size=1 rollouts_per_example=4 max_steps=2 \
  max_trajectory_tokens=81920 max_tokens=16384 max_tool_result_tokens=4096 \
  log_path=artifacts/harvey-labs/smoke
```

Before starting the smoke test, set `VLLM_MAX_MODEL_LEN` and
`OPEN_RL_TRAIN_TOKEN_BUDGET` to 81920 on the API server. Smaller context window
or per-turn token limits than those above end most episodes before the agent
writes a deliverable. If every episode in a smoke test scores 0, check whether
those limits are too tight before assuming the setup is broken.

## Development

Use `chz` for recipe configuration, data objects, and CLI entrypoints. Run the
renderer unit tests with:

```bash
uv run python -m harvey_labs.renderers.qwen35_renderer
uv run python -m harvey_labs.renderers.gemma4_renderer
```

Each renderer module includes an inline `unittest` suite that checks whether
sampled tokens survive across turns. The tests run with the real tokenizer and
need no model weights or GPU. Training must receive the exact tokens produced
during sampling. If a renderer re-renders earlier turns into different tokens,
it breaks prefix reuse and trains on text that the model never produced. The
Gemma renderer loads Google's unmodified chat template and response schema from
a pinned Hugging Face revision and parses tool calls with Transformers.
Malformed tool arguments fail parsing. The first Gemma run downloads those
files into the Hugging Face cache.

For API server configuration, see
[docs/configuration.md](../../docs/configuration.md).
