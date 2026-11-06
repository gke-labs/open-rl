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
| OpenRL deployment | A GKE deployment. See [GKE setup](../../docs/setup/gke-setup.md). | The recipe uses the model's full context window (262,144 tokens for Qwen3.5-9B), which requires a trainer with context parallelism. The default recipe uses Automodel on 4 GPUs with CP4, plus 4 vLLM samplers. We ran it on one 8 × B200 node and on 8 × H100. On smaller GPUs, lower the window; see [Train](#train). |
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

Then point the grader at a judge; see [Choose a judge](#choose-a-judge).

Check the OpenRL API server before a long run:

```bash
uv run openrl-doctor --base-url http://127.0.0.1:9003
```

## Choose a judge

LAB's grader runs on the driver's machine and asks a judge model about each
rubric criterion. Pick the judge with `judge_model`; the grader chooses the
provider from the model name and reads its credentials from the environment
or LAB's `.env` file:

| `judge_model` starts with | Provider | Set |
| --- | --- | --- |
| `gpt`, `o1`, `o3`, `o4`, `o5` | OpenAI or any OpenAI-compatible server | `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for a server other than OpenAI |
| `gemini` | Google | `GEMINI_API_KEY` |
| `claude` | Anthropic | `ANTHROPIC_API_KEY` |
| `mistral` | Mistral | `MISTRAL_API_KEY` |
| `glm` | A GLM deployment on a Vertex AI endpoint | `VERTEX_JUDGE_ENDPOINT`, `VERTEX_JUDGE_TOKENIZER` |

The default, `judge_model=gpt-glm-5.2`, is GLM-5.2 served by vLLM behind an
OpenAI-compatible API. To self-host a judge the same way, serve the model
under a name that starts with `gpt-` and point the grader at it:

```bash
vllm serve <model> --served-model-name gpt-glm-5.2 --tensor-parallel-size 8
export OPENAI_BASE_URL=http://<judge-host>:8000/v1
export OPENAI_API_KEY=unused   # any value for a self-hosted server
```

With `OPENAI_BASE_URL` set, the grader asks the model to answer without
reasoning first. A reasoning judge can spend its whole output limit thinking
about a long deliverable and return no verdict.

Two settings control grading throughput:

- `judge_parallel` sets how many criteria of one episode are graded at once.
  The default is 16 when the judge name contains `glm`, otherwise 1.
- `OPEN_RL_GRADING_CONCURRENCY` (environment, default 6) sets how many
  episodes are graded at once. A step grades 48 episodes, so raise it if the
  judge has headroom.

Separately from the judge, LAB asks Claude to match a file to its deliverable
when the file name doesn't make it clear. That needs `ANTHROPIC_API_KEY`;
without it, LAB counts those deliverables as missing.

Different judges grade with different strictness, so keep the judge fixed
across runs you compare. Before a long run, check that the judge answers:
`curl $OPENAI_BASE_URL/models` for a self-hosted server, then a one-task
[smoke test](#smoke-test).

## Train

Forward local port 9003 to the OpenRL API server with
`kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000`,
then start training:

```bash
export TINKER_API_KEY=tml-dummy-key   # OpenRL ignores it; the tinker client requires one
export TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_gpus=4,openrl.trainer_cp=4,openrl.sampler_replicas=4
uv run harvey-train base_url=http://127.0.0.1:9003 log_path=artifacts/harvey-labs/my-run
```

By default `max_trajectory_tokens` is the model's full context window, read
from its Hugging Face config: 262,144 tokens for Qwen3.5-9B and 131,072 for
Gemma 4 E4B. Before `harvey-train` creates its workers, set `VLLM_MAX_MODEL_LEN`
and `OPEN_RL_TRAIN_TOKEN_BUDGET` on the API server to at least that window:

```bash
kubectl -n openrl-system set env deployment/open-rl-api-server \
  VLLM_MAX_MODEL_LEN=262144 OPEN_RL_TRAIN_TOKEN_BUDGET=262144
```

We ran the recipe on hardware that fits the full window (8 × B200, and
8 × H100 for Qwen3.5-9B). If your GPUs run out of memory, set
`max_trajectory_tokens` lower and lower the two API server limits to match.

Pass overrides using `chz` `key=value` arguments. Run `uv run harvey-train --help`
or check [config.py](config.py) for every option. Running without overrides
uses the reference recipe defaults:

| Setting | Default |
| --- | --- |
| Model | `Qwen/Qwen3.5-9B`, with the renderer chosen from the model name |
| Batch | 8 tasks per step via `batch_size` and 6 attempts per task via `rollouts_per_example`. `stream_minibatches` streams minibatches to the trainer while sampling continues. |
| Optimizer | LoRA rank 32, constant `learning_rate=2e-4`, and no KL penalty |
| Loss and reward | Importance sampling with group-centered advantages. Reward equals passed criteria divided by total criteria, or 0 when ungraded. |
| Episode limits | `max_turns=40`, `max_tokens=32768` per turn, `max_trajectory_tokens` = the model's full window (262,144 for Qwen3.5-9B), and `max_tool_result_tokens=16384` |
| Length | `max_steps=35` across 300 training tasks |
| Evaluation | 50 held-out tasks at step 0, every 5 steps via `eval_every`, and after the final step, with 1 attempt per task via `eval_rollouts_per_task` |
| Task split | `task_split_seed=0` selects the held-out slice, and `train_split_seed=242` orders the training pool. Scenario families in the evaluation set never appear in training. |

<details>
<summary>Every setting of the reference Qwen3.5-9B run</summary>

| Setting | Value |
| --- | --- |
| Model / renderer | `Qwen/Qwen3.5-9B`, `qwen3_5` |
| Loss | `importance_sampling` (GRPO-style, group-centered advantages, no PPO clip) |
| Learning rate | 2e-4, constant, no warmup |
| LoRA | Rank 32, alpha 16 (scale 0.5), dropout 0.05, Kaiming init, bf16 adapter |
| LoRA targets | All attention, all gated-DeltaNet linear-attention projections, all MLP; not `lm_head` |
| Batch | 8 tasks × 6 attempts = 48 per step, streamed in minibatches of 8, 1 substep |
| KL penalty | None |
| Reward | Fraction of criteria passed; no process reward, no all-pass bonus; ungraded = 0 |
| Gradient clipping | None |
| Steps / tasks | 35 steps over 300 training tasks, temperature 1.0 |
| Context window | 262,144 tokens (the model's full window) |
| Per-turn cap | 32,768 tokens |
| Tool results | Truncated at 16,384 tokens |
| Turns | 40 max |
| Cut-off episodes | Graded on what was written |
| Unreadable turns | End the episode ungraded (0 in the pass rate, -0.1 training reward) |
| Judge | GLM-5.2-FP8 with reasoning off, up to 16 criteria graded in parallel |
| Held-out set | 50 tasks: `task_split_seed=0`, slice 300–350; `train_split_seed=242` |
| Eval attempts | 1 per task during training; 4 per task for the paired comparison |

</details>

Keep these runtime behaviors in mind when configuring a run:

- Each turn may generate up to `max_tokens` or whatever remains of
  `max_trajectory_tokens`, whichever is smaller. When fewer than 2,048 tokens
  of the context window remain, the episode ends with `context_overflow`
  instead of sampling a turn too short to be useful.
- When an episode stops at the per-turn token limit or fills the context
  window, the grader scores whatever deliverables the agent already wrote. A
  turn the parser can't read, or whose tool calls are all malformed, ends the
  episode ungraded: it counts as 0 in the pass rate and gets a training reward
  of -0.1.
- When the judge returns no verdicts for an episode, the recipe records a
  grading failure (`lab/reward_error=1`, `lab/graded=0`) rather than a graded
  zero. The episode's criteria still count in the pooled denominator.
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
judge graded, alongside the pass fraction on graded episodes. `harvey-results`
also prints the number of evaluation episodes whose grading failed, for
example `12 errors`. A nonzero count usually means the judge was unreachable
or returned no verdicts, so check `grading.log` before comparing pass rates.

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
| Agent Substrate | You want sandboxes that suspend while the model generates | Install the `substrate` extra and pass `sandbox_substrate_endpoint` and `sandbox_substrate_image`. See [Agent Substrate](#agent-substrate). |
| Custom backend | You use another sandbox runtime | Pass a custom factory in Python, as shown below. |

### Agent Substrate

`sandbox_substrate_endpoint` runs each episode in an
[Agent Substrate](https://github.com/agent-substrate/substrate) environment
through the `ate-env-api` service from
[sandboxenv-rl](https://github.com/ai-on-gke/sandboxenv-rl). The cluster needs:

- Substrate at or after `48f5744c`. The backend was verified with `v1.0.0-rc.1`.
- `ate-env-api` deployed from sandboxenv-rl, with
  `spiffe://cluster.local/ns/ate-env/sa/ate-env-api` in the router's ingress
  allow-list.
- A base ActorTemplate registered with `ate-env manifest template`. On
  `v1.0.0-rc.1`, per-template snapshot locations are behind the
  `PerTemplateSnapshotLocation` preview gate. Without the gate, drop
  `storageLocation` from the printed template, and snapshots go to the
  cluster's snapshot bucket.

LAB's sandbox image runs unmodified as the environment's root filesystem. Pass
it pinned by digest:

```bash
kubectl -n ate-env port-forward svc/ate-env-api 7777:7777 &
export SUBSTRATE_ENV_AUTH_TOKEN="$(kubectl -n ate-env get secret ate-env-api-token -o jsonpath='{.data.token}' | base64 -d)"
uv run --extra substrate harvey-train base_url=http://127.0.0.1:9003 \
  sandbox_substrate_endpoint=127.0.0.1:7777 \
  sandbox_substrate_image=<registry>/lab-sandbox@sha256:<digest>
```

A sandbox suspends after `sandbox_substrate_idle_suspend_seconds` (default 30)
without a tool call, which frees its worker while the model generates. The next
tool call resumes it with its files and processes intact. A sandbox never
suspends during a tool call. Set the option to `None` to keep sandboxes
running. Each episode's `metrics.json` records the number of suspensions as
`substrate_suspends`.

### Custom backends

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
manage their own leases and heartbeats. LAB's grader always runs on the
driver's machine and calls the judge endpoint from there. If
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
recipe's unit tests from `examples/`:

```bash
uv run python -m unittest harvey_labs.test_fair_grading harvey_labs.test_substrate_sandbox
uv run python -m harvey_labs.renderers.qwen35_renderer
uv run python -m harvey_labs.renderers.gemma4_renderer
```

`test_fair_grading` covers the per-turn budget floor and empty-judge error
accounting. `test_substrate_sandbox` covers the Substrate backend lifecycle and
idle suspend against a fake `ate-env-api` client.

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
