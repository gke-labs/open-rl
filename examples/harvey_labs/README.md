# Harvey LAB RL

Live-rollout RL on Harvey's Legal Agent Benchmark: the model works real LAB
tasks in a podman sandbox (documents, shell, file tools), a judge grades the
rubric, and the pass fraction is the reward. Training reuses tinker-cookbook's
GRPO loop and multi-turn tool environment.

## Results

Run 9 (Qwen3.5-9B LoRA, 20 steps, batch 8×6 rollouts, GLM judge): held-out
criterion pass rate 48.7% → **67.6%** (peak, step 15), 65.1% at the final
checkpoint — 3,151 pooled rubric criteria over the 50-task eval split.

![run 9 training curve](assets/run9.png)

## Layout

- `train.py` — run config and entrypoint (grading preflight, final eval).
- `tasks.py` — task discovery and the seeded train/eval split.
- `eval_checkpoint.py` — evaluate any saved adapter checkpoint.
- `prompts.py` — system prompt, skills, output-path contract.
- `env.py` — sandbox env construction and dataset builders.
- `reward.py` — rubric reward wrapper around LAB's judge.
- `tools.py` — LAB `ToolExecutor` adapter.
- `gemma4_renderer.py` — Gemma 4 tool-call renderer (Qwen uses stock `qwen3_5`).
- `plot_run.py` — plot a run's rewards and pass rate.
- `score_lab_run.py` — grading shim executed inside the LAB venv.

## Setup

On a bare Ubuntu GPU VM:

```bash
git clone https://github.com/ShubyM/open-rl && cd open-rl
./scripts/setup_vm.sh
```

Installs build deps, uv, the Python env, and the LAB harness (sandbox image,
pandoc, podman), then prints a health checklist. Idempotent. The pieces,
individually:

1. `uv sync --frozen --exact --extra gpu --extra vllm --extra fastpath`
   (`fastpath` builds `causal-conv1d`; without it Qwen training runs 2–5x
   slower on the eager fallback).
2. `examples/harvey_labs/setup_lab.sh` — clones the LAB fork
   (`ShubyM/harvey-labs`, which carries harness fixes from upstream PRs
   #85–#90; rewards from unfixed upstream are not comparable) and runs its
   setup.
3. Judge key: `export GEMINI_API_KEY=...` (or point `judge_model` at a
   self-hosted judge). `train.py` preflights the grading environment and
   refuses to start if it's broken.
4. Gateway — an open-rl server for the policy model:

   ```bash
   VLLM_ALLOW_RUNTIME_LORA_UPDATING=true uv run --extra gpu --extra vllm \
     vllm serve <model> --port 8000 --enable-lora --max-lora-rank 64 \
     --max-model-len 65536 --language-model-only --disable-log-requests

   SAMPLER_BASE_URL=http://127.0.0.1:8000 BASE_MODEL=<model> \
     uv run --extra gpu --extra vllm python -m uvicorn server.gateway:app --port 9003
   ```

   Keep the sampler's `--max-model-len` equal to `max_trajectory_tokens` — a
   mismatch turns over-length rollouts into silent parse failures.

## Run

One command on an 8-GPU box (sampler + gateway + trainer + typed train
command in a tmux session):

```bash
MODEL=9b ./scripts/launch_work.sh
```

Or by hand:

```bash
TINKER_API_KEY=tml-dummy-key \
uv --project examples run python examples/harvey_labs/train.py \
  model_name=Qwen/Qwen3.5-9B \
  base_url=http://127.0.0.1:9003 \
  learning_rate=2e-4 lora_rank=32 \
  batch_size=8 rollouts_per_example=6 max_steps=20 eval_every=5 \
  max_tokens=16384 max_trajectory_tokens=131072 max_tool_result_tokens=16384 \
  log_path=artifacts/harvey-labs/my-run
```

Tasks are a seeded random split of the runnable LAB pool
(`train_tasks=300 eval_tasks=50 task_split_seed=0`, disjoint). The split is
the benchmark — keep the seed fixed across runs you compare.
`task=<name>` trains a single task for smoke tests. `stream_minibatches=true`
overlaps training with sampling (identical gradients at `num_substeps=1`).

Evaluate any saved checkpoint with `eval_checkpoint.py checkpoint=...`,
passing the same split/window knobs as the training run.

## Watching a run

- `plot_run.py <run-dir>` — rewards and held-out pass rate.
- `metrics.jsonl` — every metric per step; episodes report
  `lab/criteria_passed` / `lab/criteria_total` (failures count 0/N), so
  `mean(criteria_passed) x total_episodes` gives exact pooled counts.
- `iterations/iteration_*/` — full transcripts and rollout summaries.
- `<lab_root>/results/<run-id>/scores.json` — per-episode rubric verdicts.
- Watch `by_group/frac_all_bad` (structural failures),
  `optim/kl_sample_train_v1` (~1e-4–1e-3 on-policy), `lab/reward_error`.

## Troubleshooting

- **Rollouts are `<pad>` streams** (logprobs exactly `-0.1`): sampler is in
  mock mode — vllm failed to import in that process.
- **Empty completions / `leaves no room in max_model_len`**: sampler context
  smaller than `max_trajectory_tokens`.
- **Rubric scores all zero**: judge key missing or stale LAB venv
  (`uv sync` inside the LAB checkout).
- **Episodes end after one turn with no tool call**: renderer/template
  mismatch for the model family.
- **CUDA OOM in training**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `OPEN_RL_TRAIN_TOKEN_BUDGET`, `OPEN_RL_ACTIVATION_CPU_OFFLOAD=1` (all on
  the gateway). Knob reference: [docs/configuration.md](../../docs/configuration.md).
- **Podman**: rootless podman needs `XDG_RUNTIME_DIR` in detached shells;
  sweep leaked containers with `podman rm -f $(podman ps -aq)`; small root
  disks need `graphroot` moved before pulling the sandbox image.

Run the repository unit tests with `make test unit`.
