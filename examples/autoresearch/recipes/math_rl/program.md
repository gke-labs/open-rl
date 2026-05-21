# OpenRL Math-RL Autoresearch Program

You are an autonomous researcher running inside an isolated sandbox. Your job is
to improve an OpenRL/Tinker RL run by editing only
`recipes/math_rl/config.toml`.

This mirrors the `vivekvkashyap/autoresearch-rl` loop: the human owns these
instructions, and the agent iterates on one training configuration until its
agent timeout expires.

You do not manage GPUs directly. Run against the shared OpenRL gateway exposed
by `TINKER_BASE_URL`; the model/trainer service handles GPU placement.

## Setup

Before starting:

1. Read `recipes/math_rl/autoresearch.toml`, `recipes/math_rl/train.py`, `recipes/math_rl/config.toml`, and this program.
2. Use `${LOG_ROOT:-artifacts/autoresearch/runs}` as the artifact root.
3. Use `${RESEARCHER_ID}` as your researcher id.
4. Keep concise notes in `${WORK_DIR}/notes.md`.
5. Use a local git branch named `autoresearch/${RESEARCHER_ID}`.
6. Run all commands from the repository root.
7. Stop when `AGENT_TIMEOUT_MINUTES` expires.

## Objective

Maximize the UI metric:

```text
accuracy
```

The training script keeps the environment fixed to `gsm8k` and exposes only the
simple knobs in `config.toml`. Keep `eval_enabled = true` for real comparisons,
and set `eval_interval` no higher than `max_steps` so each attempt emits an eval
metric.

## Run Command

First, run the unmodified default config with `eval "${DEFAULT_CONFIG_COMMAND}"`.
This records where the recipe starts as a normal table row. After that, edit
`config.toml`, commit that change, then run attempts with `eval "${RUN_ATTEMPT_COMMAND}"`.
The launcher provides these commands so logs, diffs, metrics, and UI events are
captured consistently.

Run attempts only in the foreground. Do not append `&`, use `nohup`, use
`disown`, or tell the shell to keep training in the background. Wait for each
attempt command to exit before inspecting metrics or starting another attempt.

Do not print full diffs or long file dumps into the agent transcript. The UI
captures config diffs automatically in the Diff tab. If you need to inspect
your change before committing, use concise commands such as `git diff --stat`,
`git diff --name-only`, or a small targeted `git diff -- config.toml | sed -n
'1,80p'`.

## Loop

Repeat until the agent timeout expires:

1. Read your notes, recent logs, and the UI.
2. Pick one concrete experiment description for the `config.toml` change.
3. Record the current commit as `start_commit`.
4. Edit `config.toml`.
5. Commit the attempted config change before running it.
6. Run the attempt command with `eval "${RUN_ATTEMPT_COMMAND}"`.
7. Inspect `metrics.jsonl`, `logs.log`, and the UI Diff tab.
8. Append a short note with commit, metric, status, and what to try next.
9. If the metric improves by a meaningful amount, keep the commit.
10. If the metric is equal or worse, run `git reset --hard "${start_commit}"` after recording the note.

Good first knobs:

- lower or raise `lr`
- change `rollouts_per_example`
- change `batch_size`
- change `max_steps` within the fixed attempt timeout
- tune `temperature`
- tune `max_tokens`
- try `loss` values supported by Tinker: `importance_sampling`, `ppo`, `cispo`, `dro`, or `cross_entropy`
