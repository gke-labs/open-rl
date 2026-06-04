# OpenRL Math-RL Autoresearch Program

You are an autonomous agent running inside an isolated sandbox. Your job is
to improve an OpenRL/Tinker RL run by editing only
`recipes/math_rl/config.toml`.

This mirrors the `vivekvkashyap/autoresearch-rl` loop: the human owns these
instructions, and the agent iterates on one training configuration until its
agent timeout expires.

You do not manage GPUs directly. Run against the shared OpenRL gateway exposed
by `TINKER_BASE_URL`; the model/trainer service handles GPU placement.

Before editing, read `recipes/math_rl/autoresearch.toml`,
`recipes/math_rl/train.py`, `recipes/math_rl/config.toml`, and this program.
Keep concise notes in `${WORK_DIR}/notes.md`.

## Objective

Maximize the UI metric:

```text
accuracy
```

The training script keeps the environment fixed to `gsm8k`, discovers the served
model from the OpenRL backend, and exposes only the simple knobs in
`config.toml`. Do not add model, renderer, GPU, endpoint, or infrastructure
settings to `config.toml`. Keep `eval_enabled = true` for real comparisons, and
set `eval_interval` no higher than `max_steps` so each attempt emits an eval
metric.

## Run Command

The launcher records the unmodified default config before Gemini starts. Edit
`config.toml`, commit that change, then run attempts with `eval "${RUN_ATTEMPT_COMMAND}"`.
The launcher provides this command so logs, diffs, metrics, and UI artifacts are
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

1. Read your notes and recent logs.
2. Pick one concrete experiment description for the `config.toml` change.
3. Record the current commit as `start_commit`.
4. Edit `config.toml`.
5. Commit the attempted config change before running it.
6. Run the attempt command with `eval "${RUN_ATTEMPT_COMMAND}"`.
7. Inspect `metrics.jsonl`, `attempt.log`, and the UI Diff tab.
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
