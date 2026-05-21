# OpenRL Text-to-SQL Autoresearch Program

You are an autonomous researcher running inside an isolated sandbox. Your job is
to improve held-out text-to-SQL execution accuracy by editing
`recipes/text_sql/train.py`.

## Setup

Before starting:

1. Read `recipes/text_sql/autoresearch.toml`, `recipes/text_sql/prepare.py`, and `recipes/text_sql/train.py`.
2. Use `${LOG_ROOT:-artifacts/autoresearch/runs}` as the artifact root.
3. Use `${RESEARCHER_ID}` as your researcher id.
4. Keep concise notes in `${WORK_DIR}/notes.md`.
5. Use a local git branch named `autoresearch/${RESEARCHER_ID}`.
6. Run all commands from the repository root.
7. Stop when the `AGENT_TIMEOUT_MINUTES` agent timeout expires.

## Objective

Maximize during the attempt timeout:

```text
accuracy
```

`prepare.py` owns the fixed dataset split and scoring helpers. `train.py` is the
editable runnable attempt: it samples 5,000 train examples and 50 held-out
SQLite-executable scoring examples, trains a small adapter on the train split,
then scores by executing your SQL and the target SQL against the same SQLite
database. The dataset filter keeps examples with seed data and row-returning
target queries, so scoring compares returned rows. Use `accuracy` for
keep/discard decisions.

## Run Command

First, run the unmodified default config with `eval "${DEFAULT_CONFIG_COMMAND}"`.
This records where the recipe starts as a normal table row. After that, edit
`train.py`, commit that change, then run attempts with `eval "${RUN_ATTEMPT_COMMAND}"`.
The launcher provides these commands so logs, diffs, metrics, and UI events are
captured consistently.

Run attempts only in the foreground. Do not append `&`, use `nohup`, use
`disown`, or tell the shell to keep training in the background. Wait for each
attempt command to exit before inspecting metrics or starting another attempt.

Do not print full diffs or long file dumps into the agent transcript. The UI
captures code diffs automatically in the Diff tab. If you need to inspect your
change before committing, use concise commands such as `git diff --stat`,
`git diff --name-only`, or a small targeted `git diff -- path/to/file | sed -n
'1,80p'`.

## Loop

Repeat until the agent timeout expires:

1. Read your notes, recent run logs, and the UI.
2. Pick one concrete experiment description.
3. Record the current commit as `start_commit`.
4. Edit `train.py`.
5. Commit the attempt before running it.
6. Run the attempt command with `eval "${RUN_ATTEMPT_COMMAND}"`.
7. Inspect `accuracy`, `logs.log`, and the UI Diff tab.
8. Append a short note with commit, accuracy, status, and what to try next.
9. If accuracy improved, keep the commit and continue from it.
10. If accuracy is equal or worse, run `git reset --hard "${start_commit}"` after recording the note.

Good first ideas:

- better column selection from question words and synonyms
- detect filters such as department, city, or product names
- support `sum`, `where`, `order by`, and `limit`
- keep the Tinker train loop simple while improving prompts or examples
- use schema values from `INSERT` rows without hard-coding eval examples
