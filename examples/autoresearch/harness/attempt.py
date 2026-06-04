"""Run one autoresearch attempt and record UI artifacts."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import tomllib

from harness.utils import (
  ATTEMPT_ARTIFACTS,
  METADATA,
  AgentPaths,
  agent_id,
  close_attempt_agent_logs,
  git_text,
  recipe_spec_hash,
  run_id,
  run_logged,
  write_json_atomic,
)

ATTEMPT_LOG = ATTEMPT_ARTIFACTS["logs"]
METRICS = ATTEMPT_ARTIFACTS["metrics"]
DIFF = ATTEMPT_ARTIFACTS["diff"]


@chz.chz
class AttemptConfig:
  recipe: Path = Path(os.getenv("RECIPE", "autoresearch.toml"))
  agent_id: str = os.getenv("AGENT_ID", "")
  baseline: bool = False
  log_root: Path = Path(os.getenv("LOG_ROOT", "artifacts/autoresearch"))
  run_name: str = os.getenv("RUN_NAME", "")
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))
  echo: bool = True


@dataclass(frozen=True)
class Recipe:
  task: str
  command: str
  editable: tuple[Path, ...]
  metric: str
  metric_label: str
  metric_mode: str


def repo_paths_for_editables(paths: tuple[Path, ...], root: Path) -> list[str]:
  resolved = []
  for path in paths:
    if path.is_absolute():
      resolved.append(str(path.resolve().relative_to(root)))
    else:
      resolved.append(str(path))
  return resolved


def extract_metrics(run_dir: Path, metric: str) -> list[tuple[float, int | float | None]]:
  found = []
  path = run_dir / METRICS
  if path.exists():
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
      if not line.strip():
        continue
      row = json.loads(line)
      if metric in row:
        found.append((float(row[metric]), row["step"]))
  return found


def ensure_attempt_is_new(
  paths: AgentPaths,
  baseline: bool,
  recipe: Recipe,
  agent: str,
  commit: dict[str, str | None],
) -> Path | None:
  attempts = [(path.parent, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(paths.attempts.glob("*/metadata.json"))]

  if baseline:
    for run_dir, row in attempts:
      if row["baseline"] and row["status"] == "completed" and row["metric"]:
        return run_dir
    return None

  root = Path(git_text("rev-parse", "--show-toplevel").strip() or Path.cwd()).resolve()
  dirty = git_text("status", "--porcelain", cwd=root).strip()

  if dirty:
    raise SystemExit("Working tree must be clean before running an attempt. Commit the edit, then rerun RUN_ATTEMPT_COMMAND.")

  commit_hash = commit["commit"]
  for run_dir, row in attempts:
    if commit_hash and row["commit"] == commit_hash:
      raise SystemExit(f"Commit {commit_hash} was already evaluated for {agent}: {run_dir}. Commit a new change before rerunning.")

  if not commit["parent"]:
    raise SystemExit("Agent-edited attempts need a parent commit so the attempted diff can be evaluated.")

  allowed = set(repo_paths_for_editables(recipe.editable, root))
  changed = set()

  for path in git_text("diff", "--name-only", "HEAD^", "HEAD", cwd=root).splitlines():
    if path:
      changed.add(path)
  extra = changed - allowed

  if extra:
    raise SystemExit("Attempt commit changes files outside recipe.editable:\n" + "\n".join(sorted(extra)))
  if not changed:
    raise SystemExit("Attempt commit does not change any recipe.editable files.")
  return None


def new_run_dir(paths: AgentPaths, baseline: bool) -> tuple[int, Path]:
  paths.attempts.mkdir(parents=True, exist_ok=True)

  if baseline:
    run_dir = paths.attempts / "000"
    attempt = 0
    if run_dir.exists():
      shutil.rmtree(run_dir)
  else:
    attempt = 1 + sum(1 for path in paths.attempts.iterdir() if path.is_dir() and path.name != "000")
    run_dir = paths.attempts / f"{attempt:03d}"

  run_dir.mkdir(parents=True, exist_ok=False)
  return attempt, run_dir


def command_for_attempt(
  args: AttemptConfig,
  paths: AgentPaths,
  recipe: Recipe,
  agent: str,
  attempt: int,
  run_dir: Path,
) -> list[str]:
  values = {
    "attempt": attempt,
    "attempt_name": run_dir.name,
    "attempt_timeout_minutes": args.attempt_timeout_minutes,
    "agent_id": agent,
    "log_root": args.log_root,
    "run_root": paths.run_root,
    "researcher": agent,
    "run_dir": run_dir,
    "run_name": args.run_name,
  }
  parts = shlex.split(os.path.expandvars(recipe.command).format(**values))
  if parts and parts[0] in {"python", "python3"}:
    parts[0] = sys.executable
  return parts


def attempt_manifest(
  args: AttemptConfig,
  status: str,
  commit: dict[str, str | None],
  started_at: float,
  exit_code: int | None,
  agent_log_start: int = 0,
  agent_log_end: int | None = None,
  metric: dict[str, Any] | None = None,
  spec_hash: str = "",
) -> dict[str, Any]:
  finished_at = None if status == "running" else time.time()
  error = None
  if status not in {"running", "timed_out"} and exit_code != 0:
    error = f"exit_code={exit_code}"
  if status == "failed" and metric is None:
    error = "metric missing"
  return {
    "spec_hash": spec_hash,
    "baseline": args.baseline,
    "status": status,
    "started_at": started_at,
    "finished_at": finished_at,
    "attempt_timeout_minutes": args.attempt_timeout_minutes,
    "agent_log_start": agent_log_start,
    "agent_log_end": agent_log_end,
    "branch": commit["branch"],
    "commit": commit["commit"],
    "parent": commit["parent"],
    "description": "Default config" if args.baseline else git_text("log", "-1", "--pretty=%s").strip(),
    "exit_code": exit_code,
    "error": error,
    "metric": metric,
  }


def capture_diffs(recipe: Recipe, run_dir: Path) -> None:
  root = Path(git_text("rev-parse", "--show-toplevel").strip() or Path.cwd()).resolve()
  paths = repo_paths_for_editables(recipe.editable, root)
  diff = git_text("diff", "--unified=3", "HEAD^", "HEAD", "--", *paths, cwd=root)
  (run_dir / DIFF).write_text(diff, encoding="utf-8")


def finish_attempt(recipe: Recipe, run_dir: Path, code: int) -> tuple[str, dict[str, Any] | None]:
  extracted = extract_metrics(run_dir, recipe.metric)
  status = "completed" if code == 0 and extracted else "timed_out" if code == 124 else "failed"

  with (run_dir / ATTEMPT_LOG).open("a", encoding="utf-8") as f:
    if not extracted:
      print(f"missing metric={recipe.metric}", file=f)
    print(f"status={status}", file=f)

  if not extracted:
    return status, None

  final_value, final_step = extracted[-1]
  return status, {
    "name": recipe.metric,
    "value": final_value,
    "step": final_step,
    "mode": recipe.metric_mode,
  }


def load_recipe(path: Path, ref: str = "") -> Recipe:
  if ref:
    text = git_text("show", f"{ref}:{path.as_posix()}")
    if not text:
      raise SystemExit(f"could not read {path} at {ref}")
  else:
    text = path.read_text(encoding="utf-8")
  raw = tomllib.loads(text)
  mode = str(raw["metric_mode"])
  if mode not in {"max", "min"}:
    raise ValueError("metric_mode must be max or min")
  return Recipe(
    task=str(raw["task"]),
    command=str(raw["command"]),
    editable=tuple(Path(value) for value in raw["editable"]),
    metric=str(raw["metric"]),
    metric_label=str(raw["metric_label"]),
    metric_mode=mode,
  )


def run_attempt(args: AttemptConfig) -> tuple[Path, str]:
  agent = agent_id(args.agent_id)
  args = chz.replace(args, run_name=run_id(args.run_name), log_root=Path(args.log_root).resolve())
  paths = AgentPaths.from_run(args.log_root, args.run_name, agent)
  commit = {
    "branch": git_text("branch", "--show-current").strip() or None,
    "commit": git_text("rev-parse", "--short=7", "HEAD").strip() or None,
    "parent": git_text("rev-parse", "--short=7", "HEAD^").strip() or None,
  }
  # spec_hash is the experiment's identity: the recipe hash at agent start, shared by every
  # attempt so the UI can confirm two runs are the same. The agent records it once; attempts
  # reuse it rather than rehashing their own edits to the editable file.
  spec_hash = recipe_spec_hash(args.recipe.parent)
  if paths.metadata.exists():
    spec_hash = str(json.loads(paths.metadata.read_text(encoding="utf-8"))["spec_hash"])
  recipe_ref = "HEAD^" if not args.baseline and commit["parent"] else ""
  recipe = load_recipe(args.recipe, recipe_ref)
  if existing := ensure_attempt_is_new(paths, args.baseline, recipe, agent, commit):
    print(f"baseline attempt already exists for {agent}: {existing}")
    return existing, "completed"

  attempt, run_dir = new_run_dir(paths, args.baseline)
  logs_path = run_dir / ATTEMPT_LOG
  started_at = time.time()
  paths.notes.touch(exist_ok=True)
  if not paths.metadata.exists():
    paths.agent_log.touch(exist_ok=True)
    write_json_atomic(
      paths.metadata,
      {
        "spec_hash": spec_hash,
        "status": "completed",
        "started_at": started_at,
        "finished_at": started_at,
        "exit_code": 0,
        "agent_model": "manual",
        "agent_timeout_minutes": 0,
      },
    )
  agent_start = paths.agent_log.stat().st_size if paths.agent_log.exists() else 0
  close_attempt_agent_logs(paths.root, agent_start)
  capture_diffs(recipe, run_dir)
  manifest = attempt_manifest(args, "running", commit, started_at, None, agent_start, spec_hash=spec_hash)
  write_json_atomic(run_dir / METADATA, manifest)

  command = command_for_attempt(args, paths, recipe, agent, attempt, run_dir)
  code = run_logged(command, logs_path, args.attempt_timeout_minutes * 60, echo=args.echo)
  capture_diffs(recipe, run_dir)
  status, metric = finish_attempt(recipe, run_dir, code)
  manifest = attempt_manifest(args, status, commit, started_at, code, agent_start, metric=metric, spec_hash=spec_hash)
  write_json_atomic(run_dir / METADATA, manifest)
  print(f"attempt {run_dir.name}: {status}", flush=True)
  if metric:
    print(f"{metric['name']}={metric['value']} step={metric['step']}", flush=True)
  print(f"logs={run_dir / ATTEMPT_LOG}", flush=True)
  print(f"metrics={run_dir / METRICS}", flush=True)
  return run_dir, status


def main() -> None:
  args = chz.entrypoint(AttemptConfig, allow_hyphens=True)
  _, status = run_attempt(args)
  raise SystemExit(0 if status == "completed" else 124 if status == "timed_out" else 1)


if __name__ == "__main__":
  main()
