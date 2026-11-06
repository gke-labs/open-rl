"""Read Harvey run results for people, agents, and plots.

  uv run harvey-results log_dir=<run-dir> json=True plot=True

The step coordinate is completed training batches: training batch 0 finishes
at step 1; its pre-update evaluation measures step 0. Stored indices stay intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import chz

ENV_NAMES = ("harvey-labs", "all")


@chz.chz
class ResultsConfig:
  log_dir: Path
  json: bool = chz.field(default=False, doc="Emit the versioned summary and unsmoothed series as JSON")
  metrics: bool = chz.field(default=False, doc="Include all original metric rows in JSON for further analysis")
  plot: bool = chz.field(default=False, doc="Also save results.json and run_plot.png from the same data")


def load_metrics(path: Path) -> list[dict]:
  lines = path.read_text().splitlines(keepends=True)
  rows = []
  for index, line in enumerate(lines):
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      # A running writer may still be appending its final record. Corruption
      # in a complete line must remain visible, rather than silently losing data.
      if index == len(lines) - 1 and not line.endswith("\n"):
        break
      raise
  return rows


def env_get(row: dict, prefix: str, suffix: str):
  for name in ENV_NAMES:
    value = row.get(f"{prefix}env/{name}/{suffix}")
    if value is not None:
      return value
  return None


def eval_result(row: dict) -> dict | None:
  # Keep numerator and denominator in the same namespace. Prefer LAB-specific
  # counts over aggregates when both are present.
  for name in ENV_NAMES:
    prefix = f"test/env/{name}/"
    passed = row.get(prefix + "lab/criteria_passed")
    total = row.get(prefix + "lab/criteria_total")
    episodes = row.get(prefix + "total_episodes")
    reward_error = row.get(prefix + "lab/reward_error")
    if passed is not None and total:
      return {
        "pass_rate": passed / total,
        "aggregation": "pooled_criteria",
        "criteria_passed": passed * episodes if episodes else None,
        "criteria_total": total * episodes if episodes else None,
        "reward_errors": reward_error * episodes if (reward_error is not None and episodes) else 0.0,
        "episodes": episodes,
      }
  rate = env_get(row, "test/", "lab/criteria_pass_fraction")
  if rate is None:
    return None
  reward_error_rate = env_get(row, "test/", "lab/reward_error") or 0.0
  episodes = env_get(row, "test/", "total_episodes")
  return {
    "pass_rate": rate,
    "aggregation": "mean_episode_fraction",
    "criteria_passed": None,
    "criteria_total": None,
    "reward_errors": reward_error_rate * episodes if episodes else 0.0,
    "episodes": episodes,
  }


def rollout_rewards(log_dir: Path) -> list[tuple[int, float]]:
  points = []
  for summary in sorted(log_dir.glob("**/iteration_*/train_rollout_summaries.jsonl")):
    step = int(summary.parent.name.split("_")[-1]) + 1
    points.extend((step, row["total_reward"]) for row in load_metrics(summary))
  return sorted(points)


def read_results(log_dir: Path, *, include_metrics: bool = False) -> dict:
  rows = load_metrics(log_dir / "metrics.jsonl")
  train, evaluations = {}, {}
  for row in rows:
    batch = row.get("progress/batch", row.get("step"))
    if batch is None:
      continue
    reward = env_get(row, "", "reward/total")
    if reward is not None:
      train[batch + 1] = {"step": batch + 1, "reward": reward}
    result = eval_result(row)
    if result is not None:
      evaluations[batch] = {"step": batch, "phase": row.get("eval_phase", "evaluation"), **result}
  train_rows = [train[key] for key in sorted(train)]
  eval_rows = [evaluations[key] for key in sorted(evaluations)]
  final = [row for row in eval_rows if row["phase"] == "final"]
  latest = eval_rows[-1] if eval_rows else None
  comparable = [row for row in eval_rows if row["aggregation"] == latest["aggregation"]] if latest else []
  return {
    "schema_version": 1,
    "run_dir": str(log_dir),
    "step_unit": "completed_training_batches",
    "training_batches_with_metrics": len(train_rows),
    "last_train": train_rows[-1] if train_rows else None,
    "baseline_eval": evaluations.get(0),
    "latest_eval": latest,
    "best_eval": max(comparable, key=lambda row: row["pass_rate"]) if comparable else None,
    "final_eval": final[-1] if final else None,
    "train": train_rows,
    "evaluations": eval_rows,
    "rollout_rewards": [{"step": step, "reward": reward} for step, reward in rollout_rewards(log_dir)],
    **({"metrics": rows} if include_metrics else {}),
  }


def format_eval(result: dict | None, label: str = "Eval") -> str:
  if result is None:
    return f"{label}: unavailable"
  counts = ""
  if result["criteria_total"] is not None:
    counts = f", {result['criteria_passed']:.0f}/{result['criteria_total']:.0f} criteria"
  errors = f", {result.get('reward_errors', 0.0):.0f} errors" if result.get("reward_errors") else ""
  step = f"; after batch {result['step']}" if "step" in result else ""
  return f"{label}: {result['pass_rate']:.1%} ({result['aggregation']}{counts}{errors}{step})"


def format_summary(results: dict) -> str:
  lines = [f"Run: {results['run_dir']}", f"Training batches with metrics: {results['training_batches_with_metrics']}"]
  if row := results["last_train"]:
    lines.append(f"Latest training reward: {row['reward']:.4f} (after batch {row['step']})")
  for key, label in (("baseline_eval", "Baseline"), ("best_eval", "Best"), ("latest_eval", "Latest"), ("final_eval", "Final")):
    lines.append(format_eval(results[key], f"{label} eval"))
  return "\n".join(lines)


def main() -> None:
  args = chz.entrypoint(ResultsConfig, allow_hyphens=True)
  if args.metrics and not args.json:
    raise SystemExit("metrics=True requires json=True")
  try:
    results = read_results(args.log_dir, include_metrics=args.metrics)
    if args.plot:
      from .plot_run import write_report  # matplotlib is only needed here

      write_report(args.log_dir, results)
  except (OSError, ValueError) as exc:
    raise SystemExit(f"Cannot read run results: {exc}") from exc
  print(json.dumps(results, indent=2) if args.json else format_summary(results))


if __name__ == "__main__":
  main()
