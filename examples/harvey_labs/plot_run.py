#!/usr/bin/env python3
"""Plot a LAB run from its log directory: raw per-rollout rewards, the
smoothed per-step mean, and held-out criterion pass rate at eval steps.

  python plot_run.py artifacts/harvey-labs/<run> [--out run.png]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RAW_COLOR = "#86b6ef"
SMOOTH_COLOR = "#2a78d6"
EVAL_COLOR = "#008300"
INK = "#3d3d3a"
MUTED = "#7f7e78"


def load_metrics(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rollout_rewards(log_dir: Path) -> list[tuple[int, float]]:
  points = []
  for summary in sorted(log_dir.glob("iterations/iteration_*/train_rollout_summaries.jsonl")):
    step = int(summary.parent.name.split("_")[-1])
    for line in summary.read_text().splitlines():
      if line.strip():
        points.append((step, json.loads(line)["total_reward"]))
  return points


def eval_pass_rate(row: dict) -> float | None:
  episodes = row.get("test/env/harvey-labs/total_episodes")
  passed = row.get("test/env/harvey-labs/lab/criteria_passed")
  total = row.get("test/env/harvey-labs/lab/criteria_total")
  graded_marker = row.get("test/env/harvey-labs/lab/graded")
  if episodes and total and graded_marker is not None:
    return (passed * episodes) / (total * episodes)
  return row.get("test/env/harvey-labs/lab/criteria_pass_fraction")


def ema(values: list[float], alpha: float = 0.4) -> list[float]:
  smoothed = []
  for value in values:
    smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
  return smoothed


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = load_metrics(args.log_dir / "metrics.jsonl")
  raw = rollout_rewards(args.log_dir)
  train_rows = [(r["step"], r["env/harvey-labs/reward/total"]) for r in rows if "env/harvey-labs/reward/total" in r and "step" in r]
  evals = [(r["step"], eval_pass_rate(r)) for r in rows if eval_pass_rate(r) is not None and "step" in r]

  fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
  rng = random.Random(0)
  if raw:
    xs = [step + rng.uniform(-0.18, 0.18) for step, _ in raw]
    ax.scatter(xs, [reward for _, reward in raw], s=14, color=RAW_COLOR, alpha=0.6, linewidths=0, label="rollout reward", zorder=2)
  if train_rows:
    steps = [step for step, _ in train_rows]
    smoothed = ema([reward for _, reward in train_rows])
    ax.plot(steps, smoothed, color=SMOOTH_COLOR, linewidth=2, label="mean reward (EMA)", zorder=3)
    ax.annotate(f"{smoothed[-1]:.2f}", (steps[-1], smoothed[-1]), textcoords="offset points", xytext=(6, -3), color=SMOOTH_COLOR, fontsize=9)
  if evals:
    ex = [step for step, _ in evals]
    ey = [rate for _, rate in evals]
    ax.plot(ex, ey, color=EVAL_COLOR, linewidth=2, linestyle=(0, (2, 3)), zorder=3)
    ax.scatter(ex, ey, s=64, color=EVAL_COLOR, marker="D", label="held-out criterion pass rate", zorder=4)
    for x, y in evals:
      ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", color=EVAL_COLOR, fontsize=9)

  ax.set_ylim(-0.15, 1.05)
  ax.set_xlabel("step", color=INK)
  ax.set_ylabel("reward / pass rate", color=INK)
  ax.set_title(args.log_dir.name, color=INK, fontsize=11, loc="left")
  ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    ax.spines[spine].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)
  ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

  out = args.out or args.log_dir / "run_plot.png"
  fig.tight_layout()
  fig.savefig(out)
  print(out)


if __name__ == "__main__":
  main()
