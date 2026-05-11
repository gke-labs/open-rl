"""Render the 4-panel text-to-SQL recipe figure from a metrics.jsonl.

Usage:
  cd examples/rl/text-to-sql
  uv run python -m utils.plot <metrics.jsonl>

  # From the repository root:
  uv --project examples run python examples/rl/text-to-sql/utils/plot.py <metrics.jsonl>

By default, the plot is written to curves.png next to the metrics file. Pass an
optional output path to override it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def ewma(values: list[float], alpha: float = 0.1) -> list[float]:
  out = []
  s = None
  for v in values:
    s = v if s is None else alpha * v + (1 - alpha) * s
    out.append(s)
  return out


def sma(values: list[float], window: int) -> list[float]:
  if not values:
    return []
  arr = np.asarray(values, dtype=float)
  if window <= 1:
    return arr.tolist()
  kernel = np.ones(window) / window
  padded = np.concatenate([np.full(window - 1, arr[0]), arr])
  return np.convolve(padded, kernel, mode="valid").tolist()


def parse_metrics_file(path: Path | str) -> list[dict[str, Any]]:
  """Parse JSONL metrics file into a list of dicts."""
  path = Path(path)
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_exec_match(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
  """Panel 1: Held-out Exec Match (SFT markers + RL markers)."""
  sft_eval = [r for r in rows if r["phase"] == "sft_eval"]
  rl_eval = [r for r in rows if r["phase"] == "rl_eval"]
  baseline = [r for r in rows if r["phase"] == "eval_baseline"]
  if not baseline and rl_eval and rl_eval[0].get("step") == 0:
    baseline = [rl_eval[0]]
    rl_eval = rl_eval[1:]

  baseline_exec = baseline[0]["execution_match"] if baseline else (sft_eval[-1]["execution_match"] if sft_eval else 0)
  sft_final_exec = sft_eval[-1]["execution_match"] if sft_eval else baseline_exec
  final_exec = rl_eval[-1]["execution_match"] if rl_eval else sft_final_exec
  rl_gain = (final_exec - sft_final_exec) * 100

  if baseline:
    ax.plot([0], [baseline_exec * 100], "o", color="gray", markersize=8, label="Baseline")
  sft_x = [r["step"] for r in sft_eval]
  sft_y = [r["execution_match"] * 100 for r in sft_eval]
  if sft_x:
    ax.plot(sft_x, sft_y, "s-", color="steelblue", markersize=7, label="SFT phase")
  rl_x = [r["step"] for r in rl_eval]
  rl_y = [r["execution_match"] * 100 for r in rl_eval]
  if rl_x:
    ax.plot(rl_x, rl_y, "o-", color="crimson", markersize=7, label="RL phase (GRPO)")
  start_label = "Post-SFT" if sft_eval else "Baseline"
  ax.set_title(f"Held-out Exec Match: {start_label} {sft_final_exec * 100:.0f}% \u2192 RL {final_exec * 100:.0f}% (+{rl_gain:.0f}pt)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Exec match (%)")
  ax.grid(alpha=0.3)
  ax.legend()


def plot_rl_reward(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
  """Panel 2: RL reward (EWMA)."""
  rl_reward_probe = [r for r in rows if r["phase"] == "rl_reward_probe"]
  rl_train = [r for r in rows if r["phase"] == "rl_train"]
  reward_rows = rl_reward_probe or rl_train

  if reward_rows:
    xs = [r["step"] for r in reward_rows]
    ys = [r["reward"] for r in reward_rows]
    ewma_ys = ewma(ys, alpha=0.1)
    start, end = ewma_ys[0] if ewma_ys else 0, ewma_ys[-1] if ewma_ys else 0
    mult = (end / start) if start and abs(start) > 1e-6 else 0
    label = "Reward Probe (EWMA)" if rl_reward_probe else "Reward (EWMA)"
    ax.plot(xs, ys, "-", color="forestgreen", alpha=0.3, label="Raw Reward")
    ax.plot(xs, ewma_ys, "-", color="forestgreen", label=label)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"RL Reward ({'climbing' if end > start else 'falling'} {mult:.1f}x)")
  else:
    ax.set_title("RL Reward (no data)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Reward")
  ax.grid(alpha=0.3)
  ax.legend()


def plot_compile_rate(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
  """Panel 3: Compile rate (EWMA)."""
  rl_reward_probe = [r for r in rows if r["phase"] == "rl_reward_probe"]
  rl_train = [r for r in rows if r["phase"] == "rl_train"]
  compile_rows = rl_reward_probe or rl_train

  if compile_rows:
    xs = [r["step"] for r in compile_rows]
    # Support both 'compile_rate' (older format) and 'compile' (notebook format)
    ys = [r.get("compile_rate", r.get("compile", 0.0)) * 100 for r in compile_rows]
    ewma_ys = ewma(ys, alpha=0.1)
    direction = "climbing" if ewma_ys and ewma_ys[-1] > ewma_ys[0] else "falling"
    label = "Compile Probe (EWMA)" if rl_reward_probe else "Compile (EWMA)"
    ax.plot(xs, ys, "-", color="purple", alpha=0.3, label="Raw Compile")
    ax.plot(xs, ewma_ys, "-", color="purple", label=label)
    ax.set_title(f"Compile Rate ({direction})")
  else:
    ax.set_title("Compile Rate (no data)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Compile rate (%)")
  ax.grid(alpha=0.3)
  ax.legend()


def plot_training_loss(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
  """Panel 4: Training loss, using SFT when present and RL otherwise."""
  sft_train = [r for r in rows if r["phase"] == "sft_train"]
  rl_train = [r for r in rows if r["phase"] == "rl_train"]
  loss_rows = sft_train or rl_train
  label = "SFT Loss (SMA 5)" if sft_train else "RL Loss (SMA 5)"
  title = "SFT Training Loss" if sft_train else "RL Training Loss"

  if loss_rows:
    xs = [r["step"] for r in loss_rows]
    ys = [r["loss"] for r in loss_rows]
    ax.plot(xs, sma(ys, 5), "-", color="royalblue", label=label)
    ax.set_title(title)
  else:
    ax.set_title("Training Loss (no data)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Loss")
  ax.grid(alpha=0.3)
  ax.legend()


def render_recipe_plots(rows: list[dict[str, Any]], out_path: Path | str | None = None) -> plt.Figure:
  """Render the standard 4-panel plot and optionally save it to disk."""
  sft_train = [r for r in rows if r["phase"] == "sft_train"]
  rl_train = [r for r in rows if r["phase"] == "rl_train"]
  sft_eval = [r for r in rows if r["phase"] == "sft_eval"]
  rl_eval = [r for r in rows if r["phase"] == "rl_eval"]
  baseline = [r for r in rows if r["phase"] == "eval_baseline"]
  if not baseline and rl_eval and rl_eval[0].get("step") == 0:
    baseline = [rl_eval[0]]

  sft_step_count = len(sft_train)
  rl_step_count = len(rl_train)

  baseline_exec = baseline[0]["execution_match"] if baseline else (sft_eval[-1]["execution_match"] if sft_eval else 0)
  sft_final_exec = sft_eval[-1]["execution_match"] if sft_eval else baseline_exec
  final_exec = rl_eval[-1]["execution_match"] if rl_eval else sft_final_exec
  rl_gain = (final_exec - sft_final_exec) * 100

  fig, axes = plt.subplots(2, 2, figsize=(13, 9))
  if sft_step_count:
    title = (
      f"Our run: {sft_step_count} SFT + {rl_step_count} RL steps | "
      f"{baseline_exec * 100:.0f}% \u2192 {sft_final_exec * 100:.0f}% (SFT) \u2192 {final_exec * 100:.0f}% (RL) "
      f"| +{rl_gain:.0f}pt RL"
    )
  else:
    title = f"Our run: {rl_step_count} RL steps | {baseline_exec * 100:.0f}% baseline \u2192 {final_exec * 100:.0f}% RL | +{rl_gain:.0f}pt RL"

  fig.suptitle(title, fontsize=12, fontweight="bold")

  plot_exec_match(axes[0, 0], rows)
  plot_rl_reward(axes[0, 1], rows)
  plot_compile_rate(axes[1, 0], rows)
  plot_training_loss(axes[1, 1], rows)

  fig.tight_layout(rect=(0, 0, 1, 0.95))

  if out_path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")

  return fig


def main() -> int:
  if len(sys.argv) < 2:
    print(
      "Usage:\n"
      "  cd examples/rl/text-to-sql && uv run python -m utils.plot <metrics.jsonl> [out.png]\n"
      "  uv --project examples run python examples/rl/text-to-sql/utils/plot.py <metrics.jsonl> [out.png]\n"
      "\n"
      "If [out.png] is omitted, the plot is written to curves.png next to the metrics file."
    )
    return 2

  path = Path(sys.argv[1])
  out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_name("curves.png")

  rows = parse_metrics_file(path)
  if not rows:
    print(f"No rows or file not found at {path}")
    return 1

  render_recipe_plots(rows, out_path)
  return 0


if __name__ == "__main__":
  sys.exit(main())
