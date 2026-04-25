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

  rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  if not rows:
    print(f"No rows in {path}")
    return 1

  sft_train = [r for r in rows if r["phase"] == "sft_train"]
  rl_train = [r for r in rows if r["phase"] == "rl_train"]
  rl_reward_probe = [r for r in rows if r["phase"] == "rl_reward_probe"]
  sft_eval = [r for r in rows if r["phase"] == "sft_eval"]
  rl_eval = [r for r in rows if r["phase"] == "rl_eval"]
  baseline = [r for r in rows if r["phase"] == "eval_baseline"]

  sft_step_count = len(sft_train)
  rl_step_count = len(rl_train)
  final_exec = rl_eval[-1]["execution_match"] if rl_eval else (sft_eval[-1]["execution_match"] if sft_eval else 0)
  baseline_exec = baseline[0]["execution_match"] if baseline else (sft_eval[-1]["execution_match"] if sft_eval else 0)
  sft_final_exec = sft_eval[-1]["execution_match"] if sft_eval else baseline_exec
  baseline_exec = baseline[0]["execution_match"] if baseline else sft_final_exec
  rl_gain = (final_exec - sft_final_exec) * 100

  fig, axes = plt.subplots(2, 2, figsize=(13, 9))
  fig.suptitle(
    f"Our run: {sft_step_count} SFT + {rl_step_count} RL steps | "
    f"{baseline_exec * 100:.0f}% \u2192 {sft_final_exec * 100:.0f}% (SFT) \u2192 {final_exec * 100:.0f}% (RL) "
    f"| +{rl_gain:.0f}pt RL",
    fontsize=12,
    fontweight="bold",
  )

  # Panel 1: Held-out Exec Match (SFT markers + RL markers)
  ax = axes[0, 0]
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
  ax.set_title(f"Held-out Exec Match: {sft_final_exec * 100:.0f}% \u2192 {final_exec * 100:.0f}% (+{rl_gain:.0f}pt)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Exec match (%)")
  ax.grid(alpha=0.3)
  ax.legend()

  # Panel 2: RL reward (EWMA)
  ax = axes[0, 1]
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

  # Panel 3: Compile rate (EWMA)
  ax = axes[1, 0]
  compile_rows = rl_reward_probe or rl_train
  if compile_rows:
    xs = [r["step"] for r in compile_rows]
    ys = [r["compile_rate"] * 100 for r in compile_rows]
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

  # Panel 4: SFT training loss (SMA 5)
  ax = axes[1, 1]
  if sft_train:
    xs = [r["step"] for r in sft_train]
    ys = [r["loss"] for r in sft_train]
    ax.plot(xs, sma(ys, 5), "-", color="royalblue", label="SFT Loss (SMA 5)")
    ax.set_title("SFT Training Loss")
  else:
    ax.set_title("SFT Training Loss (no data)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Loss")
  ax.grid(alpha=0.3)
  ax.legend()

  fig.tight_layout(rect=(0, 0, 1, 0.95))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=140)
  print(f"Wrote {out_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
