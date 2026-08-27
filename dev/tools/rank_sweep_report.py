# Analysis for the gsm8k-rl-rank-sweep scenario: does LoRA match full
# fine-tuning on RL, and does rank 1 suffice?
#
# Reads the per-arm metrics.jsonl written by each arm and emits the artifacts
# that decide the claim from docs/designs/012:
#   - a tidy CSV, one row per (arm, step)
#   - a summary table: tail-window mean reward per arm, and its gap to FullFT
#   - reward vs step, all arms overlaid (the primary plot)
#   - tail-window reward per arm against the FullFT reference (the summary plot)
#
# Usage:
#   uv run python rank_sweep_report.py --runs-dir <dir-with-per-arm-subdirs>
#   uv run python rank_sweep_report.py --runs-dir . --tail 10 --smooth 5

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REWARD_KEY = "env/all/reward/total"

# Categorical slots 1-3 of the validated default palette, assigned in fixed
# order so an arm keeps its hue no matter which arms are present.
ARM_COLORS = {
  "fullft": "#2a78d6",
  "lora-r32": "#eb6834",
  "lora-r1": "#1baf7a",
}
ARM_ORDER = ["fullft", "lora-r32", "lora-r1"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def load_arms(runs_dir: Path) -> pd.DataFrame:
  """Read every */metrics.jsonl under runs_dir into one tidy frame."""
  rows = []
  for path in sorted(runs_dir.glob("*/metrics.jsonl")):
    arm = path.parent.name.replace("gsm8k_rl_rank_sweep_", "")
    for line in path.read_text().splitlines():
      if not line.strip():
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue
      if REWARD_KEY not in rec:
        continue
      rows.append(
        {
          "arm": arm,
          "step": rec.get("step"),
          "reward": rec[REWARD_KEY],
          "entropy": rec.get("optim/entropy"),
          "kl": rec.get("optim/kl_sample_train_v1"),
          "step_seconds": rec.get("time/total"),
        }
      )
  if not rows:
    raise SystemExit(f"No metrics.jsonl with '{REWARD_KEY}' found under {runs_dir}")
  df = pd.DataFrame(rows).sort_values(["arm", "step"]).reset_index(drop=True)
  return df


def summarize(df: pd.DataFrame, tail: int) -> tuple[pd.DataFrame, int]:
  """Tail-window statistics per arm, and each arm's gap to FullFT.

  Arms are truncated to the shortest arm's step count first. They progress at
  different rates -- arms sharing a GPU take longer per step -- so a tail taken
  over unequal-length runs would compare one arm's late training against
  another's early training and report the difference as an arm effect.

  The tail window stands in for a variance estimate that repeat seeds would
  give properly. It bounds how much of a gap is attributable to step-to-step
  noise within a single run; it cannot separate a real gap from seed luck.
  """
  common = int(df.groupby("arm")["step"].count().min())
  out = []
  for arm, g in df.groupby("arm"):
    g = g.head(common)
    window = g.tail(tail)
    out.append(
      {
        "arm": arm,
        "steps": len(g),
        "tail_mean": window["reward"].mean(),
        "tail_std": window["reward"].std(),
        "final": g["reward"].iloc[-1],
        "sec_per_step": g["step_seconds"].mean(),
      }
    )
  summary = pd.DataFrame(out)
  summary["arm"] = pd.Categorical(summary["arm"], [a for a in ARM_ORDER if a in set(summary["arm"])], ordered=True)
  summary = summary.sort_values("arm").reset_index(drop=True)
  if "fullft" in set(summary["arm"]):
    ref = float(summary.loc[summary["arm"] == "fullft", "tail_mean"].iloc[0])
    summary["gap_vs_fullft"] = summary["tail_mean"] - ref
  return summary, common


def _style(ax) -> None:
  ax.set_facecolor(SURFACE)
  ax.figure.set_facecolor(SURFACE)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color("#d9d8d3")
  ax.tick_params(colors=INK_SECONDARY, labelsize=9)
  ax.grid(axis="y", color="#ececE7", linewidth=0.8)
  ax.set_axisbelow(True)


def plot_reward_curves(df: pd.DataFrame, smooth: int, out: Path) -> None:
  """Reward vs step, all arms overlaid: the plot that decides the claim."""
  fig, ax = plt.subplots(figsize=(9, 5))
  _style(ax)
  for arm in [a for a in ARM_ORDER if a in set(df["arm"])]:
    g = df[df["arm"] == arm]
    color = ARM_COLORS[arm]
    # Raw reward stays visible but recessive; the rolling mean carries the
    # comparison. Hiding the raw series would flatter the claim.
    ax.plot(g["step"], g["reward"], color=color, linewidth=1, alpha=0.25)
    rolled = g["reward"].rolling(smooth, min_periods=1).mean()
    ax.plot(g["step"], rolled, color=color, linewidth=2, label=arm)
    # Direct label at the line end: identity without relying on colour alone,
    # which the aqua slot needs anyway on this surface.
    ax.annotate(
      arm,
      (g["step"].iloc[-1], rolled.iloc[-1]),
      xytext=(6, 0),
      textcoords="offset points",
      color=INK_SECONDARY,
      fontsize=9,
      va="center",
    )
  ax.set_xlabel("step", color=INK_SECONDARY, fontsize=10)
  ax.set_ylabel(f"reward (rolling mean, {smooth} steps)", color=INK_SECONDARY, fontsize=10)
  ax.set_title("GSM8K RL: FullFT vs LoRA rank 32 vs LoRA rank 1", color=INK_PRIMARY, fontsize=13, loc="left", pad=14)
  ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="upper left")
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print(f"wrote {out}")


def plot_tail_summary(summary: pd.DataFrame, tail: int, out: Path) -> None:
  """Tail-window reward per arm against the FullFT reference line."""
  fig, ax = plt.subplots(figsize=(7, 4))
  _style(ax)
  arms = list(summary["arm"])
  xs = range(len(arms))
  if "fullft" in arms:
    ref = float(summary.loc[summary["arm"] == "fullft", "tail_mean"].iloc[0])
    ax.axhline(ref, color=ARM_COLORS["fullft"], linewidth=1, linestyle="--", alpha=0.6)
    ax.annotate(
      "FullFT reference",
      (len(arms) - 0.5, ref),
      xytext=(0, 6),
      textcoords="offset points",
      color=INK_SECONDARY,
      fontsize=9,
      ha="right",
    )
  for x, (_, row) in zip(xs, summary.iterrows(), strict=False):
    color = ARM_COLORS.get(str(row["arm"]), INK_SECONDARY)
    spread = 0.0 if pd.isna(row["tail_std"]) else row["tail_std"]
    ax.errorbar(x, row["tail_mean"], yerr=spread, fmt="o", markersize=9, color=color, ecolor=color, elinewidth=2, capsize=5)
    ax.annotate(
      f"{row['tail_mean']:.3f}",
      (x, row["tail_mean"]),
      xytext=(12, 0),
      textcoords="offset points",
      color=INK_SECONDARY,
      fontsize=9,
      va="center",
    )
  ax.set_xticks(list(xs))
  ax.set_xticklabels(arms, color=INK_SECONDARY)
  ax.set_ylabel(f"mean reward, last {tail} steps", color=INK_SECONDARY, fontsize=10)
  ax.set_title("Does rank 1 keep up?", color=INK_PRIMARY, fontsize=13, loc="left", pad=14)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print(f"wrote {out}")


def _markdown_table(summary: pd.DataFrame) -> str:
  """Render the summary as markdown without pulling in a table dependency."""
  cols = list(summary.columns)

  def cell(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)

  lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
  for _, row in summary.iterrows():
    lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
  return "\n".join(lines) + "\n"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--runs-dir", required=True, type=Path, help="directory holding one subdir per arm")
  ap.add_argument("--out-dir", type=Path, default=Path("."), help="where to write CSV and plots")
  ap.add_argument("--tail", type=int, default=10, help="tail window for the summary statistics")
  ap.add_argument("--smooth", type=int, default=5, help="rolling-mean window for the curves")
  args = ap.parse_args()

  df = load_arms(args.runs_dir)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  csv_path = args.out_dir / "rank_sweep.csv"
  df.to_csv(csv_path, index=False)
  print(f"wrote {csv_path}  ({len(df)} rows, arms: {', '.join(sorted(set(df['arm'])))})")

  summary, common = summarize(df, args.tail)
  print(f"\ncomparing the first {common} steps of every arm (shortest arm bounds the window)")
  print()
  print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
  print()
  (args.out_dir / "rank_sweep_summary.md").write_text(_markdown_table(summary))

  plot_reward_curves(df, args.smooth, args.out_dir / "rank_sweep_reward.png")
  plot_tail_summary(summary, args.tail, args.out_dir / "rank_sweep_tail.png")

  print(
    "\nOne seed per arm: the tail spread bounds within-run noise only. A gap of\n"
    "that order is not evidence of a real difference between arms -- repeat\n"
    "seeds are what separate rank from seed luck (docs/designs/012 §6)."
  )


if __name__ == "__main__":
  main()
