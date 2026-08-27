# Analysis for the gsm8k-rl-rank-sweep scenarios: does LoRA match full
# fine-tuning on RL, and does rank 1 suffice?
#
# Reads the per-run metrics.jsonl written by each run and emits the artifacts
# that decide the claim from docs/designs/012:
#   - a tidy CSV, one row per (run, step)
#   - a per-run summary: tail-window mean reward, and the gap to FullFT
#   - a per-configuration summary that folds replicate seeds together, giving
#     the replicate spread that says whether a gap means anything
#   - reward vs step, all runs overlaid (the primary plot)
#   - tail-window reward per configuration against the FullFT reference
#
# Run names carry their own structure. The replicated sweep emits
# "<group>-<tag>-<replicate>" (small-r1-a, large-fft-b); the single-seed sweep
# emits a bare configuration (fullft, lora-r1). Both parse into the same three
# fields, so one set of tools handles either shape.
#
# Usage:
#   uv run python rank_sweep_report.py --runs-dir <dir-with-per-run-subdirs>
#   uv run python rank_sweep_report.py --runs-dir . --group large --tail 10

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REWARD_KEY = "env/all/reward/total"

# Validated categorical palette. Colour tracks the configuration under test, so
# a configuration keeps its hue no matter which others are present; replicates
# of one configuration share a colour and differ by line style.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#9457c9", "#c9a227", "#3aa3a3"]
REPLICATE_STYLES = ["-", "--", ":", "-."]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

_RUN_RE = re.compile(r"^(?P<group>[a-z0-9.]+)-(?P<tag>fft|r\d+)-(?P<replicate>[a-z])$")


def parse_run_name(name: str) -> tuple[str | None, str, str | None]:
  """Split a run directory name into (group, configuration, replicate).

  Names that do not match the replicated pattern are treated as a single
  unreplicated configuration, which is what the earlier sweep produced.
  """
  m = _RUN_RE.match(name)
  if not m:
    return None, name, None
  tag = m.group("tag")
  config = "fullft" if tag == "fft" else f"lora-{tag}"
  return m.group("group"), config, m.group("replicate")


def config_order(configs) -> list[str]:
  """FullFT first, then LoRA by descending rank: the order the claim is argued in."""

  def key(c: str) -> tuple[int, int]:
    if c == "fullft":
      return (0, 0)
    m = re.match(r"lora-r(\d+)$", c)
    return (1, -int(m.group(1))) if m else (2, 0)

  return sorted(set(configs), key=key)


def config_colors(configs) -> dict[str, str]:
  return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(config_order(configs))}


def replicate_styles(replicates) -> dict[str | None, str]:
  present = sorted({r for r in replicates if r is not None})
  styles = {r: REPLICATE_STYLES[i % len(REPLICATE_STYLES)] for i, r in enumerate(present)}
  styles[None] = "-"
  return styles


def run_label(config: str, replicate: str | None) -> str:
  return config if replicate is None else f"{config} ({replicate})"


def load_runs(runs_dir: Path, group: str | None = None) -> pd.DataFrame:
  """Read every */metrics.jsonl under runs_dir into one tidy frame.

  ``group`` keeps only runs from one model size, since the replicated sweep
  writes every size into a single directory.
  """
  rows = []
  for path in sorted(runs_dir.glob("*/metrics.jsonl")):
    name = path.parent.name.replace("gsm8k_rl_rank_sweep_", "").replace("gsm8k_rl_mega_", "")
    run_group, config, replicate = parse_run_name(name)
    if group is not None and run_group != group:
      continue
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
          "run": name,
          "group": run_group,
          "config": config,
          "replicate": replicate,
          "step": rec.get("step"),
          "reward": rec[REWARD_KEY],
          "entropy": rec.get("optim/entropy"),
          "kl": rec.get("optim/kl_sample_train_v1"),
          "step_seconds": rec.get("time/total"),
        }
      )
  if not rows:
    where = f"{runs_dir}" + (f" (group={group})" if group else "")
    raise SystemExit(f"No metrics.jsonl with '{REWARD_KEY}' found under {where}")
  return pd.DataFrame(rows).sort_values(["config", "replicate", "step"]).reset_index(drop=True)


def summarize(df: pd.DataFrame, tail: int) -> tuple[pd.DataFrame, int]:
  """Tail-window statistics per run, and each run's gap to FullFT.

  Runs are truncated to the shortest run's step count first. They progress at
  different rates -- runs sharing a GPU take longer per step -- so a tail taken
  over unequal-length runs would compare one run's late training against
  another's early training and report the difference as a configuration effect.

  The tail window bounds step-to-step noise inside one run. It cannot separate
  a real gap from seed luck; that is what the replicate spread in
  ``summarize_configs`` is for.
  """
  common = int(df.groupby("run")["step"].count().min())
  out = []
  for run_name, g in df.groupby("run"):
    g = g.head(common)
    window = g.tail(tail)
    out.append(
      {
        "run": run_name,
        "config": g["config"].iloc[0],
        "replicate": g["replicate"].iloc[0],
        "steps": len(g),
        "tail_mean": window["reward"].mean(),
        "tail_std": window["reward"].std(),
        "final": g["reward"].iloc[-1],
        "sec_per_step": g["step_seconds"].mean(),
      }
    )
  summary = pd.DataFrame(out)
  order = config_order(summary["config"])
  summary["config"] = pd.Categorical(summary["config"], order, ordered=True)
  summary = summary.sort_values(["config", "replicate"]).reset_index(drop=True)
  ref_rows = summary[summary["config"] == "fullft"]
  if not ref_rows.empty:
    summary["gap_vs_fullft"] = summary["tail_mean"] - float(ref_rows["tail_mean"].mean())
  return summary, common


def summarize_configs(summary: pd.DataFrame) -> pd.DataFrame:
  """Fold replicates together: mean per configuration, and how far the replicates sit apart.

  The replicate spread is the measurement that makes a gap interpretable. Two
  runs differing only in data-shuffling seed bound how much of any difference
  is seed luck, which the within-run tail deviation cannot do.
  """
  out = []
  for config, g in summary.groupby("config", observed=True):
    means = list(g["tail_mean"])
    out.append(
      {
        "config": config,
        "replicates": len(means),
        "mean": sum(means) / len(means),
        "spread": (max(means) - min(means)) if len(means) > 1 else float("nan"),
        "sec_per_step": g["sec_per_step"].mean(),
      }
    )
  configs = pd.DataFrame(out)
  ref = configs.loc[configs["config"] == "fullft", "mean"]
  if not ref.empty:
    configs["gap_vs_fullft"] = configs["mean"] - float(ref.iloc[0])
  return configs


def noise_floor(configs: pd.DataFrame) -> float:
  """Largest spread between replicates of any one configuration.

  A gap smaller than this is not distinguishable from seed variation with the
  replicates available.
  """
  spreads = configs["spread"].dropna()
  return float(spreads.max()) if not spreads.empty else float("nan")


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
  """Reward vs step, every run overlaid: the plot that decides the claim.

  Colour is the configuration and line style is the replicate, so replicates of
  one configuration read as a band rather than as separate results.
  """
  fig, ax = plt.subplots(figsize=(9, 5))
  _style(ax)
  colors = config_colors(df["config"])
  styles = replicate_styles(df["replicate"])
  for config in config_order(df["config"]):
    sub = df[df["config"] == config]
    color = colors[config]
    for replicate, g in sub.groupby("replicate", dropna=False):
      g = g.sort_values("step")
      rep = None if pd.isna(replicate) else replicate
      ax.plot(g["step"], g["reward"], color=color, linewidth=1, alpha=0.15)
      rolled = g["reward"].rolling(smooth, min_periods=1).mean()
      ax.plot(g["step"], rolled, color=color, linewidth=2, linestyle=styles[rep], label=run_label(config, rep))
  ax.set_xlabel("step", color=INK_SECONDARY, fontsize=10)
  ax.set_ylabel(f"reward (rolling mean, {smooth} steps)", color=INK_SECONDARY, fontsize=10)
  ax.set_title("GSM8K RL: full fine-tuning vs LoRA by rank", color=INK_PRIMARY, fontsize=13, loc="left", pad=14)
  ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower right", ncol=2)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print(f"wrote {out}")


def plot_tail_summary(configs: pd.DataFrame, tail: int, out: Path) -> None:
  """Tail-window reward per configuration, with the replicate spread as the error bar.

  The error bar is the whole point: it is the distance between two runs that
  differ only in seed, so any configuration gap smaller than it is noise.
  """
  fig, ax = plt.subplots(figsize=(7, 4))
  _style(ax)
  names = [str(c) for c in configs["config"]]
  colors = config_colors(names)
  xs = range(len(names))
  ref_rows = configs[configs["config"] == "fullft"]
  if not ref_rows.empty:
    ref = float(ref_rows["mean"].iloc[0])
    ax.axhline(ref, color=colors["fullft"], linewidth=1, linestyle="--", alpha=0.6)
    ax.annotate(
      "FullFT reference",
      (len(names) - 0.5, ref),
      xytext=(0, 6),
      textcoords="offset points",
      color=INK_SECONDARY,
      fontsize=9,
      ha="right",
    )
  for x, (_, row) in zip(xs, configs.iterrows(), strict=False):
    color = colors.get(str(row["config"]), INK_SECONDARY)
    half = 0.0 if pd.isna(row["spread"]) else row["spread"] / 2
    ax.errorbar(x, row["mean"], yerr=half, fmt="o", markersize=9, color=color, ecolor=color, elinewidth=2, capsize=5)
    ax.annotate(
      f"{row['mean']:.3f}",
      (x, row["mean"]),
      xytext=(12, 0),
      textcoords="offset points",
      color=INK_SECONDARY,
      fontsize=9,
      va="center",
    )
  ax.set_xticks(list(xs))
  ax.set_xticklabels(names, color=INK_SECONDARY)
  ax.set_ylabel(f"mean reward, last {tail} steps", color=INK_SECONDARY, fontsize=10)
  ax.set_title("Does rank 1 keep up? (bars span the replicate spread)", color=INK_PRIMARY, fontsize=13, loc="left", pad=14)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print(f"wrote {out}")


def _markdown_table(frame: pd.DataFrame) -> str:
  """Render a frame as markdown without pulling in a table dependency."""
  cols = list(frame.columns)

  def cell(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)

  lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
  for _, row in frame.iterrows():
    lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
  return "\n".join(lines) + "\n"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--runs-dir", required=True, type=Path, help="directory holding one subdir per run")
  ap.add_argument("--group", help="keep only runs from one model group (e.g. small, large)")
  ap.add_argument("--out-dir", type=Path, default=Path("."), help="where to write CSV and plots")
  ap.add_argument("--tail", type=int, default=10, help="tail window for the summary statistics")
  ap.add_argument("--smooth", type=int, default=5, help="rolling-mean window for the curves")
  args = ap.parse_args()

  df = load_runs(args.runs_dir, group=args.group)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  csv_path = args.out_dir / "rank_sweep.csv"
  df.to_csv(csv_path, index=False)
  print(f"wrote {csv_path}  ({len(df)} rows, runs: {', '.join(sorted(set(df['run'])))})")

  summary, common = summarize(df, args.tail)
  configs = summarize_configs(summary)
  print(f"\ncomparing the first {common} steps of every run (shortest bounds the window)")
  print()
  print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
  print()
  print(configs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
  print()
  (args.out_dir / "rank_sweep_summary.md").write_text(_markdown_table(summary))
  (args.out_dir / "rank_sweep_configs.md").write_text(_markdown_table(configs))

  plot_reward_curves(df, args.smooth, args.out_dir / "rank_sweep_reward.png")
  plot_tail_summary(configs, args.tail, args.out_dir / "rank_sweep_tail.png")

  floor = noise_floor(configs)
  if pd.isna(floor):
    print(
      "\nOne seed per configuration: the tail spread bounds within-run noise only. A gap\n"
      "of that order is not evidence of a real difference -- repeat seeds are what\n"
      "separate rank from seed luck (docs/designs/012 §6)."
    )
  else:
    biggest = configs["gap_vs_fullft"].abs().max() if "gap_vs_fullft" in configs else float("nan")
    verdict = "within" if biggest <= floor else "larger than"
    print(
      f"\nReplicate spread (largest between two seeds of one configuration): {floor:.4f}\n"
      f"Largest gap to full fine-tuning: {biggest:.4f} -- {verdict} the replicate spread."
    )


if __name__ == "__main__":
  main()
