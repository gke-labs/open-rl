import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(file_path: str, output_file: str | None = None) -> bool:
  """Plot each numeric metric against `step`, or row order if `step` is absent."""
  if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    return False

  print(f"Reading metrics from {file_path}...")
  try:
    df = pd.read_json(file_path, lines=True)
  except Exception as e:
    print(f"Error reading JSONL file: {e}")
    return False

  if df.empty:
    print("Metrics file is empty.")
    return False

  # Most training metrics files include an explicit `step` column.
  # For generic JSONL input, fall back to plotting by row order.
  if "step" not in df.columns:
    df["step"] = range(len(df))

  metric_columns = [column for column in df.columns if column != "step" and pd.api.types.is_numeric_dtype(df[column])]
  if not metric_columns:
    print("No numeric metrics found in JSONL file.")
    return False

  fig, axes = plt.subplots(len(metric_columns), 1, figsize=(10, 4 * len(metric_columns)), sharex=True)
  if len(metric_columns) == 1:
    axes = [axes]

  for axis, column in zip(axes, metric_columns):
    axis.plot(df["step"], df[column], marker="o", markersize=3, label=column)
    axis.set_ylabel(column)
    axis.set_title(column.replace("_", " ").title())
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

  axes[-1].set_xlabel("Step")
  plt.tight_layout()

  output_path = output_file or str(Path(file_path).with_suffix(".png"))
  plt.savefig(output_path)
  print(f"Plot saved to {output_path}")
  plt.close(fig)
  return True


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot numeric metrics from a JSONL file against `step` (or row order if `step` is missing).")
  parser.add_argument(
    "file_path",
    nargs="?",
    default="/tmp/tinker-examples/sl-loop/metrics.jsonl",
    help="Path to the metrics JSONL file. Numeric columns are plotted against `step` when present.",
  )
  parser.add_argument("--output", default=None, help="Optional output path for the generated PNG")
  args = parser.parse_args()
  plot_metrics(args.file_path, args.output)
