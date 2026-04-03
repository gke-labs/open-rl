import argparse
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def parse_logs(log_file):
  if not os.path.exists(log_file):
    print(f"Error: Log file '{log_file}' not found.")
    return None

  metric_pattern = re.compile(r"^\[([\w-]+)\s*\]\s+(?:(\d{2}:\d{2}:\d{2})\s+\|\s+)?(\d+)\s+\|\s+([-\d\.]+)\s+\|\s+([-\d\.]+)%?")

  data = defaultdict(lambda: {"iter": [], "time": [], "reward": [], "acc": []})

  with open(log_file) as f:
    for line in f:
      line = line.strip()
      match = metric_pattern.match(line)
      if match:
        tag = match.group(1)
        time_str = match.group(2)
        iteration = int(match.group(3))
        reward = float(match.group(4))
        acc = float(match.group(5))

        dt = datetime.strptime(time_str, "%H:%M:%S") if time_str else None

        data[tag]["iter"].append(iteration)
        data[tag]["time"].append(dt)
        data[tag]["reward"].append(reward)
        data[tag]["acc"].append(acc)

  return data


def plot_combined(data, x_axis="iter", output_file="combined_metrics.png"):
  if not data:
    print("No metrics found in log file.")
    return

  tags = sorted(data.keys())
  n_tags = len(tags)

  if n_tags == 0:
    return

  fig, axes = plt.subplots(n_tags, 2, figsize=(12, 4 * n_tags), sharex=True)
  if n_tags == 1:
    axes = [axes]

  for i, tag in enumerate(tags):
    iters = data[tag]["iter"]
    times = data[tag]["time"]

    has_time = times and times[0] is not None
    actual_x_axis = "time" if (x_axis == "time" and has_time) else "iter"
    x_data = times if actual_x_axis == "time" else iters

    rewards = data[tag]["reward"]
    accs = data[tag]["acc"]

    ax_reward = axes[i][0]
    ax_reward.plot(x_data, rewards, "b-o", label="Reward")
    ax_reward.set_title(f"Job: {tag} - Reward")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True)

    ax_acc = axes[i][1]
    ax_acc.plot(x_data, accs, "g-o", label="Accuracy")
    ax_acc.set_title(f"Job: {tag} - Accuracy")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True)

    if actual_x_axis == "time":
      ax_reward.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
      ax_acc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    if i == n_tags - 1:
      label = "Time" if actual_x_axis == "time" else "Iteration"
      ax_reward.set_xlabel(label)
      ax_acc.set_xlabel(label)

  plt.tight_layout()
  plt.savefig(output_file)
  print(f"[{actual_x_axis.upper()}] plot saved to {output_file}")
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot parallel RLVR metrics from log file.")
  parser.add_argument("log_file", nargs="?", default="rlvr_parallel_results.log", help="Path to the rlvr_parallel_results.log file")
  parser.add_argument("--watch", action="store_true", help="Monitor log file and update plot continuously")
  parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")

  args = parser.parse_args()

  if args.watch:
    import time

    print(f"Monitoring {args.log_file} every {args.interval} seconds...")
    try:
      while True:
        if not os.path.exists(args.log_file):
          print(f"Waiting for log file: {args.log_file} (retrying in {args.interval}s)...", end="\r")
        else:
          data = parse_logs(args.log_file)
          if data:
            plot_combined(data, x_axis="iter", output_file="combined_metrics_steps.png")
            plot_combined(data, x_axis="time", output_file="combined_metrics_time.png")
        time.sleep(args.interval)
    except KeyboardInterrupt:
      print("\nStopped monitoring.")
  else:
    data = parse_logs(args.log_file)
    if data:
      plot_combined(data, x_axis="iter", output_file="combined_metrics_steps.png")
      plot_combined(data, x_axis="time", output_file="combined_metrics_time.png")
