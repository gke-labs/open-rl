import json
import os


def main():
  metrics_path = "/mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/fft_gsm8k_rl/metrics.jsonl"
  if not os.path.exists(metrics_path):
    print("Waiting for step 0 metrics...")
    return

  with open(metrics_path) as f:
    rows = [json.loads(line) for line in f if line.strip()]

  print("Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time")
  print("-" * 80)
  for row in rows:
    if "env/all/correct" in row:
      step = row.get("progress/batch", "?")
      corr = row.get("env/all/correct", 0.0)
      rew = row.get("env/all/reward/total", 0.0)
      t_samp = row.get("time/sampling", 0.0)
      t_train = row.get("time/train_step", 0.0)
      t_save = row.get("time/save_checkpoint", 0.0)
      t_total = row.get("time/total", 0.0)
      print(f"{str(step):>4} | {corr:>7.2%}  | {rew:>6.4f} | {t_samp:>7.1f}s | {t_train:>9.1f}s | {t_save:>9.1f}s | {t_total:>14.1f}s")


if __name__ == "__main__":
  main()
