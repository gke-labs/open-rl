#!/usr/bin/env python3
"""Utility script to print clean telemetry progression tables from Open-RL metrics.jsonl runs."""

import argparse
import json
import os
import subprocess


def parse_args():
  parser = argparse.ArgumentParser(description="Print Open-RL telemetry progression table.")
  parser.add_argument(
    "--scenario",
    type=str,
    default="fft-gsm8k-rl",
    help="E2E scenario name (e.g. fft-gsm8k-rl, lora-textsql, tiny-rl, fft-gsm8k-x2, fft-gsm8k-rl-x2).",
  )
  parser.add_argument(
    "--job",
    type=str,
    default="job-a",
    help="Sub-job name for concurrent runs (e.g. job-a or job-b).",
  )
  parser.add_argument(
    "--metrics-path",
    type=str,
    default="",
    help="Optional explicit path to metrics.jsonl file.",
  )
  return parser.parse_args()


def resolve_metrics_path(scenario: str, job: str, explicit_path: str) -> str:
  if explicit_path:
    return explicit_path
  if "x2" in scenario:
    return f"/mnt/shared/open-rl/runs/{scenario}/open-rl-tmp/fft_gsm8k_rl_{job}/metrics.jsonl"
  return f"/mnt/shared/open-rl/runs/{scenario}/open-rl-tmp/{scenario.replace('-', '_')}/metrics.jsonl"


def fetch_from_k8s(metrics_path: str) -> str:
  # Try fetching from client pod first, then gateway pod
  cmd_client = f"kubectl exec job/open-rl-e2e-client -- cat {metrics_path} 2>/dev/null"
  res = subprocess.run(cmd_client, shell=True, capture_output=True, text=True)
  if res.returncode == 0 and res.stdout.strip():
    return res.stdout

  cmd_gateway = f"kubectl exec deployment/open-rl-gateway -- cat {metrics_path} 2>/dev/null"
  res_gw = subprocess.run(cmd_gateway, shell=True, capture_output=True, text=True)
  if res_gw.returncode == 0 and res_gw.stdout.strip():
    return res_gw.stdout
  return ""


def print_table(raw_jsonl: str):
  rows = []
  for line in raw_jsonl.splitlines():
    line = line.strip()
    if line:
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        pass

  if not rows:
    print("No telemetry metric rows found.")
    return

  print("Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time")
  print("-" * 80)
  for row in rows:
    if "env/all/correct" in row or "env/all/reward/total" in row:
      step = row.get("progress/batch", "?")
      corr = row.get("env/all/correct", row.get("env/all/reward/total", 0.0))
      rew = row.get("env/all/reward/total", 0.0)
      t_samp = row.get("time/sampling", 0.0)
      t_train = row.get("time/train_step", 0.0)
      t_save = row.get("time/save_checkpoint", 0.0)
      t_total = row.get("time/total", 0.0)
      print(f"{str(step):>4} | {corr:>7.2%}  | {rew:>6.4f} | {t_samp:>7.1f}s | {t_train:>9.1f}s | {t_save:>9.1f}s | {t_total:>14.1f}s")


def main():
  args = parse_args()
  metrics_path = resolve_metrics_path(args.scenario, args.job, args.metrics_path)

  if os.path.exists(metrics_path):
    with open(metrics_path) as f:
      raw_content = f.read()
  else:
    raw_content = fetch_from_k8s(metrics_path)

  print_table(raw_content)


if __name__ == "__main__":
  main()
