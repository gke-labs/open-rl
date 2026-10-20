"""Per-job step timings for a demo run, from the Jobs and the recipes' metrics.

  python3 scripts/demo_report.py --run demo-01 [--namespace openrl-system]

For every Job labeled openrl.io/demo-run=<run>: when the pod started, when the
first and last step finished, and the recipe's own per-step times
(time/sampling, time/train_step, time/total) from metrics.jsonl on the shared
volume, read through the API server pod. Prints one row per job and the share
of jobs that finished N steps within a budget.
"""

import argparse
import json
import subprocess
from datetime import datetime


def sh(*args: str) -> str:
  return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def ts(value: str | None) -> datetime | None:
  return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--namespace", default="openrl-system")
  parser.add_argument("--budget", type=float, default=180.0, help="seconds from pod start for the finish-rate line")
  parser.add_argument("--steps", type=int, default=2, help="steps that count as finished for the finish-rate line")
  parser.add_argument("--per-step", action="store_true", help="also print every step of every job")
  args = parser.parse_args()
  ns = args.namespace
  jobs = json.loads(sh("kubectl", "-n", ns, "get", "jobs", "-l", f"openrl.io/demo-run={args.run}", "-o", "json"))["items"]
  pods = json.loads(sh("kubectl", "-n", ns, "get", "pods", "-l", f"openrl.io/demo-run={args.run}", "-o", "json"))["items"]
  pod_by_job = {p["metadata"]["labels"].get("job-name"): p for p in pods}
  # One cat per job through the API server pod, which mounts the volume.
  names = [j["metadata"]["name"] for j in jobs]
  job_list = " ".join(names)
  script = (
    f"for n in {job_list}; do for f in /mnt/shared/open-rl/runs/demo/{args.run}/$n/open-rl-tmp/*_gsm8k_rl/metrics.jsonl; "
    'do [ -f "$f" ] && { echo "== $n"; cat "$f"; }; done; done'
  )
  raw = sh("kubectl", "-n", ns, "exec", "deploy/open-rl-api-server", "--", "sh", "-c", script) if names else ""
  metrics: dict[str, list[dict]] = {}
  current = None
  for line in raw.splitlines():
    if line.startswith("== "):
      current = line[3:].strip()
      metrics[current] = []
    elif current and line.strip():
      try:
        metrics[current].append(json.loads(line))
      except ValueError:
        pass
  print(f"| job | pod start | status | steps | first step at +s | step {args.steps} at +s | sampling s/step | train s/step | total s/step |")
  print("|---|---|---|---|---|---|---|---|---|")
  finished = 0
  for job in sorted(jobs, key=lambda j: j["metadata"]["name"]):
    name = job["metadata"]["name"]
    pod = pod_by_job.get(name, {})
    start = ts(pod.get("status", {}).get("startTime"))
    status = (
      "succeeded" if job["status"].get("succeeded") else "failed" if job["status"].get("failed") else pod.get("status", {}).get("phase", "?").lower()
    )
    rows = [r for r in metrics.get(name, []) if "time/total" in r]
    first = step_n = None
    if rows and start:
      # metrics rows carry no wall clock; derive from the pod start plus cumulative time/total,
      # which the recipe measures per step including startup of the first.
      elapsed = 0.0
      for i, r in enumerate(rows, 1):
        elapsed += float(r.get("time/total", 0))
        if i == 1:
          first = elapsed
        if i == args.steps:
          step_n = elapsed
    if step_n is not None and step_n <= args.budget:
      finished += 1

    def mean(key: str, rows=rows) -> str:
      return f"{sum(float(r.get(key, 0)) for r in rows) / len(rows):.1f}" if rows else ""

    print(
      f"| {name} | {start.strftime('%H:%M:%S') if start else ''} | {status} | {len(rows)} | {'' if first is None else f'{first:.0f}'} | "
      f"{'' if step_n is None else f'{step_n:.0f}'} | {mean('time/sampling')} | {mean('time/train_step')} | {mean('time/total')} |"
    )
  if jobs:
    print(f"\n{finished}/{len(jobs)} jobs reached step {args.steps} within {args.budget:.0f} s of pod start ({100 * finished / len(jobs):.0f}%).")
  if args.per_step:
    print("\n| job | step | sampling s | train s | save s | total s |")
    print("|---|---|---|---|---|---|")
    for name in sorted(metrics):
      for i, r in enumerate([r for r in metrics[name] if "time/total" in r], 1):
        print(
          f"| {name} | {i} | {float(r.get('time/sampling', 0)):.1f} | {float(r.get('time/train_step', 0)):.1f} | "
          f"{float(r.get('time/save_checkpoint', 0)):.1f} | {float(r.get('time/total', 0)):.1f} |"
        )


if __name__ == "__main__":
  main()
