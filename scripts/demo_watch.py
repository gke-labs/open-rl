"""One screen for the fleet demo: the jobs, their steps, the workers, the GPUs, the storage.

  python3 scripts/demo_watch.py --run drill-1 [--interval 5] [--once]

Refreshes until Ctrl-C. Steps come from each job's metrics.jsonl on the shared
volume (one `wc -l` through the API server pod per refresh), workers from the
Workload objects, GPU occupancy from Workloads per node, storage from `df` on
the volume. Everything shown is what kubectl would show; this only puts it on
one page.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime


def sh(*args: str, timeout: float = 25) -> str:
  try:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout
  except Exception:
    return ""


def ts(value):
  return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def snapshot(ns: str, run: str) -> dict:
  jobs = json.loads(sh("kubectl", "-n", ns, "get", "jobs", "-l", f"openrl.io/demo-run={run}", "-o", "json") or '{"items":[]}')["items"]
  pods = json.loads(sh("kubectl", "-n", ns, "get", "pods", "-l", f"openrl.io/demo-run={run}", "-o", "json") or '{"items":[]}')["items"]
  workloads = json.loads(sh("kubectl", "-n", ns, "get", "workloads", "-o", "json") or '{"items":[]}')["items"]
  names = [j["metadata"]["name"] for j in jobs]
  script = (
    f"cd /mnt/shared/open-rl/runs/demo/{run} 2>/dev/null || exit 0; "
    + "for n in "
    + " ".join(names)
    + "; do f=$(ls $n/open-rl-tmp/*_gsm8k_rl/metrics.jsonl 2>/dev/null | head -1); "
    + 'if [ -n "$f" ]; then echo "$n $(wc -l < $f)"; fi; done; df -h /mnt/shared | tail -1'
  )
  raw = sh("kubectl", "-n", ns, "exec", "deploy/open-rl-api-server", "--", "sh", "-c", script) if names else ""
  steps, disk = {}, ""
  for line in raw.splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[0] in names:
      steps[parts[0]] = int(parts[1])
    elif line.startswith("10.") or "/mnt/shared" in line:
      disk = line
  pod_by_job = {p["metadata"]["labels"].get("job-name"): p for p in pods}
  return {"jobs": jobs, "pod_by_job": pod_by_job, "workloads": workloads, "steps": steps, "disk": disk}


def render(snap: dict, run: str, budget: float, target_steps: int) -> str:
  now = datetime.now(UTC)
  lines = [f"OpenRL fleet demo  run={run}  {now.strftime('%H:%M:%S')} UTC", ""]
  lines.append(f"{'job':44} {'status':10} {'age':>6} {'steps':>5}")
  done_in_time = total = 0
  for job in sorted(snap["jobs"], key=lambda j: j["metadata"]["name"]):
    name = job["metadata"]["name"]
    pod = snap["pod_by_job"].get(name, {})
    start = ts(pod.get("status", {}).get("startTime")) or ts(job["metadata"]["creationTimestamp"])
    age = (now - start).total_seconds() if start else 0
    status = (
      "done" if job["status"].get("succeeded") else "FAILED" if job["status"].get("failed") else pod.get("status", {}).get("phase", "pending").lower()
    )
    n = snap["steps"].get(name, 0)
    total += 1
    if n >= target_steps and age <= budget or (job["status"].get("succeeded") and n >= target_steps and age <= budget):
      done_in_time += 1
    bar = "#" * n + "." * max(0, target_steps - n)
    lines.append(f"{name[:44]:44} {status:10} {age:6.0f} {n:>3} {bar}")
  lines.append("")
  by_node: Counter = Counter()
  phases: Counter = Counter()
  for wl in snap["workloads"]:
    st = wl.get("status", {})
    phases[st.get("phase", "Pending")] += 1
    if st.get("nodeName"):
      by_node[st["nodeName"].replace("gke-open-rl-dra-", "")] += st.get("deviceCount") or 1
  lines.append(f"workers: {len(snap['workloads'])}  " + "  ".join(f"{k}={v}" for k, v in sorted(phases.items())))
  lines.append("GPUs in use by node: " + ", ".join(f"{n}:{c}" for n, c in sorted(by_node.items())))
  lines.append(f"storage: {snap['disk'] or 'n/a'}")
  lines.append("")
  lines.append(f"{done_in_time}/{total} jobs at >= {target_steps} steps within {budget:.0f} s")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--namespace", default="openrl-system")
  parser.add_argument("--interval", type=float, default=5.0)
  parser.add_argument("--budget", type=float, default=180.0)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--once", action="store_true")
  args = parser.parse_args()
  while True:
    text = render(snapshot(args.namespace, args.run), args.run, args.budget, args.steps)
    if args.once:
      print(text)
      return
    os.system("clear")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
