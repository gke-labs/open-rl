#!/usr/bin/env python3
"""Compare OpenRL performance between two sets of images on a Kubernetes cluster.

Runs the same multi-job GSM8K GRPO workload (k8s/perf/workload.conf) once per version and writes
a report of before vs after: total time, time to the first and last step per job, step time split
into sampling, training and weight sync, and peak GPU memory per worker.

OpenRL must already be deployed with FFT enabled (k8s/deploy/fft) on the kubectl context's cluster,
with enough GPUs for the workload (8 for the default one). The script swaps the deployed images
between runs. Build each version first, for example on each branch:

  make cloud-build-server cloud-build-api-server GCP_PROJECT=<project> \\
    CLOUD_REGISTRY=<registry> CLOUD_IMAGE_TAG=<tag>

Then:

  python3 scripts/perf_compare.py \\
    --before api-server=<registry>/open-rl-api-server:<old>,worker=<registry>/open-rl-server:<old> \\
    --after api-server=<registry>/open-rl-api-server:<new>,worker=<registry>/open-rl-server:<new>

A version may also name scheduler=<image>; without it the deployed scheduler stays. Each version
gets one unscored 1-step warm-up run that pulls its images and fills the compile cache. The scored
runs then alternate before, after, after, before, ... so cluster drift hits both versions alike.
Results go to --out: report.md, and per run timeline.json plus the client logs. Stdlib only.
"""

import argparse
import datetime
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = REPO_ROOT / "k8s" / "perf" / "workload.conf"
CLIENT_IMAGE = "ghcr.io/gke-labs/open-rl/client:latest"
APP_LABEL = "open-rl-perf"
RUN_LABEL = "openrl.io/perf-run"
WORKER_SELECTOR = "app in (open-rl-trainer-worker,open-rl-sampler-worker)"
VERSION_KEYS = ("api-server", "worker", "scheduler")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# A row of the per-step metrics table the client prints: "│ time/total │ 117.589038 │".
METRIC_ROW = re.compile(r"│\s*(\S+)\s*│\s*(\S+)\s*│")
# Step time columns: report label and the client metric behind it.
STEP_PARTS = (("Whole step", "time/total"), ("Sampling", "time/sampling"), ("Training", "time/train_step"), ("Weight sync", "time/save_checkpoint"))


@dataclass(frozen=True)
class Job:
  name: str
  group: str
  scenario: str
  model: str
  tags: str

  @property
  def shared_lora(self) -> bool:
    return self.scenario.startswith("lora") and "openrl.exclusive=true" not in self.tags.split(",")


def parse_version(spec: str) -> dict[str, str]:
  """api-server=<img>,worker=<img>[,scheduler=<img>] as a dict."""
  images = {}
  for part in spec.split(","):
    key, sep, image = part.partition("=")
    if not sep or key not in VERSION_KEYS or not image:
      raise argparse.ArgumentTypeError(f"bad image {part!r}; expected one of {', '.join(VERSION_KEYS)} as key=<image>")
    images[key] = image
  if "api-server" not in images or "worker" not in images:
    raise argparse.ArgumentTypeError(f"{spec!r} needs both api-server=<image> and worker=<image>")
  return images


def load_workload(path: Path) -> list[Job]:
  jobs = []
  for line in path.read_text(encoding="utf-8").splitlines():
    fields = line.split("#", 1)[0].split()
    if not fields:
      continue
    count, group, scenario, model, tags = fields
    tags = "" if tags == "-" else tags
    names = [group] if int(count) == 1 else [f"{group}-{i}" for i in range(1, int(count) + 1)]
    jobs += [Job(name, group, scenario, model, tags) for name in names]
  return jobs


def job_manifest(job: Job, run: str, steps: int, pvc: str, namespace: str) -> dict[str, Any]:
  labels = {"app": APP_LABEL, RUN_LABEL: run}
  args = {
    "scenario": job.scenario,
    "uv_extra": "cpu",
    "base_url": "http://open-rl-api-server-service:8000",
    "base_model": job.model,
    "steps": steps,
    "group_size": 8,
    "groups_per_batch": 8,
    "max_tokens": 512,
    "log_dir": f"/mnt/shared/open-rl/runs/perf/{run}/{job.name}",
  }
  return {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": f"perf-{run}-{job.name}", "namespace": namespace, "labels": labels},
    "spec": {
      "backoffLimit": 0,
      "ttlSecondsAfterFinished": 86400,
      "template": {
        "metadata": {"labels": labels},
        "spec": {
          "restartPolicy": "Never",
          "serviceAccountName": "open-rl-sa",
          "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
          "containers": [
            {
              "name": "client",
              "image": CLIENT_IMAGE,
              "args": [f"{k}={v}" for k, v in args.items()],
              "env": [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "HF_HOME", "value": "/mnt/shared/open-rl/huggingface"},
                {"name": "TINKER_TAGS", "value": job.tags},
              ],
              # Small requests: every job's client runs at once on the CPU nodes.
              "resources": {"requests": {"cpu": "500m", "memory": "2Gi"}, "limits": {"memory": "8Gi"}},
              "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
            }
          ],
          "volumes": [{"name": "shared-storage", "persistentVolumeClaim": {"claimName": pvc}}],
        },
      },
    },
  }


def log_time(stamp: str) -> float:
  """Unix time of a `kubectl logs --timestamps` stamp (RFC 3339, nanoseconds)."""
  head, _, frac = stamp.rstrip("Z").partition(".")
  return datetime.datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00").timestamp()


def parse_client_log(log: str) -> list[dict[str, float]]:
  """One dict per finished step from the client's metrics tables, with "t" = when the table printed."""
  steps, rows, last = [], {}, 0.0
  for line in log.splitlines():
    stamp, _, text = line.partition(" ")
    if m := METRIC_ROW.search(text):
      try:
        rows[m.group(1)] = float(m.group(2))
      except ValueError:
        continue
      last = log_time(stamp)
    elif "└" in text and "time/total" in rows:
      steps.append({**rows, "t": last})
      rows = {}
  return steps


class Cluster:
  def __init__(self, namespace: str):
    self.namespace = namespace

  def kubectl(self, *args: str, check: bool = True, timeout: float = 120, stdin: str | None = None) -> str:
    r = subprocess.run(["kubectl", "-n", self.namespace, *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
      raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout

  def get_json(self, *args: str) -> dict[str, Any]:
    return json.loads(self.kubectl("get", *args, "-o", "json"))

  def shared_pvc(self) -> str:
    deploy = self.get_json("deploy", "open-rl-api-server")
    for volume in deploy["spec"]["template"]["spec"]["volumes"]:
      if volume["name"] == "shared-storage":
        return volume["persistentVolumeClaim"]["claimName"]
    raise RuntimeError("open-rl-api-server has no shared-storage volume")

  def container_image(self, kind: str, name: str, container: str) -> str | None:
    obj = json.loads(self.kubectl("get", kind, name, "-o", "json", check=False) or "null")
    if obj is None:
      return None
    return next(c["image"] for c in obj["spec"]["template"]["spec"]["containers"] if c["name"] == container)

  def set_version(self, images: dict[str, str]) -> None:
    """Point the API server, worker pods, time-slicer and (if given) scheduler at a version's images."""
    worker_images = self.get_json("configmap", "open-rl-images").get("data", {})
    changed = []
    if self.container_image("deploy", "open-rl-api-server", "api-server") != images["api-server"]:
      self.kubectl("set", "image", "deploy/open-rl-api-server", f"api-server={images['api-server']}")
      changed.append("deploy/open-rl-api-server")
    # The API server reads these at start-up; the dispatcher runs the API server image.
    want = {"worker": images["worker"], "dispatcher": images["api-server"]}
    if any(worker_images.get(k) != v for k, v in want.items()):
      self.kubectl("patch", "configmap", "open-rl-images", "--type", "merge", "-p", json.dumps({"data": want}))
      self.kubectl("rollout", "restart", "deploy/open-rl-api-server")
      changed.append("deploy/open-rl-api-server")
    timeslicer = self.container_image("ds", "open-rl-accel-timeslicer", "accel-timeslicer")
    if timeslicer is not None and timeslicer != images["worker"]:
      self.kubectl("set", "image", "ds/open-rl-accel-timeslicer", f"accel-timeslicer={images['worker']}")
      changed.append("ds/open-rl-accel-timeslicer")
    if "scheduler" in images and self.container_image("deploy", "open-rl-scheduler", "manager") != images["scheduler"]:
      self.kubectl("set", "image", "deploy/open-rl-scheduler", f"manager={images['scheduler']}")
      changed.append("deploy/open-rl-scheduler")
    for obj in dict.fromkeys(changed):
      log(f"waiting for {obj} rollout")
      self.kubectl("rollout", "status", obj, "--timeout=1200s", timeout=1260)

  def wait_idle(self, timeout: float = 1800) -> None:
    """Wait until no OpenRL Workloads or worker pods are left from earlier jobs."""
    deadline = time.time() + timeout
    while time.time() < deadline:
      workloads = self.get_json("workloads.openrl.io")["items"]
      pods = self.get_json("pods", "-l", WORKER_SELECTOR)["items"]
      if not workloads and not pods:
        return
      time.sleep(10)
    raise RuntimeError(f"cluster not idle after {timeout:.0f} s: Workloads or worker pods are still running")

  def gpu_memory(self, pod: str) -> tuple[str, int | None, str | None]:
    out = self.kubectl(
      "exec", pod, "-c", "worker", "--", "nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits", check=False, timeout=20
    )
    try:
      mem, name = out.strip().splitlines()[0].split(", ", 1)
      return pod, int(mem), name
    except (IndexError, ValueError):
      return pod, None, None


def log(msg: str) -> None:
  print(f"[perf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_once(cluster: Cluster, run: str, version: str, images: dict[str, str], jobs: list[Job], steps: int, out: Path, timeout: float) -> dict:
  """Launch every job at once, watch until all finish, and save the run's timeline and client logs."""
  cluster.wait_idle()
  cluster.set_version(images)
  cluster.kubectl("delete", "jobs", "-l", f"{RUN_LABEL}={run}", "--ignore-not-found", "--wait=true", timeout=300)
  pvc = cluster.shared_pvc()
  manifest = {"apiVersion": "v1", "kind": "List", "items": [job_manifest(j, run, steps, pvc, cluster.namespace) for j in jobs]}
  run_dir = out / run
  (run_dir / "logs").mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  log(f"{run} ({version}): launching {len(jobs)} jobs, {steps} step(s) each")
  cluster.kubectl("apply", "-f", "-", stdin=json.dumps(manifest))

  workloads: dict[str, dict] = {}
  workers: dict[str, dict] = {}
  finished: dict[str, dict] = {}
  last_print = 0.0
  while len(finished) < len(jobs):
    now = time.time()
    if now - t0 > timeout:
      log(f"{run}: timed out after {timeout:.0f} s; deleting its jobs and workers")
      cluster.kubectl("delete", "jobs", "-l", f"{RUN_LABEL}={run}", "--wait=false", check=False)
      cluster.kubectl("delete", "workloads.openrl.io", "--all", "--wait=false", check=False)
      break
    try:
      for w in cluster.get_json("workloads.openrl.io")["items"]:
        workloads[w["metadata"]["name"]] = w["spec"]
      running = []
      for p in cluster.get_json("pods", "-l", WORKER_SELECTOR)["items"]:
        pod = p["metadata"]["name"]
        workload = p["metadata"]["labels"].get("openrl.io/worker", "")
        spec = workloads.get(workload, {})
        w = workers.setdefault(pod, {"workload": workload, "peak_mib": 0, "gpu": None})
        w.update(role=spec.get("role"), model_id=spec.get("modelID"), training_kind=spec.get("trainingKind"), node=p["spec"].get("nodeName"))
        if p["status"].get("phase") == "Running":
          running.append(pod)
      with ThreadPoolExecutor(16) as ex:
        for pod, mem, gpu in ex.map(cluster.gpu_memory, running):
          if mem is not None:
            workers[pod]["peak_mib"] = max(workers[pod]["peak_mib"], mem)
            workers[pod]["gpu"] = gpu
      for j in cluster.get_json("jobs", "-l", f"{RUN_LABEL}={run}")["items"]:
        status = j.get("status", {})
        name = j["metadata"]["name"][len(f"perf-{run}-") :]
        if (status.get("succeeded") or status.get("failed")) and name not in finished:
          finished[name] = {"ok": bool(status.get("succeeded")), "done_s": now - t0}
          log(f"{run}: {name} {'done' if status.get('succeeded') else 'FAILED'} at {now - t0:.0f} s")
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
      log(f"{run}: poll error, retrying: {e}")
    if now - last_print > 60:
      last_print = now
      log(f"{run}: {len(finished)}/{len(jobs)} jobs finished, {len(workers)} workers seen, {now - t0:.0f} s")
    time.sleep(5)

  record = {
    "run": run,
    "version": version,
    "images": images,
    "t0": t0,
    "steps": steps,
    "valid": len(finished) == len(jobs),
    "jobs": {},
    "workers": workers,
  }
  for job in jobs:
    text = cluster.kubectl("logs", "--timestamps", f"job/perf-{run}-{job.name}", check=False, timeout=300)
    (run_dir / "logs" / f"{job.name}.log").write_text(text, encoding="utf-8")
    record["jobs"][job.name] = {
      "group": job.group,
      "model": job.model,
      "shared_lora": job.shared_lora,
      **finished.get(job.name, {"ok": False, "done_s": None}),
      "steps": [{**s, "t": s["t"] - t0} for s in parse_client_log(text)],
      "model_ids": sorted(set(UUID.findall(text))),
    }
  (run_dir / "timeline.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
  return record


def total_time(record: dict) -> float | None:
  """Launch until the last job finishes its last step; None if any job did not."""
  ends = []
  for job in record["jobs"].values():
    if len(job["steps"]) < record["steps"]:
      return None
    ends.append(job["steps"][record["steps"] - 1]["t"])
  return max(ends) if ends else None


def worker_groups(record: dict) -> dict[tuple[str, str], list[dict]]:
  """Workers by (job group, role). An FFT or exclusive worker serves the job that logged its model id;
  a shared LoRA worker's model id is its base model, so it serves that model's shared LoRA jobs."""
  by_id = {mid: job["group"] for job in record["jobs"].values() for mid in job["model_ids"]}
  out: dict[tuple[str, str], list[dict]] = {}
  for w in record["workers"].values():
    mid = w.get("model_id")
    if mid in by_id:
      group = by_id[mid]
    else:
      groups = sorted({j["group"] for j in record["jobs"].values() if j["shared_lora"] and j["model"] == mid})
      group = " + ".join(groups) or f"? ({w['workload']})"
    out.setdefault((group, w.get("role") or "?"), []).append(w)
  return out


def mean(xs: list[float]) -> float | None:
  return statistics.mean(xs) if xs else None


def fmt(x: float | None) -> str:
  return "-" if x is None else f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"


def change(before: float | None, after: float | None) -> str:
  return "-" if not before or after is None else f"{(after - before) / before * 100:+.0f}%"


def compare(before: float | None, after: float | None, pct: bool = True) -> str:
  """One report cell: before → after, and the change unless pct is False."""
  if before is None and after is None:
    return "-"
  return f"{fmt(before)} → {fmt(after)}" + (f" ({change(before, after)})" if pct else "")


def build_report(records: list[dict], labels: dict[str, str]) -> str:
  """labels: plain-word description per job group, from describe_groups."""
  steps = records[0]["steps"]
  runs = {v: [r for r in records if r["version"] == v and r["valid"]] for v in ("before", "after")}
  out = ["# OpenRL performance: before vs after", ""]
  for v in ("before", "after"):
    images = next((r["images"] for r in records if r["version"] == v), {})
    out.append(f"- **{v}**: " + ", ".join(f"{k} `{img}`" for k, img in images.items()))
  totals = {v: mean([t for r in rs if (t := total_time(r)) is not None]) for v, rs in runs.items()}
  out += [
    "",
    "Values are means over each version's valid runs. On a shared runtime, times include waiting for other jobs' turns.",
    "",
    "## Total time",
    "",
    f"Launch until the last job finishes step {steps}: **{compare(totals['before'], totals['after'])}** seconds.",
    "",
    "| Run | Version | Total time (s) | Failed or unfinished jobs |",
    "|---|---|---|---|",
  ]
  for r in records:
    bad = [name for name, j in r["jobs"].items() if not j["ok"] or len(j["steps"]) < steps]
    out.append(f"| {r['run']} | {r['version']}{'' if r['valid'] else ' (timed out)'} | {fmt(total_time(r))} | {', '.join(bad) or '-'} |")

  def group_values(version: str, group: str) -> dict[str, float | None]:
    """Per metric: mean over the group's jobs in every run of the version."""
    vals: dict[str, list[float]] = {}
    for r in runs[version]:
      for job in (j for j in r["jobs"].values() if j["group"] == group and j["steps"]):
        s = job["steps"]
        vals.setdefault("first", []).append(s[0]["t"])
        if len(s) >= steps:
          vals.setdefault("last", []).append(s[steps - 1]["t"])
        # Steady state: step 1 includes start-up, so use the later steps when there are any.
        later = s[1:steps] or s[:1]
        for label, key in STEP_PARTS:
          xs = [row[key] for row in later if key in row]
          if xs:
            vals.setdefault(label, []).append(statistics.median(xs))
    return {k: mean(v) for k, v in vals.items()}

  groups = list(dict.fromkeys(j["group"] for r in records for j in r["jobs"].values()))
  values = {g: (group_values("before", g), group_values("after", g)) for g in groups}
  out += [
    "",
    "## When jobs finish",
    "",
    "Seconds from launch until the job finishes that step, before → after.",
    "",
    f"| Jobs | First step done | All {steps} steps done |",
    "|---|---|---|",
  ]
  for g in groups:
    b, a = values[g]
    out.append(f"| {labels.get(g, g)} | {compare(b.get('first'), a.get('first'))} | {compare(b.get('last'), a.get('last'))} |")

  parts = [label for label, _ in STEP_PARTS]
  out += [
    "",
    "## How long a step takes",
    "",
    f"Seconds, median of steps 2-{steps} (step 1 includes start-up), before → after. Weight sync is `time/save_checkpoint`.",
    "",
    "| Jobs | " + " | ".join(parts) + " |",
    "|---|" + "---|" * len(parts),
  ]
  for g in groups:
    b, a = values[g]
    cells = [compare(b.get(parts[0]), a.get(parts[0]))] + [compare(b.get(p), a.get(p), pct=False) for p in parts[1:]]
    out.append(f"| {labels.get(g, g)} | " + " | ".join(cells) + " |")

  out += [
    "",
    "## Peak GPU memory",
    "",
    "nvidia-smi `memory.used` in each worker pod, GiB, max over the group's workers, before → after. Time-sliced workers",
    "share a GPU, so theirs is the memory of everything on that GPU.",
    "",
    "| Jobs | GPU | Trainer | Sampler |",
    "|---|---|---|---|",
  ]
  peaks: dict[tuple[str, str], dict[str, list[float]]] = {}
  gpus: dict[str, set[str]] = {}
  for v, rs in runs.items():
    for r in rs:
      for (group, role), ws in worker_groups(r).items():
        peaks.setdefault((group, role), {}).setdefault(v, []).append(max(w["peak_mib"] for w in ws) / 1024)
        gpus.setdefault(group, set()).update(w["gpu"].replace("NVIDIA ", "") for w in ws if w.get("gpu"))
  for g in [*groups, *sorted(set(gpus) - set(groups))]:
    if g not in gpus:
      continue
    cells = [compare(*(mean(peaks.get((g, role), {}).get(v, [])) for v in ("before", "after"))) for role in ("trainer", "sampler")]
    out.append(f"| {labels.get(g, g)} | {', '.join(sorted(gpus[g])) or '-'} | {cells[0]} | {cells[1]} |")
  return "\n".join(out) + "\n"


def describe_groups(jobs: list[Job]) -> dict[str, str]:
  """Each job group in plain words, e.g. "gemma-4-E2B LoRA, shared runtime (6 jobs)"."""
  out = {}
  for group in dict.fromkeys(j.group for j in jobs):
    members = [j for j in jobs if j.group == group]
    job = members[0]
    lora = job.scenario.startswith("lora")
    if "openrl.exclusive=true" in job.tags.split(","):
      sharing = "exclusive"
    else:
      sharing = "shared runtime" if lora else "time-sliced"
    count = f"{len(members)} job{'s' if len(members) > 1 else ''}"
    out[group] = f"{job.model.split('/')[-1]} {'LoRA' if lora else 'FFT'}, {sharing} ({count})"
  return out


def run_order(repeats: int) -> list[tuple[str, str]]:
  """(run id, version) for the scored runs: before, after, after, before, ..."""
  order, count = [], {"before": 0, "after": 0}
  for i in range(repeats):
    for version in ("before", "after") if i % 2 == 0 else ("after", "before"):
      count[version] += 1
      order.append((f"{version[0]}{count[version]}", version))
  return order


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--before", type=parse_version, required=True, help="Images of the version to compare against.")
  parser.add_argument("--after", type=parse_version, required=True, help="Images of the changed version.")
  parser.add_argument("--repeats", type=int, default=2, help="Scored runs per version.")
  parser.add_argument("--steps", type=int, default=3, help="Training steps per job.")
  parser.add_argument("--no-warmup", action="store_true", help="Skip the unscored 1-step warm-up run per version.")
  parser.add_argument("--workload", type=Path, default=WORKLOAD, help="Job mix to launch.")
  parser.add_argument("--namespace", default="openrl-system")
  parser.add_argument("--run-timeout", type=float, default=3600, help="Seconds before a run is stopped and marked timed out.")
  parser.add_argument("--out", type=Path, default=None, help="Results directory (default runs/perf-compare-<UTC time>).")
  args = parser.parse_args()

  out = args.out or REPO_ROOT / "runs" / f"perf-compare-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
  out.mkdir(parents=True, exist_ok=True)
  jobs = load_workload(args.workload)
  labels = describe_groups(jobs)
  cluster = Cluster(args.namespace)
  context = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True).stdout.strip()
  log(f"context {context}, namespace {args.namespace}, {len(jobs)} jobs, results in {out}")

  images = {"before": args.before, "after": args.after}
  if not args.no_warmup:
    for version in ("before", "after"):
      run_once(cluster, f"warmup-{version}", version, images[version], jobs, 1, out, args.run_timeout)
  records = []
  for run, version in run_order(args.repeats):
    records.append(run_once(cluster, run, version, images[version], jobs, args.steps, out, args.run_timeout))
    (out / "report.md").write_text(build_report(records, labels), encoding="utf-8")
  print(build_report(records, labels))
  log(f"report: {out / 'report.md'}")
  if not all(r["valid"] for r in records):
    sys.exit(1)


if __name__ == "__main__":
  main()
