"""Measure one OpenRL cold start from primary records, and build its timeline.

Primary records, and the clock each one carries:

  Workload and pod objects   kube-apiserver clock, 1 s resolution (creationTimestamp,
                             condition lastTransitionTime, containerStatuses startedAt)
  Kubernetes events          kubelet or scheduler clock; the Pulled message carries the
                             pull duration kubelet measured
  worker container logs      containerd on the node, ns resolution; vLLM and the trainer
                             print their own phase timings into them
  Redis ready keys           first seen by a 1 s poller running inside the Redis pod,
                             stamped with that pod's clock
  API server access log      the API server pod's clock
  client phases              the laptop's clock (scripts/cold_start_client.py)

All GKE nodes and the laptop run NTP; the report prints one cross-check
(container Started event versus the first log line) so a skew would show.

Usage, one run:

  python3 scripts/cold_start_timeline.py watch   --out runs/<dir>/<case>      # start recorders
  uv --project examples run python scripts/cold_start_client.py base_model=... out=runs/<dir>/<case>/client.jsonl
  python3 scripts/cold_start_timeline.py collect --out runs/<dir>/<case>      # logs, events, snapshots; stops recorders
  python3 scripts/cold_start_timeline.py timeline --out runs/<dir>/<case>     # timeline.json + timeline.md

Needs kubectl pointed at the cluster. Namespace: openrl-system (OPENRL_NAMESPACE).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NS = os.getenv("OPENRL_NAMESPACE", "openrl-system")

WORKLOAD_JSONPATH = (
  '{range .items[*]}{.metadata.name}{"\\t"}{.metadata.creationTimestamp}{"\\t"}{.status.phase}{"\\t"}{.status.reason}'
  '{"\\t"}{.status.claimName}{"\\t"}{.status.podName}{"\\t"}{.status.nodeName}{"\\t"}{.metadata.deletionTimestamp}{"\\n"}{end}'
)
POD_JSONPATH = (
  '{range .items[*]}{.metadata.name}{"\\t"}{.metadata.creationTimestamp}{"\\t"}{.status.phase}{"\\t"}{.spec.nodeName}'
  '{"\\t"}{.status.containerStatuses[0].state.waiting.reason}{"\\t"}{.status.containerStatuses[0].state.running.startedAt}'
  '{"\\t"}{.status.containerStatuses[0].ready}{"\\t"}{.metadata.deletionTimestamp}{"\\n"}{end}'
)


def now() -> str:
  return datetime.now(UTC).isoformat(timespec="milliseconds")


def sh(*args: str, check: bool = True) -> str:
  return subprocess.run(args, check=check, capture_output=True, text=True).stdout


def kubectl(*args: str, check: bool = True) -> str:
  return sh("kubectl", "-n", NS, *args, check=check)


def stamped_follower(cmd: list[str], out: Path) -> subprocess.Popen:
  """Run cmd and prefix every output line with the local receive time."""
  script = (
    "import sys,datetime\n"
    "for line in sys.stdin:\n"
    "  sys.stdout.write(datetime.datetime.now(datetime.UTC).isoformat(timespec='milliseconds')+'\\t'+line); sys.stdout.flush()\n"
  )
  f = open(out, "a", encoding="utf-8")
  producer = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  consumer = subprocess.Popen([sys.executable, "-u", "-c", script], stdin=producer.stdout, stdout=f, stderr=subprocess.STDOUT, text=True)
  producer.stdout.close()  # type: ignore[union-attr]
  consumer.producer = producer  # type: ignore[attr-defined]
  return consumer


def poll_loop(cmd_args: list[str], out: Path, every: float = 1.0) -> subprocess.Popen:
  """Run a kubectl get on an interval, each sample prefixed with the local time."""
  script = (
    "import subprocess,sys,time,datetime\n"
    f"args={cmd_args!r}; every={every}\n"
    "while True:\n"
    "  t=datetime.datetime.now(datetime.UTC).isoformat(timespec='milliseconds')\n"
    "  r=subprocess.run(args,capture_output=True,text=True)\n"
    "  for line in r.stdout.splitlines():\n"
    "    if line.strip(): sys.stdout.write(t+'\\t'+line+'\\n')\n"
    "  sys.stdout.flush(); time.sleep(every)\n"
  )
  f = open(out, "a", encoding="utf-8")
  return subprocess.Popen([sys.executable, "-u", "-c", script], stdout=f, stderr=subprocess.STDOUT, text=True)


def node_images() -> dict[str, list[str]]:
  data = json.loads(sh("kubectl", "get", "nodes", "-o", "json"))
  out = {}
  for node in data["items"]:
    names = [n for img in node["status"].get("images", []) for n in img.get("names", []) if "open-rl" in n and "@sha256" not in n]
    out[node["metadata"]["name"]] = sorted(set(names))
  return out


# -- watch ------------------------------------------------------------------


def cmd_watch(out: Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "node-images-before.json").write_text(json.dumps(node_images(), indent=1))
  (out / "worker-image.txt").write_text(kubectl("get", "cm", "open-rl-images", "-o", "jsonpath={.data.worker}"))
  (out / "watch-started.txt").write_text(now() + "\n")
  procs = [
    poll_loop(["kubectl", "-n", NS, "get", "workloads", "-o", f"jsonpath={WORKLOAD_JSONPATH}"], out / "workloads.tsv"),
    poll_loop(["kubectl", "-n", NS, "get", "pods", "-o", f"jsonpath={POD_JSONPATH}"], out / "pods.tsv"),
    stamped_follower(
      [
        "kubectl",
        "-n",
        NS,
        "exec",
        "deploy/redis-store",
        "--",
        "sh",
        "-c",
        "while true; do t=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
        'for k in $(redis-cli --scan --pattern "open_rl:*_ready:*"); do echo "$t $k"; done; sleep 1; done',
      ],
      out / "redis-ready.log",
    ),
    stamped_follower(["kubectl", "-n", NS, "logs", "-f", "--timestamps", "--tail=0", "deploy/open-rl-api-server"], out / "api-server.log"),
    stamped_follower(["kubectl", "-n", NS, "logs", "-f", "--timestamps", "--tail=0", "deploy/open-rl-scheduler"], out / "scheduler.log"),
  ]
  pids = [p.pid for p in procs] + [p.producer.pid for p in procs if hasattr(p, "producer")]  # type: ignore[attr-defined]
  (out / "watch.pids").write_text("\n".join(map(str, pids)) + "\n")
  print(f"watching; recorders {pids} write into {out}")


def stop_watch(out: Path) -> None:
  pid_file = out / "watch.pids"
  if not pid_file.exists():
    return
  for pid in pid_file.read_text().split():
    try:
      os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
      pass
  pid_file.unlink()


# -- collect ------------------------------------------------------------------


def cmd_collect(out: Path) -> None:
  (out / "collect-started.txt").write_text(now() + "\n")
  pods = json.loads(kubectl("get", "pods", "-o", "json"))
  workers = [p for p in pods["items"] if p["metadata"]["name"].startswith("orw-")]
  (out / "pods.json").write_text(json.dumps(pods, indent=1))
  (out / "workloads.json").write_text(kubectl("get", "workloads", "-o", "json"))
  (out / "events.json").write_text(kubectl("get", "events", "-o", "json"))
  for pod in workers:
    name = pod["metadata"]["name"]
    (out / f"log-{name}.txt").write_text(kubectl("logs", "--timestamps", name, check=False))
  (out / "node-images-after.json").write_text(json.dumps(node_images(), indent=1))
  time.sleep(2)
  stop_watch(out)
  print(f"collected {len(workers)} worker log(s) into {out}")


# -- timeline -----------------------------------------------------------------


def parse_ts(value: str | None) -> datetime | None:
  if not value:
    return None
  value = value.strip()
  if value.endswith("Z"):
    value = value[:-1] + "+00:00"
  # Container log timestamps carry nanoseconds; Python takes microseconds.
  m = re.match(r"(.*\.\d{6})\d+(\+00:00)$", value)
  if m:
    value = m.group(1) + m.group(2)
  try:
    return datetime.fromisoformat(value)
  except ValueError:
    return None


def duration_s(text: str) -> float | None:
  """'3m48.333s' or '19.2s' or '1.275s' from a kubelet Pulled message."""
  m = re.search(r"in ((?:\d+h)?(?:\d+m)?[\d.]+s)", text)
  if not m:
    return None
  total, s = 0.0, m.group(1)
  for num, unit in re.findall(r"([\d.]+)([hms])", s):
    total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
  return total


SAMPLER_MARKS = [
  ("sampler.process_first_log", None),
  ("sampler.engine_init_start", r"Initializing a V1 LLM engine"),
  ("sampler.load_start", r"Loading model from scratch"),
  ("sampler.prefetch_start", r"Prefetching checkpoint files into page cache started"),
  ("sampler.weights_loaded", r"Loading weights took ([\d.]+) seconds"),
  ("sampler.model_loaded", r"Model loading took .* and ([\d.]+) seconds"),
  ("sampler.compile_done", r"torch\.compile took ([\d.]+) s"),
  ("sampler.kv_cache_sized", r"Available KV cache memory"),
  ("sampler.graphs_captured", r"Graph capturing finished in (\d+) secs"),
  ("sampler.engine_init_done", r"init engine .* took ([\d.]+) s"),
]
TRAINER_MARKS = [
  ("trainer.process_first_log", None),
  ("trainer.load_start", r"Loading base model"),
  ("trainer.load_done", r"Successfully loaded\."),
  ("trainer.processor_started", r"Training requests processor started"),
]


def scan_log(path: Path, marks: list[tuple[str, str | None]]) -> dict[str, dict[str, Any]]:
  found: dict[str, dict[str, Any]] = {}
  if not path.exists():
    return found
  for line in path.read_text(errors="replace").splitlines():
    ts, _, text = line.partition(" ")
    when = parse_ts(ts)
    if when is None:
      continue
    for name, pattern in marks:
      if pattern is None:
        found.setdefault(name, {"ts": when, "source": "container log, first line"})
        continue
      if name in found:
        continue
      m = re.search(pattern, text)
      if m:
        entry: dict[str, Any] = {"ts": when, "source": f"container log: {text.strip()[:120]}"}
        if m.groups():
          entry["self_reported_s"] = float(m.group(1))
        found[name] = entry
  return found


def first_seen(tsv: Path, key_index: int, predicate) -> dict[str, datetime]:
  """First local time each key satisfied predicate, from a poll_loop file."""
  seen: dict[str, datetime] = {}
  if not tsv.exists():
    return seen
  for line in tsv.read_text().splitlines():
    parts = line.split("\t")
    if len(parts) < 2:
      continue
    when = parse_ts(parts[0])
    fields = parts[1:]
    if when is None or len(fields) <= key_index:
      continue
    key = fields[key_index]
    if key and key not in seen and predicate(fields):
      seen[key] = when
  return seen


def ready_keys(path: Path) -> dict[str, datetime]:
  seen: dict[str, datetime] = {}
  if not path.exists():
    return seen
  for line in path.read_text().splitlines():
    parts = line.split("\t", 1)
    if len(parts) < 2:
      continue
    body = parts[1].strip().split(" ", 1)
    if len(body) == 2 and body[1] not in seen:
      seen[body[1]] = parse_ts(body[0]) or parse_ts(parts[0])  # type: ignore[assignment]
  return seen


def pod_events(events: dict[str, Any], pod: dict[str, Any]) -> list[dict[str, Any]]:
  """Events for this pod object, not for an earlier pod of the same name: the
  workload names repeat across runs, so match the pod's uid."""
  name, uid = pod["metadata"]["name"], pod["metadata"].get("uid")
  rows = []
  for e in events.get("items", []):
    obj = e["involvedObject"]
    if obj.get("name") != name or (uid and obj.get("uid") and obj["uid"] != uid):
      continue
    rows.append({"reason": e["reason"], "ts": parse_ts(e.get("firstTimestamp") or e.get("eventTime")), "message": e.get("message", "")})
  return sorted(rows, key=lambda r: r["ts"] or datetime.min.replace(tzinfo=UTC))


def cmd_timeline(out: Path) -> None:
  client = [json.loads(l) for l in (out / "client.jsonl").read_text().splitlines() if l.strip()] if (out / "client.jsonl").exists() else []
  marks: dict[str, dict[str, Any]] = {}
  for row in client:
    marks[f"client.{row['phase']}"] = {
      "ts": parse_ts(row["ts"]),
      "source": "client.jsonl (laptop clock)",
      **{k: v for k, v in row.items() if k not in ("ts", "t", "phase")},
    }

  pods = json.loads((out / "pods.json").read_text()) if (out / "pods.json").exists() else {"items": []}
  events = json.loads((out / "events.json").read_text()) if (out / "events.json").exists() else {"items": []}
  workloads = json.loads((out / "workloads.json").read_text()) if (out / "workloads.json").exists() else {"items": []}
  ready = ready_keys(out / "redis-ready.log")
  wl_first_pod = first_seen(out / "workloads.tsv", 0, lambda f: len(f) > 5 and bool(f[5]))
  wl_first_node = first_seen(out / "workloads.tsv", 0, lambda f: len(f) > 6 and bool(f[6]))
  wl_first_seen = first_seen(out / "workloads.tsv", 0, lambda f: True)

  workers: dict[str, dict[str, Any]] = {}
  for wl in workloads.get("items", []):
    role = wl["spec"]["role"]
    name = wl["metadata"]["name"]
    w: dict[str, Any] = {
      "workload": name,
      "model_id": wl["spec"].get("modelID"),
      "node": wl.get("status", {}).get("nodeName"),
      "pod": wl.get("status", {}).get("podName"),
      "marks": {},
    }
    w["marks"][f"{role}.workload_created"] = {"ts": parse_ts(wl["metadata"]["creationTimestamp"]), "source": "Workload creationTimestamp (apiserver)"}
    if name in wl_first_seen:
      w["marks"][f"{role}.workload_first_seen_by_poller"] = {"ts": wl_first_seen[name], "source": "workloads.tsv poller (laptop clock, 1 s)"}
    if name in wl_first_pod:
      w["marks"][f"{role}.workload_reports_pod"] = {"ts": wl_first_pod[name], "source": "workloads.tsv poller: status.podName set"}
    if name in wl_first_node:
      w["marks"][f"{role}.workload_reports_node"] = {"ts": wl_first_node[name], "source": "workloads.tsv poller: status.nodeName set"}
    for cond in wl.get("status", {}).get("conditions", []):
      w["marks"][f"{role}.condition_{cond['type']}_{cond['status']}"] = {
        "ts": parse_ts(cond.get("lastTransitionTime")),
        "source": f"Workload condition {cond.get('reason', '')}",
      }
    workers[role] = w

  for pod in pods.get("items", []):
    name = pod["metadata"]["name"]
    if not name.startswith("orw-"):
      continue
    role = "sampler" if name.endswith("sampler") or "-sampler-" in name else "trainer"
    w = workers.setdefault(role, {"workload": None, "pod": name, "node": pod["spec"].get("nodeName"), "marks": {}})
    w["pod"] = name
    w["node"] = pod["spec"].get("nodeName")
    w["marks"][f"{role}.pod_created"] = {"ts": parse_ts(pod["metadata"]["creationTimestamp"]), "source": "pod creationTimestamp (apiserver)"}
    for cond in pod.get("status", {}).get("conditions", []):
      if cond["type"] in ("PodScheduled", "Initialized", "ContainersReady", "Ready") and cond["status"] == "True":
        w["marks"][f"{role}.pod_{cond['type']}"] = {"ts": parse_ts(cond.get("lastTransitionTime")), "source": f"pod condition {cond['type']}"}
    for cs in pod.get("status", {}).get("containerStatuses", []):
      running = (cs.get("state") or {}).get("running") or {}
      if running.get("startedAt"):
        w["marks"][f"{role}.container_started"] = {"ts": parse_ts(running["startedAt"]), "source": "containerStatuses.state.running.startedAt"}
      w["image_id"] = cs.get("imageID")
    for ev in pod_events(events, pod):
      key = f"{role}.event_{ev['reason']}"
      if key in w["marks"]:
        continue
      entry = {"ts": ev["ts"], "source": f"event: {ev['message'][:140]}"}
      if ev["reason"] == "Pulled":
        entry["kubelet_pull_s"] = duration_s(ev["message"])
        entry["pulled_from_cache"] = "already present" in ev["message"]
      w["marks"][key] = entry
    log = out / f"log-{name}.txt"
    w["marks"].update(scan_log(log, SAMPLER_MARKS if role == "sampler" else TRAINER_MARKS))
    for key, when in ready.items():
      if key.startswith(f"open_rl:{role}_ready:"):
        w["marks"][f"{role}.ready_key"] = {"ts": when, "source": f"redis poller: {key}"}

  report = {"case": out.name, "collected": now(), "client": marks, "workers": workers}
  (out / "timeline.json").write_text(json.dumps(report, indent=1, default=str))
  (out / "timeline.md").write_text(render(report))
  print((out / "timeline.md").read_text())


def fmt(ts: datetime | None) -> str:
  return ts.strftime("%H:%M:%S.%f")[:-3] if ts else "unmeasured"


def render(report: dict[str, Any]) -> str:
  client = report["client"]
  t0 = client.get("client.create_model.start", {}).get("ts")
  lines = [f"# Cold start timeline: {report['case']}", ""]
  if t0:
    lines.append(
      f"t=0 is the client's `create_model` call at {t0.isoformat(timespec='milliseconds')} UTC. "
      "Offsets are seconds after t=0; a negative offset is before it."
    )
  lines.append("")
  for role in ("trainer", "sampler"):
    w = report["workers"].get(role)
    if not w:
      lines += [f"## {role}", "", "no worker observed", ""]
      continue
    lines += [
      f"## {role}: `{w.get('pod')}` on `{w.get('node')}`",
      "",
      "| Mark | Clock (UTC) | t+ s | Self-reported s | Source |",
      "|---|---|---|---|---|",
    ]
    rows = sorted(w["marks"].items(), key=lambda kv: (kv[1].get("ts") is None, kv[1].get("ts") or datetime.max.replace(tzinfo=UTC)))
    for name, m in rows:
      ts = m.get("ts")
      off = f"{(ts - t0).total_seconds():.1f}" if (ts and t0) else ""
      self_rep = m.get("kubelet_pull_s", m.get("self_reported_s", ""))
      lines.append(f"| {name} | {fmt(ts)} | {off} | {self_rep} | {m.get('source', '')} |")
    lines.append("")
  lines += ["## client", "", "| Phase | Clock (UTC) | t+ s |", "|---|---|---|"]
  for name, m in client.items():
    ts = m.get("ts")
    lines.append(f"| {name} | {fmt(ts)} | {(ts - t0).total_seconds():.1f} |" if (ts and t0) else f"| {name} | {fmt(ts)} | |")
  lines += ["", "## Critical path (derived from the marks above)", ""]
  lines += critical_path(report)
  return "\n".join(lines) + "\n"


def critical_path(report: dict[str, Any]) -> list[str]:
  """Consecutive phases per worker, each with its start and end mark named."""
  out = []
  chains = {
    "trainer": [
      ("client.create_model.start", "trainer.pod_created", "placement: Workload to pod"),
      ("trainer.pod_created", "trainer.pod_PodScheduled", "pod scheduled"),
      ("trainer.pod_PodScheduled", "trainer.event_Pulled", "image pull (0 if cached)"),
      ("trainer.event_Pulled", "trainer.container_started", "container create"),
      ("trainer.container_started", "trainer.load_start", "process start to load start (imports)"),
      ("trainer.load_start", "trainer.load_done", "weights load"),
      ("trainer.load_done", "trainer.ready_key", "ready key"),
      ("client.create_model.start", "client.create_model.done", "client: create_model blocked"),
      ("client.forward_backward.start", "client.forward_backward.done", "client: first forward_backward"),
    ],
    "sampler": [
      ("client.save_weights_and_get_sampling_client.start", "sampler.pod_created", "placement: Workload to pod"),
      ("sampler.pod_created", "sampler.pod_PodScheduled", "pod scheduled"),
      ("sampler.pod_PodScheduled", "sampler.event_Pulled", "image pull (0 if cached)"),
      ("sampler.event_Pulled", "sampler.container_started", "container create"),
      ("sampler.container_started", "sampler.process_first_log", "process start to first log"),
      ("sampler.process_first_log", "sampler.engine_init_start", "imports until engine init"),
      ("sampler.engine_init_start", "sampler.load_start", "engine init until load start"),
      ("sampler.load_start", "sampler.weights_loaded", "weights load (vLLM prefetch into page cache + load)"),
      ("sampler.weights_loaded", "sampler.compile_done", "torch.compile"),
      ("sampler.compile_done", "sampler.engine_init_done", "profiling, KV cache, graph capture"),
      ("sampler.engine_init_done", "sampler.ready_key", "ready key"),
      ("client.sample.start", "client.sample.done", "client: first sample blocked"),
    ],
  }
  all_marks = dict(report["client"])
  for w in report["workers"].values():
    all_marks.update(w["marks"])
  for role, chain in chains.items():
    out += [f"### {role}", "", "| Phase | From | To | Seconds |", "|---|---|---|---|"]
    for a, b, label in chain:
      ta, tb = all_marks.get(a, {}).get("ts"), all_marks.get(b, {}).get("ts")
      secs = f"{(tb - ta).total_seconds():.1f}" if (ta and tb) else "unmeasured"
      out.append(f"| {label} | {a} | {b} | {secs} |")
    out.append("")
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("command", choices=["watch", "collect", "timeline", "stop"])
  parser.add_argument("--out", required=True, type=Path)
  args = parser.parse_args()
  {"watch": cmd_watch, "collect": cmd_collect, "timeline": cmd_timeline, "stop": stop_watch}[args.command](args.out)


if __name__ == "__main__":
  main()
