"""The one view every page and fact reads: runs joined to their pods and workloads, worker and placement records, and nodes."""

import asyncio
import json
import time
from collections import OrderedDict

from server import openrl_status
from server.dashboard import sources
from server.model_metadata import decode_model_metadata
from server.store import get_state_store
from server.telemetry import key
from server.worker_manager import STOPPED, WAITING_FOR_PLACEMENT, owner_id

CACHE_SECONDS, OPS_SECONDS, CACHE_ENTRIES = 5, 10, 32
cache: OrderedDict[str, dict] = OrderedDict()  # least recently used first
STALE_SECONDS = 60


def succeeded(task: asyncio.Future | None) -> bool:
  return task is not None and task.done() and not task.cancelled() and task.exception() is None


async def cached(name: str, seconds: float, read) -> object:
  """One read serves every caller. Once it is older than `seconds`, or failed, a fresh read starts and callers get the last
  good result until it lands, unless that is far older than the window. Only the CACHE_ENTRIES most recently used are kept."""
  entry = cache.setdefault(name, {"at": 0.0, "task": None, "good": None, "good_at": 0.0})
  cache.move_to_end(name)
  while len(cache) > CACHE_ENTRIES:
    cache.popitem(last=False)
  task = entry["task"]
  if succeeded(task):
    entry["good"], entry["good_at"] = task, entry["at"]
  if task is None or (task.done() and (not succeeded(task) or time.monotonic() - entry["at"] > seconds)):
    entry["at"], entry["task"] = time.monotonic(), asyncio.ensure_future(read())
    entry["task"].add_done_callback(lambda t: t.cancelled() or t.exception())  # retrieved, so asyncio does not log it
  if succeeded(entry["task"]):
    return entry["task"].result()
  # A result much older than its window is no answer: after an idle spell the caller waits for the fresh read.
  if entry["good"] is not None and time.monotonic() - entry["good_at"] <= max(STALE_SECONDS, 6 * seconds):
    return entry["good"].result()
  return await asyncio.shield(entry["task"])


async def snapshot() -> dict:
  return await cached("fleet", CACHE_SECONDS, read)


async def operations(run_id: str) -> list[dict]:
  """A run's command records, trainer ops and failed samples, oldest first."""

  async def read():
    store = get_state_store()
    ops, samples = await asyncio.gather(store.read_samples(key("operations", run_id)), store.read_samples(key("operations", run_id, "samples")))
    return sorted(ops + samples, key=lambda r: r["at"])

  return await cached(run_id, OPS_SECONDS, read)


async def node_gpus() -> dict[str, list[dict]]:
  """Each node's GPUs now, from DCGM: its UUID and slot, in slot order."""
  found: dict[str, list[dict]] = {}
  now = time.time()
  for row in await sources.prom(f'max by (Hostname, UUID, gpu) (DCGM_FI_DEV_GPU_UTIL{{UUID!=""{sources.scope()}}})', now, now):
    found.setdefault(row["metric"].get("Hostname"), []).append({"id": row["metric"]["UUID"], "name": f"gpu-{row['metric'].get('gpu')}"})
  return {node: sorted(gpus, key=lambda d: (len(d["name"]), d["name"])) for node, gpus in found.items()}


def lifecycle(run: dict, live: bool, workloads: list | None) -> str:
  """ended: finished, or nothing serves it and no session is live. waiting: a worker is not placed yet.
  placed: every worker is placed. active: the scheduler cannot be read, so placement is unknown."""
  states = {w["state"] for w in run["workloads"]}
  if run["status"] != "active" or (states and states <= {STOPPED}) or not (states or live):
    return "ended"
  return "active" if workloads is None else "waiting" if not states or WAITING_FOR_PLACEMENT in states else "placed"


def join(run_id: str, raw: str, steps: dict, live: set, cluster: dict) -> dict:
  """A run is its model. LoRA runs may share a base-model runtime; an FFT or exclusive run's runtime is the run itself."""
  row, meta = json.loads(raw), decode_model_metadata(raw)
  runtime, kind = meta.runtime(run_id), "lora" if meta.fine_tuning_type == "lora" else "fft"
  shared = runtime != run_id
  # A finished run on a shared runtime lets go of it; the runtime's pods and workloads now serve others.
  keep = meta.status == "active" or not shared
  owner = owner_id(runtime).rstrip("-")
  pods = [p for p in cluster["pods"] if keep and p["owner"] == owner]
  workloads = [w for w in cluster["workloads"] or [] if keep and w["model_id"] == runtime and w["training_kind"] == kind]
  run = {"run_id": run_id, "name": row.get("name") or row.get("run_name") or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}"}
  run.update(model=meta.base_model, display_name=row.get("name") or row.get("run_name"), recipe_name=row.get("recipe_name"))
  run.update(fine_tuning_type=meta.fine_tuning_type, status=meta.status, created_at=meta.created_at, completed_at=meta.completed_at)
  run.update(steps=int(steps.get(key("steps", run_id), 0)), settings=openrl_status.settings_of(meta), session_ids=meta.session_ids)
  run.update(runtime_id=runtime, shared_runtime=shared, pods=pods, workloads=workloads, nodes=sorted({p["node"] for p in pods if p["node"]}))
  run["lifecycle"] = lifecycle(run, any(s in live for s in meta.session_ids), cluster["workloads"])
  return run


async def read() -> dict:
  store, now = get_state_store(), time.time()
  cluster, gpus = await asyncio.gather(sources.kubernetes(), node_gpus(), return_exceptions=True)
  if isinstance(gpus, Exception):
    gpus, cluster["errors"] = {}, [*cluster["errors"], f"GPU inventory unavailable: {gpus}"]
  metas, steps, workers, live = await asyncio.gather(*(store.scan_values(key(kind, "")) for kind in ("model_meta", "steps", "worker", "session")))
  live = {name.removeprefix(key("session", "")) for name in live}
  runs = sorted(
    (join(name.removeprefix(key("model_meta", "")), raw, steps, live, cluster) for name, raw in metas.items()), key=lambda r: -r["created_at"]
  )
  placements = await asyncio.gather(*(store.read_samples(key("placements", runtime)) for runtime in {r["runtime_id"] for r in runs}))

  async def last_op(run):  # when a run that never said it finished last did anything
    run["last_completed_at"] = max((s["at"] for s in await store.read_samples(key("operations", run["run_id"]), last=1)), default=None)

  await asyncio.gather(*(last_op(run) for run in runs if run["status"] == "active"))
  nodes = [{**n, "devices": gpus.get(n["name"], []), "gpu_capacity": max(n["gpu_capacity"], len(gpus.get(n["name"], [])))} for n in cluster["nodes"]]
  fleet = {"observed_at": now, "runs": runs, "workers": [json.loads(w) for w in workers.values()], "placements": [p for ps in placements for p in ps]}
  return fleet | {"nodes": nodes, "pods": cluster["pods"], "workloads": cluster["workloads"], "errors": cluster["errors"]}


def lanes(fleet: dict, run_id: str | None = None, node_id: str | None = None) -> list[dict]:
  """One lane per placement record, with its worker process's phase and GPU turns. A lane on a shared runtime
  becomes one lane per run, which shows only that run's ops."""
  found = []
  for record in fleet["placements"]:
    start, end = record["placed_at"], record["released_at"] or fleet["observed_at"]
    members = [r for r in fleet["runs"] if r["runtime_id"] == record["model_id"] and r["created_at"] <= end and (r["completed_at"] or end) >= start]
    # Pod names repeat across placements of one Workload; the process ran inside this placement's span.
    procs = [p for p in fleet["workers"] if record["pod"] and p.get("pod") == record["pod"] and p["started_at"] <= end and p["heartbeat_at"] >= start]
    ids = [r["run_id"] for r in members]
    lane = {"id": f"{record['workload']}@{record['placed_at']:.0f}", "name": record["workload"], "run_ids": ids, "runtime_id": record["model_id"]}
    lane.update(label=members[0]["name"] if len(members) == 1 else record["model_id"], shared_runtime=record["model_id"] not in ids if ids else None)
    lane.update(role=record["role"], node=record["node"] or "", pod=record["pod"], devices=record["gpu_uuids"], device_count=len(record["gpu_uuids"]))
    lane.update(phase=max(procs, key=lambda p: p["started_at"])["phase"] if procs else None, turns=[t for p in procs for t in p.get("turns") or []])
    lane.update(waits=[w for p in procs for w in p.get("waits") or []], terminations=record.get("terminations") or [])
    lane.update(start=start, end=end, ended=record["released_at"] is not None)
    for member in members if lane["shared_runtime"] else [None]:
      if member is None:
        found.append(lane)
      else:
        label = f"{(member['model'] or lane['label']).split('/')[-1]} · {member['run_id'][:8]}"
        found.append({**lane, "id": f"{lane['id']}:{member['run_id']}", "allocation_id": lane["id"], "run_ids": [member["run_id"]], "label": label})
  return [p for p in found if (not run_id or run_id in p["run_ids"]) and (not node_id or p["node"] == node_id)]


def merge(spans: list) -> list:
  result = []
  for start, end in sorted(spans):
    if result and start <= result[-1][1]:
      result[-1][1] = max(end, result[-1][1])
    else:
      result.append([start, end])
  return result


async def activity(fleet: dict, low: float, high: float, run_id: str | None = None, node_id: str | None = None) -> list[dict]:
  """Each lane in the window with its GPU turns and waits for them (none on a shared lane) and the ops it ran, clipped to it. A sampler lane's
  samples are the groups it served and its failed sends. Never utilization."""
  shown = [p for p in lanes(fleet, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  wanted = sorted({n for lane in shown for n in lane["run_ids"]})
  records = dict(zip(wanted, await asyncio.gather(*(operations(n) for n in wanted))))
  for lane in shown:
    lo, hi, turns, waits = max(low, lane["start"]), min(high, lane["end"]), lane.pop("turns"), lane.pop("waits")
    turns = [] if lane["shared_runtime"] else [[max(lo, a), min(hi, b)] for a, b in turns if a < hi and b > lo]
    # A grant inside half a second is the slicer's own round trip, not a wait for another holder.
    waits = [] if lane["shared_runtime"] else [[max(lo, a), min(hi, b)] for a, b in waits if a < hi and b > lo and b - a >= 0.5]
    ops: dict[str, list] = {}
    for r in (r for n in lane["run_ids"] for r in records[n]):
      start = max(lo, r.get("started_at") or r["at"] - r.get("elapsed_seconds", 0))
      # A sample lands on the replica pod that answered it.
      if r.get("role") == lane["role"] and r.get("pod") in (None, lane["pod"]) and start < min(hi, r["at"]):
        ops.setdefault(r["operation"], []).append([start, min(hi, r["at"])])
    lane.update(ops={n: merge(s) for n, s in ops.items()}, exact=bool(turns), intervals=merge(turns + [s for group in ops.values() for s in group]))
    lane["waits"] = merge(waits)
  return shown


def steps(ops: list[dict]) -> list[dict]:
  """Trainer ops grouped by the step they belong to: the step's window, and per op its count, time and metric spread."""
  found: dict[int, dict] = {}
  completed = 0  # records from before ops carried their step count optim_steps instead
  for s in (s for s in ops if s["operation"] != "sample"):
    number = s["step"] if isinstance(s.get("step"), int) else completed
    completed += s["operation"] == "optim_step" and s.get("status") == "succeeded"
    step = found.setdefault(number, {"step": number, "started_at": s["started_at"], "ended_at": s["at"], "ops": {}, "nonfinite": set()})
    step["started_at"], step["ended_at"] = min(step["started_at"], s["started_at"]), max(step["ended_at"], s["at"])
    op = step["ops"].setdefault(s["operation"], {"count": 0, "failed": 0, "seconds": 0.0, "metrics": {}})
    op["count"] += 1
    op["failed"] += s.get("status") != "succeeded"
    op["seconds"] += s.get("elapsed_seconds", 0)
    for name, value in s.get("metrics", {}).items():
      op["metrics"].setdefault(name, []).append(value)
    step["nonfinite"].update(s.get("nonfinite", ()))
  for step in found.values():
    optim = step["ops"].get("optim_step", {"count": 0, "failed": 0})
    step.update(nonfinite=sorted(step["nonfinite"]), complete=optim["count"] > optim["failed"])
    for op in step["ops"].values():
      op["metrics"] = {name: {"min": min(v), "max": max(v), "mean": sum(v) / op["count"]} for name, v in op["metrics"].items()}
  return [found[k] for k in sorted(found)]
