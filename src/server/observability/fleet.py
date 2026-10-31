import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from server import openrl_status
from server.model_metadata import decode_model_metadata
from server.observability.kubernetes import READ_TIMEOUT_SECONDS, Cluster, kubernetes
from server.store import get_state_store
from server.worker_manager import FAILED, STOPPED, WAITING_FOR_PLACEMENT, owner_id

CACHE_SECONDS, STALE_SECONDS = 5, 60
PLACEMENT_FIELDS = ("name", "role", "node", "pod", "gpu_uuids", "placed_at", "released_at", "training_kind")
Lifecycle = Literal["ended", "failed", "waiting", "placed", "active"]


@dataclass
class Entry:
  at: float = 0.0
  task: asyncio.Future | None = None
  last_good: asyncio.Future | None = None
  last_good_at: float = 0.0


cache: dict[str, Entry] = {}


def succeeded(task: asyncio.Future | None) -> bool:
  return task is not None and task.done() and not task.cancelled() and task.exception() is None


async def cached(name: str, seconds: float, read) -> object:
  entry = cache.setdefault(name, Entry())
  task = entry.task
  if succeeded(task):
    entry.last_good, entry.last_good_at = task, entry.at
  if task is None or (task.done() and (not succeeded(task) or time.monotonic() - entry.at > seconds)):
    entry.at, entry.task = time.monotonic(), asyncio.ensure_future(read())
    entry.task.add_done_callback(lambda t: t.cancelled() or t.exception())  # retrieved, so asyncio does not log it
  if succeeded(entry.task):
    return entry.task.result()
  if entry.last_good is not None and time.monotonic() - entry.last_good_at <= STALE_SECONDS:
    return entry.last_good.result()
  return await asyncio.shield(entry.task)


async def snapshot() -> dict:
  return await cached("inventory", CACHE_SECONDS, read)


def owner(runtime: str) -> str:
  """The owner label on the runtime's pods; the scheduler trims a truncated label value as well."""
  return owner_id(runtime).rstrip("-")


def lifecycle(status: str, live: bool, states: set[str] | None) -> Lifecycle:
  if status != "active" or (states and states <= {STOPPED}) or not (states or live):
    return "ended"
  if states and FAILED in states:
    return "failed"
  return "active" if states is None else "waiting" if not states or WAITING_FOR_PLACEMENT in states else "placed"


def join(run_id: str, raw: str, live: set[str], cluster: Cluster) -> dict:
  row, meta = json.loads(raw), decode_model_metadata(raw)
  runtime, kind = meta.runtime(run_id), "lora" if meta.fine_tuning_type == "lora" else "fft"
  shared = runtime != run_id
  claims_runtime = meta.status == "active" or not shared
  pods = [p for p in cluster["pods"] if claims_runtime and p["owner"] == owner(runtime)]
  states = None
  if cluster["workloads"] is not None:
    states = {w["state"] for w in cluster["workloads"] if claims_runtime and w["model_id"] == runtime and w["training_kind"] == kind}
  name = row.get("name") or row.get("run_name")
  run = {"run_id": run_id, "name": name or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}", "display_name": name}
  run.update(model=meta.base_model, recipe_name=row.get("recipe_name"), fine_tuning_type=meta.fine_tuning_type, client_status=meta.status)
  run["lifecycle"] = lifecycle(meta.status, any(s in live for s in meta.session_ids), states)
  run.update(created_at=meta.created_at, completed_at=meta.completed_at, session_ids=meta.session_ids, settings=openrl_status.settings_of(meta))
  run.update(runtime=runtime, shared_runtime=shared, nodes=sorted({p["node"] for p in pods if p["node"]}))
  return run | {"pods": [{"name": p["name"], "problem": p["problem"]} for p in pods]}


def redis_failed(exc: Exception) -> str:
  return f"Redis read failed: {f'no reply in {READ_TIMEOUT_SECONDS} s' if isinstance(exc, TimeoutError) else exc}"


async def read() -> dict:
  store, observed_at = get_state_store(), datetime.now(UTC).isoformat(timespec="seconds")
  scans = [asyncio.wait_for(store.scan_values(prefix), READ_TIMEOUT_SECONDS) for prefix in ("open_rl:model_meta:", "open_rl:session:")]
  cluster, (metas, sessions) = await asyncio.gather(kubernetes(), asyncio.gather(*scans, return_exceptions=True))
  errors, runs = cluster["errors"], []
  if failed := next((r for r in (metas, sessions) if isinstance(r, Exception)), None):
    errors.append(redis_failed(failed))
    metas, sessions = {}, {}
  live = {name.removeprefix("open_rl:session:") for name in sessions}
  for name, raw in metas.items():
    run_id = name.removeprefix("open_rl:model_meta:")
    try:
      runs.append(join(run_id, raw, live, cluster))
    except Exception as exc:
      errors.append(f"model record {run_id} unreadable: {exc}")
  runs.sort(key=lambda r: -r["created_at"])
  pods = [{"name": p["name"], "problem": p["problem"], "restarts": p["restarts"]} for p in cluster["pods"] if p["problem"]]
  view = {"available": cluster["available"], "nodes": cluster["nodes"], "pods": pods}
  return {"observed_at": observed_at, "errors": errors, "runs": runs, "cluster": view, "placements": placements(cluster["workloads"], metas)}


def placements(workloads: list[dict] | None, metas: dict[str, str]) -> list[dict]:
  """The Workloads holding GPUs now, each with the runs it serves: those whose runtime is its model ID and whose kind
  matches. A shared runtime serves only its active runs."""
  served = []
  for name, raw in metas.items():
    run_id = name.removeprefix("open_rl:model_meta:")
    try:
      meta = decode_model_metadata(raw)
    except Exception:
      continue  # read reports the unreadable record
    if meta.status == "active" or not meta.shares_runtime():
      served.append((meta.runtime(run_id), "lora" if meta.fine_tuning_type == "lora" else "fft", run_id))
  fields = ("name", "role", "node", "pod", "gpu_uuids", "placed_at")
  return [
    {field: w[field] for field in fields}
    | {"run_ids": sorted(r for runtime, kind, r in served if (runtime, kind) == (w["model_id"], w["training_kind"]))}
    for w in workloads or []
    if w["holds_gpus"]
  ]


async def holdings() -> dict:
  return await cached("holdings", CACHE_SECONDS, read_holdings)


async def read_holdings() -> dict:
  """Every placement known: live ones from Workload status, released ones from open_rl:placements:<runtime>; and the runs
  that could have used them."""
  now = time.time()
  history = asyncio.ensure_future(asyncio.wait_for(read_history(), READ_TIMEOUT_SECONDS))
  cluster = await kubernetes()
  live = [{f: w[f] for f in PLACEMENT_FIELDS} | {"runtime": w["model_id"]} for w in cluster["workloads"] or [] if w["holds_gpus"]]
  try:
    runs, released, workers, errors = await history
  except Exception as exc:
    runs, released, workers, errors = [], [], [], [redis_failed(exc)]
  return {"observed_at": now, "runs": runs, "placements": live + released, "workers": workers, "errors": cluster["errors"] + errors}


async def read_history() -> tuple[list[dict], list[dict], list[dict], list[str]]:
  """From Redis: the runs, the placements their runtimes released, the worker processes' records, and an error per
  unreadable history."""
  store, runs, released, errors = get_state_store(), [], [], []
  workers = []
  for raw in (await store.scan_values("open_rl:worker:")).values():
    try:
      workers.append(json.loads(raw))
    except ValueError as exc:
      errors.append(f"worker record unreadable: {exc!r}")
  for name, raw in (await store.scan_values("open_rl:model_meta:")).items():
    try:
      meta, run_id = decode_model_metadata(raw), name.removeprefix("open_rl:model_meta:")
    except Exception:
      continue  # the inventory reports it
    kind = "lora" if meta.fine_tuning_type == "lora" else "fft"
    runs.append({"run_id": run_id, "runtime": meta.runtime(run_id), "kind": kind, "created_at": meta.created_at, "completed_at": meta.completed_at})
  runtimes = sorted({r["runtime"] for r in runs})
  for runtime, records in zip(runtimes, await asyncio.gather(*(store.recent(f"open_rl:placements:{r}") for r in runtimes))):
    try:
      released += [{f: record[f] for f in PLACEMENT_FIELDS} | {"runtime": runtime} for record in map(json.loads, records)]
    except (ValueError, KeyError) as exc:
      errors.append(f"placement history of {runtime} unreadable: {exc!r}")
  return runs, released, workers, errors


def lanes(held: dict, low: float, high: float, run_id: str | None = None, node_id: str | None = None) -> list[dict]:
  """One lane per Workload and the GPUs it held in [low, high]: the intervals it held them, clipped to the window, the
  runs on its runtime that lived then, and from its worker processes' records the time-slicer turns it was on the GPUs
  and its waits for them. Holding is the placement; a turn is a slice of it, so a lane with turns was busy only during
  them. Lanes that held one GPU at overlapping times are shared_gpu."""
  now, found = held["observed_at"], {}
  for p in held["placements"]:
    start, end = p["placed_at"], p["released_at"] or now
    if start >= high or end <= low:
      continue
    fresh = {k: p[k] for k in ("name", "role", "node", "pod", "gpu_uuids")} | {"run_ids": set(), "ended": True}
    fresh |= {"intervals": [], "turns": [], "waits": []}
    lane = found.setdefault((p["name"], p["node"], tuple(p["gpu_uuids"])), fresh)
    lane["run_ids"].update(
      r["run_id"]
      for r in held["runs"]
      if (r["runtime"], r["kind"]) == (p["runtime"], p["training_kind"]) and r["created_at"] <= end and (r["completed_at"] or now) >= start
    )
    lane["intervals"].append([max(low, start), min(high, end)])
    # Pod names repeat across a Workload's placements; a process ran inside the placement it heartbeat during.
    lo, hi = max(low, start), min(high, end)
    for proc in held.get("workers") or []:
      if p["pod"] and proc.get("pod") == p["pod"] and proc.get("started_at", 0) <= end and proc.get("heartbeat_at", 0) >= start:
        for kind in ("turns", "waits"):
          lane[kind] += [[max(lo, a), min(hi, b)] for a, b in proc.get(kind) or [] if a < hi and b > lo]
    lane["ended"] &= p["released_at"] is not None
  result = [lane | {"run_ids": sorted(lane["run_ids"]), **{k: merge(lane[k]) for k in ("intervals", "turns", "waits")}} for lane in found.values()]
  for lane in result:
    lane["shared_gpu"] = any(
      other is not lane and set(lane["gpu_uuids"]) & set(other["gpu_uuids"]) and overlap(lane["intervals"], other["intervals"]) for other in result
    )
  return [lane for lane in result if (not run_id or run_id in lane["run_ids"]) and (not node_id or lane["node"] == node_id)]


def merge(spans: list[list[float]]) -> list[list[float]]:
  merged: list[list[float]] = []
  for start, end in sorted(spans):
    if merged and start <= merged[-1][1]:
      merged[-1][1] = max(end, merged[-1][1])
    else:
      merged.append([start, end])
  return merged


def overlap(a: list[list[float]], b: list[list[float]]) -> bool:
  return any(a0 < b1 and b0 < a1 for a0, a1 in a for b0, b1 in b)
