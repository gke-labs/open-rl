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
  run.update(shared_runtime=shared, nodes=sorted({p["node"] for p in pods if p["node"]}))
  return run | {"pods": [{"name": p["name"], "problem": p["problem"]} for p in pods]}


async def read() -> dict:
  store, observed_at = get_state_store(), datetime.now(UTC).isoformat(timespec="seconds")
  scans = [asyncio.wait_for(store.scan_values(prefix), READ_TIMEOUT_SECONDS) for prefix in ("open_rl:model_meta:", "open_rl:session:")]
  cluster, (metas, sessions) = await asyncio.gather(kubernetes(), asyncio.gather(*scans, return_exceptions=True))
  errors, runs = cluster["errors"], []
  if failed := next((r for r in (metas, sessions) if isinstance(r, Exception)), None):
    errors.append(f"Redis read failed: {f'no reply in {READ_TIMEOUT_SECONDS} s' if isinstance(failed, TimeoutError) else failed}")
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
