import asyncio
import os
import time
from datetime import datetime
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query

from server.model_metadata import decode_model_metadata
from server.observability import cloud_logging, facts, fleet, metrics
from server.observability.kubernetes import namespace
from server.observability.log_selector import ClientLines, WorkerPods
from server.observability.steps import steps
from server.store import get_state_store
from server.telemetry import key
from server.worker_manager import owner_id

router = APIRouter(tags=["inspection"])
BOUNDS = "ISO-8601 or Unix seconds; since also takes a span before until, e.g. 1h. Default: the 30 minutes ending now; at most 7 days."
Bound = Annotated[str | None, Query(max_length=64, description=BOUNDS)]
CURSOR_EXPIRED = "Invalid or expired log cursor; repeat the query without it"
# OPEN_RL_LOG_STORE is <store> or <store>:<resource>, the resource in the store's own terms. A store reads what a selector names.
LOG_STORES = {"cloud-logging": cloud_logging.log_page}


@router.get("/api/v1/inspect/inventory")
async def inventory():
  """Runs with their lifecycle (ended, failed, waiting, placed, active), pods and nodes; the cluster's nodes with their GPUs
  and its pods with problems; and an error per unavailable source or unreadable run record. client_status is what the
  client last reported, active until it deletes the model, so a crashed run stays active there; lifecycle is what the
  cluster shows."""
  return await fleet.snapshot()


def bounds(since: str | None, until: str | None) -> tuple[str, str]:
  try:
    return cloud_logging.window(since, until)
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 or Unix-second since/until and a positive window of at most seven days") from exc


def epoch(bound: str) -> float:
  return datetime.fromisoformat(bound).timestamp()


@router.get("/api/v1/inspect/activity")
async def activity(since: Bound = None, until: Bound = None, run_id: str | None = None, node_id: str | None = None):
  """One lane per Workload that held GPUs in the window, a run's or a node's: its role, node, pod, GPU UUIDs, the runs on
  its runtime that lived then, the intervals it held them in Unix seconds clipped to the window, and whether it let go.
  turns are the time-slicer turns its worker process was on the GPUs within those intervals, and waits its waits for a
  turn, both from the worker's record; a lane with no slicer has none. shared_gpu means another lane held one of its GPUs
  at an overlapping time, so its GPUs' metrics are not its own."""
  since, until = bounds(since, until)
  held = await fleet.holdings()
  return {"since": since, "until": until, "errors": held["errors"], "lanes": fleet.lanes(held, epoch(since), epoch(until), run_id, node_id)}


@router.get("/api/v1/inspect/query")
async def query(
  name: Literal[tuple(metrics.QUERIES)],
  run_id: str | None = None,
  node_id: str | None = None,
  since: Bound = None,
  until: Bound = None,
  points: int = Query(300, ge=1, le=1000),
):
  """A named DCGM query over the GPUs a run's or a node's placements held in the window, from OPEN_RL_PROMETHEUS_URL:
  series of [Unix seconds, value] `step` seconds apart. points=1 evaluates once over the whole window. metric is the DCGM
  metric that answered; gpu_busy falls back to DCGM_FI_DEV_GPU_UTIL where DCGM_FI_PROF_GR_ENGINE_ACTIVE has no data."""
  since, until = bounds(since, until)
  result = {"name": name, "since": since, "until": until, "series": []}
  if not os.getenv("OPEN_RL_PROMETHEUS_URL"):
    return {**result, "available": False, "error": "No metrics source configured. Set OPEN_RL_PROMETHEUS_URL."}
  low, high = epoch(since), epoch(until)
  gpus = sorted({uuid for lane in fleet.lanes(await fleet.holdings(), low, high, run_id, node_id) for uuid in lane["gpu_uuids"]})
  if not gpus:
    return {**result, "available": True, "error": "No GPUs held by this run or node in this window"}
  try:
    return {**result, **await metrics.query(name, gpus, low, high, points), "available": True}
  except httpx.HTTPStatusError as exc:
    return {**result, "available": False, "error": f"The metrics source refused the query (HTTP {exc.response.status_code})"}
  except Exception as exc:
    return {**result, "available": False, "error": f"The metrics source is unavailable: {type(exc).__name__}"}


@router.get("/api/v1/inspect/runs")
async def runs(ended: bool = False):
  """Every live run in one line, longest since its last op (a trainer op or a sample) first, so a stuck run is one read
  away. ended=true adds ended runs, which show no since_last_op_s or since_last_step_s. failures counts failed trainer
  requests, sample_failures failed samples, and sdk_errors the SDK's error events across the run's sessions."""
  state, found, now = await fleet.snapshot(), await facts.workers(), time.time()
  chosen = [r for r in state["runs"] if ended or r["lifecycle"] != "ended"]
  limit = asyncio.Semaphore(64)  # each read in flight holds a store connection

  async def line(run: dict) -> dict:
    async with limit:
      return {"run_id": run["run_id"], "name": run["name"], **await facts.line(run, facts.workers_of(found, run), now)}

  lines = await asyncio.gather(*(line(r) for r in chosen))
  return {"runs": sorted(lines, key=lambda r: -r.get("since_last_op_s", -1))}


@router.get("/api/v1/inspect/runs/{run_id}/summary")
async def run_summary(run_id: str):
  """The run's facts, each block only when it has content: steps and pace; goodput, the share of the trainer's span spent
  training, its idle seconds split into gpu_wait, queued, rollout (a sample group in flight) and client (the rest);
  requests, with the ledger and unanswered requests with their holder; failures and relaunches with a logs read;
  saves; optimizer; samplers; config mismatches; placement; incidents; sdk_errors. A run_id may name a session with one
  run. In the ledger, lost = enqueued - popped - queued, shown once the gap has lasted over a flush across reads."""
  state, found, now = await fleet.snapshot(), await facts.workers(), time.time()
  chosen = [r for r in state["runs"] if r["run_id"] == run_id] or [r for r in state["runs"] if run_id in r["session_ids"]]
  if len(chosen) != 1:
    raise HTTPException(404 if not chosen else 409, f"No run {run_id}" if not chosen else f"Session {run_id} has {len(chosen)} runs")
  run = chosen[0]
  return {"run_id": run["run_id"], **await facts.summary(run, facts.workers_of(found, run), now)}


@router.get("/api/v1/inspect/sessions/{session_id}/summary")
async def session_summary(session_id: str):
  """The session's record (created_at and redacted metadata, or recorded false when it has none), when the API server
  last heard from it, whether its live key is present, its SDK errors, and a /runs line per model."""
  state, found, now = await fleet.snapshot(), await facts.workers(), time.time()
  if (body := await facts.session(session_id, state["runs"], found, now)) is None:
    raise HTTPException(404, f"No session {session_id}")
  return body


@router.get("/api/v1/inspect/runs/{run_id}/steps")
async def run_steps(
  run_id: str,
  last: int = Query(20, ge=1, le=1000),
  fields: Annotated[str | None, Query(description="Comma-separated prefixes of the op metrics to keep, e.g. loss,grad_norm. Default: all")] = None,
):
  """The run's last steps, oldest first, from the trainer's op records: each step's window, whether its optim_step
  succeeded, the metrics that were NaN or infinite, and per op its count, failures, seconds and metric spread."""
  store = get_state_store()
  if await store.get_value(key("model_meta", run_id)) is None:
    raise HTTPException(404, f"No run {run_id}")
  every = steps(await store.read_samples(key("operations", run_id)))
  if keep := tuple(p for p in (fields or "").split(",") if p):
    for step in every[-last:]:
      for op in step["ops"].values():
        op["metrics"] = {name: spread for name, spread in op["metrics"].items() if name.startswith(keep)}
  return {"run_id": run_id, "total": len(every), "steps": every[-last:]}


@router.get("/api/v1/inspect/logs")
async def logs(
  run_id: str,
  pod: Annotated[str | None, Query(description="One of the run's pods, or 'client' for the lines labelled openrl_client=<client_log_id>")] = None,
  q: str = Query("", max_length=1024),
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: Bound = None,
  until: Bound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: Annotated[str | None, Query(max_length=16384, description="next_cursor of a page with the same filters and window")] = None,
):
  """Newest first, from the OPEN_RL_LOG_STORE log store: the run's worker pods, or with pod=client its client's lines."""
  since, until = bounds(since, until)
  if (raw := await get_state_store().get_value(f"open_rl:model_meta:{run_id}")) is None:
    raise HTTPException(404, f"No run {run_id}")
  meta = decode_model_metadata(raw)
  store = os.getenv("OPEN_RL_LOG_STORE", "").strip()
  kind, _, resource = store.partition(":")
  result = {"run_id": run_id, "source": store, "since": since, "until": until, "records": [], "next_cursor": None}
  if not store:
    return {**result, "available": False, "error": "No log store configured. Set OPEN_RL_LOG_STORE; see docs/configuration.md."}
  if (page_of := LOG_STORES.get(kind)) is None:
    return {**result, "available": False, "error": f"Unknown OPEN_RL_LOG_STORE {store!r}; supported: {', '.join(LOG_STORES)}"}
  if pod == "client":
    if not meta.client_log_id:
      return {**result, "available": False, "error": "This run set no openrl.client_log_id; see docs/configuration.md."}
    selector = ClientLines(meta.client_log_id)
  else:
    # The scheduler names a worker pod orw-<Workload>, and scheduler_worker_manager.workload_name starts it <kind>-<owner>-.
    prefix = f"orw-{'lora' if meta.fine_tuning_type == 'lora' else 'fft'}-{owner_id(meta.runtime(run_id))}-"
    selector = WorkerPods(namespace(), prefix, pod)
  try:
    page = await page_of(resource or None, selector, since, until, cursor=cursor, limit=limit, q=q, severity=severity)
    return {**result, **page, "available": True}
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 400 and cursor:
      raise HTTPException(400, CURSOR_EXPIRED) from None
    return {**result, "available": False, "error": f"{store} refused the read (HTTP {exc.response.status_code}); check credentials and IAM"}
  except Exception as exc:
    return {**result, "available": False, "error": str(exc) if isinstance(exc, LookupError) else f"{store} unavailable"}
