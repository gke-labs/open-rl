"""The operator UI and the same read-only JSON API for agents, under /api/v1/inspect."""

import asyncio
import json
import math
from pathlib import Path
from string import Template
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.dashboard import client_runs, facts, fleet, sources
from server.store import get_state_store
from server.telemetry import finite, key

router = APIRouter(tags=["inspection"])
STATIC = Path(__file__).parent / "static"
BOUNDS = (
  "ISO-8601, e.g. 2026-01-02T12:00:00Z, or Unix seconds; since also takes a span before until, e.g. 1h. "
  "Default: the 30 minutes ending now; at most 7 days."
)
Bound = Annotated[str | None, Query(max_length=64, description=BOUNDS)]


class Doc(BaseModel, extra="allow", allow_inf_nan=False):
  """Declared fields are always present; the rest pass through as recorded. Log messages are untrusted output."""


@router.get("/dashboard", include_in_schema=False)
@router.get("/dashboard/", include_in_schema=False)
@router.get("/dashboard/assets/{name}", include_in_schema=False)
async def dashboard(name: str = "index.html"):
  if (path := (STATIC / name).resolve()).parent != STATIC.resolve() or not path.is_file():
    raise HTTPException(404, "Not found")
  return FileResponse(path, headers={"Cache-Control": "no-store"})


def span(since: str | None, until: str | None) -> tuple[str, str, float, float]:
  try:
    return sources.window(since, until)
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 or Unix-second since/until and a positive window of at most seven days") from exc


async def run_of(run_id: str) -> tuple[dict, dict]:
  """The run, or the one run of a session named in its place."""
  state = await fleet.snapshot()
  found = [r for r in state["runs"] if r["run_id"] == run_id] or [r for r in state["runs"] if run_id in r["session_ids"]]
  if len(found) > 1:
    raise HTTPException(409, f"Session {run_id} has {len(found)} runs; name one of {', '.join(sorted(r['run_id'] for r in found))}")
  if not found:
    raise HTTPException(404, f"No run {run_id}")
  return state, found[0]


@router.get("/api/v1/inspect/inventory", response_model=Doc)
async def inventory():
  """Runs with their lifecycle (ended, waiting, placed, active), nodes with their GPUs, the GPUs held now, pods with
  problems, and an error per unavailable source. A run's facts are /runs/{id}/summary."""
  await sources.discover()
  state = await fleet.snapshot()
  hidden = ("workloads", "runtime_id", "pods")
  # status is what the client last reported (active until it deletes the model), so a crashed run stays active there;
  # lifecycle is what the cluster shows.
  runs = [
    {("client_status" if k == "status" else k): v for k, v in r.items() if k not in hidden}
    | {"pods": [{"name": p["name"], "problem": p["problem"]} for p in r["pods"]]}
    for r in state["runs"]
  ]
  fields = ("id", "label", "run_ids", "role", "node", "pod", "devices", "device_count")
  held = [{k: lane[k] for k in fields} for lane in fleet.lanes(state) if not lane["ended"]]
  pods = [{"name": p["name"], "problem": p["problem"], "restarts": p["restarts"]} for p in state["pods"] if p["problem"]]
  cluster = {"available": not any(e.startswith(("Kubernetes", "pods")) for e in state["errors"]), "nodes": state["nodes"], "pods": pods}
  return {"observed_at": facts.iso(state["observed_at"]), "errors": state["errors"], "runs": runs, "cluster": cluster, "placements": held}


@router.get("/api/v1/inspect/activity", response_model=Doc)
async def activity(since: Bound = None, until: Bound = None, run_id: str | None = None, node_id: str | None = None):
  """Each placement's GPU turns, waits (from asking the time slicer for the GPU to the grant, half a second or more), recorded
  ops and its pod's container terminations in the window (Unix seconds), fleet-wide, a node's or a run's.
  A sampler's `sample` spans are the groups of samples it served, one record per run each. Gaps do not prove idleness."""
  since, until, low, high = span(since, until)
  if run_id:
    state, run = await run_of(run_id)
    run_id = run["run_id"]
  else:
    state = await fleet.snapshot()
  placements = await fleet.activity(state, low, high, run_id, node_id)
  return {"placements": placements, "since": since, "until": until, "observed_at": facts.iso(state["observed_at"])}


VLLM = "{pod=~$pods$scope}"


# The DCGM exporter's collection interval in k8s/deploy.
DCGM_SECONDS = 10


def utilization(form: str, by: str = "UUID, gpu") -> str:
  """A GPU's busy percent. GR_ENGINE_ACTIVE averages DCGM's whole collection interval. GPU_UTIL, the fallback where it is
  missing, covers about the last second of each interval, so a load with that period aliases."""
  active, util = "DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID=~$gpus$scope}", "DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus$scope}"
  return f"max by ({by}) (100 * {form.format(active)}) or max by ({by}) ({form.format(util)})"


QUERIES = {
  # By UUID and gpu: DCGM's slot index names a GPU after its node is gone.
  "gpu_util": utilization("{}"),
  "gpu_memory_mib": "max by (UUID, gpu) (DCGM_FI_DEV_FB_USED{UUID=~$gpus$scope})",
  "gpu_memory_peak_mib": "max by (UUID, gpu) (max_over_time(DCGM_FI_DEV_FB_USED{UUID=~$gpus$scope}[${window}s]))",
  "gpu_memory_total_mib": "max by (UUID) (DCGM_FI_DEV_FB_TOTAL{UUID=~$gpus$scope})",
  # Utilization comes in bursts; the last two minutes say whether a GPU is working now better than one sample.
  "gpu_util_now": utilization("avg_over_time({}[2m])", "Hostname, UUID, gpu, modelName"),
  # Duty cycle over the window: the share of time a kernel ran, one number per GPU with points=1.
  "gpu_busy": utilization("avg_over_time({}[${{window}}s])", "Hostname, UUID, gpu, modelName"),
  # From vLLM's and llm-d's published Grafana dashboards.
  "vllm_generation_tokens_per_second": f"sum(rate(vllm:generation_tokens_total{VLLM}[2m]))",
  "vllm_prompt_tokens_per_second": f"sum(rate(vllm:prompt_tokens_total{VLLM}[2m]))",
  "vllm_requests_running": f"sum(vllm:num_requests_running{VLLM})",
  "vllm_requests_waiting": f"sum(vllm:num_requests_waiting{VLLM})",
  "vllm_kv_cache_usage": f"max(vllm:kv_cache_usage_perc{VLLM})",
  "vllm_preemptions_per_second": f"sum(rate(vllm:num_preemptions_total{VLLM}[2m]))",
  "vllm_time_to_first_token_p99_seconds": f"histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{VLLM}[2m])))",
  "vllm_request_latency_p99_seconds": f"histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{VLLM}[2m])))",
}


def subject(state: dict, run_id: str | None, node_id: str | None, low: float, high: float) -> tuple[list[str] | None, list[str]]:
  """The GPUs and pods a run's or a node's placements held in the window, plus a node's GPUs now; None GPUs is the fleet."""
  if not (run_id or node_id):
    return None, []
  held = [p for p in fleet.lanes(state, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  now = [d["id"] for n in state["nodes"] if not run_id and n["name"] == node_id for d in n["devices"]]
  return sorted({uuid for p in held for uuid in p["devices"]} | set(now)), sorted({p["pod"] for p in held if p["pod"]})


@router.get("/api/v1/inspect/query", response_model=Doc)
async def query(
  name: Annotated[str | None, Query(description="One of the named queries; leave it out to list them")] = None,
  run_id: str | None = None,
  node_id: str | None = None,
  since: Bound = None,
  until: Bound = None,
  points: int = Query(300, ge=1, le=1000),
):
  """A named PromQL query over a run's GPUs and pods, a node's, or (GPU queries only) the fleet's, at `points` resolution,
  `step` seconds apart. points=1 evaluates once over the whole window, which is how a duty cycle is one number per GPU.
  A gpu_util point is a DCGM reading placed at the start of the reading_seconds it averaged, so it lines up with turns."""
  if name not in QUERIES:
    raise HTTPException(400, f"{'Unknown query' if name else 'Name a query'}; use one of {sorted(QUERIES)}")
  since, until, low, high = span(since, until)
  await sources.discover()
  gpus, pods = subject(await fleet.snapshot(), run_id, node_id, low, high)
  step = sources.step_seconds(low, high, points)
  result = {"name": name, "since": since, "until": until, "step": step, "series": [], "source": sources.prometheus_source()}
  if gpus == [] and "$gpus" in QUERIES[name] or not pods and "$pods" in QUERIES[name]:
    return {**result, "error": "No GPUs or pods recorded for this subject in this window"}
  values = {"gpus": '".+"' if gpus is None else sources.any_of(gpus), "pods": sources.any_of(pods), "window": max(1, int(high - low))}
  result["query"] = text = Template(QUERIES[name]).substitute(values, scope=sources.scope())
  try:
    rows = await sources.prom(text, high if points == 1 else low, high, points)
  except Exception as exc:
    return {**result, "error": str(exc) if isinstance(exc, LookupError) else f"Query failed: {type(exc).__name__}"}
  # A reading averages the interval before it was collected, and a range query returns it up to one reading later.
  shift = 2 * DCGM_SECONDS if name == "gpu_util" else 0
  series = [{"labels": r["metric"], "points": [[float(t) - shift, float(v)] for t, v in r["values"] if math.isfinite(float(v))]} for r in rows]
  return result | {"series": series} | ({"reading_seconds": DCGM_SECONDS} if shift else {})


@router.get("/api/v1/inspect/accelerators", response_model=Doc)
async def accelerators(run_id: str | None = None, node_id: str | None = None, since: Bound = None, until: Bound = None):
  """Every GPU in the fleet, a node's, or a run's in the window: node, slot, model, holders (the newest placement on it
  and any time-sliced with it, with their pods' container terminations), busy (percent of the window DCGM saw the GPU
  engine active), and peak and total memory in MiB. The window defaults to the last hour, or an ended run's last hour."""
  if run_id and since is None and until is None:
    state, run = await run_of(run_id)
    run_id, ends = run["run_id"], [p["end"] for p in fleet.lanes(state, run["run_id"])]
    if run["lifecycle"] == "ended" and ends:
      since, until = "1h", facts.iso(max(ends))
  since, until, low, high = span(since or ("1h" if until is None else None), until)
  names = {"busy": "gpu_busy", "memory_peak_mib": "gpu_memory_peak_mib", "memory_total_mib": "gpu_memory_total_mib"}
  found = dict(zip(names, await asyncio.gather(*(query(name, run_id, node_id, since, until, 1) for name in names.values()))))
  result = {"since": since, "until": until, "gpus": [], "source": found["busy"]["source"]}
  if error := next((r["error"] for r in found.values() if r.get("error")), None):
    return {**result, "error": error}
  value = {k: {s["labels"]["UUID"]: round(s["points"][-1][1], 1) for s in r["series"] if s["points"]} for k, r in found.items()}
  held = sorted((p for p in fleet.lanes(await fleet.snapshot(), run_id) if p["start"] <= high and p["end"] >= low), key=lambda p: p["start"])
  placed: dict[str, list[dict]] = {}
  for lane in held:
    for uuid in lane["devices"]:
      placed.setdefault(uuid, []).append(lane)
  # A time-sliced GPU has several holders at once, every placement that overlaps the newest one.
  holders = {uuid: [p for p in lanes if p["end"] >= lanes[-1]["start"]] for uuid, lanes in placed.items()}
  labels_of = {series["labels"]["UUID"]: series["labels"] for series in found["busy"]["series"]}
  for uuid, labels in labels_of.items():
    node = labels.get("Hostname") or next((lane["node"] for lane in reversed(holders.get(uuid, []))), None)
    gpu = {"uuid": uuid, "node": node, "slot": labels.get("gpu"), "model": labels.get("modelName")}
    gpu |= {name: value[name][uuid] for name in names if uuid in value[name]}
    if uuid in holders:
      gpu["holders"] = [{k: lane[k] for k in ("run_ids", "role", "pod", "start", "end", "ended", "terminations")} for lane in holders[uuid]]
    result["gpus"].append(gpu)
  result["gpus"].sort(key=lambda g: (g["node"] or "", int(g["slot"] or 0)))
  return result


# Prefixes of the cookbook's metrics an agent reads first; the rest is a fields= away.
KEY_METRICS = (
  *("env/all/reward", "test/env/all/reward", "env/all/lab/criteria_pass_fraction", "test/env/all/lab/criteria_pass_fraction"),
  *("optim/kl", "optim/entropy", "time/total", "progress/"),
)


async def client_directory(run: dict) -> str | None:
  """The run's client directory under OPEN_RL_CLIENT_RUNS_ROOT, if exactly one matches it."""
  candidates = await fleet.cached("client directories", 30, lambda: asyncio.to_thread(client_runs.directories))
  return await asyncio.to_thread(client_runs.match, run, candidates)


@router.get("/api/v1/inspect/runs/{run_id}/steps", response_model=Doc)
async def steps(
  run_id: str,
  last: int = Query(20, ge=1, le=1000),
  fields: Annotated[str | None, Query(description="Comma-separated prefixes of client metrics to keep, e.g. env/all/reward; * keeps all")] = None,
):
  """The run's last steps, oldest first: window, completion, trainer ops (count, failures, seconds, metric spread), and the
  `client` metrics its client wrote to metrics.jsonl for the step: by default the key ones (rewards, KL, entropy, step time,
  progress), with every name the client logged in client_metric_names."""
  _, run = await run_of(run_id)
  run_id = run["run_id"]
  keep = () if fields == "*" else tuple(p for p in fields.split(",") if p) if fields else KEY_METRICS
  every, result = fleet.steps(await fleet.operations(run_id)), {}
  # Some runs keep their client metrics in Redis; metrics.jsonl wins a step both have.
  client = {row["step"]: finite(row["metrics"]) for row in await get_state_store().read_samples(f"open_rl:run_metrics:{run_id}")}
  if directory := await client_directory(run):
    try:
      client |= await asyncio.to_thread(client_runs.metrics, directory)
    except (OSError, ValueError) as exc:  # steps stand without the client's metrics
      result["client_error"] = f"{directory}/metrics.jsonl unreadable: {type(exc).__name__}"
  result["client_directory"] = directory
  for step in every[-last:]:
    step["client"] = {k: v for k, v in client.get(step["step"], {}).items() if not keep or k.startswith(keep)}
  result["client_metric_names"] = sorted({name for metrics in client.values() for name in metrics})
  return result | {"run_id": run_id, "total": len(every), "steps": every[-last:]}


@router.get("/api/v1/inspect/runs/{run_id}/commands", response_model=Doc)
async def commands(run_id: str, since: Bound = None, until: Bound = None, role: str | None = None, limit: int = Query(100, ge=1, le=5000)):
  """The run's raw command records (trainer ops, the groups of samples a sampler served, and failed samples),
  newest first; without bounds, all of them. A group's record has its count of samples."""
  run_id = (await run_of(run_id))[1]["run_id"]
  low, high = span(since, until)[2:] if since or until else (0, math.inf)
  wanted = (lambda r: True) if role is None else (lambda r: (r.get("role") == "sampler") == (role == "sampler"))
  found = [r for r in reversed(await fleet.operations(run_id)) if low <= r["at"] <= high and wanted(r)]
  return {"run_id": run_id, "total": len(found), "commands": found[:limit]}


@router.get("/api/v1/inspect/runs", response_model=Doc)
async def runs(ended: bool = False):
  """Every live run in one line, longest since its last op first, so a stuck run is one read away. An op includes a sample.
  ended=true adds ended runs."""
  state = await fleet.snapshot()
  chosen = [r for r in state["runs"] if ended or r["lifecycle"] != "ended"]
  # A few runs at a time, since each read in flight holds a store connection.
  limit = asyncio.Semaphore(128)

  async def line(run):
    async with limit:
      return await facts.fleet_line(state, run)

  found = await asyncio.gather(*(line(r) for r in chosen))
  lines = [{"run_id": r["run_id"], "name": r["name"], **line} for r, line in zip(chosen, found)]
  return {"runs": sorted(lines, key=lambda r: -(r["since_last_op_s"] if r["since_last_op_s"] is not None else math.inf))}


@router.get("/api/v1/inspect/runs/{run_id}/summary", response_model=Doc)
async def summary(run_id: str):
  """The run's facts, each block only when it has content: pace, goodput, unanswered requests and their holders, the ledger
  (lost = enqueued - popped - queued), failures and relaunches with log reads, saves, optimizer, samplers, config,
  placement, incidents, and GPU duty by role over the run's span (gpu_busy, percent). No verdict; reason from the facts.

  A vLLM sampler keeps one record per group of samples it served and one per failed sample. In samplers, sends is the
  run's sample count, failed and cancelled count its records, and median_s is the median time of the groups it served.
  In goodput, rollout is trainer idle time while one of those groups was being served."""
  state, run = await run_of(run_id)
  run_id = run["run_id"]
  ctx = await facts.context(state, run)
  if relaunched := facts.relaunches(ctx):
    # The scheduler names a Workload's pod orw-<workload>, so the Workload's name matches both; older records hold the pod's.
    names = sorted({lane["name"].removeprefix("orw-") for lane in ctx["lanes"]})
    try:
      ctx["changes"] = await sources.audit_events(names, relaunched[0]["stopped_at"] - 300, relaunched[-1]["started_at"] + 60)
    except Exception as exc:
      ctx["changes"] = {"error": f"Audit log unavailable: {exc if isinstance(exc, LookupError) else type(exc).__name__}"}
  found = {}
  if ops := ctx["ops"]:
    # Duty over the run's own span: averaging in the idle days after it ended would understate it.
    end = min(state["observed_at"], facts.last_op_at(ctx) + 60)
    duty = await query("gpu_busy", run_id, None, facts.iso(max(ops[0]["at"] - 60, end - 7 * 86400)), facts.iso(end), 1)
    roles: dict[str, set[str]] = {}
    for lane in ctx["lanes"]:
      for uuid in lane["devices"]:
        roles.setdefault(uuid, set()).add(lane["role"])
    busy: dict[str, list[float]] = {}
    for series in (s for s in duty["series"] if s["points"]):
      # A GPU that time-sliced roles shared counts under the roles together, e.g. sampler+trainer.
      busy.setdefault("+".join(sorted(roles.get(series["labels"].get("UUID")) or {"other"})), []).append(series["points"][-1][1])
    if busy:
      found["gpu_busy"] = {role: round(sum(v) / len(v), 1) for role, v in busy.items()}
      # On a shared runtime these are the runtime's GPUs, busy with every run it serves.
      found["gpu_busy_window"] = {k: duty[k] for k in ("since", "until")} | {"scope": "runtime" if run["shared_runtime"] else "run"}
  return {"run_id": run_id, **facts.summary(ctx), **found}


@router.get("/api/v1/inspect/sessions/{session_id}/summary", response_model=Doc)
async def session_summary(session_id: str):
  """The session's record, whether its registry key is live, its SDK errors, and a line per model over the session's span."""
  store = get_state_store()
  if (record := json.loads(await store.get_value(key("session_record", session_id)) or "null")) is None:
    raise HTTPException(404, f"No session {session_id}")
  live = await store.get_value(key("session", session_id)) is not None
  # A model's ops belong to this session while it was alive, which ends when its live key expires after last_seen_at.
  low, high = record["created_at"], math.inf if live else record["last_seen_at"] + 120
  state, models = await fleet.snapshot(), []
  for run in (r for r in state["runs"] if r["run_id"] in record["model_ids"]):
    models.append({"model_id": run["run_id"], **facts.line(await facts.context(state, run, low, high))})
  errors = await store.read_samples(key("sdk_errors", session_id))
  return {**record, "live": live, "models": models} | ({"sdk_errors": {"count": len(errors), "last": errors[-1].get("message")}} if errors else {})


@router.get("/api/v1/inspect/logs", response_model=Doc)
async def logs(
  run_id: str,
  pod: Annotated[str | None, Query(description="One of the run's pods, or 'client' for its client's lines")] = None,
  q: str = Query("", max_length=1024),
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: Bound = None,
  until: Bound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: Annotated[str | None, Query(max_length=16384, description="next_cursor of a page with the same filters and window")] = None,
):
  """Newest first: the run's worker pods by owner label from Cloud Logging, and with pod=client its client's logs.log. That
  file's lines carry no time, so since and until do not apply to it and its cursor is a byte offset."""
  since, until, low, high = span(since, until)
  await sources.discover()
  state, run = await run_of(run_id)
  run_id = run["run_id"]
  pods = {lane["pod"]: {"pod": lane["pod"], "node": lane["node"], "role": lane["role"]} for lane in fleet.lanes(state, run_id) if lane["pod"]}
  result = {"run_id": run_id, "source": "gke", "since": since, "until": until, "sources": list(pods.values()), "records": [], "next_cursor": None}
  if directory := await client_directory(run):
    result["sources"].append({"pod": sources.CLIENT, "node": None, "role": sources.CLIENT})
  if directory and pod == sources.CLIENT:
    if cursor and not cursor.isdigit():
      raise HTTPException(400, "Invalid or expired log cursor; repeat the query without it")
    result["source"] = path = f"{directory}/logs.log"
    try:
      page = await asyncio.to_thread(client_runs.log_page, directory, int(cursor) if cursor else None, limit, q, severity)
      return {**result, **page, "available": True}
    except OSError as exc:
      return {**result, "available": False, "error": f"{path} unreadable: {type(exc).__name__}"}
  try:
    return {**result, **await sources.log_page(run, since, until, cursor=cursor, limit=limit, q=q, pod=pod, severity=severity), "available": True}
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 400 and cursor:
      raise HTTPException(400, "Invalid or expired log cursor; repeat the query without it") from None
    return {**result, "available": False, "error": f"Cloud Logging refused the read (HTTP {exc.response.status_code}); check credentials and IAM"}
  except Exception as exc:
    return {**result, "available": False, "error": str(exc) if isinstance(exc, LookupError) else "Cloud Logging unavailable"}
