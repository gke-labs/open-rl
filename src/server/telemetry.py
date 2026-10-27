"""The records OpenRL keeps about its own runs, written where things happen, for the inspection API to read.

A run's events go through record(), which keeps them in the store and also prints each as one JSON line labelled with
the run, for history. Sessions, placements and SDK events print the same lines through log_event().

Telemetry never changes what it observes. Every hook swallows its own errors, every store write runs in the background
and is dropped after WRITE_SECONDS, and no hook changes what an op returns or raises.
"""

import asyncio
import functools
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from numbers import Real

from server.store import get_state_store

TTL_SECONDS = float(os.getenv("OPEN_RL_OPS_TTL_DAYS", "7")) * 86400 or None  # 0 keeps records forever
SAMPLE_LIMIT = int(os.getenv("OPEN_RL_OPS_SAMPLE_LIMIT", "2000"))  # command records kept per list
STEP_TIMES = 64
SDK_ERRORS = 200
MAX_TURNS = 200
HEARTBEAT_SECONDS = 5
STALE_SECONDS = 30  # a worker record not rewritten for this long belongs to a gone process
WRITE_SECONDS = 5
FINISH_SECONDS = 5  # how long an exiting process waits for its last writes
OPEN_PLACEMENTS = "open_rl:placements:open"  # models holding a placement not yet released
FAILED = "RequestFailedResponse"

SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)
CONFIG_ENV = ("OPEN_RL_", "OPENRL_", "VLLM_", "NCCL_", "CUDA_", "PYTORCH_", "TORCH_")


def key(*parts: str) -> str:
  return ":".join(("open_rl", *parts))


def epoch(stamp: str | None) -> float | None:
  return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() if stamp else None


def rank_zero() -> bool:
  # Every torchrun rank runs the same ops; rank 0 records them.
  return os.getenv("RANK", "0") == "0"


def identity() -> dict[str, str]:
  """The pod this process runs in, as the scheduler passes it."""
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


def run_of(request: dict) -> str | None:
  return request.get("logical_run_id") or request.get("model_id")


def operation_name(request: dict, role: str) -> str:
  # A forward-only call travels as forward_backward, and its loss must not mix with training losses.
  return "forward" if request.get("forward_only") else str(request.get("op") or ("sample" if role == "sampler" else "unknown"))


def redact(value):
  # The SDK attaches full request and response headers to fatal errors, and the request carries the API key.
  if isinstance(value, dict):
    return {k: "[redacted]" if SECRET.search(str(k)) else redact(v) for k, v in value.items() if k not in ("request_headers", "response_headers")}
  return [redact(v) for v in value] if isinstance(value, list) else value


writes: set[asyncio.Task] = set()  # held so a write in flight is not garbage collected


def best_effort(hook):
  @functools.wraps(hook)
  def quiet(*args, **kwargs):
    with suppress(Exception):
      hook(*args, **kwargs)

  return quiet


def write(coroutine: Coroutine, timeout: float | None = None) -> None:
  """Run a store write in the background. One that fails or takes longer than timeout (WRITE_SECONDS) is dropped."""

  async def bounded():
    with suppress(Exception):
      await asyncio.wait_for(coroutine, timeout or WRITE_SECONDS)

  try:
    task = asyncio.get_running_loop().create_task(bounded())
  except RuntimeError:  # no event loop to run it on
    coroutine.close()
    return
  writes.add(task)
  task.add_done_callback(writes.discard)


# Worker records


def gpu_uuids() -> list[str] | None:
  """The GPUs this process can see, in CUDA order, as the UUIDs DCGM labels its series with. None when NVML cannot be
  read. pynvml comes with vLLM, so a trainer may lack it, which says nothing about how many GPUs it has."""
  try:
    import pynvml

    pynvml.nvmlInit()
    uuids = [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())]
  except Exception:
    return None
  visible = os.getenv("CUDA_VISIBLE_DEVICES")
  return [gpu for i, gpu in enumerate(uuids) if visible is None or str(i) in visible.split(",") or gpu in visible]


def alive(record: dict, now: float) -> bool:
  return record.get("phase") != "exited" and now - record.get("heartbeat_at", 0) < STALE_SECONDS


@dataclass
class Outstanding:
  """A request this worker popped and has not answered."""

  op: str
  run_id: str | None
  since: float
  observed: bool = False  # observe_operation records the op, so answered need not


class Worker:
  """This process's worker record. A thread rewrites it, so it stays fresh while loading a model holds the event loop."""

  def __init__(self, role: str, runtime_id: str | None) -> None:
    now = time.time()
    self.record = {
      # One record per process, not per pod, so a container restarted in place does not overwrite its predecessor's.
      "id": uuid.uuid4().hex[:16],
      "node": socket.gethostname(),
      **identity(),
      "role": role,
      "runtime_id": runtime_id,
      "gpus": gpu_uuids(),
      "started_at": now,
      "phase": "starting",
      "phase_since": now,
      "turns": [],
      "waits": [],
      # The settings the scheduler passed as env, which is what this process runs with.
      "config": {"env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(CONFIG_ENV) and not SECRET.search(k)}},
    }
    self.outstanding: dict[str, Outstanding] = {}
    self.stopped = threading.Event()
    self.heartbeat = threading.Thread(target=self.beat, name="worker-heartbeat", daemon=True)

  def snapshot(self) -> str:
    """The record as JSON, with outstanding requests counted by op and run, each group showing its oldest."""
    groups: dict[tuple, dict] = {}
    for entry in list(self.outstanding.values()):  # a copy, since the event loop changes it while the heartbeat reads
      group = groups.setdefault((entry.op, entry.run_id), {"op": entry.op, "run_id": entry.run_id, "since": entry.since, "count": 0})
      group["count"] += 1
    return json.dumps({**self.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())})

  @best_effort
  def save(self) -> None:
    write(get_state_store().set_value(key("worker", self.record["id"]), self.snapshot(), ttl_seconds=TTL_SECONDS))

  def beat(self) -> None:
    while not self.stopped.wait(HEARTBEAT_SECONDS):
      with suppress(Exception):
        get_state_store().set_value_sync(key("worker", self.record["id"]), self.snapshot(), ttl_seconds=TTL_SECONDS)


current: Worker | None = None  # this process's record while it serves

# The sampler pod's port named http, which the sampler PodMonitoring scrapes.
SAMPLER_METRICS_PORT = 9095


@best_effort
def serve_metrics() -> None:
  """Serve this process's Prometheus registry, where an in-process vLLM engine records its own metrics. Always in the
  0.0.4 text format, which keeps vLLM's vllm: names; offered OpenMetrics, the scraper asks for them escaped to vllm_."""
  from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

  from prometheus_client.exposition import CONTENT_TYPE_PLAIN_0_0_4, generate_latest  # in the sampler image through vLLM

  class Metrics(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
      body = generate_latest()
      self.send_response(200)
      self.send_header("Content-Type", CONTENT_TYPE_PLAIN_0_0_4)
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def log_message(self, *args) -> None:  # one line per scrape would flood the sampler's log
      pass

  server = ThreadingHTTPServer(("", SAMPLER_METRICS_PORT), Metrics)
  threading.Thread(target=server.serve_forever, name="sampler-metrics", daemon=True).start()


@asynccontextmanager
async def serving(role: str, runtime_id: str | None):
  """Keep this process's worker record while the body runs, and mark it exited after."""
  global current
  if rank_zero():
    with suppress(Exception):
      current = Worker(role, runtime_id)
      current.heartbeat.start()
      current.save()
  try:
    yield current
  finally:
    await finish()


async def finish() -> None:
  """Mark this process exited and land the writes in flight, within FINISH_SECONDS. Also called before an os._exit."""
  global current
  worker, current = current, None
  with suppress(Exception):
    async with asyncio.timeout(FINISH_SECONDS):
      if worker is not None:
        worker.stopped.set()  # stop the heartbeat first, so no late beat overwrites the exit
        await asyncio.to_thread(worker.heartbeat.join, WRITE_SECONDS)
        worker.record.update(phase="exited", phase_since=time.time())
        worker.save()
      await asyncio.gather(*writes, return_exceptions=True)


@best_effort
def set_phase(phase: str) -> None:
  if current is not None and current.record["phase"] != phase:
    current.record.update(phase=phase, phase_since=time.time())


@asynccontextmanager
async def gpu_turn(slot):
  """Take the time slicer's slot for the body and show it on the worker record: the wait from asking to the grant, then
  the turn, which is the hold."""
  asked = time.time()
  async with slot:
    started = time.time()
    if current is not None:
      current.record["waits"] = [*current.record["waits"][1 - MAX_TURNS :], [asked, started]]
    set_phase("on_gpu")
    try:
      yield
    finally:
      if current is not None:
        current.record["turns"] = [*current.record["turns"][1 - MAX_TURNS :], [started, time.time()]]
      set_phase("ready")


@best_effort
def popped(requests: list[dict]) -> None:
  """Requests this worker took from its queue, each outstanding until answered. A shared trainer's delete_model is queued
  without being counted as enqueued, so it is not counted here either."""
  if current is None:
    return
  for request in requests:
    if request.get("request_id") and request.get("op") != "delete_model":
      run_id, op = run_of(request), operation_name(request, current.record["role"])
      current.outstanding[request["request_id"]] = Outstanding(op, run_id, time.time())
      record(run_id, "popped", request_id=request["request_id"], op=op)


@best_effort
def answered(request_id: str, result: dict | None = None) -> None:
  """The request's answer is written. A failure answered without its op running, such as a failed wake or parse, gets its
  command record here, so every failed answer has one."""
  if current is None or (entry := current.outstanding.pop(request_id, None)) is None:
    return
  record(entry.run_id, "answered", request_id=request_id, op=entry.op)
  if not entry.observed and (result or {}).get("type") == FAILED:
    request = {"op": entry.op, "request_id": request_id, "logical_run_id": entry.run_id}
    op_finished(request, current.record["role"], current.record["runtime_id"], entry.since, FAILED)


# Command records


@dataclass
class RunSteps:
  """A run's optimizer steps as this process counts them. start is the run's count when this process first ran one of
  its ops, read in the background, and done_here the steps finished here since."""

  start: asyncio.Task
  done_here: int = 0


run_steps: dict[str, RunSteps] = {}


async def read_step_count(run_id: str) -> int | None:
  with suppress(Exception):
    return int(await asyncio.wait_for(get_state_store().get_value(key("steps", run_id)), WRITE_SECONDS) or 0)
  return None


async def observe_operation(request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one op and record how it went. Returns or raises exactly as the op does."""
  if not rank_zero():
    return await call()
  started_at = time.time()
  with suppress(Exception):
    if current is not None and (entry := current.outstanding.get(request.get("request_id"))):
      entry.observed = True
    # Read the run's step count before this op can change it.
    if (run_id := run_of(request)) and run_id not in run_steps and operation_name(request, role) != "sample":
      run_steps[run_id] = RunSteps(asyncio.ensure_future(read_step_count(run_id)))
  try:
    result = await call()
  except Exception as error:
    op_finished(request, role, runtime_id, started_at, type(error).__name__)
    raise
  op_finished(request, role, runtime_id, started_at, result=result)
  return result


def numbers(metrics: dict | None) -> dict[str, float]:
  """The metric values that are numbers, as floats. A bool is not one."""
  return {k: float(v) for k, v in (metrics or {}).items() if isinstance(v, Real) and not isinstance(v, bool)}


def finite(metrics: dict | None) -> dict[str, float]:
  return {k: v for k, v in numbers(metrics).items() if math.isfinite(v)}


def command_record(request: dict, role: str, runtime_id: str | None, started_at: float, error_type: str | None, result) -> dict:
  """How one op went. It failed with error_type, or as its result says. No payload, output or error message is kept."""
  result = result if isinstance(result, dict) else {}
  if error_type is None and result.get("type") == FAILED:
    error_type = FAILED
  now, values = time.time(), numbers(result.get("metrics"))
  record = {
    "at": now,
    "started_at": started_at,
    "operation": operation_name(request, role),
    "request_id": request.get("request_id"),
    "run_id": run_of(request),
    "runtime_id": runtime_id,
    "role": role,
    "status": "failed" if error_type else "succeeded",
    "error_type": error_type,
    "elapsed_seconds": now - started_at,
    **identity(),
    "metrics": finite(values),
  }
  # JSON has no NaN or infinity, so name the metrics that were, rather than leave them missing.
  if nonfinite := sorted(k for k, v in values.items() if not math.isfinite(v)):
    record["nonfinite"] = nonfinite
  return record


@best_effort
def op_finished(request: dict, role: str, runtime_id: str | None, started_at: float, error_type: str | None = None, result=None) -> None:
  """Record how one op went. A sample a vLLM sampler answered is only counted, with the time of the last, since vLLM's own
  metrics time what it answers."""
  op = command_record(request, role, runtime_id, started_at, error_type, result)
  if op["operation"] == "sample" and role == "sampler" and op["status"] == "succeeded":
    count_sample(op["run_id"], op["at"])
  else:
    record(op["run_id"], "op", **op)


@best_effort
def sampled_group(requests: list[dict], started_at: float) -> None:
  """One record per run for the samples a vLLM sampler served together, so its lane shows when it sampled without a
  record per sample. op_finished counted each sample, and a failed one keeps its own record too."""
  store, groups = get_state_store(), defaultdict(list)
  for request in requests:
    groups[run_of(request)].append(request)
  for run_id, group in groups.items():
    if run_id:
      op = command_record(group[0], "sampler", group[0].get("model_id"), started_at, None, None) | {"request_id": None, "count": len(group)}
      write(store.append_sample(key("operations", run_id, "samples"), op, SAMPLE_LIMIT, TTL_SECONDS))
      log_event("op", op, run_id=run_id)


def count_sample(run_id: str | None, at: float) -> None:
  if run_id:
    store = get_state_store()
    write(store.increment(key("sampled", run_id), TTL_SECONDS))
    write(store.set_value(key("sampled_at", run_id), str(at), ttl_seconds=TTL_SECONDS))


def keep_op(run_id: str, op: dict) -> None:
  """Keep one op's command record, and for a trainer op the run's failure and step counts."""
  store = get_state_store()
  if op["operation"] == "sample":
    count_sample(run_id, op["at"])
    # Samples keep their own list, so a run's sampling never evicts its training ops.
    write(store.append_sample(key("operations", run_id, "samples"), op, SAMPLE_LIMIT, TTL_SECONDS))
    return
  steps = run_steps.get(run_id)
  # done_here is taken before this op's own step is counted below. The write may wait WRITE_SECONDS on the step read.
  write(append_trainer_op(op, steps, steps.done_here if steps else 0), 2 * WRITE_SECONDS)
  # The few records a run's one-line summary reads instead of its command history.
  if op["status"] == "failed":
    count_request(run_id, "failed")
  elif op["operation"] == "optim_step":
    if steps is not None:
      steps.done_here += 1
    write(store.increment(key("steps", run_id), TTL_SECONDS))
    write(store.append_sample(key("step_times", run_id), op["at"], STEP_TIMES, TTL_SECONDS))


async def append_trainer_op(record: dict, steps: RunSteps | None, done_here: int) -> None:
  """Keep a trainer op, stamped with the step it belongs to once the run's count is read. An unread count leaves it off."""
  if steps is not None and (start := await steps.start) is not None:
    record["step"] = start + done_here
  await get_state_store().append_sample(key("operations", record["run_id"]), record, SAMPLE_LIMIT, TTL_SECONDS)


def count_request(run_id: str, event: str) -> None:
  """Count one more of the run's requests reaching event, which is enqueued, popped, answered or failed. A request
  enqueued but never popped shows up as a gap between the counts."""
  write(get_state_store().increment_field(key("ledger", run_id), event, TTL_SECONDS))


# Run events


@best_effort
def record(run_id: str | None, kind: str, /, **fields) -> None:
  """Keep one event about a run and log it. kind is enqueued, popped or answered, which count the run's requests, or op,
  whose fields are a command record from command_record. The store write runs in the background."""
  if not run_id:
    return
  with suppress(Exception):
    if kind == "op":
      keep_op(run_id, fields)
    else:
      count_request(run_id, kind)
  log_event(kind, fields, run_id=run_id)


@best_effort
def log_event(kind: str, fields: dict, *, run_id: str | None = None, session_id: str | None = None, severity: str | None = None) -> None:
  """Print one event as a JSON line, labelled with its run or session, for the year of history Cloud Logging keeps. The
  inspection API never reads these lines. A failed op logs as an error."""
  labels = {"openrl_source": "telemetry"} | ({"openrl_run_id": run_id} if run_id else {}) | ({"openrl_session_id": session_id} if session_id else {})
  line = {"severity": severity or ("ERROR" if fields.get("status") == "failed" else "INFO"), "message": f"telemetry {kind}"}
  print(json.dumps(line | redact(fields) | {"logging.googleapis.com/labels": labels}, default=str), flush=True)


# Placement records

placements_lock = asyncio.Lock()  # sweeps run in the background and may overlap


def placement(workload: dict, terminations: dict[str, list[dict]]) -> dict:
  """A Workload's placement, with the container terminations its pod has shown since, so they outlive the pod."""
  spec, status = workload["spec"], workload["status"]
  placed_at = epoch(status["placedAt"])
  return {
    "workload": workload["metadata"]["name"],
    "role": spec["role"],
    "model_id": spec["modelID"],
    "owner": spec.get("ownerID"),
    "pod": status.get("podName"),
    "node": status.get("nodeName"),
    "gpu_uuids": status.get("gpuUUIDs") or [],
    "placed_at": placed_at,
    "released_at": epoch(status.get("releasedAt")),
    "terminations": [t for t in terminations.get(status.get("podName"), []) if t["at"] >= placed_at],
  }


def merge_placements(records: list[dict], live: dict[str, dict], now: float) -> list[dict]:
  """A model's placement records after a sweep. A record whose Workload still holds that placement takes its current
  state, an open one whose Workload is gone or placed again is released now, and a new placement is added."""
  fresh, merged = dict(live), []
  for old in records:
    if (new := fresh.get(old["workload"])) and new["placed_at"] == old["placed_at"]:
      # A restarted container's earlier termination drops out of its pod status; the record keeps it.
      kept = old.get("terminations", [])
      merged.append(fresh.pop(old["workload"]) | {"terminations": kept + [t for t in new["terminations"] if t not in kept]})
    elif old["released_at"] is None:
      merged.append({**old, "released_at": now})
    else:
      merged.append(old)
  return merged + list(fresh.values())


async def record_placements(store, workloads: list[dict], terminations: dict[str, list[dict]]) -> None:
  """Keep one record per Workload placement, which outlives the Workload, and the set of models still holding one."""
  now = time.time()
  live: defaultdict[str, dict[str, dict]] = defaultdict(dict)  # model -> Workload name -> placement
  for workload in workloads:
    if (workload.get("status") or {}).get("placedAt"):  # the scheduler stamps it on bind
      p = placement(workload, terminations)
      live[p["model_id"]][p["workload"]] = p
  async with placements_lock:
    opened = await store.set_members(OPEN_PLACEMENTS)
    for model_id in opened | live.keys():
      records_key = key("placements", model_id)
      records = await store.read_samples(records_key)
      merged = merge_placements(records, live[model_id], now)
      held = any(r["released_at"] is None for r in merged)
      if merged != records:
        await store.replace_samples(records_key, merged, TTL_SECONDS)
        for changed in (p for p in merged if p not in records):  # placed, moved or released
          log_event("placement", changed, run_id=model_id)
      elif held and TTL_SECONDS:  # a placement held longer than the TTL must outlive it
        await store.expire(records_key, TTL_SECONDS)
      if held and model_id not in opened:
        await store.add_to_set(OPEN_PLACEMENTS, model_id)
      elif not held and model_id in opened:
        await store.remove_from_set(OPEN_PLACEMENTS, model_id)


# SDK events


@best_effort
def ingest_sdk_events(store, body: dict) -> None:
  """Log the tinker SDK's own telemetry, one redacted JSON line per event, and keep a session's error events."""
  context = {k: body.get(k) for k in ("session_id", "process_uuid", "sdk_version", "platform")}
  session_id, events = body.get("session_id"), body.get("events")
  if not isinstance(events, list):
    return
  session_id = session_id if isinstance(session_id, str) and session_id else None
  for event in [redact(e) for e in events if isinstance(e, dict)]:
    severity = str(event.get("severity") or "INFO").upper()
    log_event("sdk_event", {**context, "event": event}, session_id=session_id, severity=severity)
    if severity in ("ERROR", "CRITICAL") and session_id:
      write(store.append_sample(key("sdk_errors", session_id), sdk_error(event, severity), SDK_ERRORS, TTL_SECONDS))


def sdk_error(event: dict, severity: str) -> dict:
  data = event.get("event_data") if isinstance(event.get("event_data"), dict) else {}
  name = str(event.get("error_type") or event.get("event_name") or event.get("event") or "event").rsplit(".", 1)[-1]
  status = f"HTTP {data['status_code']}: " if data.get("status_code") else ""
  message = status + str(event.get("error_message") or data.get("detail") or data.get("exception") or data.get("message") or "")
  try:
    at = datetime.fromisoformat(str(event.get("timestamp"))).timestamp()
  except ValueError:
    at = time.time()
  return {"at": at, "severity": severity, "reason": f"SDK {name}", "message": message[:2000]}
