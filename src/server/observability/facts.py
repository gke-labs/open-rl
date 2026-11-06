"""Facts about a run or a session, joined from their records, with no verdicts. A block appears only when it has content,
so an absent block means nothing of its kind was recorded. A log pointer is a read against /api/v1/inspect to run as
given. Ages count from the request, and a run that ended shows none."""

import asyncio
import json
import math
import statistics
from datetime import UTC, datetime

from server.model_metadata import pool_name
from server.observability.steps import TRAINING
from server.store import get_state_store, get_store
from server.telemetry import FLUSH_SECONDS, key

STALE_SECONDS = 6 * FLUSH_SECONDS  # a worker that has not flushed in this long no longer reports
BUCKETS = {"train": TRAINING, "checkpoint": {"save_state", "save_weights_for_sampler"}, "forward": {"forward"}}
SETTING_ENV = {"openrl.fine_tuning_type": "OPEN_RL_FINE_TUNING_TYPE", "openrl.weight_sync": "OPEN_RL_WEIGHT_SYNC_STRATEGY"}

# Each run's ledger gaps as this API server's reads saw them, (at, gap) oldest first. A process's counts land up to a
# flush late, so a gap is a loss only once it has held across reads more than a flush apart.
gaps: dict[str, list[tuple[float, int]]] = {}


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")


def read(route: str, **args) -> dict:
  return {"read": route, "args": {k: v for k, v in args.items() if v is not None}}


def present(**blocks) -> dict:
  return {k: v for k, v in blocks.items() if v not in (None, [], {})}


def age(now: float, at: float | None) -> int | None:
  return None if at is None else max(0, round(now - at))


async def workers() -> list[dict]:
  return [json.loads(raw) for raw in (await get_state_store().scan_values(key("worker", ""))).values()]


def workers_of(found: list[dict], run: dict) -> list[dict]:
  ids = {run["run_id"], run["runtime"], pool_name(run["runtime"])}
  return [w for w in found if w.get("runtime_id") in ids]


def alive(worker: dict, now: float) -> bool:
  return worker.get("phase") != "exited" and now - worker.get("heartbeat_at", 0) < STALE_SECONDS


def unanswered(run: dict, workers: list[dict], now: float) -> list[dict]:
  """Trainer requests the client waits on, with the pod holding each and whether its worker still reports."""
  return [
    {"op": g["op"], "count": g["count"], "age_s": age(now, g["since"]), "pod": w.get("pod") or w["id"], "holder_alive": alive(w, now)}
    for w in workers
    for g in w.get("outstanding") or []
    if g["run_id"] == run["run_id"]
  ]


def relaunches(run: dict, workers: list[dict]) -> list[dict]:
  """Worker processes that replaced an earlier one in the same pod during the run: a container restarted in place, or a
  new pod after one died holding requests. A clean exit is not a relaunch."""
  by_pod: dict[str, list[dict]] = {}
  low, high = run["created_at"] - 60, run["completed_at"] or math.inf
  for w in sorted(workers, key=lambda w: w["started_at"]):
    if w.get("pod") and low <= w["started_at"] <= high:
      by_pod.setdefault(w["pod"], []).append(w)
  return [
    {"pod": b["pod"], "stopped_at": a.get("heartbeat_at", b["started_at"]), "started_at": b["started_at"]}
    for p in by_pod.values()
    for a, b in zip(p, p[1:])
    if a.get("phase") != "exited" and (a.get("pod_uid") == b.get("pod_uid") or a.get("outstanding"))
  ]


def pace(times: list[float], now: float, ended: bool) -> dict:
  gaps_s = [b - a for a, b in zip(times, times[1:])]
  return present(
    step_seconds={"median": round(statistics.median(gaps_s), 1), "last": round(gaps_s[-1], 1)} if gaps_s else None,
    since_last_step_s=None if ended or not times else age(now, times[-1]),
  )


async def sdk_errors(session_ids: list[str]) -> list[dict]:
  found = await asyncio.gather(*(get_state_store().read_samples(key("sdk_errors", s)) for s in session_ids))
  return sorted((e for errors in found for e in errors), key=lambda e: e["at"])


def errors_block(errors: list[dict]) -> dict | None:
  return {"count": len(errors), "last": errors[-1]} if errors else None


async def line(run: dict, workers: list[dict], now: float) -> dict:
  """A run in one line, enough to see which run to open, from small records rather than its op history."""
  store, run_id, ended = get_state_store(), run["run_id"], run["lifecycle"] == "ended"
  steps, times, ledger, last_ops, samples, errors = await asyncio.gather(
    store.get_value(key("steps", run_id)),
    store.read_samples(key("step_times", run_id)),
    store.read_counts(key("ledger", run_id)),
    store.read_samples(key("operations", run_id), last=1),
    store.read_samples(key("operations", run_id, "samples")),
    sdk_errors(run["session_ids"]),
  )
  held, last = unanswered(run, workers, now), max((r["at"] for r in last_ops + samples[-1:]), default=None)
  found = pace(times, now, ended)
  return {
    "lifecycle": run["lifecycle"],
    "steps": int(steps or 0),
    **present(since_last_op_s=None if ended else age(now, last), since_last_step_s=found.get("since_last_step_s")),
    "median_step_s": found.get("step_seconds", {}).get("median"),
    "trainer_requests_unanswered": sum(o["count"] for o in held),
    "held_by_dead_worker": sum(o["count"] for o in held if not o["holder_alive"]),
    "failures": ledger.get("failed", 0),
    "sample_failures": sum(s["status"] == "failed" for s in samples),
    "sdk_errors": len(errors),
    "relaunches": len(relaunches(run, workers)),
  }


def lost(run_id: str, gap: int, now: float) -> int | None:
  """Requests enqueued that no worker popped and the queue no longer holds, once the gap has lasted over a flush."""
  seen, cutoff = gaps.setdefault(run_id, []), now - FLUSH_SECONDS
  seen.append((now, gap))
  while len(seen) > 1 and seen[1][0] < cutoff:
    seen.pop(0)
  return max(0, min(g for _, g in seen)) if seen[0][0] < cutoff else None


def covered(spans: list[tuple[float, float]], low: float, high: float) -> float:
  """Seconds of [low, high] inside at least one of the spans."""
  total, reach = 0.0, low
  for a, b in sorted((max(a, low), min(b, high)) for a, b in spans):
    if (top := max(a, reach)) < b:
      total, reach = total + b - top, b
  return total


def goodput(ops: list[dict], samples: list[dict], workers: list[dict], now: float, ended: bool) -> dict | None:
  """Share of the trainer's span spent training (Google ML goodput), and where the rest went."""
  if not ops:
    return None
  start = min(s["started_at"] for s in ops)
  end = max(start + 1, max(s["at"] for s in ops) if ended else now)
  spent = {kind: sum(s["elapsed_seconds"] for s in ops if s["operation"] in names) for kind, names in BUCKETS.items()}
  # Each idle second counts once, in the first of these that covers it; client is the rest.
  layers = {
    "busy": [(s["started_at"], s["at"]) for s in ops],
    "gpu_wait": [tuple(w) for worker in workers if worker.get("role") == "trainer" for w in worker.get("waits") or []],
    "queued": [(s["started_at"] - s["queue_seconds"], s["started_at"]) for s in ops if "queue_seconds" in s],
    "rollout": [(s["started_at"], s["at"]) for s in samples if s["status"] == "succeeded"],
  }
  split, spans, before = {}, [], 0.0
  for name, more in layers.items():
    spans += more
    reached = covered(spans, start, end)
    split[name], before = reached - before, reached
  del split["busy"]
  steps = [s["at"] for s in ops if s["operation"] == "optim_step" and s["status"] == "succeeded"]
  return present(
    goodput=round(spent["train"] / (end - start), 3),
    seconds={k: round(v) for k, v in ({"wall": end - start} | spent | split | {"client": end - start - before}).items()},
    first_step_after_s=round(min(steps) - start) if steps else None,
  )


def failures(ops: list[dict], run_id: str) -> dict | None:
  if not (failed := [s for s in ops if s["status"] == "failed"]):
    return None
  last = failed[-1]
  return {
    "count": len(failed),
    "ops": [{"operation": s["operation"], "error_type": s["error_type"], "at": iso(s["at"]), "pod": s.get("pod")} for s in reversed(failed[-5:])],
    "logs": read("logs", run_id=run_id, pod=last.get("pod"), since=iso(last["started_at"] - 60), until=iso(last["at"] + 60)),
  }


def saves(ops: list[dict], steps: list[dict]) -> dict | None:
  if not (found := [s for s in ops if s["operation"] == "save_state"]):
    return None
  last = max((s["at"] for s in found if s["status"] == "succeeded"), default=None)
  return present(
    count=len(found),
    failed=sum(s["status"] == "failed" for s in found),
    last_at=iso(last) if last else None,
    steps_since_last=sum(s["at"] > (last or 0) for s in steps),
  )


def samplers(samples: list[dict]) -> dict | None:
  """A vLLM sampler keeps one record per group it served each flush, with the group's count, and one per failed sample."""
  if not samples:
    return None
  groups = [s for s in samples if s["status"] == "succeeded"]
  failed = len(samples) - len(groups)
  found = {"sends": sum(s.get("count", 1) for s in groups) + failed, "failed": failed}
  return found | ({"median_s": round(statistics.median(s["elapsed_seconds"] for s in groups), 2)} if groups else {})


def config(run: dict, workers: list[dict]) -> list[dict] | None:
  """Workers whose settings env differs from what the run asked for."""
  found = []
  for w in workers:
    env = (w.get("config") or {}).get("env") or {}
    for setting, name in SETTING_ENV.items():
      asked, running = run["settings"].get(setting), env.get(name)
      if asked is not None and running is not None and str(asked) != running:
        found.append({"pod": w.get("pod"), "setting": setting, "asked": asked, "running": running})
  return found or None


async def summary(run: dict, workers: list[dict], now: float) -> dict:
  store, run_id, ended = get_state_store(), run["run_id"], run["lifecycle"] == "ended"
  ops, samples, count, times, ledger, queued, errors = await asyncio.gather(
    store.read_samples(key("operations", run_id)),
    store.read_samples(key("operations", run_id, "samples")),
    store.get_value(key("steps", run_id)),
    store.read_samples(key("step_times", run_id)),
    store.read_counts(key("ledger", run_id)),
    get_store().queued(run_id),
    sdk_errors(run["session_ids"]),
  )
  steps = [s for s in ops if s["operation"] == "optim_step" and s["status"] == "succeeded"]
  if "enqueued" in ledger and (gone := lost(run_id, ledger["enqueued"] - ledger.get("popped", 0) - queued, now)) is not None:
    ledger["lost"] = gone
  poisoned = [s for s in ops if s.get("nonfinite")]
  first = present(operation=poisoned[0]["operation"], step=poisoned[0].get("step"), metrics=poisoned[0]["nonfinite"]) if poisoned else None
  norms = [float(f"{s['metrics']['grad_norm:mean']:.4g}") for s in steps if "grad_norm:mean" in s["metrics"]]
  restarts = None
  if relaunched := relaunches(run, workers):
    p = relaunched[0]
    restarts = {
      "workers": [{"pod": p["pod"], "stopped_reporting_at": iso(p["stopped_at"]), "restarted_at": iso(p["started_at"])} for p in relaunched],
      "logs": read("logs", run_id=run_id, pod=p["pod"], since=iso(p["stopped_at"] - 120), until=iso(p["started_at"] + 60)),
    }
  last_op = max(ops + samples[-1:], key=lambda s: s["at"], default=None)
  return present(
    lifecycle=run["lifecycle"],
    steps=int(count) if count else len(steps),
    **pace(times, now, ended),
    since_last_op_s=None if ended or last_op is None else age(now, last_op["at"]),
    last_op={"operation": last_op["operation"], "status": last_op["status"], "at": iso(last_op["at"])} if last_op else None,
    goodput=goodput(ops, samples, workers, now, ended),
    requests=present(unanswered=unanswered(run, workers, now), ledger=ledger),
    failures=failures(ops, run_id),
    relaunches=restarts,
    saves=saves(ops, steps),
    optimizer=present(nonfinite={"count": len(poisoned), "first": first} if poisoned else None, grad_norm_last=norms[-5:]),
    samplers=samplers(samples),
    config=config(run, workers),
    placement=present(nodes=run["nodes"], pods=[p["name"] for p in run["pods"]]),
    incidents=[{"pod": p["name"], "problem": p["problem"]} for p in run["pods"] if p["problem"]],
    sdk_errors=errors_block(errors),
  )


async def session(session_id: str, runs: list[dict], workers: list[dict], now: float) -> dict | None:
  """The session's record, or what its runs and live key show when it has none, with a line per run."""
  store = get_state_store()
  record, seen, live, errors = await asyncio.gather(
    store.get_value(key("session_record", session_id)),
    store.get_value(key("session_seen", session_id)),
    store.get_value(key("session", session_id)),
    store.read_samples(key("sdk_errors", session_id)),
  )
  mine = [r for r in runs if session_id in r["session_ids"]]
  if record is None and live is None and not mine:
    return None
  lines = await asyncio.gather(*(line(r, workers_of(workers, r), now) for r in mine))
  saved = json.loads(record or "{}")
  return present(
    session_id=session_id,
    created_at=saved.get("created_at"),
    metadata=saved.get("metadata"),
    last_seen_at=float(seen) if seen else None,
    live=live is not None,
    recorded=record is not None,
    models=[{"model_id": r["run_id"], **found} for r, found in zip(mine, lines)],
    sdk_errors=errors_block(errors),
  )
