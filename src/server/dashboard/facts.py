"""Facts about a run, joined from its records, with no verdicts. A block appears only when it has content, so an absent
block means nothing of its kind was recorded. A log pointer is a read against /api/v1/inspect to run as given."""

import asyncio
import statistics
from datetime import UTC, datetime

from server.dashboard import fleet
from server.store import get_state_store
from server.telemetry import alive, key

BUCKETS = {"train": {"forward_backward", "optim_step"}, "checkpoint": {"save_state", "save_weights_for_sampler"}, "forward": {"forward"}}


async def context(state: dict, run: dict, low: float = 0, high: float = float("inf")) -> dict:
  """What the facts read about one run: its ops in [low, high], its workers, request and sample counters and queue."""
  store, run_id, finished = get_state_store(), run["run_id"], run["lifecycle"] == "ended"
  ops, counts, queued, samples_queued, sampled, sampled_at = await asyncio.gather(
    fleet.operations(run_id),
    store.read_counts(key("ledger", run_id)),
    store.list_length(request_queue(run_id)),
    queued_samples(store, run, finished),
    store.get_value(key("sampled", run_id)),
    store.get_value(key("sampled_at", run_id)),
  )
  ops = [s for s in ops if low <= s["at"] <= high]
  trainer = [s for s in ops if s["operation"] != "sample"]  # a single-process trainer answers samples too
  ctx = {
    "run": run,
    "ops": ops,
    "trainer": trainer,
    "steps": [s for s in trainer if s["operation"] == "optim_step" and s.get("status") == "succeeded"],
  }
  ctx.update(ledger=counts, queued=queued, samples_queued=samples_queued, now=state["observed_at"], finished=finished, window=(low, high))
  ctx.update(sampled=int(sampled) if sampled else None, sampled_at=float(sampled_at) if sampled_at else None)
  ctx.update(lanes=fleet.lanes(state, run_id), workers=workers_of(state, run))
  return ctx


def request_queue(run_id: str) -> str:
  return f"open_rl:queue:{run_id}"  # RedisStore's names for the queues it pushes to


def sampler_queue(runtime_id: str) -> str:
  return f"open_rl:sampler_queue:{runtime_id}"


def workers_of(state: dict, run: dict) -> list[dict]:
  return [w for w in state["workers"] if w.get("runtime_id") in (run["runtime_id"], run["run_id"])]


async def queued_samples(store, run: dict, finished: bool) -> int | None:
  # Samples wait in the runtime's queue, which co-tenant runs share; a finished run's share of it is gone.
  return None if finished else await store.list_length(sampler_queue(run["runtime_id"]))


def read(route: str, **args) -> dict:
  return {"read": route, "args": {k: v for k, v in args.items() if v is not None}}


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")


def present(**blocks) -> dict:
  return {k: v for k, v in blocks.items() if v not in (None, [], {})}


def unanswered(ctx: dict) -> list[dict]:
  """Trainer requests the client waits on, with the pod holding each and whether its worker still reports."""
  found = []
  for w in ctx["workers"]:
    for g in (g for g in w.get("outstanding") or [] if g.get("run_id") == ctx["run"]["run_id"] and g["op"] != "sample"):
      found.append({"op": g["op"], "count": g["count"], "age_s": round(ctx["now"] - g["since"]), "pod": w.get("pod") or w["id"]})
      found[-1]["holder_alive"] = alive(w, ctx["now"])
  return found


def failures(ctx: dict) -> dict | None:
  if not (failed := [s for s in ctx["trainer"] if s.get("status") == "failed"]):
    return None
  ops = [{"operation": s["operation"], "error_type": s.get("error_type"), "at": iso(s["at"]), "pod": s.get("pod")} for s in reversed(failed[-5:])]
  last, run_id = failed[-1], ctx["run"]["run_id"]
  logs = read("logs", run_id=run_id, pod=last.get("pod"), q=last.get("error_type"), since=iso(last["started_at"] - 60), until=iso(last["at"] + 60))
  return {"count": len(failed), "ops": ops, "logs": logs}


def relaunches(ctx: dict) -> list[dict]:
  """Worker processes that replaced an earlier one in the same pod during this run: a container restarted in place, or a
  new pod after one died holding requests. A clean exit, or a shared runtime's idle pod recreated, is not a crash."""
  by_pod: dict[str, list[dict]] = {}
  low, high = (ctx["run"]["created_at"] or 0) - 60, ctx["run"].get("completed_at") or float("inf")
  for w in sorted(ctx["workers"], key=lambda w: w["started_at"]):
    if w.get("pod") and low <= w["started_at"] <= high:
      by_pod.setdefault(w["pod"], []).append(w)
  found = [
    {"pod": b["pod"], "stopped_at": a.get("heartbeat_at", b["started_at"]), "started_at": b["started_at"]}
    for p in by_pod.values()
    for a, b in zip(p, p[1:])
    if a.get("phase") != "exited" and (a.get("pod_uid") == b.get("pod_uid") or any(g["op"] != "sample" for g in a.get("outstanding") or []))
  ]
  return sorted(found, key=lambda p: p["started_at"])


def saves(ctx: dict) -> dict | None:
  if not (found := [s for s in ctx["trainer"] if s["operation"] == "save_state"]) and not ctx["steps"]:
    return None
  last = max((s["at"] for s in found if s.get("status") == "succeeded"), default=None)
  return present(
    count=len(found),
    failed=sum(s.get("status") == "failed" for s in found),
    last_at=iso(last) if last else None,
    steps_since_last=sum(s["at"] > (last or 0) for s in ctx["steps"]),
    slowest_s=round(max(s.get("elapsed_seconds", 0) for s in found)) if found else None,
  )


def sends(ctx: dict) -> int:
  """The samples the run sent. A run recorded before the sample count kept a record per send."""
  return ctx["sampled"] if ctx["sampled"] is not None else sum(s["operation"] == "sample" for s in ctx["ops"])


def answered(ctx: dict) -> list[dict]:
  """The answered samples that kept a record: a torch trainer's, or a group a vLLM sampler served, timed as a whole."""
  return [s for s in ctx["ops"] if s["operation"] == "sample" and s.get("status") == "succeeded"]


def samplers(ctx: dict) -> dict | None:
  """Sends and failures from the run's own count and records, and the median time of the groups its samplers served."""
  if not (sent := sends(ctx)):
    return None
  records = [s for s in ctx["ops"] if s["operation"] == "sample"]
  found = {
    "sends": sent,
    "cancelled": sum(s.get("status") == "cancelled" for s in records),
    "failed": sum(s.get("status") == "failed" for s in records),
  }
  if recorded := answered(ctx):
    found["median_s"] = round(statistics.median(s.get("elapsed_seconds", 0) for s in recorded), 2)
  return found


def config(ctx: dict) -> list[dict] | str | None:
  """Workers whose echoed settings differ from what the run asked for."""
  if not (echoed := [w for w in ctx["workers"] if w.get("config")]):
    return "no echo"
  settings, found = ctx["run"]["settings"], []
  for w in echoed:
    asked, running = settings.get("openrl.fine_tuning_type"), w["config"].get("env", {}).get("OPEN_RL_FINE_TUNING_TYPE")
    if asked and running and asked != running:
      found.append({"pod": w.get("pod"), "setting": "openrl.fine_tuning_type", "asked": asked, "running": running})
    # gpus is None when the worker could not count them.
    gpus, wanted = w.get("gpus"), settings.get("openrl.trainer_gpus")
    if w.get("role") == "trainer" and wanted and gpus is not None and len(gpus) != int(wanted):
      found.append({"pod": w.get("pod"), "setting": "openrl.trainer_gpus", "asked": wanted, "running": len(gpus)})
  return found or None


def trainer_span(ctx: dict) -> tuple[float, float] | None:
  """From the run's first trainer op to its last once the run has ended, else to now."""
  if not (ops := ctx["trainer"]):
    return None
  start = ops[0]["started_at"]
  return start, start + max((ops[-1]["at"] if ctx["finished"] else ctx["now"]) - start, 1.0)


def goodput(ctx: dict) -> dict | None:
  """Share of the run's span the trainer spent training, and where the rest went (Google ML goodput)."""
  if not (span := trainer_span(ctx)):
    return None
  ops, (start, end) = ctx["trainer"], span
  wall = end - start
  spent = {kind: sum(s.get("elapsed_seconds", 0) for s in ops if s["operation"] in names) for kind, names in BUCKETS.items()}
  idle = max(0.0, wall - sum(spent.values()))
  split = {"other": round(idle)}
  # Trainer idle time with a sample in flight is rollout; with none the server waits on the client.
  if running := [(s["started_at"], s["at"]) for s in answered(ctx)]:
    busy = [(s["started_at"], s["at"]) for s in ops]
    rollout = min(idle, max(0.0, covered(busy + running, start, end) - covered(busy, start, end)))
    split = {"rollout": round(rollout), "client": round(idle - rollout)}
  first = round(ctx["steps"][0]["at"] - start) if ctx["steps"] else None
  return {
    "goodput": round(spent["train"] / wall, 3),
    "seconds": {"wall": round(wall), **{k: round(v) for k, v in spent.items()}, **split},
    "first_step_after_s": first,
  }


def covered(spans: list, low: float, high: float) -> float:
  """Seconds of [low, high] inside at least one of the spans."""
  return sum(b - a for a, b in fleet.merge([(max(a, low), min(b, high)) for a, b in spans if b > low and a < high]))


def last_op_at(ctx: dict) -> float | None:
  """When the run last did anything in the context's window: its last recorded op, or its last sample."""
  low, high = ctx["window"]
  sampled = [at for at in [ctx["sampled_at"]] if at is not None and low <= at <= high]
  return max([s["at"] for s in ctx["ops"][-1:]] + sampled, default=None)


def summary(ctx: dict) -> dict:
  steps, ops, run, ledger = ctx["steps"], ctx["trainer"], ctx["run"], dict(ctx["ledger"])
  gaps = [b["at"] - a["at"] for a, b in zip(steps, steps[1:])]
  if "enqueued" in ledger and ctx["queued"] is not None:
    ledger["lost"] = ledger["enqueued"] - ledger.get("popped", 0) - ctx["queued"]
  poisoned = [s for s in ops if s.get("nonfinite")]
  first = present(operation=poisoned[0]["operation"], step=poisoned[0].get("step"), metrics=poisoned[0]["nonfinite"]) if poisoned else None
  norms = [float(f"{s['metrics']['grad_norm:mean']:.4g}") for s in steps if "grad_norm:mean" in s.get("metrics", {})]
  restarts = None
  if relaunched := relaunches(ctx):
    p = relaunched[0]
    logs = read("logs", run_id=run["run_id"], pod=p["pod"], since=iso(p["stopped_at"] - 120), until=iso(p["started_at"] + 60))
    restarts = {
      "workers": [{"pod": p["pod"], "stopped_reporting_at": iso(p["stopped_at"]), "restarted_at": iso(p["started_at"])} for p in relaunched],
      "logs": logs,
      # Who created or deleted the run's Workloads and pods around the relaunches, from the cluster's audit log.
      "cluster_changes": ctx.get("changes"),
    }
  unplaced = [
    {"name": w["name"], "role": w["role"], "age_s": round(w["age_seconds"] or 0), "message": w["message"]} for w in run["workloads"] if not w["node"]
  ]
  # A live pod's restarts, and the terminations placement records kept after their pods were gone.
  kept = [
    {"at": t["at"], "pod": lane["pod"], "reason": t.get("reason") or "Terminated", "message": exit_text(t)}
    for lane in ctx["lanes"]
    for t in lane.get("terminations") or []
  ]
  found = {(i["pod"], round(i["at"])): i for i in [*kept, *(i for pod in run["pods"] for i in pod["incidents"])]}
  incidents = sorted(found.values(), key=lambda i: -i["at"])[:5]
  return present(
    lifecycle=run["lifecycle"],
    steps=len(steps),
    step_seconds={"median": round(statistics.median(gaps), 1), "last": round(gaps[-1], 1)} if gaps else None,
    since_last_step_s=round(ctx["now"] - steps[-1]["at"]) if steps and not ctx["finished"] else None,
    since_last_op_s=max(0, round(ctx["now"] - last)) if (last := last_op_at(ctx)) is not None and not ctx["finished"] else None,
    trainer_idle_s=max(0, round(ctx["now"] - ops[-1]["at"])) if ops and not ctx["finished"] else None,
    last_op={"operation": ops[-1]["operation"], "status": ops[-1].get("status"), "at": iso(ops[-1]["at"])} if ops else None,
    last_metrics={k: float(f"{v:.6g}") for s in ops[-50:] for k, v in s.get("metrics", {}).items()},
    goodput=goodput(ctx),
    requests=present(unanswered=unanswered(ctx), ledger=ledger, samples_queued=ctx["samples_queued"] or None),
    failures=failures(ctx),
    relaunches=restarts,
    saves=saves(ctx),
    optimizer=present(nonfinite={"count": len(poisoned), "first": first} if poisoned else None, grad_norm_last=norms[-5:]),
    samplers=samplers(ctx),
    config=config(ctx),
    placement=present(recorded=len(ctx["lanes"]), unplaced=unplaced) or None,
    incidents=[present(at=iso(i["at"]), pod=i["pod"], reason=i["reason"], message=(i.get("message") or "")[:200]) for i in incidents],
  )


def exit_text(termination: dict) -> str:
  signal = f", signal {termination['signal']}" if termination.get("signal") else ""
  return f"{termination.get('container') or 'container'} exited with code {termination.get('exit_code')}{signal}"


def line(ctx: dict) -> dict:
  """A run in one line from its full context, for a session over the session's span."""
  failures = sum(s.get("status") == "failed" for s in ctx["trainer"])
  return one_line(ctx, len(ctx["steps"]), [s["at"] for s in ctx["steps"]], last_op_at(ctx), failures)


async def fleet_line(state: dict, run: dict) -> dict:
  """A run in one line for /runs from a few small records, never its command history. A run recorded before step times
  and the failure count existed leaves those null."""
  store, run_id, finished = get_state_store(), run["run_id"], run["lifecycle"] == "ended"
  counts, last_ops, last_samples, times, samples_queued, sampled_at = await asyncio.gather(
    store.read_counts(key("ledger", run_id)),
    store.read_samples(key("operations", run_id), last=1),
    store.read_samples(key("operations", run_id, "samples"), last=1),
    store.read_samples(key("step_times", run_id)),
    queued_samples(store, run, finished),
    store.get_value(key("sampled_at", run_id)),
  )
  ctx = {"run": run, "workers": workers_of(state, run), "now": state["observed_at"], "samples_queued": samples_queued}
  last = max([r["at"] for r in last_ops + last_samples] + ([float(sampled_at)] if sampled_at else []), default=None)
  return one_line(ctx, run["steps"], times, last, counts.get("failed", 0) if counts else None)


def one_line(ctx: dict, steps: int, step_times: list[float], last_op_at: float | None, failures: int | None) -> dict:
  """A run in one line, enough to see which run to open. ctx holds the run, its workers, now and its queued samples."""
  held, now = unanswered(ctx), ctx["now"]
  return {
    "lifecycle": ctx["run"]["lifecycle"],
    "steps": steps,
    # The snapshot time lags records written after it, so a fresh op can look a few seconds in the future.
    "since_last_op_s": max(0, round(now - last_op_at)) if last_op_at is not None else None,
    "since_last_step_s": round(now - step_times[-1]) if step_times else None,
    "median_step_s": round(statistics.median(b - a for a, b in zip(step_times, step_times[1:]))) if len(step_times) > 1 else None,
    "trainer_requests_unanswered": sum(o["count"] for o in held),
    "held_by_dead_worker": sum(o["count"] for o in held if not o["holder_alive"]),
    "samples_queued": ctx["samples_queued"],
    "failures": failures,
    "relaunches": len(relaunches(ctx)),
  }
