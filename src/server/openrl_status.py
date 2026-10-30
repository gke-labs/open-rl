"""The model view: what the API server knows about one model, for clients
without kubectl (design 016, section 5.2).

Most clients run on a laptop behind a port-forward. For a model id the client
owns, this answers what `kubectl get workloads` would: which workers the
model has, where each one is in its life, and whether its session is live.
Nothing here changes state. The worker manager answers the cluster side; the
stores answer the rest.

The view is scoped to one model on purpose. A model id is a random UUID the
server minted for the client that created it, so knowing the id is the
capability, as for every model-scoped Tinker call. Fleet-wide views wait for
a tenancy design.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from server.model_metadata import TrainingModelMetadata, get_model_metadata
from server.session_registry import SessionRegistry
from server.store import StateStore
from server.worker_manager import READY, STARTING, WorkerManager, owner_id

# What a worker entry exposes. The launchers know more (pod names, pids,
# owners); none of that is the client's contract.
WORKER_FIELDS = ("role", "state", "detail", "workload", "node", "age_seconds")


@dataclass
class StatusContext:
  state: StateStore
  registry: SessionRegistry
  manager: WorkerManager | None
  worker_launcher: str  # as get_server_capabilities reports it


def describe_worker(worker: dict[str, Any]) -> str:
  """One line for a human or an example's progress output."""
  age = f", {int(worker['age_seconds'])}s" if worker.get("age_seconds") is not None else ""
  return f"{worker['role']} {worker['state']}: {worker['detail']}{age}"


async def workers_of_owner(ctx: StatusContext, owner: str) -> tuple[list[dict[str, Any]], str | None]:
  """The owner's workers, or an empty list and the error when the launcher
  cannot answer. The rest of the view comes from the store and is still worth
  returning when the cluster API is not."""
  if ctx.manager is None:
    return [], None
  try:
    found_workers = await asyncio.to_thread(ctx.manager.inspect_owner, owner)
  except Exception as exc:
    return [], f"{type(exc).__name__}: {exc}"
  out = []
  for found in found_workers:
    worker = {field: found.get(field) for field in WORKER_FIELDS}
    # The launcher sees a pod run minutes before the model is loaded, so a
    # trainer announces itself when it can serve (open_rl:trainer_ready:<runtime>).
    announced = found["role"] != "trainer" or await ctx.state.get_value(f"open_rl:trainer_ready:{found['model_id']}") is not None
    if worker["state"] == READY and not announced:
      worker["state"] = STARTING
      worker["detail"] = f"{worker['detail']}, loading the model"
    worker["summary"] = describe_worker(worker)
    out.append(worker)
  return out, None


async def any_live(ctx: StatusContext, session_ids: Any) -> bool:
  for session_id in session_ids:
    if await ctx.registry.live(session_id):
      return True
  return False


async def workers_in_use_by_others(ctx: StatusContext, owner: str) -> bool:
  """Whether any session still holds the owner's workers. Used only to word
  the summary; which session it is stays out of the response. Read-only:
  in_use() drops dead sessions, this only looks."""
  return await any_live(ctx, await ctx.state.set_members(f"open_rl:owner:{owner}"))


def settings_of(meta: TrainingModelMetadata) -> dict[str, Any]:
  """The model's settings as the client could have passed them."""
  return {
    "openrl.fine_tuning_type": meta.fine_tuning_type,
    "openrl.weight_sync": meta.weight_sync_config.strategy,
    "openrl.exclusive": meta.exclusive,
    "openrl.sampler_replicas": meta.sampler_replicas,
    "openrl.trainer_backend": meta.trainer_backend,
    "openrl.trainer_gpus": meta.trainer_gpus,
    "openrl.trainer_cp": meta.trainer_cp,
    "openrl.recipe": meta.recipe,
  }


def summarize(ctx: StatusContext, entry: dict[str, Any], workers: list[dict[str, Any]], others_hold_workers: bool) -> str:
  """What an agent should know first, in one line."""
  if entry["status"] != "active":
    return f"model {entry['status']}"
  if ctx.worker_launcher == "single_process":
    return "the trainer runs inside the API server; there are no separate workers to wait for"
  if ctx.worker_launcher == "none":
    return "workers are standing deployments this server does not launch or track"
  if entry["workers_error"]:
    return f"worker state unavailable ({entry['workers_error']}); settings and session are from the store"
  if not workers:
    if entry["session_live"]:
      return "no workers yet; the API server has not created them or they were released"
    return "no live session and no workers; the model's workers were released"
  if not entry["session_live"]:
    if entry["shares_workers"] and others_hold_workers:
      then = "the workers stay up while other jobs use them, but this model's session is gone"
    else:
      then = "the reaper releases unused workers within about two minutes"
    return f"{'; '.join(w['summary'] for w in workers)}. Its session expired; {then}"
  return "; ".join(w["summary"] for w in workers)


async def model_status(ctx: StatusContext, model_id: str) -> dict[str, Any] | None:
  meta = await get_model_metadata(ctx.state, model_id)
  if meta is None:
    return None
  owner = owner_id(meta.runtime(model_id))
  workers, workers_error = await workers_of_owner(ctx, owner)
  entry: dict[str, Any] = {
    "model_id": model_id,
    "base_model": meta.base_model,
    "fine_tuning_type": meta.fine_tuning_type,
    "status": meta.status,
    "created_at": meta.created_at,
    "shares_workers": meta.shares_runtime(),
    # The model's own sessions: on a shared LoRA runtime another client's
    # session keeps the workers up but says nothing about this one.
    "session_live": await any_live(ctx, meta.session_ids),
    "settings": settings_of(meta),
    "workers": workers,
    "workers_error": workers_error,
  }
  entry["summary"] = summarize(ctx, entry, workers, await workers_in_use_by_others(ctx, owner))
  return entry
