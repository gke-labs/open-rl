"""The one view of the cluster the pages and agents read: runs joined to their
workers, current placements on GPUs, nodes with their devices, and the
placement history that outlives them."""

import asyncio
import copy
import time

from server.dashboard import cluster, history

CACHE_SECONDS = 5
HISTORY_WINDOW_SECONDS = 24 * 3600


def iso(ts: float) -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1e6):06d}+00:00"


def observed_status(metadata: dict, workloads: list[dict], pods: list[dict], available: bool) -> str:
  """Recorded lifecycle first; otherwise what Kubernetes shows right now."""
  recorded = str(metadata.get("status", "")).lower()
  if recorded in {"completed", "failed", "ended"}:
    return recorded.capitalize()
  if not available:
    return "Unknown"
  if any(pod.get("problem") for pod in pods):
    return "Needs attention"
  if any(pod.get("phase") == "Running" for pod in pods):
    return "Running"
  if pods or any(w.get("node_name") for w in workloads):
    return "Starting"
  if workloads:
    return "Queued"
  return "Unassigned"


def join_runs(metadata: list[dict], state: dict) -> list[dict]:
  """LoRA workers serve a base-model runtime shared by every LoRA run on it;
  an FFT worker's modelID is the run itself."""
  workloads = state["scheduler"]["workloads"]
  claims = state["devices"]["claims"]
  runs = []
  for row in metadata:
    run_id = row["model_id"]
    lora = row.get("fine_tuning_type", "lora") == "lora"
    runtime = row.get("base_model") if lora else run_id
    matched = [w for w in workloads if w.get("model_id") == runtime and w.get("training_kind") == ("lora" if lora else "fft")]
    pods = []
    for pod in state["pods"]:
      owned = [w for w in matched if w["uid"] in pod["owner_uids"] or (w.get("pod_name") == pod["name"] and pod.get("worker") == w["name"])]
      if owned:
        pod = copy.deepcopy(pod)
        pod.update(role=owned[0].get("role"), shared_runtime=lora, runtime_id=runtime)
        pod["devices"] = sorted({device for w in owned for device in claims.get(w.get("claim_name"), [])})
        pods.append(pod)
    runs.append(
      {
        "run_id": run_id,
        "name": row.get("name") or row.get("run_name") or f"{(row.get('base_model') or 'Run').split('/')[-1]} · {run_id[:8]}",
        "model": row.get("base_model"),
        "display_name": row.get("name") or row.get("run_name"),
        "recipe_name": row.get("recipe_name"),
        "fine_tuning_type": row.get("fine_tuning_type"),
        "runtime_id": runtime,
        "shared_runtime": lora,
        "status": row.get("status", "unknown"),
        "display_status": observed_status(row, matched, pods, state["available"]),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "completed_at": row.get("completed_at"),
        "steps": row.get("total_steps_completed"),
        "max_steps": row.get("max_steps"),
        "pods": pods,
        "workloads": matched,
        "nodes": sorted({p["node"] for p in pods if p.get("node")}),
      }
    )
  for run in runs:
    run["runtime_run_ids"] = [r["run_id"] for r in runs if r["runtime_id"] == run["runtime_id"]]
  return sorted(runs, key=lambda r: str(r.get("created_at") or ""), reverse=True)


def placements_of(state: dict, runs: list[dict]) -> list[dict]:
  """Every placed workload as one allocation: who, where, on which devices."""
  claims = state["devices"]["claims"]
  placements = []
  for workload in state["scheduler"]["workloads"]:
    if not workload.get("node_name"):
      continue
    associated = [r for r in runs if workload in r["workloads"]]
    placements.append(
      {
        "id": workload.get("uid") or workload["name"],
        "name": workload["name"],
        "label": associated[0]["name"] if len(associated) == 1 else workload.get("model_id") or workload["name"],
        "run_ids": [r["run_id"] for r in associated],
        "runtime_id": workload.get("model_id"),
        "node": workload["node_name"],
        "pod": workload.get("pod_name"),
        "role": workload.get("role"),
        "owner_id": workload.get("owner_id"),
        "phase": workload.get("phase"),
        "device_count": workload.get("device_count", 0),
        "devices": claims.get(workload.get("claim_name"), []),
      }
    )
  return placements


class Snapshot:
  """Reads the cluster at most once per CACHE_SECONDS however many pages poll."""

  def __init__(self) -> None:
    self.lock = asyncio.Lock()
    self.latest: dict | None = None
    self.updated = 0.0

  async def current(self, store) -> dict:
    async with self.lock:
      if self.latest is not None and time.monotonic() - self.updated < CACHE_SECONDS:
        return self.latest
      state = await asyncio.to_thread(cluster.read)
      store_error = None
      try:
        metadata = await asyncio.wait_for(store.list_jobs_metadata(), timeout=5)
      except Exception:
        metadata, store_error = [], "Run store unavailable"
      runs = join_runs(metadata, state)
      placements = placements_of(state, runs)
      now = time.time()
      try:
        past = await history.read(store, now - HISTORY_WINDOW_SECONDS)
      except Exception:
        past = []
      self.latest = {
        "schema_version": 2,
        "observed_at": iso(now),
        "cluster": state,
        "runs": runs,
        "store_error": store_error,
        "placements": placements,
        "history": past,
      }
      self.updated = time.monotonic()
      return self.latest

  async def placements(self, store) -> list[dict]:
    return (await self.current(store))["placements"]


snapshot = Snapshot()
