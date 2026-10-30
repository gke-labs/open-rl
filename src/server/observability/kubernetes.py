import asyncio
import json
import os
from functools import cache
from typing import TypedDict

READ_TIMEOUT_SECONDS = 6


class Cluster(TypedDict):
  available: bool  # whether pods could be listed
  pods: list[dict]
  nodes: list[dict]
  workloads: list[dict] | None  # None when the scheduler's Workloads cannot be listed
  errors: list[str]


def namespace() -> str:
  return os.getenv("OPEN_RL_WORKER_NAMESPACE") or os.getenv("POD_NAMESPACE") or "openrl-system"


def problem(status: dict) -> str | None:
  if status.get("phase") == "Failed":
    return f"Failed: {': '.join(p for p in (status.get('reason'), status.get('message')) if p) or 'see logs'}"
  for cs in status.get("containerStatuses") or []:
    waiting, ended = cs["state"].get("waiting"), cs["state"].get("terminated")
    if waiting and waiting.get("reason") not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting['reason']}: {waiting.get('message') or ''}".strip(": ")
    if ended and (ended.get("exitCode") or ended.get("reason") not in (None, "Completed")):
      return f"{ended.get('reason') or 'Terminated'}: exit code {ended.get('exitCode')}" + (f" — {ended['message']}" if ended.get("message") else "")
  if status.get("phase") == "Pending":
    unscheduled = [c for c in status.get("conditions") or [] if c["type"] == "PodScheduled" and c["status"] != "True"]
    return f"Unschedulable: {unscheduled[0].get('message') or 'no node available'}" if unscheduled else "Pending"
  return None


def gpu_capacity(node: dict, slices: list[dict]) -> int:
  """A node using DRA publishes no nvidia.com/gpu capacity; its GPUs are the devices in its NVIDIA ResourceSlices. Only
  each pool's newest generation counts: while the driver restarts, old and new generations coexist."""
  name, capacity = node["metadata"]["name"], int((node["status"].get("capacity") or {}).get("nvidia.com/gpu", 0))
  mine = [s["spec"] for s in slices if s["spec"].get("nodeName") == name and s["spec"].get("driver") == "gpu.nvidia.com"]
  newest: dict[str, int] = {}
  for pool in (spec["pool"] for spec in mine):
    newest[pool["name"]] = max(newest.get(pool["name"], 0), pool["generation"])
  devices = sum(len(spec.get("devices") or []) for spec in mine if spec["pool"]["generation"] == newest[spec["pool"]["name"]])
  return max(capacity, devices)


@cache
def kubernetes_api() -> tuple:
  from kubernetes import client, config

  try:
    config.load_incluster_config()
  except Exception:
    config.load_kube_config()
  return client.CoreV1Api(), client.CustomObjectsApi()


async def kubernetes() -> Cluster:
  try:
    from server.scheduler_worker_manager import holds_gpus, placement_of, worker_state

    core, custom = kubernetes_api()
  except Exception as exc:
    return {"available": False, "pods": [], "nodes": [], "workloads": None, "errors": [f"Kubernetes unavailable: {exc}"]}
  ns, errors = namespace(), []

  async def items(name: str, call, *args) -> list[dict] | None:
    try:
      # Read as raw JSON, because the client's own decoding of a large pod list costs more than the request.
      found = await asyncio.to_thread(call, *args, _preload_content=False, _request_timeout=READ_TIMEOUT_SECONDS)
      return json.loads(found.data)["items"]
    except Exception as exc:
      if getattr(exc, "status", None) != 404:  # no scheduler CRD or no DRA on a local stack
        errors.append(f"{name} read failed: {exc}")
      return None

  pods, nodes, workloads, slices = await asyncio.gather(
    items("pods", core.list_namespaced_pod, ns),
    items("nodes", core.list_node),
    items("workloads", custom.list_namespaced_custom_object, "openrl.io", "v1alpha1", ns, "workloads"),
    items("resourceslices", custom.list_cluster_custom_object, "resource.k8s.io", "v1", "resourceslices"),
  )
  result: Cluster = {"available": pods is not None, "pods": [], "nodes": [], "workloads": None if workloads is None else [], "errors": errors}
  for pod in pods or []:
    labels, statuses = pod["metadata"].get("labels") or {}, pod["status"].get("containerStatuses") or []
    row = {"name": pod["metadata"]["name"], "node": pod["spec"].get("nodeName"), "owner": labels.get("timeslice.io/owner")}
    result["pods"].append(row | {"problem": problem(pod["status"]), "restarts": sum(cs.get("restartCount") or 0 for cs in statuses)})
  for node in nodes or []:
    labels, status = node["metadata"].get("labels") or {}, node["status"]
    row = {"name": node["metadata"]["name"], "ready": any(c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions") or [])}
    row.update(accelerator=labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"))
    row["gpu_capacity"] = gpu_capacity(node, slices or [])
    result["nodes"].append(row | {"openrl_enabled": labels.get("openrl.io/enabled") == "true"})
  for item in workloads or []:
    spec, status = item["spec"], item.get("status") or {}
    deleting = bool(item["metadata"].get("deletionTimestamp"))
    state, _ = worker_state(phase=status.get("phase") or "Pending", reason=status.get("reason") or "", node=status.get("nodeName"), deleting=deleting)
    placement = placement_of(item)
    row = {"model_id": spec["modelID"], "training_kind": spec.get("trainingKind"), "state": state, "holds_gpus": holds_gpus(placement)}
    result["workloads"].append(row | placement)
  return result
