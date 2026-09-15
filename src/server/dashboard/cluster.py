"""What the dashboard reads from Kubernetes, reduced to the fields the pages use.

Everything here is blocking and meant to run in a thread. A missing client,
missing credentials, or a missing scheduler CRD are supported states and come
back as explicit errors, never exceptions.
"""

import concurrent.futures
import functools
import os
from typing import Any

REQUEST_TIMEOUT = 6
SCHEDULER_GROUP, SCHEDULER_VERSION = "openrl.io", "v1alpha1"
MAX_POD_EVENTS = 10
FAILED_TERMINATIONS = {"OOMKilled", "Error", "ContainerCannotRun"}


def namespace() -> str:
  return os.getenv("OPEN_RL_WORKER_NAMESPACE") or os.getenv("POD_NAMESPACE") or "openrl-system"


@functools.cache
def clients() -> tuple[Any, Any, str | None]:
  """(CoreV1Api, CustomObjectsApi, None) or (None, None, reason). Cached for the process."""
  try:
    from kubernetes import client, config
  except ImportError:
    return None, None, "kubernetes python client not installed"
  try:
    config.load_incluster_config()
  except Exception:
    try:
      config.load_kube_config()
    except Exception as exc:
      return None, None, f"no cluster credentials: {exc}"
  return client.CoreV1Api(), client.CustomObjectsApi(), None


def stamp(value: Any) -> str | None:
  return value.isoformat() if value is not None and hasattr(value, "isoformat") else value


# ---- pods -------------------------------------------------------------------


def termination(state: Any) -> dict | None:
  if state is None:
    return None
  return {"reason": state.reason, "exit_code": state.exit_code, "finished_at": stamp(state.finished_at), "started_at": stamp(state.started_at)}


def container_summary(status: Any) -> dict:
  state, reason, exit_code, started_at, finished_at = "unknown", None, None, None, None
  if status.state and status.state.running:
    state, started_at = "running", stamp(status.state.running.started_at)
  elif status.state and status.state.waiting:
    state, reason = "waiting", status.state.waiting.reason
  elif status.state and status.state.terminated:
    ended = status.state.terminated
    state, reason, exit_code = "terminated", ended.reason, ended.exit_code
    started_at, finished_at = stamp(ended.started_at), stamp(ended.finished_at)
  return {
    "name": status.name,
    "state": state,
    "reason": reason,
    "exit_code": exit_code,
    "started_at": started_at,
    "finished_at": finished_at,
    "restart_count": status.restart_count or 0,
    "last_termination": termination(status.last_state.terminated if status.last_state else None),
  }


def pod_problem(pod: Any) -> str | None:
  """The one line that explains a pod that is not simply running."""
  phase = pod.status.phase or "Unknown"
  if phase == "Failed":
    detail = ": ".join(part for part in (pod.status.reason, pod.status.message) if part)
    return f"Failed: {detail or 'see logs'}"
  for cs in pod.status.container_statuses or []:
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting.reason}: {waiting.message or ''}".strip(": ")
    ended = cs.state.terminated if cs.state else None
    if ended and (ended.exit_code or ended.reason not in (None, "Completed")):
      return f"{ended.reason or 'Terminated'}: exit code {ended.exit_code}{f' — {ended.message}' if ended.message else ''}"
    previous = cs.last_state.terminated if cs.last_state else None
    if previous and (cs.restart_count or 0) and previous.reason in FAILED_TERMINATIONS:
      return f"{previous.reason}: {cs.name} exited {previous.exit_code} and restarted"
  if phase == "Pending":
    for cond in pod.status.conditions or []:
      if cond.type == "PodScheduled" and cond.status != "True":
        return f"Unschedulable: {cond.message or cond.reason or 'no node available'}"
    return "Pending"
  return None


def pod_summary(pod: Any) -> dict:
  statuses = pod.status.container_statuses or []
  containers = [container_summary(cs) for cs in statuses] or [
    {
      "name": c.name,
      "state": "unknown",
      "reason": None,
      "exit_code": None,
      "started_at": None,
      "finished_at": None,
      "restart_count": 0,
      "last_termination": None,
    }
    for c in pod.spec.containers or []
  ]
  return {
    "name": pod.metadata.name,
    "uid": str(pod.metadata.uid),
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "worker": (pod.metadata.labels or {}).get("openrl.io/worker"),
    "owner_uids": [str(owner.uid) for owner in pod.metadata.owner_references or []],
    "restarts": sum(cs.restart_count or 0 for cs in statuses),
    "created_at": stamp(pod.metadata.creation_timestamp),
    "problem": pod_problem(pod),
    "containers": containers,
    "events": [],
  }


def event_summary(event: Any) -> dict:
  series = event.series
  return {
    "reason": event.reason,
    "message": event.message,
    "type": event.type,
    "count": event.count or (series.count if series else None) or 1,
    "first_seen_at": stamp(event.first_timestamp or event.metadata.creation_timestamp),
    "last_seen_at": stamp((series.last_observed_time if series else None) or event.event_time or event.last_timestamp),
    "pod_uid": event.involved_object.uid,
  }


def node_summary(node: Any) -> dict:
  labels = node.metadata.labels or {}
  return {
    "name": node.metadata.name,
    "ready": any(c.type == "Ready" and c.status == "True" for c in node.status.conditions or []),
    "accelerator": labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"),
    "gpu_capacity": int((node.status.capacity or {}).get("nvidia.com/gpu", 0)),
    "devices": [],
  }


# ---- scheduler and DRA --------------------------------------------------------


def workload_summary(item: dict) -> dict:
  metadata, spec, status = item.get("metadata") or {}, item.get("spec") or {}, item.get("status") or {}
  placed = next((c for c in status.get("conditions") or [] if c.get("type") == "Placed"), None) or {}
  return {
    "name": metadata.get("name"),
    "uid": metadata.get("uid"),
    "created_at": metadata.get("creationTimestamp"),
    "role": spec.get("role"),
    "model_id": spec.get("modelID"),
    "owner_id": spec.get("ownerID"),
    "training_kind": spec.get("trainingKind"),
    "exclusive": spec.get("exclusive", False),
    "requested_memory": (spec.get("accelerator") or {}).get("memory"),
    "phase": status.get("phase") or "Pending",
    "reason": status.get("reason"),
    "claim_name": status.get("claimName"),
    "pod_name": status.get("podName"),
    "node_name": status.get("nodeName"),
    "device_count": status.get("deviceCount", 0),
    "placed_reason": placed.get("reason"),
    "placed_message": placed.get("message"),
  }


def ledger_summary(item: dict) -> dict:
  spec = item.get("spec") or {}
  return {
    "name": (item.get("metadata") or {}).get("name"),
    "claim_name": spec.get("claimName"),
    "seats": [
      {
        "workload": seat.get("workload"),
        "workload_uid": seat.get("workloadUID"),
        "owner": seat.get("ownerID"),
        "exclusive": seat.get("exclusive", False),
      }
      for seat in spec.get("seats") or []
    ],
  }


def scheduler_state(custom: Any, ns: str) -> dict:
  def listing(plural: str) -> list[dict]:
    return custom.list_namespaced_custom_object(SCHEDULER_GROUP, SCHEDULER_VERSION, ns, plural, _request_timeout=REQUEST_TIMEOUT).get("items", [])

  try:
    workloads = [workload_summary(item) for item in listing("workloads")]
    ledgers = [ledger_summary(item) for item in listing("claimledgers")]
  except Exception as exc:
    if getattr(exc, "status", None) == 404:
      return {"installed": False, "available": False, "error": None, "workloads": [], "ledgers": []}
    return {"installed": True, "available": False, "error": f"scheduler read failed: {exc}", "workloads": [], "ledgers": []}
  return {"installed": True, "available": True, "error": None, "workloads": workloads, "ledgers": ledgers}


def complete_slices(slices: list[dict], driver: str) -> tuple[list[dict], bool]:
  """Only the latest complete generation of each DRA pool describes its devices."""
  pools: dict[tuple, list[dict]] = {}
  for item in slices:
    spec = item.get("spec", {})
    if spec.get("driver") == driver and spec.get("nodeName"):
      pools.setdefault((spec["nodeName"], spec["pool"]["name"]), []).append(spec)
  complete, incomplete = [], False
  for specs in pools.values():
    generation = max(spec["pool"].get("generation", -1) for spec in specs)
    current = [spec for spec in specs if spec["pool"].get("generation", -1) == generation]
    if generation < 0 or any(spec["pool"].get("resourceSliceCount") != len(current) for spec in current):
      incomplete = True
    else:
      complete.extend(current)
  return complete, incomplete


def device_uuid(attributes: dict) -> str | None:
  for key, value in attributes.items():
    if key.rsplit("/", 1)[-1].lower() == "uuid" and isinstance(value, dict):
      return value.get("string")
  return None


def device_inventory(custom: Any, ns: str) -> dict:
  """GPUs per node from ResourceSlices, and the devices each ResourceClaim holds."""
  driver = os.getenv("OPEN_RL_GPU_DRA_DRIVER", "gpu.nvidia.com")
  for version in ("v1", "v1beta1"):
    try:
      claims = custom.list_namespaced_custom_object("resource.k8s.io", version, ns, "resourceclaims", _request_timeout=REQUEST_TIMEOUT)["items"]
      slices = custom.list_cluster_custom_object("resource.k8s.io", version, "resourceslices", _request_timeout=REQUEST_TIMEOUT)["items"]
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        continue
      return {"available": False, "error": f"DRA discovery failed (status {getattr(exc, 'status', 'unavailable')})", "nodes": {}, "claims": {}}
    slices, incomplete = complete_slices(slices, driver)
    nodes: dict[str, list[dict]] = {}
    for spec in slices:
      for device in spec.get("devices", []):
        attributes = device.get("attributes") or device.get("basic", {}).get("attributes", {})
        identity = "/".join((spec["driver"], spec["pool"]["name"], device["name"]))
        nodes.setdefault(spec["nodeName"], []).append({"id": identity, "name": device["name"], "uuid": device_uuid(attributes)})
    return {
      "available": True,
      "error": "Some DRA pools have incomplete device inventory" if incomplete else None,
      "nodes": nodes,
      "claims": {
        item["metadata"]["name"]: [
          "/".join((d["driver"], d["pool"], d["device"])) for d in item.get("status", {}).get("allocation", {}).get("devices", {}).get("results", [])
        ]
        for item in claims
      },
    }
  return {"available": False, "error": "DRA API unavailable", "nodes": {}, "claims": {}}


# ---- the one read -------------------------------------------------------------


def unavailable(ns: str, error: str) -> dict:
  return {
    "available": False,
    "namespace": ns,
    "error": error,
    "pods": [],
    "nodes": [],
    "nodes_error": None,
    "events_error": None,
    "scheduler": {"installed": None, "available": False, "error": error, "workloads": [], "ledgers": []},
    "devices": {"available": False, "error": error, "nodes": {}, "claims": {}},
  }


def read() -> dict:
  """Pods, nodes, events, scheduler objects and DRA devices in one pass."""
  core, custom, error = clients()
  ns = namespace()
  if core is None:
    return unavailable(ns, error)
  try:
    pods = [pod_summary(p) for p in core.list_namespaced_pod(ns, _request_timeout=REQUEST_TIMEOUT).items]
  except Exception as exc:
    return unavailable(ns, f"pod list failed: {exc}")

  def attempt(call, *args, **kwargs):
    try:
      return call(*args, **kwargs), None
    except Exception as exc:
      return None, str(exc)

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    nodes_f = pool.submit(attempt, core.list_node, _request_timeout=REQUEST_TIMEOUT)
    events_f = pool.submit(
      attempt, core.list_namespaced_event, ns, field_selector="involvedObject.kind=Pod", limit=200, _request_timeout=REQUEST_TIMEOUT
    )
    scheduler_f = pool.submit(scheduler_state, custom, ns)
    devices_f = pool.submit(device_inventory, custom, ns)
    node_list, nodes_error = nodes_f.result()
    event_list, events_error = events_f.result()
    scheduler = scheduler_f.result()
    devices = devices_f.result()

  nodes = [node_summary(n) for n in node_list.items] if node_list else []
  for node in nodes:
    node["devices"] = devices["nodes"].get(node["name"], [])
    node["gpu_capacity"] = max(node["gpu_capacity"], len(node["devices"]))
  events_by_pod: dict[str, list[dict]] = {}
  for event in event_list.items if event_list else []:
    summary = event_summary(event)
    events_by_pod.setdefault(summary.pop("pod_uid"), []).append(summary)
  for pod in pods:
    pod["events"] = sorted(events_by_pod.get(pod["uid"], []), key=lambda e: e["last_seen_at"] or "")[-MAX_POD_EVENTS:]
  return {
    "available": True,
    "namespace": ns,
    "error": None,
    "pods": pods,
    "nodes": nodes,
    "nodes_error": f"node list failed: {nodes_error}" if nodes_error else None,
    "events_error": f"event list failed: {events_error}" if events_error else None,
    "scheduler": scheduler,
    "devices": devices,
  }


def pod_logs(pod: str, container: str | None, tail: int, previous: bool = False) -> dict:
  core, _, error = clients()
  if core is None:
    raise RuntimeError(error or "kubernetes unavailable")
  text = core.read_namespaced_pod_log(
    pod,
    namespace(),
    container=container,
    previous=previous,
    tail_lines=tail,
    timestamps=True,
    limit_bytes=128 * 1024,
    _request_timeout=REQUEST_TIMEOUT + 4,
  )
  return {"pod": pod, "container": container, "previous": previous, "text": text}
