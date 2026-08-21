# Real data sources for the operational dashboard: gateway process state, Redis, the shared
# filesystem, and (when reachable) the Kubernetes API. Every accessor degrades to an explicit
# "unavailable" result instead of raising so the dashboard can always render something truthful.

import asyncio
import functools
import json
import os
import shutil
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from server.store import InMemoryStore, RedisStore, RequestStore
from server.worker_launch_processor import FFTWorkerManager

START_TIME = time.time()
K8S_REQUEST_TIMEOUT = 4
NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def demo_mode_enabled() -> bool:
  return os.getenv("OPEN_RL_DASHBOARD_DEMO", "").lower() in {"1", "true", "yes"}


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def iso_timestamp(ts: float | str | None) -> str | None:
  if ts is None:
    return None
  if isinstance(ts, str):
    return ts
  return datetime.fromtimestamp(ts, tz=UTC).isoformat()


# *** Kubernetes ***


def k8s_namespace() -> str:
  if ns := os.getenv("OPEN_RL_WORKER_NAMESPACE"):
    return ns
  try:
    with open(NAMESPACE_FILE) as f:
      return f.read().strip()
  except OSError:
    return "default"


@functools.cache
def k8s_core_v1() -> tuple[Any, str | None]:
  """Return (CoreV1Api, None) or (None, reason). The client library and cluster credentials
  are both optional; the first outcome is cached for the lifetime of the process."""
  try:
    from kubernetes import client, config
  except ImportError:
    return None, "kubernetes python client not installed"
  try:
    config.load_incluster_config()
  except Exception:
    try:
      config.load_kube_config()
    except Exception as exc:
      return None, f"no cluster credentials: {exc}"
  return client.CoreV1Api(), None


def pod_problem(pod: Any) -> str | None:
  phase = pod.status.phase or "Unknown"
  if phase == "Failed":
    return f"Failed: {pod.status.reason or 'see logs'}"
  for cs in pod.status.container_statuses or []:
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting.reason}: {waiting.message or ''}".strip(": ")
  if phase == "Pending":
    for cond in pod.status.conditions or []:
      if cond.type == "PodScheduled" and cond.status != "True":
        return f"Unschedulable: {cond.message or cond.reason or 'no node available'}"
    return "Pending"
  return None


def pod_gpu_count(pod: Any) -> int:
  """GPUs a pod claims: nvidia.com/gpu requests/limits, or DRA resource claims (1 device each)."""
  gpus = 0
  for container in pod.spec.containers or []:
    resources = container.resources
    for source in (resources.requests if resources else None, resources.limits if resources else None):
      if source and "nvidia.com/gpu" in source:
        gpus += int(float(source["nvidia.com/gpu"]))
        break
  if gpus == 0:
    gpus = len(pod.spec.resource_claims or [])
  return gpus


def pod_to_dict(pod: Any) -> dict:
  statuses = pod.status.container_statuses or []
  containers = []
  for cs in statuses:
    state = "unknown"
    if cs.state:
      state = "running" if cs.state.running else "waiting" if cs.state.waiting else "terminated" if cs.state.terminated else "unknown"
    containers.append({"name": cs.name, "image": cs.image, "ready": bool(cs.ready), "state": state})
  if not containers:
    containers = [{"name": c.name, "image": c.image, "ready": False, "state": "unknown"} for c in pod.spec.containers or []]
  ready_count = sum(1 for c in statuses if c.ready)
  return {
    "name": pod.metadata.name,
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "app": (pod.metadata.labels or {}).get("app"),
    "labels": pod.metadata.labels or {},
    "ready": f"{ready_count}/{len(pod.spec.containers or [])}",
    "restarts": sum(cs.restart_count or 0 for cs in statuses),
    "created_at": pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
    "problem": pod_problem(pod),
    "containers": containers,
    "gpus": pod_gpu_count(pod),
  }


def node_to_dict(node: Any) -> dict:
  labels = node.metadata.labels or {}
  capacity = node.status.capacity or {}
  allocatable = node.status.allocatable or {}
  conditions = node.status.conditions or []
  return {
    "name": node.metadata.name,
    "ready": any(c.type == "Ready" and c.status == "True" for c in conditions),
    "memory_pressure": any(c.type == "MemoryPressure" and c.status == "True" for c in conditions),
    "disk_pressure": any(c.type == "DiskPressure" and c.status == "True" for c in conditions),
    "unschedulable": bool(node.spec.unschedulable),
    "instance_type": labels.get("node.kubernetes.io/instance-type") or labels.get("beta.kubernetes.io/instance-type"),
    "accelerator": labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"),
    "gpu_capacity": int(capacity.get("nvidia.com/gpu", 0)),
    "gpu_allocatable": int(allocatable.get("nvidia.com/gpu", 0)),
  }


def k8s_snapshot() -> dict:
  """List pods in our namespace and (when RBAC allows) cluster nodes. Blocking; call in a thread."""
  api, err = k8s_core_v1()
  namespace = k8s_namespace()
  if api is None:
    return {"available": False, "namespace": namespace, "error": err, "pods": [], "nodes": []}
  try:
    pods = [pod_to_dict(p) for p in api.list_namespaced_pod(namespace, _request_timeout=K8S_REQUEST_TIMEOUT).items]
  except Exception as exc:
    return {"available": False, "namespace": namespace, "error": f"pod list failed: {exc}", "pods": [], "nodes": []}
  try:
    nodes = [node_to_dict(n) for n in api.list_node(_request_timeout=K8S_REQUEST_TIMEOUT).items]
  except Exception:
    # Namespaced service accounts often cannot list nodes; pods alone are still useful.
    nodes = []
  return {"available": True, "namespace": namespace, "error": None, "pods": pods, "nodes": nodes}


def k8s_pod_logs(pod: str, container: str | None, tail: int) -> dict:
  api, err = k8s_core_v1()
  if api is None:
    raise RuntimeError(err or "kubernetes unavailable")
  text = api.read_namespaced_pod_log(
    pod,
    k8s_namespace(),
    container=container,
    tail_lines=tail,
    _request_timeout=K8S_REQUEST_TIMEOUT + 4,
  )
  return {"demo": False, "pod": pod, "container": container, "text": text}


# *** GPU duty cycle ***
#
# Allocation duty per pool, broken down by job: GPUs claimed by non-terminal pods / pool GPU
# capacity, attributed to runs via the timeslice job-id labels the k8s worker manager stamps,
# sampled into an in-memory ring buffer whenever the cluster is polled. This is truthful
# scheduler state, not device utilization — DCGM owns that. History lives for the gateway's
# lifetime. A series sample is [unix_ts, {job: claimed_gpus}].

TERMINAL_POD_PHASES = {"Succeeded", "Failed"}


def pool_gpu_capacity(pool: dict) -> int:
  return sum(node["gpu_capacity"] for node in pool["nodes"])


def pod_job(pod: dict) -> str:
  """The run a pod's GPUs belong to: the model id from its timeslice job-id label, else its
  app label, else 'other'. Trainer and sampler pods of one run share one job."""
  job_id = (pod.get("labels") or {}).get("timeslice.io/job-id", "")
  for role in ("trainer-", "sampler-"):
    if job_id.startswith(role):
      return job_id.removeprefix(role)
  return pod.get("app") or "other"


class DutyTracker:
  """Ring buffer of per-job allocation-duty samples per GPU pool."""

  def __init__(self, max_samples: int = 120, min_interval_s: float = 5.0):
    self.max_samples = max_samples
    self.min_interval_s = min_interval_s
    self.series: dict[str, list[list]] = {}
    self.last_sample_at = 0.0

  def record(self, pools: list[dict], pods: list[dict], now: float | None = None) -> None:
    """Append one duty sample per GPU pool, throttled so overlapping polls don't stack points."""
    now = now if now is not None else time.time()
    if now - self.last_sample_at < self.min_interval_s:
      return
    self.last_sample_at = now
    claims_by_node: dict[str, dict[str, int]] = {}
    for pod in pods:
      gpus = pod.get("gpus", 0)
      if not gpus or not pod["node"] or pod["phase"] in TERMINAL_POD_PHASES:
        continue
      node_claims = claims_by_node.setdefault(pod["node"], {})
      job = pod_job(pod)
      node_claims[job] = node_claims.get(job, 0) + gpus
    for pool in pools:
      if not pool_gpu_capacity(pool):
        continue
      claims: dict[str, int] = {}
      for node in pool["nodes"]:
        for job, gpus in claims_by_node.get(node["name"], {}).items():
          claims[job] = claims.get(job, 0) + gpus
      series = self.series.setdefault(pool["id"], [])
      series.append([int(now), claims])
      del series[: -self.max_samples]

  def duty(self, pool: dict) -> dict | None:
    capacity = pool_gpu_capacity(pool)
    if not capacity:
      return None
    series = self.series.get(pool["id"], [])
    jobs: list[str] = []
    for _, claims in series:
      for job in claims:
        if job not in jobs:
          jobs.append(job)
    # Deliberately unclamped: time-sliced pools can be overcommitted, and current > 1 is the
    # honest way to report that.
    current = sum(series[-1][1].values()) / capacity if series else 0.0
    return {"capacity": capacity, "current": round(current, 4), "jobs": jobs, "series": series}


duty_tracker = DutyTracker()


# *** Cluster assembly ***


def gateway_summary() -> dict:
  from server import gateway

  return {
    "title": "open-rl gateway",
    "mode": "single-process" if gateway.is_single_process_mode() else "distributed",
    "fft_enabled": gateway.is_fft_enabled(),
    "redis_configured": bool(os.getenv("REDIS_URL")),
    "vllm_url": gateway.VLLM_URL if gateway.get_sampler_backend() == "vllm" else None,
    "sampler_backend": gateway.get_sampler_backend(),
  }


async def ping_redis(store: RequestStore) -> bool | None:
  """True/False for a Redis-backed store, None when the store is in-memory."""
  if not isinstance(store, RedisStore):
    return None
  try:
    return bool(await store.redis.ping())
  except Exception:
    return False


async def cluster_snapshot(store: RequestStore, k8s: dict) -> dict:
  gateway_card = gateway_summary()
  redis_ok = await ping_redis(store)

  shared = tmp_dir()
  services = [
    {
      "id": "redis",
      "label": "Redis",
      "configured": gateway_card["redis_configured"],
      "ok": redis_ok,
      "detail": os.getenv("REDIS_URL") or "not set — in-memory store",
    },
    {
      "id": "storage",
      "label": "Shared storage",
      "configured": True,
      "ok": os.path.isdir(shared) and os.access(shared, os.W_OK),
      "detail": shared,
    },
  ]
  if gateway_card["vllm_url"]:
    services.append({"id": "vllm", "label": "vLLM worker", "configured": True, "ok": None, "detail": gateway_card["vllm_url"]})

  # Edges only where we know the gateway actually connects: its configured Redis and vLLM URLs.
  edges = []
  if gateway_card["redis_configured"]:
    edges.append({"from": "gateway", "to": "redis", "reason": "REDIS_URL configured"})
  if gateway_card["vllm_url"]:
    edges.append({"from": "gateway", "to": "vllm", "reason": "VLLM_URL configured"})

  pools: dict[str, dict] = {}
  pods_by_node: dict[str, list[str]] = {}
  for pod in k8s["pods"]:
    if pod["node"]:
      pods_by_node.setdefault(pod["node"], []).append(pod["name"])
  for node in k8s["nodes"]:
    pool_id = node["accelerator"] or ("gpu" if node["gpu_capacity"] else "cpu")
    pool = pools.setdefault(pool_id, {"id": pool_id, "label": pool_id, "nodes": []})
    pool["nodes"].append(
      {**{k: node[k] for k in ("name", "ready", "instance_type", "gpu_capacity", "gpu_allocatable")}, "pods": pods_by_node.get(node["name"], [])}
    )
  duty_tracker.record(list(pools.values()), k8s["pods"])
  for pool in pools.values():
    pool["duty"] = duty_tracker.duty(pool)

  known_nodes = {n["name"] for n in k8s["nodes"]}
  unplaced = [p["name"] for p in k8s["pods"] if not p["node"] or p["node"] not in known_nodes]
  if unplaced:
    pools["unscheduled"] = {
      "id": "unscheduled",
      "label": "not scheduled",
      "duty": None,
      "nodes": [{"name": "—", "ready": None, "instance_type": None, "gpu_capacity": 0, "gpu_allocatable": 0, "pods": unplaced}],
    }

  return {
    "demo": False,
    "kubernetes": {"available": k8s["available"], "namespace": k8s["namespace"], "error": k8s["error"]},
    "gateway": gateway_card,
    "services": services,
    "edges": edges,
    "pools": sorted(pools.values(), key=lambda p: (p["id"] == "cpu", p["id"] == "unscheduled", p["id"])),
    "pods": k8s["pods"],
  }


# *** Runs ***


def filesystem_runs() -> dict[str, dict]:
  found: dict[str, dict] = {}
  peft_dir = os.path.join(tmp_dir(), "peft")
  if os.path.isdir(peft_dir):
    for entry in os.scandir(peft_dir):
      if not entry.is_dir():
        continue
      info = found.setdefault(entry.name, {"sources": set()})
      info["sources"].add("adapter")
      info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
      metadata_path = os.path.join(entry.path, "metadata.json")
      if os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            meta = json.load(f)
          info.setdefault("name", meta.get("alias"))
          info.setdefault("base_model", meta.get("base_model"))
          info.setdefault("wandb_url", meta.get("wandb_url"))
        except Exception:
          pass
  ckpt_dir = os.path.join(tmp_dir(), "checkpoints")
  if os.path.isdir(ckpt_dir):
    for entry in os.scandir(ckpt_dir):
      if entry.is_dir():
        info = found.setdefault(entry.name, {"sources": set()})
        info["sources"].add("checkpoint")
        info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
  return found


async def redis_runs(store: RequestStore) -> dict[str, dict]:
  found: dict[str, dict] = {}
  if isinstance(store, RedisStore):
    try:
      async for key in store.redis.scan_iter(match="open_rl:queue:*", count=200):
        model_id = key.removeprefix("open_rl:queue:")
        if model_id != "default":
          found.setdefault(model_id, {"sources": set()})["sources"].add("queue")
      async for key in store.redis.scan_iter(match="open_rl:model_meta:*", count=200):
        model_id = key.removeprefix("open_rl:model_meta:")
        info = found.setdefault(model_id, {"sources": set()})
        info["sources"].add("registered")
        try:
          meta = json.loads(await store.redis.get(key) or "{}")
          info.setdefault("base_model", meta.get("base_model"))
          info.setdefault("created_at", iso_timestamp(meta.get("created_at")))
          info.setdefault("wandb_url", meta.get("wandb_url"))
        except Exception:
          pass
    except Exception:
      return found
  elif isinstance(store, InMemoryStore):
    for model_id, queue in store.queues.items():
      if model_id != "default" and not queue.empty():
        found.setdefault(model_id, {"sources": set()})["sources"].add("queue")
  return found


def worker_processes(worker_manager: FFTWorkerManager | None) -> dict[str, bool]:
  """model_id -> process alive, for gateway-launched local FFT workers."""
  if worker_manager is None:
    return {}
  return {model_id: proc.poll() is None for model_id, proc in worker_manager.processes.items()}


def sanitize_job_id(model_id: str) -> str:
  return "".join(c if c.isalnum() else "-" for c in model_id.lower()).strip("-")


def model_pods(model_id: str, pods: list[dict]) -> list[dict]:
  """Pods launched for this model, matched exactly on the timeslice job-id labels the k8s
  worker manager stamps. Exact match only: prefix matching would cross-match runs that share
  an id prefix, and stop_run deletes what this returns. (Branches whose sanitize_job_id
  hash-truncates long ids should swap that helper in here.)"""
  wanted = {f"{role}-{sanitize_job_id(model_id)}" for role in ("trainer", "sampler")}
  return [p for p in pods if (p.get("labels") or {}).get("timeslice.io/job-id") in wanted]


async def runs_snapshot(store: RequestStore, worker_manager: FFTWorkerManager | None, pods: list[dict]) -> dict:
  found = await redis_runs(store)
  for model_id, info in filesystem_runs().items():
    merged = found.setdefault(model_id, {"sources": set()})
    merged["sources"] |= info.pop("sources")
    for key, value in info.items():
      if merged.get(key) is None:
        merged[key] = value
  workers = worker_processes(worker_manager)
  for model_id, alive in workers.items():
    info = found.setdefault(model_id, {"sources": set()})
    info["sources"].add("worker")
    info["worker_alive"] = alive

  runs = []
  for model_id, info in found.items():
    run_pods = model_pods(model_id, pods)
    stoppable = bool(info.get("worker_alive") or "queue" in info["sources"] or run_pods)
    runs.append(
      {
        "run_id": model_id,
        "name": info.get("name") or f"run-{model_id[:8]}",
        "base_model": info.get("base_model"),
        "created_at": info.get("created_at"),
        "wandb_url": info.get("wandb_url"),
        "stoppable": stoppable,
        "sources": sorted(info["sources"]),
        "pods": [p["name"] for p in run_pods],
      }
    )
  runs.sort(key=lambda r: (r["created_at"] is not None, r["created_at"] or "", r["run_id"]), reverse=True)
  return {"demo": False, "runs": runs}


async def run_detail(
  store: RequestStore,
  worker_manager: FFTWorkerManager | None,
  run_id: str,
  k8s: dict,
  log_tail: int = 0,
) -> dict | None:
  """Everything about one run in a single payload: its record, full pod state, queue depth,
  current GPU claims per pool, and (when log_tail > 0) a log tail per pod."""
  snapshot = await runs_snapshot(store, worker_manager, k8s["pods"])
  run = next((r for r in snapshot["runs"] if r["run_id"] == run_id), None)
  if run is None:
    return None

  queue_depth = 0
  if isinstance(store, RedisStore):
    try:
      queue_depth = await store.redis.llen(f"open_rl:queue:{run_id}")
    except Exception:
      pass
  elif isinstance(store, InMemoryStore):
    queue = store.queues.get(run_id)
    queue_depth = queue.qsize() if queue else 0

  gpu_claims = {}
  for pool_id, series in duty_tracker.series.items():
    if series and series[-1][1].get(run_id):
      gpu_claims[pool_id] = series[-1][1][run_id]

  pods = model_pods(run_id, k8s["pods"])
  detail = {**run, "demo": False, "pods": pods, "queue_depth": queue_depth, "gpu_claims": gpu_claims}
  if log_tail:
    logs = {}
    for pod in pods:
      try:
        logs[pod["name"]] = (await asyncio.to_thread(k8s_pod_logs, pod["name"], None, log_tail))["text"]
      except Exception as exc:
        logs[pod["name"]] = f"(logs unavailable: {exc})"
    detail["logs"] = logs
  return detail


async def stop_run(store: RequestStore, worker_manager: FFTWorkerManager | None, model_id: str) -> dict:
  """Stop everything we can truthfully stop for a run: the gateway-launched worker process,
  queued work in Redis, and any pods labeled for the model. Reports each action taken."""
  actions = []

  proc = worker_manager.processes.get(model_id) if worker_manager is not None else None
  if proc is not None and proc.poll() is None:
    proc.terminate()
    actions.append("terminated local worker process")

  if isinstance(store, RedisStore):
    try:
      removed = await store.redis.delete(
        f"open_rl:queue:{model_id}",
        f"open_rl:sampler_queue:{model_id}",
        f"open_rl:sampler_ready:{model_id}",
      )
      await store.redis.lrem(store.active_list, 0, model_id)
      await store.redis.srem(store.active_set, model_id)
      if removed:
        actions.append(f"cleared {removed} queue key(s) in Redis")
    except Exception as exc:
      actions.append(f"redis cleanup failed: {exc}")
  elif isinstance(store, InMemoryStore):
    async with store.active_tenants_cv:
      if model_id in store.queues:
        del store.queues[model_id]
        actions.append("cleared in-memory queue")
      if model_id in store.active_tenants:
        store.active_tenants.remove(model_id)

  api, _ = k8s_core_v1()
  if api is not None:
    try:
      k8s = k8s_snapshot()
      for pod in model_pods(model_id, k8s["pods"]):
        api.delete_namespaced_pod(pod["name"], k8s["namespace"], _request_timeout=K8S_REQUEST_TIMEOUT)
        actions.append(f"deleted pod {pod['name']}")
    except Exception as exc:
      actions.append(f"pod deletion failed: {exc}")

  return {"run_id": model_id, "stopped": bool(actions), "actions": actions}


# *** Operational load stats ***


def format_bytes(n: float) -> str:
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if n < 1024 or unit == "TiB":
      return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
    n /= 1024
  return f"{n:.1f} TiB"


def gateway_rss_bytes() -> int | None:
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    pass
  return None


def stat_entry(stat_id: str, label: str, value: str, detail: str = "") -> dict:
  return {"id": stat_id, "label": label, "value": value, "detail": detail}


async def operational_stats(store: RequestStore, k8s: dict, worker_manager: FFTWorkerManager | None) -> tuple[list[dict], list[dict]]:
  """Load numbers for the Health screen: queue depths, active runs, Redis and gateway memory,
  disk, and pod totals. Everything is measured, never estimated."""
  queues: list[dict] = []
  launch_depth = 0
  redis_stats: list[dict] = []
  if isinstance(store, RedisStore):
    try:
      async for key in store.redis.scan_iter(match="open_rl:queue:*", count=200):
        depth = await store.redis.llen(key)
        if depth:
          queues.append({"model_id": key.removeprefix("open_rl:queue:"), "depth": depth})
      launch_depth = await store.redis.llen(store.worker_launch_queue)
      memory = await store.redis.info("memory")
      used = memory.get("used_memory", 0)
      maxmemory = memory.get("maxmemory", 0)
      value = f"{used / maxmemory:.0%} of {format_bytes(maxmemory)}" if maxmemory else format_bytes(used)
      limit = f"limit {format_bytes(maxmemory)}" if maxmemory else "no maxmemory limit"
      redis_stats.append(stat_entry("redis.memory", "Redis memory", value, f"peak {format_bytes(memory.get('used_memory_peak', 0))} · {limit}"))
      clients = await store.redis.info("clients")
      redis_stats.append(stat_entry("redis.clients", "Redis clients", str(clients.get("connected_clients", 0)), "connected"))
    except Exception:
      pass
  elif isinstance(store, InMemoryStore):
    queues = [{"model_id": model_id, "depth": queue.qsize()} for model_id, queue in store.queues.items() if queue.qsize()]
  queues.sort(key=lambda q: -q["depth"])

  workers = worker_processes(worker_manager)
  active = {model_id for model_id, alive in workers.items() if alive} | {q["model_id"] for q in queues if q["model_id"] != "default"}

  stats = [
    stat_entry("runs.active", "Active runs", str(len(active)), "live worker or queued work"),
    stat_entry(
      "queue.requests", "Queued requests", str(sum(q["depth"] for q in queues)), f"across {len(queues)} queue{'' if len(queues) == 1 else 's'}"
    ),
    stat_entry("queue.launch", "Launches pending", str(launch_depth), "worker launch queue"),
    *redis_stats,
  ]
  rss = gateway_rss_bytes()
  if rss is not None:
    stats.append(stat_entry("gateway.rss", "Gateway memory", format_bytes(rss), "resident set size"))
  shared = tmp_dir()
  if os.path.isdir(shared):
    usage = shutil.disk_usage(shared)
    stats.append(stat_entry("storage.disk", "Disk free", format_bytes(usage.free), f"of {format_bytes(usage.total)} at {shared}"))
  if k8s["available"]:
    phases: dict[str, int] = {}
    for pod in k8s["pods"]:
      phases[pod["phase"]] = phases.get(pod["phase"], 0) + 1
    running = phases.pop("Running", 0)
    others = " · ".join(f"{count} {phase.lower()}" for phase, count in sorted(phases.items())) or "no other phases"
    stats.append(stat_entry("pods.running", "Pods running", str(running), others))
    total_gpus = sum(node["gpu_capacity"] for node in k8s["nodes"])
    if total_gpus:
      claimed = sum(pod.get("gpus", 0) for pod in k8s["pods"] if pod["node"] and pod["phase"] not in TERMINAL_POD_PHASES)
      stats.append(stat_entry("gpus.claimed", "GPUs claimed", f"{min(claimed, total_gpus)}/{total_gpus}", "across all pools"))
  return stats, queues


# *** Health ***


def check_entry(check_id: str, group: str, label: str, status: str, detail: str) -> dict:
  return {"id": check_id, "group": group, "label": label, "status": status, "detail": detail}


async def health_checks(store: RequestStore, k8s: dict) -> list[dict]:
  from server import gateway

  checks = []

  uptime = int(time.time() - START_TIME)
  mode = "single-process" if gateway.is_single_process_mode() else "distributed"
  fft = "FFT enabled" if gateway.is_fft_enabled() else "LoRA mode"
  checks.append(check_entry("gateway", "Gateway", "Gateway process", "ok", f"{mode}, {fft}, up {uptime // 3600}h {uptime % 3600 // 60}m"))

  if isinstance(store, RedisStore):
    started = time.perf_counter()
    ok = await ping_redis(store)
    latency_ms = (time.perf_counter() - started) * 1000
    if ok:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "ok", f"PING {latency_ms:.1f} ms — {os.getenv('REDIS_URL')}"))
    else:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "error", f"PING failed — {os.getenv('REDIS_URL')}"))
  else:
    checks.append(check_entry("storage.redis", "Storage", "Redis", "off", "REDIS_URL not set — using in-memory store"))

  shared = tmp_dir()
  if os.path.isdir(shared) and os.access(shared, os.W_OK):
    free_gib = shutil.disk_usage(shared).free / 2**30
    status = "warn" if free_gib < 20 else "ok"
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", status, f"{shared} writable, {free_gib:.0f} GiB free"))
  else:
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", "warn", f"{shared} missing or not writable"))

  if k8s["available"]:
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", "ok", f"{len(k8s['pods'])} pods visible in namespace {k8s['namespace']}"))
  else:
    status = "off" if "not installed" in (k8s["error"] or "") or "credentials" in (k8s["error"] or "") else "error"
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", status, k8s["error"] or "unavailable"))

  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "ok", "GCP Cloud Trace exporter configured"))
  else:
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "off", "ENABLE_GCP_TRACE=0 — tracing not configured"))

  if gateway.get_sampler_backend() == "vllm" and gateway.is_single_process_mode():
    healthz = f"{gateway.VLLM_URL.rstrip('/')}/healthz"
    try:
      async with httpx.AsyncClient(timeout=2.0) as client:
        (await client.get(healthz)).raise_for_status()
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "ok", f"reachable at {gateway.VLLM_URL}"))
    except Exception:
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "error", f"unreachable at {gateway.VLLM_URL}"))

  return checks


def derive_problems(checks: list[dict], k8s: dict) -> list[dict]:
  problems = []
  for check in checks:
    if check["status"] in {"warn", "error"}:
      problems.append({"severity": check["status"], "source": check["label"], "message": check["detail"]})
  for pod in k8s["pods"]:
    if pod["problem"]:
      severity = "error" if pod["phase"] == "Failed" or "BackOff" in pod["problem"] else "warn"
      problems.append({"severity": severity, "source": f"pod/{pod['name']}", "message": pod["problem"]})
    elif pod["restarts"] >= 3:
      problems.append({"severity": "warn", "source": f"pod/{pod['name']}", "message": f"{pod['restarts']} container restarts"})
  for node in k8s["nodes"]:
    if not node["ready"]:
      problems.append({"severity": "warn", "source": f"node/{node['name']}", "message": "Node not ready"})
    if node["memory_pressure"]:
      problems.append({"severity": "warn", "source": f"node/{node['name']}", "message": "Node under memory pressure"})
    if node["disk_pressure"]:
      problems.append({"severity": "warn", "source": f"node/{node['name']}", "message": "Node under disk pressure"})
  problems.sort(key=lambda p: p["severity"] != "error")
  return problems
