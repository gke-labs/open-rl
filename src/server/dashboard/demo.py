# Fictional demo payloads for the dashboard. Everything here is invented so the UI can be
# developed and demoed without a cluster; every payload carries demo=True and the UI is
# required to label it as fictional.

import math
import time

DEMO_NOTICE = "Demo data — every machine, pod, and run on this page is fictional."


def demo_duty_series(capacity: int, jobs: dict[str, tuple[int, int]], seed: int) -> dict:
  """A plausible fictional per-job duty series: 10 minutes of samples every 10s. `jobs` maps
  each job to (base GPUs, wobble amplitude); claims are clamped to pool capacity."""
  now = int(time.time())
  series = []
  for i in range(60):
    claims = {}
    for j, (job, (base, amplitude)) in enumerate(jobs.items()):
      value = round(base + amplitude * math.sin((i + seed * 5 + j * 9) / 6))
      if value > 0:
        claims[job] = value
    while sum(claims.values()) > capacity:
      biggest = max(claims, key=claims.get)
      claims[biggest] -= 1
    series.append([now - (59 - i) * 10, {job: gpus for job, gpus in claims.items() if gpus > 0}])
  current = round(sum(series[-1][1].values()) / capacity, 4)
  return {"capacity": capacity, "current": current, "jobs": list(jobs), "series": series}


def demo_cluster() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "kubernetes": {"available": True, "namespace": "open-rl-demo", "error": None},
    "gateway": {
      "title": "open-rl gateway",
      "mode": "distributed",
      "fft_enabled": True,
      "redis_configured": True,
      "vllm_url": None,
      "sampler_backend": "vllm",
    },
    "services": [
      {"id": "redis", "label": "Redis", "configured": True, "ok": True, "detail": "redis://demo-redis:6379"},
      {"id": "storage", "label": "Shared storage", "configured": True, "ok": True, "detail": "/mnt/shared/open-rl"},
    ],
    "edges": [{"from": "gateway", "to": "redis", "reason": "REDIS_URL configured"}],
    "pools": [
      {
        "id": "nvidia-h100-80gb",
        "label": "nvidia-h100-80gb",
        "duty": demo_duty_series(16, {"demo-run-1": (8, 0), "demo-run-2": (4, 3), "other": (1, 1)}, seed=1),
        "nodes": [
          {
            "name": "demo-h100-node-1",
            "ready": True,
            "instance_type": "a3-highgpu-8g",
            "gpu_capacity": 8,
            "gpu_allocatable": 8,
            "pods": ["open-rl-trainer-demo-run-1", "open-rl-trainer-demo-run-2"],
          },
          {
            "name": "demo-h100-node-2",
            "ready": True,
            "instance_type": "a3-highgpu-8g",
            "gpu_capacity": 8,
            "gpu_allocatable": 8,
            "pods": ["open-rl-sampler-demo-run-1"],
          },
        ],
      },
      {
        "id": "nvidia-l4",
        "label": "nvidia-l4",
        "duty": demo_duty_series(4, {"demo-run-2": (2, 1), "other": (1, 1)}, seed=2),
        "nodes": [
          {
            "name": "demo-l4-node-1",
            "ready": True,
            "instance_type": "g2-standard-24",
            "gpu_capacity": 2,
            "gpu_allocatable": 2,
            "pods": ["open-rl-sampler-demo-run-2"],
          },
          {
            "name": "demo-l4-node-2",
            "ready": False,
            "instance_type": "g2-standard-24",
            "gpu_capacity": 2,
            "gpu_allocatable": 0,
            "pods": [],
          },
        ],
      },
      {
        "id": "cpu",
        "label": "cpu",
        "nodes": [
          {
            "name": "demo-cpu-node-1",
            "ready": True,
            "instance_type": "n2-standard-8",
            "gpu_capacity": 0,
            "gpu_allocatable": 0,
            "pods": ["open-rl-gateway-7f9c4", "demo-redis-0"],
          }
        ],
      },
    ],
    "pods": [
      {
        "name": "open-rl-gateway-7f9c4",
        "phase": "Running",
        "node": "demo-cpu-node-1",
        "app": "open-rl-gateway",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-27T09:12:00+00:00",
        "problem": None,
        "containers": [{"name": "gateway", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "demo-redis-0",
        "phase": "Running",
        "node": "demo-cpu-node-1",
        "app": "redis",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-27T09:10:00+00:00",
        "problem": None,
        "containers": [{"name": "redis", "image": "redis:7", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-trainer-demo-run-1",
        "phase": "Running",
        "node": "demo-h100-node-1",
        "app": "open-rl-trainer-worker",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-29T06:02:00+00:00",
        "problem": None,
        "containers": [{"name": "trainer", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-trainer-demo-run-2",
        "phase": "Pending",
        "node": "demo-h100-node-1",
        "app": "open-rl-trainer-worker",
        "ready": "0/1",
        "restarts": 0,
        "created_at": "2026-07-29T07:41:00+00:00",
        "problem": "Unschedulable: waiting for a free GPU claim",
        "containers": [{"name": "trainer", "image": "gcr.io/demo/open-rl-server:demo", "ready": False, "state": "waiting"}],
      },
      {
        "name": "open-rl-sampler-demo-run-1",
        "phase": "Running",
        "node": "demo-h100-node-2",
        "app": "open-rl-sampler-worker",
        "ready": "1/1",
        "restarts": 2,
        "created_at": "2026-07-29T06:03:00+00:00",
        "problem": None,
        "containers": [{"name": "sampler", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-sampler-demo-run-2",
        "phase": "Failed",
        "node": "demo-l4-node-1",
        "app": "open-rl-sampler-worker",
        "ready": "0/1",
        "restarts": 4,
        "created_at": "2026-07-28T22:17:00+00:00",
        "problem": "CrashLoopBackOff: CUDA out of memory",
        "containers": [{"name": "sampler", "image": "gcr.io/demo/open-rl-server:demo", "ready": False, "state": "terminated"}],
      },
    ],
  }


def demo_runs() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "runs": [
      {
        "run_id": "demo-run-1",
        "name": "math-rl-qwen3-8b",
        "base_model": "Qwen/Qwen3-8B",
        "created_at": "2026-07-29T06:02:00+00:00",
        "wandb_url": "https://wandb.ai/example/open-rl/runs/demo-run-1",
        "stoppable": True,
        "sources": ["worker", "queue"],
      },
      {
        "run_id": "demo-run-2",
        "name": "sft-gemma-warmup",
        "base_model": "google/gemma-3-4b-it",
        "created_at": "2026-07-29T07:41:00+00:00",
        "wandb_url": None,
        "stoppable": True,
        "sources": ["worker"],
      },
      {
        "run_id": "demo-run-3",
        "name": "run-9f31ab02",
        "base_model": "Qwen/Qwen3-8B",
        "created_at": "2026-07-26T18:20:00+00:00",
        "wandb_url": "https://wandb.ai/example/open-rl/runs/demo-run-3",
        "stoppable": False,
        "sources": ["checkpoint"],
      },
    ],
  }


def demo_health() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "checks": [
      {"id": "gateway", "group": "Gateway", "label": "Gateway process", "status": "ok", "detail": "distributed mode, FFT enabled"},
      {"id": "storage.redis", "group": "Storage", "label": "Redis", "status": "ok", "detail": "PING 0.8 ms — redis://demo-redis:6379"},
      {
        "id": "storage.shared",
        "group": "Storage",
        "label": "Shared filesystem",
        "status": "ok",
        "detail": "/mnt/shared/open-rl writable, 412 GiB free",
      },
      {"id": "kubernetes", "group": "Kubernetes", "label": "API server", "status": "ok", "detail": "6 pods visible in namespace open-rl-demo"},
      {
        "id": "visibility.trace",
        "group": "Visibility",
        "label": "Trace export",
        "status": "off",
        "detail": "ENABLE_GCP_TRACE=0 — tracing not configured",
      },
      {
        "id": "visibility.sampler",
        "group": "Visibility",
        "label": "vLLM sampler",
        "status": "error",
        "detail": "open-rl-sampler-demo-run-2 is failing",
      },
    ],
    "stats": [
      {"id": "runs.active", "label": "Active runs", "value": "2", "detail": "live worker or queued work"},
      {"id": "queue.requests", "label": "Queued requests", "value": "7", "detail": "across 2 queues"},
      {"id": "queue.launch", "label": "Launches pending", "value": "0", "detail": "worker launch queue"},
      {"id": "redis.memory", "label": "Redis memory", "value": "48.2 MiB", "detail": "peak 61.0 MiB · no maxmemory limit"},
      {"id": "redis.clients", "label": "Redis clients", "value": "9", "detail": "connected"},
      {"id": "gateway.rss", "label": "Gateway memory", "value": "213.4 MiB", "detail": "resident set size"},
      {"id": "storage.disk", "label": "Disk free", "value": "412.0 GiB", "detail": "of 1.0 TiB at /mnt/shared/open-rl"},
      {"id": "pods.running", "label": "Pods running", "value": "4", "detail": "1 failed · 1 pending"},
      {"id": "gpus.claimed", "label": "GPUs claimed", "value": "13/20", "detail": "across all pools"},
    ],
    "queues": [
      {"model_id": "demo-run-1", "depth": 5},
      {"model_id": "demo-run-2", "depth": 2},
    ],
  }


def demo_problems() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "problems": [
      {"severity": "error", "source": "pod/open-rl-sampler-demo-run-2", "message": "CrashLoopBackOff: CUDA out of memory (4 restarts)"},
      {"severity": "warn", "source": "pod/open-rl-trainer-demo-run-2", "message": "Pending: waiting for a free GPU claim"},
      {"severity": "warn", "source": "node/demo-l4-node-2", "message": "Node not ready"},
    ],
  }


def demo_run_detail(run_id: str, log_tail: int = 0) -> dict | None:
  run = next((r for r in demo_runs()["runs"] if r["run_id"] == run_id), None)
  if run is None:
    return None
  cluster = demo_cluster()
  pods = [pod for pod in cluster["pods"] if run_id in pod["name"]]
  gpu_claims = {}
  for pool in cluster["pools"]:
    duty = pool.get("duty")
    if duty and duty["series"] and duty["series"][-1][1].get(run_id):
      gpu_claims[pool["id"]] = duty["series"][-1][1][run_id]
  queue_depth = next((q["depth"] for q in demo_health()["queues"] if q["model_id"] == run_id), 0)
  detail = {**run, "demo": True, "notice": DEMO_NOTICE, "pods": pods, "queue_depth": queue_depth, "gpu_claims": gpu_claims}
  if log_tail:
    detail["logs"] = {pod["name"]: demo_pod_logs(pod["name"])["text"] for pod in pods}
  return detail


def demo_pod_logs(pod: str) -> dict:
  lines = [
    "[demo] fictional log output — this pod does not exist",
    f"[demo] {pod} starting",
    "[demo] loading base model weights (shard 1/4)",
    "[demo] loading base model weights (4/4) done in 41.2s",
    "[demo] worker ready, polling queue open_rl:queue:demo-run-1",
    "[demo] forward_backward batch=32 seq_len=4096 loss=0.8312",
    "[demo] optim_step lr=1e-5 grad_norm=0.42",
    "[demo] forward_backward batch=32 seq_len=4096 loss=0.7981",
  ]
  return {"demo": True, "notice": DEMO_NOTICE, "pod": pod, "container": "demo", "text": "\n".join(lines)}
