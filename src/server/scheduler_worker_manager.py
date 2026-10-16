"""Worker manager that creates one Workload per worker process and lets the
scheduler place it. Part of the cluster extra: importing it assumes the
Kubernetes client is installed.

Two FFT jobs and two LoRA jobs on the same base model come in:

  model_id (job)   owner               Workload (process)          shares
  fft-job 1a2b     1a2b                fft-1a2b-trainer            nothing
  fft-job 3c4d     3c4d                fft-3c4d-trainer            nothing
  lora-job 5e6f    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   one runtime
  lora-job 7a8b    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   with 5e6f
  automodel 9c0d   automodel-qwen-qwen2-5-0-5b   lora-automodel-qwen-qwen2-5-0-5b-0-trainer   one Automodel
                                                                                  runtime, apart from PyTorch's
  image 1d2e       image-<hash>-qwen-qwen2-5-0-5b   lora-image-<hash>-qwen-qwen2-5-0-5b-0-trainer   one runtime
                                                                                  per trainer image

The Workload name is what the pod label and the time-slicer call job_id.
"""

import dataclasses
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from kubernetes import client, config

from server.estimator import Footprint, footprint
from server.sampling_dispatcher import revision_file
from server.vllm_options import serve_args
from server.worker_manager import (
  FAILED,
  READY,
  STARTING,
  STOPPED,
  WAITING_FOR_PLACEMENT,
  base_model_of,
  owner_id,
  runtime_of,
  worker_args,
  worker_env,
  worker_module,
)

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"

EPP_IMAGE = "ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0"
ENVOY_IMAGE = "docker.io/envoyproxy/envoy:distroless-v1.33.2"
ROUTER_CONFIG = "/etc/sampler-router"


def workload_name(role: str, owner: str, is_lora: bool, index: int = 0) -> str:
  # A second compatible request renders the same name, and the create's
  # AlreadyExists is the reuse. The index numbers sampler replicas; the first
  # keeps the name it always had.
  if is_lora:
    return f"lora-{owner}-{index}-{role}"
  return f"fft-{owner}-{role}" if index == 0 else f"fft-{owner}-{role}-{index}"


@dataclass(frozen=True)
class Worker:
  """Everything the API server knows about one worker process before placement."""

  role: str
  runtime: str
  base_model: str
  is_lora: bool
  exclusive: bool
  meta: Any
  footprint: Footprint
  # GPUs the worker drives as one torchrun group.
  devices: int = 1

  # Which replica of its role this is.
  index: int = 0

  # The sampling pool a sampler serves.
  pool: str | None = None

  @property
  def owner(self) -> str:
    return owner_id(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora, self.index)


def replicas_of(model_id: str, role: str) -> int:
  """How many workers of this role the model asked for. Each sampler replica
  is its own Workload in the model's sampling pool."""
  if role != "sampler":
    return 1
  meta, _, _ = runtime_of(model_id)
  return meta.sampler_replicas


def pool_labels(component: str, pool: str) -> dict[str, str]:
  return {"app.kubernetes.io/component": component, "app.kubernetes.io/instance": pool}


def pool_selector(component: str, pool: str) -> str:
  return ",".join(f"{key}={value}" for key, value in pool_labels(component, pool).items())


def downward(name: str, path: str) -> dict[str, Any]:
  return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": path}}}


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  pool = meta.pool(runtime) if role == "sampler" else None
  # A pool sampler keeps its GPUs between batches; nothing time-slices it.
  exclusive = not meta.shares_gpu() or role == "sampler"
  devices = meta.trainer_gpus if role == "trainer" else 1
  size = footprint(base_model, meta.fine_tuning_type, role)
  # Each torchrun rank is its own process with its own host memory.
  size = replace(size, host_request_bytes=size.host_request_bytes * devices, host_limit_bytes=size.host_limit_bytes * devices)
  return Worker(role, runtime, base_model, is_lora, exclusive, meta, size, devices, pool=pool)


def pod_env(worker: Worker) -> list[dict[str, Any]]:
  """The shared worker env plus what only the cluster knows. The time-slice
  group is placement's and the scheduler stamps it."""
  tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")
  values = {
    "REDIS_URL": os.environ["REDIS_URL"],
    "OPEN_RL_TMP_DIR": tmp_dir,
    "HF_HOME": os.getenv("HF_HOME", f"{tmp_dir}/huggingface"),
    **worker_env(worker.meta, worker.base_model, worker.runtime, worker.is_lora, worker.role),
    "OPEN_RL_WORKLOAD_ID": worker.name,
    # The llmd snapshot agent still discovers processes by the older name.
    "OPEN_RL_TIME_SLICE_JOB_ID": worker.name,
    "OPEN_RL_ACCEL_TIMESLICER_PORT": os.getenv("OPEN_RL_ACCEL_TIMESLICER_PORT", "9753"),
  }
  # MAX_JOBS caps FlashInfer's JIT build, which otherwise runs one ~3GB
  # compiler per core and blows through the pod's host memory limit.
  for name in ("VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_MODEL_LEN", "OPEN_RL_TRAIN_TOKEN_BUDGET", "MAX_JOBS"):
    if os.getenv(name):
      values[name] = os.environ[name]
  # No other worker shares an exclusive worker's GPUs, so it never parks.
  if worker.exclusive:
    values["OPEN_RL_TIME_SLICING"] = "off"
  env: list[dict[str, Any]] = [{"name": name, "value": value} for name, value in values.items()]
  env.append(downward("OPEN_RL_ACCEL_TIMESLICER_HOST", "status.hostIP"))
  if worker.pool:
    env.append(downward("POD_IP", "status.podIP"))
  return env


def worker_container(worker: Worker) -> tuple[str, list[str]]:
  """The image and command. An Automodel trainer, or one from an image the job
  names, runs the python on its image's PATH."""
  if worker.role == "trainer" and worker.meta.trainer_backend != "pytorch":
    image = worker.meta.trainer_image() or os.getenv("OPEN_RL_AUTOMODEL_IMAGE", "ghcr.io/gke-labs/open-rl/automodel:latest")
    if worker.devices > 1:
      torchrun = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={worker.devices}"]
      return image, ["python", "-u", *torchrun, "-m", worker_module(worker.role)]
    return image, ["python", "-u", "-m", worker_module(worker.role)]
  image = os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest")
  return image, ["uv", "run", "python", "-u", "-m", worker_module(worker.role)]


def pod_template(worker: Worker) -> dict[str, Any]:
  """The complete worker pod minus placement. Node selection and claims are
  the scheduler's; it rejects a template that carries them."""
  image, command = worker_container(worker)
  template = {
    "spec": {
      "restartPolicy": "OnFailure",
      "containers": [
        {
          "name": "worker",
          "image": image,
          "command": command,
          "args": worker_args(worker.runtime, worker.role, worker.is_lora),
          "env": pod_env(worker),
          "resources": worker.footprint.resources,
          "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
        }
      ],
      "volumes": [
        {
          "name": "shared-storage",
          "persistentVolumeClaim": {"claimName": os.getenv("OPEN_RL_SHARED_PVC", "open-rl-shared-pvc")},
        }
      ],
      "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    },
  }

  # NCCL moves data between ranks through /dev/shm, which is 64Mi by default.
  if worker.devices > 1:
    template["spec"]["containers"][0]["volumeMounts"].append({"name": "dshm", "mountPath": "/dev/shm"})
    template["spec"]["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory"}})

  # A pool sampler is plain `vllm serve`, ready only while it serves the version the pool's dispatcher expects.
  if worker.pool:
    container = template["spec"]["containers"][0]
    container["command"] = ["uv", "run", "vllm", "serve"]
    # The pool's EPP subscribes to these events; the topic names the pod and the served model.
    kv_events = {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": "tcp://*:5557", "topic": f"kv@$(POD_IP):8000@{worker.base_model}"}
    env = {e["name"]: e.get("value") for e in container["env"]}
    container["args"] = [*serve_args(worker.base_model, worker.is_lora, env), "--kv-events-config", json.dumps(kv_events)]
    container["ports"] = [{"name": "http", "containerPort": 8000}, {"name": "kv-events", "containerPort": 5557}]
    container["readinessProbe"] = {"exec": {"command": readiness_command(worker.pool)}, "periodSeconds": 2, "timeoutSeconds": 2}
    template["metadata"] = {"labels": pool_labels("sampler", worker.pool)}

  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    template["spec"]["containers"][0]["imagePullPolicy"] = pull_policy
  return template


def readiness_command(pool: str) -> list[str]:
  probe = (
    "import json, sys, urllib.request\n"
    'version = json.load(urllib.request.urlopen("http://127.0.0.1:8000/weight_info", timeout=1))["weight_version"]\n'
    f"sys.exit(version != open({str(revision_file(pool))!r}).read())"
  )
  return ["/app/.venv/bin/python", "-c", probe]


def dispatcher_body(pool: str) -> dict[str, Any]:
  """The pool's CPU pod, named after the pool: the dispatcher, Envoy and the llm-d
  EPP. Recreate stops the old pod before the new one starts, so two dispatchers
  never send for one pool."""
  labels = pool_labels("dispatcher", pool)
  env = [
    {"name": "REDIS_URL", "value": os.environ["REDIS_URL"]},
    {"name": "OPEN_RL_TMP_DIR", "value": os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")},
    {"name": "OPEN_RL_SAMPLER_POOL", "value": pool},
    downward("OPEN_RL_WORKER_NAMESPACE", "metadata.namespace"),
  ]
  if concurrency := os.getenv("OPEN_RL_DISPATCH_CONCURRENCY"):
    env.append({"name": "OPEN_RL_DISPATCH_CONCURRENCY", "value": concurrency})
  dispatcher = {
    "name": "dispatcher",
    # The dispatcher is a small CPU process; the API-server image carries it without the GPU stack.
    "image": os.getenv("OPEN_RL_DISPATCHER_IMAGE") or os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest"),
    "command": ["uv", "run", "python", "-u", "-m", "server.sampling_dispatcher"],
    "env": env,
    "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
    "resources": {"requests": {"cpu": "250m", "memory": "512Mi"}, "limits": {"memory": "2Gi"}},
  }
  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    dispatcher["imagePullPolicy"] = pull_policy
  envoy = {
    "name": "envoy",
    "image": ENVOY_IMAGE,
    "args": ["-c", f"{ROUTER_CONFIG}/envoy.yaml", "--log-level", "warn"],
    "volumeMounts": [{"name": "router", "mountPath": ROUTER_CONFIG}],
    "resources": {"requests": {"cpu": "250m", "memory": "256Mi"}, "limits": {"memory": "1Gi"}},
  }
  epp = {
    "name": "epp",
    "image": EPP_IMAGE,
    "args": [
      *("--endpoint-selector", pool_selector("sampler", pool), "--endpoint-target-ports", "8000"),
      *("--config-file", f"{ROUTER_CONFIG}/epp.yaml", "--health-checking", "--metrics-endpoint-auth=false"),
    ],
    "env": [downward("POD_NAME", "metadata.name"), downward("NAMESPACE", "metadata.namespace")],
    "ports": [{"name": "grpc", "containerPort": 9002}, {"name": "grpc-health", "containerPort": 9003}, {"name": "metrics", "containerPort": 9090}],
    "livenessProbe": {"grpc": {"port": 9003, "service": "liveness"}, "periodSeconds": 10},
    "resources": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"memory": "4Gi"}},
    "volumeMounts": [{"name": "router", "mountPath": ROUTER_CONFIG}],
  }
  return {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": pool, "labels": {**labels, "app.kubernetes.io/managed-by": "open-rl-api-server"}},
    "spec": {
      "replicas": 1,
      "strategy": {"type": "Recreate"},
      "selector": {"matchLabels": labels},
      "template": {
        "metadata": {"labels": labels},
        "spec": {
          "serviceAccountName": "open-rl-sampler-dispatcher",
          "containers": [dispatcher, envoy, epp],
          "volumes": [
            {"name": "router", "configMap": {"name": "open-rl-sampler-router"}},
            {"name": "shared-storage", "persistentVolumeClaim": {"claimName": os.getenv("OPEN_RL_SHARED_PVC", "open-rl-shared-pvc")}},
          ],
        },
      },
    },
  }


def network_policy_body(pool: str) -> dict[str, Any]:
  """Only the pool's CPU pod reaches its samplers, for inference and KV events."""
  return {
    "apiVersion": "networking.k8s.io/v1",
    "kind": "NetworkPolicy",
    "metadata": {"name": pool, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
    "spec": {
      "podSelector": {"matchLabels": pool_labels("sampler", pool)},
      "policyTypes": ["Ingress"],
      "ingress": [{"from": [{"podSelector": {"matchLabels": pool_labels("dispatcher", pool)}}], "ports": [{"port": 8000}, {"port": 5557}]}],
    },
  }


def accelerator_spec(worker: Worker) -> dict[str, Any]:
  """A torchrun group asks for its devices, each sized for a whole replica."""
  if worker.devices > 1:
    return {"mode": "MultiGPU", "devices": worker.devices, "memory": worker.footprint.accelerator}
  return {"mode": "SingleGPU", "memory": worker.footprint.accelerator}


def workload_body(worker: Worker) -> dict[str, Any]:
  labels = {"app.kubernetes.io/managed-by": "open-rl-api-server", **(pool_labels("sampler", worker.pool) if worker.pool else {})}
  return {
    "apiVersion": f"{GROUP}/{VERSION}",
    "kind": "Workload",
    "metadata": {"name": worker.name, "labels": labels},
    "spec": {
      "role": worker.role,
      "trainingKind": "lora" if worker.is_lora else "fft",
      "exclusive": worker.exclusive,
      "modelID": worker.runtime,
      "ownerID": worker.owner,
      "accelerator": accelerator_spec(worker),
      "workerContainerName": "worker",
      "template": pod_template(worker),
    },
  }


class SchedulerWorkerManager:
  """Runs trainer and sampler workers by creating Workload objects."""

  def __init__(self, custom_api: Any = None, apps_api: Any = None, networking_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
      apps_api = client.AppsV1Api()
      networking_api = client.NetworkingV1Api()
    self.custom_api = custom_api
    # The model view may be polled; one read per few seconds is plenty.
    self._workers_cache: tuple[float, list[dict[str, Any]]] | None = None
    self.view_cache_seconds = 5.0
    self.apps_api = apps_api
    self.networking_api = networking_api

  def ensure(self, model_id: str, role: str) -> None:
    worker = describe_worker(model_id, role)
    for index in range(replicas_of(model_id, role)):
      self.ensure_workload(dataclasses.replace(worker, index=index))
    if worker.pool:
      self.ensure_dispatcher(worker.pool)

  def ensure_dispatcher(self, pool: str) -> None:
    # A released pool's Deployment stays until its pod is gone, so waiting
    # here keeps an old and a new dispatcher from overlapping. The
    # NetworkPolicy of a released pool is gone long before that.
    deadline = time.monotonic() + 180
    while not self.create_once(self.apps_api.create_namespaced_deployment, dispatcher_body(pool)):
      try:
        if not self.apps_api.read_namespaced_deployment(pool, self.namespace).metadata.deletion_timestamp:
          break
      except Exception as exc:
        if getattr(exc, "status", None) != 404:
          raise
      if time.monotonic() > deadline:
        raise RuntimeError(f"dispatcher {pool} has been terminating for over three minutes")
      time.sleep(2)
    self.create_once(self.networking_api.create_namespaced_network_policy, network_policy_body(pool))

  def create_once(self, create: Any, body: dict[str, Any]) -> bool:
    try:
      create(self.namespace, body)
      return True
    except Exception as exc:
      if getattr(exc, "status", None) != 409:
        raise
      return False

  def ensure_workload(self, worker: Worker) -> None:
    role = worker.role
    deadline = time.monotonic() + 180
    while True:
      try:
        self.custom_api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, workload_body(worker))
        logger.info("requested %s workload %s (%s, owner %s)", role, worker.name, worker.footprint.accelerator, worker.owner)
        return
      except Exception as exc:
        if getattr(exc, "status", None) != 409:
          raise
      # AlreadyExists is the reuse, unless the old one is still being deleted.
      if not self.still_deleting(worker.name):
        return
      if time.monotonic() > deadline:
        raise RuntimeError(f"workload {worker.name} has been terminating for over three minutes")
      time.sleep(2)

  def still_deleting(self, name: str) -> bool:
    try:
      workload = self.custom_api.get_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return True  # gone since the create failed, so the next create will go through
      raise
    return bool(workload["metadata"].get("deletionTimestamp"))

  def release(self, model_id: str) -> None:
    try:
      meta, runtime, _ = runtime_of(model_id)
      shared = meta.shares_runtime()
    except Exception:
      runtime, shared = model_id, False
    if shared:
      return  # a shared runtime outlives any one job
    self.release_owner(owner_id(runtime))

  def release_owner(self, owner: str) -> set[str]:
    """Delete the owner's workloads. The scheduler's finalizer frees the seats.
    A pool's CPU pod goes first, so a retry still finds it through the samplers."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    ours = [item for item in found["items"] if item["spec"]["ownerID"] == owner]
    for pool in {item["metadata"]["labels"].get("app.kubernetes.io/instance") for item in ours} - {None}:
      for delete in (self.networking_api.delete_namespaced_network_policy, self.apps_api.delete_namespaced_deployment):
        try:
          delete(pool, self.namespace, propagation_policy="Foreground")
        except Exception as exc:
          if getattr(exc, "status", None) != 404:
            raise
    for item in ours:
      self.delete_workload(item["metadata"]["name"])
    return {item["spec"]["modelID"] for item in ours}

  def close(self) -> None:
    pass  # Workloads outlive the API server; the scheduler owns them from here

  def delete_workload(self, name: str) -> None:
    try:
      self.custom_api.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise

  # -- the read-only model view (design 016, section 5.2) ---------------------

  def inspect_owner(self, owner: str) -> list[dict[str, Any]]:
    return [worker for worker in self.workers() if worker["owner"] == owner]

  def workers(self) -> list[dict[str, Any]]:
    now = time.monotonic()
    if self._workers_cache is not None and now - self._workers_cache[0] < self.view_cache_seconds:
      return self._workers_cache[1]
    workers = self.load_workers()
    self._workers_cache = (now, workers)
    return workers

  def load_workers(self) -> list[dict[str, Any]]:
    """Our Workloads, one entry per worker, in the client vocabulary. The pod
    is the scheduler's; what the API server knows about it is what the
    scheduler wrote into Workload status."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    now = time.time()
    workers = []
    for item in found["items"]:
      spec, status, meta = item["spec"], item.get("status") or {}, item["metadata"]
      state, detail = worker_state(
        phase=status.get("phase") or "Pending",
        reason=status.get("reason") or "",
        node=status.get("nodeName"),
        deleting=bool(meta.get("deletionTimestamp")),
      )
      workers.append(
        {
          "workload": meta["name"],
          "role": spec["role"],
          "owner": spec.get("ownerID"),
          "model_id": spec["modelID"],
          "state": state,
          "detail": detail,
          "node": status.get("nodeName"),
          "age_seconds": age_seconds(meta.get("creationTimestamp"), now),
        }
      )
    return workers


def worker_state(phase: str, reason: str, node: str | None, deleting: bool) -> tuple[str, str]:
  """Map a Workload's status onto the client vocabulary. The raw facts go
  into `detail`, for a human; `state` is what a client branches on. A node
  assignment is the placement boundary; phase Running is the readiness
  boundary (the model view refines it for samplers)."""
  facts = phase + (f" ({reason})" if reason else "")
  if node:
    facts += f" on {node}"
  if deleting:
    return STOPPED, f"being released; {facts}"
  if phase == "Succeeded":
    return STOPPED, facts
  if phase == "Failed":
    return FAILED, facts
  if not node:
    return WAITING_FOR_PLACEMENT, facts
  if phase == "Running":
    return READY, facts
  return STARTING, facts


def age_seconds(timestamp: str | None, now: float) -> float | None:
  if not timestamp:
    return None
  created = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
  return max(0.0, now - created)
