"""Worker manager that creates one Workload per worker process and lets the
scheduler place it. Part of the cluster extra: importing it assumes the
Kubernetes client is installed.

Two FFT jobs and two LoRA jobs on the same base model come in:

  model_id (job)   owner               Workload (process)          shares
  fft-job 1a2b     1a2b                fft-1a2b-trainer            nothing
  fft-job 3c4d     3c4d                fft-3c4d-trainer            nothing
  lora-job 5e6f    qwen-qwen2-5-0-5b-<hash>   lora-qwen-qwen2-5-0-5b-<hash>-0-trainer   one runtime
  lora-job 7a8b    qwen-qwen2-5-0-5b-<hash>   lora-qwen-qwen2-5-0-5b-<hash>-0-trainer   with 5e6f
  automodel 9c0d   automodel-qwen-qwen2-5-0-5b-<hash>   lora-automodel-qwen-qwen2-5-0-5b-<hash>-0-trainer   one Automodel
                                                                                  runtime, apart from PyTorch's
  image 1d2e       image-<hash>-qwen-qwen2-5-0-5b-<hash>   lora-image-<hash>-qwen-qwen2-5-0-5b-<hash>-0-trainer   one runtime
                                                                                  per trainer image

A base model is not a valid Kubernetes name, so its owner keeps a hash of the
runtime, and no two runtimes share an owner. The Workload name is what the pod
label and the time-slicer call job_id.
"""

import dataclasses
import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from kubernetes import client, config

from server.accelerators import PodSpec, pod_spec
from server.estimator import Footprint, footprint
from server.lora_snapshots import snapshot_root
from server.sampling_dispatcher import revision_file, tmp_root
from server.store import get_state_store
from server.vllm_options import ENGINE_ENV
from server.worker_manager import (
  FAILED,
  READY,
  STARTING,
  STOPPED,
  TRAINER_MODULE,
  WAITING_FOR_PLACEMENT,
  base_model_of,
  owner_id,
  runtime_of,
  worker_args,
  worker_env,
)

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"

EPP_IMAGE = "ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0"
ENVOY_IMAGE = "docker.io/envoyproxy/envoy:distroless-v1.33.2"
ROUTER_CONFIG = "/etc/sampler-router"
# On a pool's Deployment and NetworkPolicy, so release_owner finds them after the pool's Workloads are gone.
OWNER_LABEL = "openrl.io/owner"
PLACEMENT_HISTORY, PLACEMENT_HISTORY_SECONDS = 200, 7 * 24 * 3600


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
  pod_spec: PodSpec
  # GPUs the worker drives as one torchrun group.
  devices: int = 1

  # Which replica of its role this is.
  index: int = 0

  # The sampling pool a sampler serves.
  pool: str | None = None

  @property
  def replicas(self) -> int:
    # A sampler on its trainer's GPU is its pool's only replica, since only one fits beside the trainer.
    if self.role != "sampler" or self.meta.shares_gpu():
      return 1
    return self.meta.sampler_replicas

  @property
  def routed(self) -> bool:
    # A pool several jobs share is always routed, so a job that asks for more replicas only adds them. A pool of one
    # replica that a job owns alone has nothing to route, so that replica serves the pool itself.
    return self.role == "sampler" and (self.replicas > 1 or self.meta.shares_runtime())

  @property
  def owner(self) -> str:
    return owner_id(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora, self.index)


def pool_labels(component: str, pool: str) -> dict[str, str]:
  return {"app.kubernetes.io/component": component, "app.kubernetes.io/instance": pool}


def downward(name: str, path: str) -> dict[str, Any]:
  return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": path}}}


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  pool = meta.pool(runtime) if role == "sampler" else None
  exclusive = not meta.shares_gpu()
  accelerator = meta.accelerator_for(role)
  devices = meta.trainer_gpus if role == "trainer" else 1
  size = footprint(base_model, meta.fine_tuning_type, role, accelerator)
  # Each torchrun rank is its own process with its own host memory.
  size = replace(size, host_request_bytes=size.host_request_bytes * devices, host_limit_bytes=size.host_limit_bytes * devices)
  return Worker(role, runtime, base_model, is_lora, exclusive, meta, size, pod_spec(accelerator, role), devices, pool=pool)


def pod_env(worker: Worker) -> list[dict[str, Any]]:
  """The shared worker env plus what only the cluster knows. The time-slice
  group is placement's and the scheduler stamps it."""
  tmp_dir = tmp_root()
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
  passed = (*ENGINE_ENV, "OPEN_RL_TRAIN_TOKEN_BUDGET", "MAX_JOBS")
  for name in (*passed, "OPEN_RL_SAMPLE_DEADLINE_SECONDS"):
    if os.getenv(name):
      values[name] = os.environ[name]
  # vLLM keys its torch.compile artifacts by model, GPU type and engine config.
  # Inside the container they die with the pod and every sampler compiles
  # again; on the shared volume the first sampler of a key compiles and the
  # rest load the result.
  if worker.role == "sampler":
    values["VLLM_CACHE_ROOT"] = f"{tmp_dir}/vllm-cache"
    # Triton writes each kernel to a temp dir and renames it into place, so samplers can share one cache.
    values["TRITON_CACHE_DIR"] = f"{tmp_dir}/vllm-cache/triton"
  values.update(worker.pod_spec.env)
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
  names, runs the python on its image's PATH. Automodel is GPU-only
  (check_supported); a job's image for a TPU trainer must carry torch_tpu."""
  if worker.role == "trainer" and worker.meta.trainer_backend != "pytorch":
    image = worker.meta.trainer_image() or os.getenv("OPEN_RL_AUTOMODEL_IMAGE", "ghcr.io/gke-labs/open-rl/automodel:latest")
    if worker.devices > 1:
      torchrun = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={worker.devices}"]
      return image, ["python", "-u", *torchrun, "-m", TRAINER_MODULE]
    return image, ["python", "-u", "-m", TRAINER_MODULE]
  spec = worker.pod_spec
  image = os.getenv(spec.image_env, spec.default_image)
  if not image:
    raise RuntimeError(f"{spec.image_env} is not set, so the API server has no image for this {worker.role}")
  return image, ["uv", "run", "python", "-u", "-m", TRAINER_MODULE]


def pod_template(worker: Worker) -> dict[str, Any]:
  """The complete worker pod minus placement. Node selection and claims are
  the scheduler's; it rejects a template that carries them."""
  image, command = worker_container(worker)
  spec = worker.pod_spec
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
          "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}, *spec.volume_mounts],
        }
      ],
      "volumes": [
        {
          "name": "shared-storage",
          "persistentVolumeClaim": {"claimName": os.getenv("OPEN_RL_SHARED_PVC", "open-rl-shared-pvc")},
        },
        *spec.volumes,
      ],
      "tolerations": [{"key": spec.toleration_key, "operator": "Exists", "effect": "NoSchedule"}],
    },
  }

  # Per-accelerator worker settings, so TPU tuning never reaches GPU workers.
  if configmap := os.getenv(spec.env_configmap_env):
    template["spec"]["containers"][0]["envFrom"] = [{"configMapRef": {"name": configmap}}]

  # NCCL moves data between ranks through /dev/shm, which is 64Mi by default.
  if worker.devices > 1:
    template["spec"]["containers"][0]["volumeMounts"].append({"name": "dshm", "mountPath": "/dev/shm"})
    template["spec"]["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory"}})

  container = template["spec"]["containers"][0]
  if worker.pool and not worker.routed:
    container["command"] = ["uv", "run", "python", "-u", "-m", "server.vllm_sampler"]
    container["args"] = ["--pool", worker.pool]
    container["readinessProbe"] = {"exec": {"command": ["test", "-f", "/tmp/open-rl-sampler-ready"]}, "periodSeconds": 2}
    template["metadata"] = {"labels": pool_labels("sampler", worker.pool)}
  # A pool sampler is `vllm serve`, ready only while it serves the version the pool's dispatcher expects.
  elif worker.pool:
    container["command"] = ["uv", "run", "python", "-u", "-m", "server.vllm_serve"]
    container["args"] = []
    if worker.is_lora:
      # vLLM loads each frozen adapter snapshot by name from this directory.
      container["env"] += [
        {"name": "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "value": "true"},
        {"name": "VLLM_LORA_RESOLVER_CACHE_DIR", "value": str(snapshot_root())},
      ]
    probe = (
      "import json, sys, urllib.request\n"
      'version = json.load(urllib.request.urlopen("http://127.0.0.1:8000/weight_info", timeout=1))["weight_version"]\n'
      f"sys.exit(version != open({str(revision_file(worker.pool))!r}).read())"
    )
    container["readinessProbe"] = {"exec": {"command": ["/app/.venv/bin/python", "-c", probe]}, "periodSeconds": 2, "timeoutSeconds": 2}
    template["metadata"] = {"labels": pool_labels("sampler", worker.pool)}

  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    template["spec"]["containers"][0]["imagePullPolicy"] = pull_policy
  return template


def dispatcher_body(pool: str, owner: str) -> dict[str, Any]:
  """The pool's CPU pod, named after the pool: the dispatcher, Envoy and the llm-d
  EPP. Recreate stops the old pod before the new one starts. An eviction or a
  partition can still overlap two dispatchers, and set_result's transaction keeps
  each request to one result."""
  labels = pool_labels("dispatcher", pool)
  env = [
    {"name": "REDIS_URL", "value": os.environ["REDIS_URL"]},
    {"name": "OPEN_RL_TMP_DIR", "value": tmp_root()},
    {"name": "OPEN_RL_SAMPLER_POOL", "value": pool},
    downward("OPEN_RL_WORKER_NAMESPACE", "metadata.namespace"),
  ]
  for name in ("OPEN_RL_DISPATCH_CONCURRENCY", "OPEN_RL_SAMPLE_DEADLINE_SECONDS"):
    if value := os.getenv(name):
      env.append({"name": name, "value": value})
  # Requests near the pod's peaks in a 23-job B200 run (dispatcher 0.2 cores and 145 MiB, Envoy 0.04 and 47, EPP 0.24 and
  # 113), so it fits on a busy node instead of waiting for a new one. CPU bursts above a request when the node has room.
  dispatcher = {
    "name": "dispatcher",
    # The dispatcher is a small CPU process; the API-server image carries it without the GPU stack.
    "image": os.getenv("OPEN_RL_DISPATCHER_IMAGE") or os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest"),
    "command": ["uv", "run", "python", "-u", "-m", "server.sampling_dispatcher"],
    "env": env,
    "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"memory": "2Gi"}},
  }
  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    dispatcher["imagePullPolicy"] = pull_policy
  envoy = {
    "name": "envoy",
    "image": ENVOY_IMAGE,
    "args": ["-c", f"{ROUTER_CONFIG}/envoy.yaml", "--log-level", "warn"],
    "volumeMounts": [{"name": "router", "mountPath": ROUTER_CONFIG}],
    "resources": {"requests": {"cpu": "25m", "memory": "64Mi"}, "limits": {"memory": "1Gi"}},
  }
  epp = {
    "name": "epp",
    "image": EPP_IMAGE,
    "args": [
      *("--endpoint-selector", f"app.kubernetes.io/component=sampler,app.kubernetes.io/instance={pool}", "--endpoint-target-ports", "8000"),
      *("--config-file", f"{ROUTER_CONFIG}/epp.yaml", "--metrics-endpoint-auth=false"),
    ],
    "env": [downward("POD_NAME", "metadata.name"), downward("NAMESPACE", "metadata.namespace")],
    "livenessProbe": {"grpc": {"port": 9003, "service": "liveness"}, "periodSeconds": 10},
    "resources": {"requests": {"cpu": "125m", "memory": "256Mi"}, "limits": {"memory": "4Gi"}},
    "volumeMounts": [{"name": "router", "mountPath": ROUTER_CONFIG}],
  }
  return {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": pool, "labels": {**labels, "app.kubernetes.io/managed-by": "open-rl-api-server", OWNER_LABEL: owner}},
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


def network_policy_body(pool: str, owner: str) -> dict[str, Any]:
  """Only the pool's CPU pod reaches its samplers, for inference and KV events."""
  return {
    "apiVersion": "networking.k8s.io/v1",
    "kind": "NetworkPolicy",
    "metadata": {"name": pool, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server", OWNER_LABEL: owner}},
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
  # A pool label marks a Workload whose pool has a dispatcher, which release_owner deletes with it.
  labels = {"app.kubernetes.io/managed-by": "open-rl-api-server", **(pool_labels("sampler", worker.pool) if worker.routed else {})}
  accelerator = accelerator_spec(worker)
  if worker.pod_spec.workload_type:
    accelerator["type"] = worker.pod_spec.workload_type
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
      "accelerator": accelerator,
      "workerContainerName": "worker",
      "template": pod_template(worker),
    },
  }


class SchedulerWorkerManager:
  """Runs trainer and sampler workers by creating Workload objects."""

  def __init__(self, custom_api: Any = None, apps_api: Any = None, networking_api: Any = None, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    if not os.getenv("OPEN_RL_TMP_DIR"):
      raise RuntimeError("The scheduler worker manager needs OPEN_RL_TMP_DIR on the shared volume every worker pod mounts")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
      apps_api, networking_api, core_api = client.AppsV1Api(), client.NetworkingV1Api(), client.CoreV1Api()
    self.custom_api = custom_api
    self.apps_api, self.networking_api, self.core_api = apps_api, networking_api, core_api
    # The model view may be polled; one read per few seconds is plenty.
    self._workers_cache: tuple[float, list[dict[str, Any]]] | None = None
    self.view_cache_seconds = 5.0

  def ensure(self, model_id: str, role: str) -> None:
    worker = describe_worker(model_id, role)
    if worker.routed and worker.is_lora:
      # vLLM's filesystem LoRA resolver exits at startup without its directory, and no adapter may be frozen yet.
      # The API server mounts the shared volume the replicas read it from.
      snapshot_root().mkdir(parents=True, exist_ok=True)
    for index in range(worker.replicas):
      self.ensure_workload(dataclasses.replace(worker, index=index))
    if worker.routed:
      self.ensure_dispatcher(worker.pool, worker.owner)

  def ensure_dispatcher(self, pool: str, owner: str) -> None:
    # A released pool's Deployment stays until its pod is gone, so waiting
    # here keeps an old and a new dispatcher from overlapping in the usual case.
    # The NetworkPolicy of a released pool is gone long before that.
    started, waited = time.monotonic(), False
    while not self.create_once(self.apps_api.create_namespaced_deployment, dispatcher_body(pool, owner)):
      try:
        if not self.apps_api.read_namespaced_deployment(pool, self.namespace).metadata.deletion_timestamp:
          break
      except Exception as exc:
        if getattr(exc, "status", None) != 404:
          raise
      if time.monotonic() > started + 180:
        raise RuntimeError(f"dispatcher {pool} has been terminating for over three minutes")
      if not waited:
        # A warning, because the API server configures no logging and drops info lines.
        logger.warning("Dispatcher %s of a released pool is still terminating, so this sampling session waits up to 180s for it", pool)
        waited = True
      time.sleep(2)
    if waited:
      logger.warning("Waited %.0fs for the old dispatcher %s to terminate", time.monotonic() - started, pool)
    self.create_once(self.networking_api.create_namespaced_network_policy, network_policy_body(pool, owner))

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
    A pool's CPU pod goes first. It is found through the samplers, or by its owner label once they are gone."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    ours = [item for item in found["items"] if item["spec"]["ownerID"] == owner]
    record_placements(ours)
    pools = {item["metadata"]["labels"].get("app.kubernetes.io/instance") for item in ours} - {None}
    owned = f"{OWNER_LABEL}={owner}"
    pools |= {d.metadata.name for d in self.apps_api.list_namespaced_deployment(self.namespace, label_selector=owned).items}
    pools |= {p.metadata.name for p in self.networking_api.list_namespaced_network_policy(self.namespace, label_selector=owned).items}
    for pool in pools:
      for delete in (self.networking_api.delete_namespaced_network_policy, self.apps_api.delete_namespaced_deployment):
        try:
          delete(pool, self.namespace, propagation_policy="Foreground")
        except Exception as exc:
          if getattr(exc, "status", None) != 404:
            raise
    for item in ours:
      self.delete_workload(item["metadata"]["name"])
    return {item["spec"]["modelID"] for item in ours}

  def sampler_exists(self, model_id: str) -> bool:
    """Whether the model's pool is up: a sampler Workload of its runtime not being deleted and, when the pool is routed,
    its dispatcher Deployment."""
    _, runtime, _ = runtime_of(model_id)
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    samplers = [
      item
      for item in found["items"]
      if item["spec"].get("role") == "sampler" and item["spec"]["modelID"] == runtime and not item["metadata"].get("deletionTimestamp")
    ]
    if not samplers:
      return False
    pool = samplers[0]["metadata"].get("labels", {}).get("app.kubernetes.io/instance")
    if pool is None:
      return True
    try:
      return not self.apps_api.read_namespaced_deployment(pool, self.namespace).metadata.deletion_timestamp
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise
      return False

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
    scheduler wrote into Workload status, plus whether a running sampler's
    pod is Ready and its pool's dispatcher has a pod available."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    now = time.time()
    workers = []
    sampler_pods = None
    dispatchers: dict[str, bool] = {}
    for item in found["items"]:
      spec, status, meta = item["spec"], item.get("status") or {}, item["metadata"]
      state, detail = worker_state(
        phase=status.get("phase") or "Pending",
        reason=status.get("reason") or "",
        node=status.get("nodeName"),
        deleting=bool(meta.get("deletionTimestamp")),
      )
      if state == READY and spec["role"] == "sampler":
        if sampler_pods is None:
          pods = self.core_api.list_namespaced_pod(self.namespace, label_selector="app.kubernetes.io/component=sampler", _request_timeout=10)
          sampler_pods = {pod.metadata.name: pod for pod in pods.items}
        state, detail = sampler_state(detail, sampler_pods.get(status.get("podName")))
      # The readiness probe reads a revision file that outlives the pool's dispatcher, and without the dispatcher no sample is sent.
      if state == READY and spec["role"] == "sampler" and (pool := meta.get("labels", {}).get("app.kubernetes.io/instance")):
        if pool not in dispatchers:
          dispatchers[pool] = self.dispatcher_available(pool)
        if not dispatchers[pool]:
          state, detail = STARTING, f"{detail}, dispatcher {pool} has no available pod"
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

  def dispatcher_available(self, pool: str) -> bool:
    try:
      deployment = self.apps_api.read_namespaced_deployment(pool, self.namespace)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise
      return False
    return bool(deployment.status and deployment.status.available_replicas)


def sampler_state(detail: str, pod: Any) -> tuple[str, str]:
  """A pool sampler's Workload is Running as soon as its container starts, and
  stays Running while the container crash-loops. Its readiness probe passes only
  while vLLM serves the weights the pool expects, so the pod's Ready condition is
  what makes it ready."""
  status = getattr(pod, "status", None)
  if any(c.type == "Ready" and c.status == "True" for c in getattr(status, "conditions", None) or []):
    return READY, detail
  for container in getattr(status, "container_statuses", None) or []:
    waiting = container.state.waiting if container.state else None
    if waiting and waiting.reason:
      return STARTING, f"{detail}, {waiting.reason}, restart count {container.restart_count}"
  return STARTING, f"{detail}, loading the model"


def worker_state(phase: str, reason: str, node: str | None, deleting: bool) -> tuple[str, str]:
  """Map a Workload's status onto the client vocabulary. The raw facts go
  into `detail`, for a human; `state` is what a client branches on. A node
  assignment is the placement boundary; phase Running is the readiness
  boundary (sampler_state refines it for samplers)."""
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


def epoch_seconds(timestamp: str | None) -> float | None:
  return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() if timestamp else None


def age_seconds(timestamp: str | None, now: float) -> float | None:
  created = epoch_seconds(timestamp)
  return None if created is None else max(0.0, now - created)


def placement_of(workload: dict[str, Any]) -> dict[str, Any]:
  """Where a Workload runs and the GPUs it holds, as the scheduler stamps them on its status."""
  meta, spec, status = workload["metadata"], workload["spec"], workload.get("status") or {}
  return {
    "name": meta["name"],
    "role": spec.get("role"),
    "node": status.get("nodeName"),
    "pod": status.get("podName"),
    "gpu_uuids": status.get("gpuUUIDs") or [],
    "placed_at": epoch_seconds(status.get("placedAt")),
    "released_at": epoch_seconds(status.get("releasedAt")),
    "deleting": bool(meta.get("deletionTimestamp")),
  }


def holds_gpus(placement: dict[str, Any]) -> bool:
  return placement["placed_at"] is not None and placement["released_at"] is None and not placement["deleting"]


def record_placements(workloads: list[dict[str, Any]]) -> None:
  """Keep the final placement of each Workload still holding GPUs in open_rl:placements:<runtime>, newest first, so it
  outlives the Workload. A daemon thread writes it, so a slow or failing Redis neither delays nor fails the release."""
  records: dict[str, list[str]] = {}
  for item in workloads:
    if holds_gpus(placement := placement_of(item)):
      del placement["deleting"]
      placement.update(training_kind=item["spec"].get("trainingKind"), released_at=time.time())
      records.setdefault(f"open_rl:placements:{item['spec']['modelID']}", []).append(json.dumps(placement))
  if records:
    threading.Thread(target=push_placements, args=(records,), daemon=True).start()


def push_placements(records: dict[str, list[str]]) -> None:
  try:
    for key, values in records.items():
      get_state_store().push_recent_sync(key, values, PLACEMENT_HISTORY, PLACEMENT_HISTORY_SECONDS)
  except Exception as exc:
    logger.warning("Could not record the placements of released workloads: %s", exc)
