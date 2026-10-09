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
import hashlib
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from kubernetes import client, config

from server.estimator import Footprint, footprint
from server.worker_manager import base_model_of, owner_id, runtime_of, worker_args, worker_env, worker_module

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"

# With openrl.sampler_router=llmd every sampler of a set serves HTTP on
# SAMPLER_HTTP_PORT, and the set's first sampler also runs the llm-d router:
# Envoy on ROUTER_PORT, asking the endpoint picker beside it which replica
# should serve each request. The picker finds the set by SAMPLER_SET_LABEL.
SAMPLER_HTTP_PORT = 8000
ROUTER_PORT = 8081
SAMPLER_SET_LABEL = "openrl.io/sampler-set"
ROUTER_LABEL = "openrl.io/sampler-router"
ROUTER_CONFIG = "openrl-llmd-router"
ROUTER_LOOKUP_TIMEOUT = 10


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

  @property
  def owner(self) -> str:
    return owner_id(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora, self.index)

  @property
  def routed(self) -> bool:
    return self.role == "sampler" and self.meta.sampler_router == "llmd"


def sampler_set(runtime: str) -> str:
  """A label-safe name for a runtime's samplers."""
  return "set-" + hashlib.sha256(runtime.encode()).hexdigest()[:16]


def replicas_of(model_id: str, role: str) -> int:
  """How many workers of this role the model asked for. Each sampler replica
  is its own Workload draining the shared sampling queue."""
  if role != "sampler":
    return 1
  meta, _, _ = runtime_of(model_id)
  return meta.sampler_replicas


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  exclusive = not meta.shares_gpu()
  devices = meta.trainer_gpus if role == "trainer" else 1
  size = footprint(base_model, meta.fine_tuning_type, role)
  # Each torchrun rank is its own process with its own host memory.
  size = replace(size, host_request_bytes=size.host_request_bytes * devices, host_limit_bytes=size.host_limit_bytes * devices)
  return Worker(role, runtime, base_model, is_lora, exclusive, meta, size, devices)


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
  if worker.routed:
    values["OPEN_RL_SAMPLER_HTTP_PORT"] = str(SAMPLER_HTTP_PORT)
  # MAX_JOBS caps FlashInfer's JIT build, which otherwise runs one ~3GB
  # compiler per core and blows through the pod's host memory limit.
  for name in ("VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_MODEL_LEN", "OPEN_RL_TRAIN_TOKEN_BUDGET", "MAX_JOBS"):
    if os.getenv(name):
      values[name] = os.environ[name]
  # No other worker shares an exclusive worker's GPUs, so it never parks.
  if worker.exclusive:
    values["OPEN_RL_TIME_SLICING"] = "off"
  env: list[dict[str, Any]] = [{"name": name, "value": value} for name, value in values.items()]
  env.append({"name": "OPEN_RL_ACCEL_TIMESLICER_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}})
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

  if worker.routed:
    add_router(template, worker)

  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    template["spec"]["containers"][0]["imagePullPolicy"] = pull_policy
  return template


def add_router(template: dict[str, Any], worker: Worker) -> None:
  """Label the sampler into its set and open its HTTP port; the set's first
  sampler also gets the router. The router's config and its permission to
  watch pods come from k8s/deploy/llmd-router."""
  name = sampler_set(worker.runtime)
  labels = {SAMPLER_SET_LABEL: name}
  sampler = template["spec"]["containers"][0]
  sampler["ports"] = [{"name": "http", "containerPort": SAMPLER_HTTP_PORT}]
  sampler["readinessProbe"] = {"httpGet": {"path": "/health", "port": SAMPLER_HTTP_PORT}, "periodSeconds": 10}
  if worker.index == 0:
    labels[ROUTER_LABEL] = name
    pod_identity = [
      {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
      {"name": "NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
    ]
    template["spec"]["serviceAccountName"] = ROUTER_CONFIG
    template["spec"]["containers"] += [
      {
        "name": "router-proxy",
        "image": os.getenv("OPEN_RL_LLMD_PROXY_IMAGE", "docker.io/envoyproxy/envoy:distroless-v1.33.2"),
        "args": ["--service-node", "envoy-sidecar", "--log-level", "warn", "--concurrency", "4", "-c", "/etc/envoy/envoy.yaml"],
        "ports": [{"name": "router", "containerPort": ROUTER_PORT}],
        "readinessProbe": {"httpGet": {"path": "/ready", "port": 19001}, "periodSeconds": 5},
        "volumeMounts": [{"name": "router-config", "mountPath": "/etc/envoy", "readOnly": True}],
        "resources": {"requests": {"cpu": "500m", "memory": "512Mi"}, "limits": {"memory": "1Gi"}},
      },
      {
        "name": "router-picker",
        "image": os.getenv("OPEN_RL_LLMD_PICKER_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0"),
        "args": [
          "--endpoint-selector",
          f"{SAMPLER_SET_LABEL}={name}",
          "--endpoint-target-ports",
          str(SAMPLER_HTTP_PORT),
          "--config-file",
          "/etc/router/plugins.yaml",
          "--grpc-health-port",
          "9003",
          "--zap-encoder",
          "json",
          "--tracing=false",
        ],
        "env": pod_identity,
        "readinessProbe": {"grpc": {"port": 9003, "service": "inference-extension"}, "periodSeconds": 2},
        "volumeMounts": [{"name": "router-config", "mountPath": "/etc/router", "readOnly": True}],
        "resources": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"memory": "2Gi"}},
      },
    ]
    template["spec"]["volumes"].append({"name": "router-config", "configMap": {"name": ROUTER_CONFIG}})
  template["metadata"] = {"labels": labels}


def accelerator_spec(worker: Worker) -> dict[str, Any]:
  """A torchrun group asks for its devices, each sized for a whole replica."""
  if worker.devices > 1:
    return {"mode": "MultiGPU", "devices": worker.devices, "memory": worker.footprint.accelerator}
  return {"mode": "SingleGPU", "memory": worker.footprint.accelerator}


def workload_body(worker: Worker) -> dict[str, Any]:
  return {
    "apiVersion": f"{GROUP}/{VERSION}",
    "kind": "Workload",
    "metadata": {"name": worker.name, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
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

  def __init__(self, custom_api: Any = None, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
    self.custom_api = custom_api
    # Only routed samplers need pods; the config loaded above serves it too.
    self.core_api = core_api

  def router_url(self, model_id: str) -> str | None:
    """The llm-d router on the model's first sampler, once its containers are ready."""
    _, runtime, _ = runtime_of(model_id)
    if self.core_api is None:
      self.core_api = client.CoreV1Api()
    # Bounded, so a stale API connection cannot hang the gateway's sample path.
    pods = self.core_api.list_namespaced_pod(
      self.namespace, label_selector=f"{ROUTER_LABEL}={sampler_set(runtime)}", _request_timeout=ROUTER_LOOKUP_TIMEOUT
    ).items
    for pod in pods:
      if (
        pod.status.phase == "Running"
        and pod.status.pod_ip
        and not pod.metadata.deletion_timestamp
        and any(c.type == "Ready" and c.status == "True" for c in pod.status.conditions or [])
      ):
        return f"http://{pod.status.pod_ip}:{ROUTER_PORT}"
    return None

  def ensure(self, model_id: str, role: str) -> None:
    for index in range(replicas_of(model_id, role)):
      self.ensure_workload(dataclasses.replace(describe_worker(model_id, role), index=index))

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
      if self.can_reuse(worker):
        return
      if time.monotonic() > deadline:
        raise RuntimeError(f"workload {worker.name} has been terminating for over three minutes")
      time.sleep(2)

  def can_reuse(self, worker: Worker) -> bool:
    try:
      workload = self.custom_api.get_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, worker.name)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return False  # gone since the create failed, so the next create will go through
      raise
    if workload["metadata"].get("deletionTimestamp"):
      return False
    labels = workload["spec"]["template"].get("metadata", {}).get("labels", {})
    if worker.role == "sampler" and (SAMPLER_SET_LABEL in labels) != worker.routed:
      raise ValueError(f"Shared sampler {worker.name} has a different sampler_router setting; use matching settings or openrl.exclusive=true")
    return True

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
    """Delete the owner's workloads. The scheduler's finalizer frees the seats."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    ours = [item for item in found["items"] if item["spec"]["ownerID"] == owner]
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
