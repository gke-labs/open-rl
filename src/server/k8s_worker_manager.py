"""Kubernetes manager for dedicated per-model trainer workers.

Cluster-mode counterpart of FFTWorkerManager: instead of a local subprocess, each
FFT model gets its own trainer worker pod, labeled with a stable per-model id.
The pod spec comes from a ConfigMap-mounted YAML template; this class only stamps the
per-model name, labels, job-id env, and --model-id argument. The labels follow
the time-slicing convention used by the node-local snapshot agent. DRA pinning
is handled by the shared ResourceClaim in the pod template; the accel-timeslicer
coordinates which colocated worker process may access CUDA.

This module is part of the cluster extra; importing it assumes Kubernetes
dependencies are installed.
"""

import copy
import os
import re
import time
from typing import Any

import yaml
from kubernetes import client, config

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

POD_NAME_PREFIX = "open-rl-trainer-"
TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
# Label values allow at most 63 chars of [a-z0-9A-Z-_.]; we also reuse the
# sanitized id in the pod name, which is stricter (lowercase DNS).
_LABEL_SAFE = re.compile(r"[^a-z0-9-]+")


def sanitize_job_id(model_id: str) -> str:
  cleaned = _LABEL_SAFE.sub("-", model_id.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {model_id!r} has no label-safe characters")
  return cleaned[:63]


class KubernetesFFTWorkerManager:
  """Runs one trainer worker pod per FFT model."""

  def __init__(self, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    trainer_path = os.getenv("OPEN_RL_TRAINER_POD_TEMPLATE") or os.getenv("OPEN_RL_WORKER_POD_TEMPLATE")
    if not trainer_path:
      raise RuntimeError("OPEN_RL_WORKER_MANAGER=kubernetes requires OPEN_RL_TRAINER_POD_TEMPLATE or OPEN_RL_WORKER_POD_TEMPLATE")
    with open(trainer_path, encoding="utf-8") as f:
      self.trainer_template: dict[str, Any] = yaml.safe_load(f)

    sampler_path = os.getenv("OPEN_RL_SAMPLER_POD_TEMPLATE") or trainer_path
    with open(sampler_path, encoding="utf-8") as f:
      self.sampler_template: dict[str, Any] = yaml.safe_load(f)

    self.pod_template = self.trainer_template

    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "default")

    if core_api is None:
      config.load_incluster_config()
      core_api = client.CoreV1Api()
    self.core_api = core_api

  def launch(self, model_id: str) -> None:
    self.launch_trainer(model_id)

  def launch_trainer(self, model_id: str) -> None:
    self._launch_pod(model_id, role="trainer")

  def launch_sampler(self, model_id: str) -> None:
    self._launch_pod(model_id, role="sampler")

  def _launch_pod(self, model_id: str, role: str) -> None:
    job_id = sanitize_job_id(model_id)
    prefix = "open-rl-trainer-" if role == "trainer" else "open-rl-sampler-"
    pod_name = prefix + job_id

    existing = self.read_pod(pod_name)
    if existing is not None:
      if existing.status.phase not in TERMINAL_POD_PHASES:
        return
      self.delete_pod_and_wait(pod_name)

    try:
      pod_body = self.render_pod(pod_name, model_id, job_id, role=role)
      self.core_api.create_namespaced_pod(namespace=self.namespace, body=pod_body)
    except Exception as exc:
      if getattr(exc, "status", None) != 409:
        raise

  def shutdown(self, model_id: str) -> None:
    job_id = sanitize_job_id(model_id)
    for prefix in ("open-rl-trainer-", "open-rl-sampler-"):
      pod_name = prefix + job_id
      try:
        self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
      except Exception as exc:
        if getattr(exc, "status", None) != 404:
          raise

  def shutdown_all(self) -> None:
    pass

  def render_pod(
    self,
    pod_name: str,
    model_id: str,
    job_id: str,
    role: str = "trainer",
  ) -> dict[str, Any]:
    from server.worker_manager import _fetch_metadata_from_store

    meta = _fetch_metadata_from_store(model_id)
    base_tmpl = self.trainer_template if role == "trainer" else self.sampler_template
    pod = copy.deepcopy(base_tmpl)
    metadata = pod.setdefault("metadata", {})
    metadata["name"] = pod_name
    app_label = "open-rl-trainer-worker" if role == "trainer" else "open-rl-sampler-worker"
    role_group = TRAINER_TIME_SLICE_GROUP if role == "trainer" else SAMPLER_TIME_SLICE_GROUP
    role_job_id = workload_job_id(role, job_id)
    metadata.setdefault("labels", {}).update(
      {
        "app": app_label,
        "accel-timeslicer": "true",
        "timeslice.io/group": role_group,
        "timeslice.io/job-id": role_job_id,
      }
    )

    container = pod["spec"]["containers"][0]
    worker_image = os.getenv("OPEN_RL_WORKER_IMAGE")
    if worker_image:
      container["image"] = worker_image
    if role == "sampler":
      ft_type = meta.fine_tuning_type if (meta and hasattr(meta, "fine_tuning_type")) else None
      is_lora = (ft_type == "lora") if ft_type is not None else False
      sampler_module = "server.lora_sampler" if is_lora else "server.vllm_sampler"
      container["command"] = ["uv", "run", "python", "-u", "-m", sampler_module]
    container.setdefault("args", []).extend(["--model-id", model_id])
    if meta and meta.base_model:
      set_env(container, "BASE_MODEL", meta.base_model)
      set_env(container, "OPEN_RL_BASE_MODEL", meta.base_model)
      if "gemma-4" in meta.base_model.lower() or "gemma4" in meta.base_model.lower():
        set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", "Gemma4ForCausalLM")
    arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
    if arch_override:
      set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", arch_override)
    # Keep env aligned with labels so process discovery and llm-d target the
    # same workload identity.
    set_env(container, "OPEN_RL_TIME_SLICE_JOB_ID", role_job_id)
    set_env(container, "OPEN_RL_TIME_SLICE_GROUP", role_group)
    from server.model_metadata import WeightSyncConfig

    weight_sync_cfg = meta.weight_sync_config if meta else WeightSyncConfig()
    set_env(container, "OPEN_RL_WEIGHT_SYNC_STRATEGY", weight_sync_cfg.strategy)
    if weight_sync_cfg.strategy == "delta":
      set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT", weight_sync_cfg.delta_format)
      set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD", weight_sync_cfg.delta_apply_method)
    return pod

  def read_pod(self, pod_name: str) -> Any | None:
    try:
      return self.core_api.read_namespaced_pod(name=pod_name, namespace=self.namespace)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return None
      raise

  def delete_pod_and_wait(self, pod_name: str, timeout: float = 60.0) -> None:
    self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
    deadline = time.monotonic() + timeout
    while self.read_pod(pod_name) is not None:
      if time.monotonic() > deadline:
        raise RuntimeError(f"pod {pod_name} did not terminate within {timeout:.0f}s; cannot relaunch worker")
      time.sleep(0.5)


def set_env(container: dict[str, Any], name: str, value: str) -> None:
  env = container.setdefault("env", [])
  for item in env:
    if item.get("name") == name:
      item.clear()
      item.update({"name": name, "value": value})
      return
  env.append({"name": name, "value": value})
