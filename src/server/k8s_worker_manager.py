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
import logging
import os
import re
import time
import uuid
from typing import Any

import yaml
from kubernetes import client, config

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

logger = logging.getLogger(__name__)

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


class KubernetesWorkerManager:
  """Runs trainer and sampler worker pods on Kubernetes."""

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

    self._resource_slice_cache: dict[str, str] = {}
    self._resource_slice_cache_time: float = 0.0
    try:
      self._discover_cluster_gpu_products()
    except Exception as exc:
      logger.debug("Initial ResourceSlice auto-scan skipped: %s", exc)

  def launch(self, model_id: str) -> None:
    self.launch_trainer(model_id)

  def launch_trainer(self, model_id: str) -> None:
    self._launch_pod(model_id, role="trainer")

  def launch_sampler(self, model_id: str) -> None:
    self._launch_pod(model_id, role="sampler")

  def _launch_pod(self, model_id: str, role: str) -> None:
    from server.worker_manager import get_model_target_info

    meta, target_id, is_lora = get_model_target_info(model_id)
    job_id = sanitize_job_id(target_id)
    prefix = "open-rl-trainer-" if role == "trainer" else "open-rl-sampler-"
    pod_name = f"{prefix}{job_id}-1"

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
    from server.worker_manager import get_model_target_info

    try:
      _, target_id, _ = get_model_target_info(model_id)
    except Exception:
      target_id = model_id

    job_id = sanitize_job_id(target_id)
    for prefix in ("open-rl-trainer-", "open-rl-sampler-"):
      pod_name = f"{prefix}{job_id}-1"
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
    from server.worker_manager import get_model_target_info

    meta, target_id, is_lora = get_model_target_info(model_id)
    if is_lora:
      return self.render_lora_pod(pod_name, model_id, target_id, job_id, role=role, meta=meta)
    return self.render_fft_pod(pod_name, model_id, target_id, job_id, role=role, meta=meta)

  def _discover_eligible_claims(self, workload_type: str, role: str, memory_tier: str | None = None) -> list[str]:
    try:
      custom_api = client.CustomObjectsApi(self.core_api.api_client)
      label_selector = f"open-rl.io/workload-type={workload_type},open-rl.io/role={role}"
      if memory_tier:
        label_selector += f",open-rl.io/memory-tier={memory_tier}"
      res = custom_api.list_namespaced_custom_object(
        group="resource.k8s.io",
        version="v1",
        namespace=self.namespace,
        plural="resourceclaims",
        label_selector=label_selector,
      )
      items = res.get("items", []) if isinstance(res, dict) else getattr(res, "items", [])
      claim_names = [item["metadata"]["name"] for item in items if isinstance(item, dict) and "metadata" in item and "name" in item["metadata"]]
      return sorted(claim_names)
    except Exception as exc:
      logger.debug("DRA label discovery failed for %s/%s (%s): %s", workload_type, role, memory_tier, exc)
      return []

  def _get_claim_locks(self) -> dict[str, set[str]]:
    locks: dict[str, set[str]] = {}
    try:
      pods = self.core_api.list_namespaced_pod(
        self.namespace,
        label_selector="app in (open-rl-trainer-worker, open-rl-sampler-worker)",
      )
      items = pods.items if hasattr(pods, "items") else pods.get("items", [])
      for pod in items:
        status_phase = ""
        metadata = {}
        if isinstance(pod, dict):
          status_phase = pod.get("status", {}).get("phase", "")
          metadata = pod.get("metadata", {})
        else:
          status_phase = getattr(getattr(pod, "status", None), "phase", "")
          metadata = getattr(pod, "metadata", {})

        if status_phase in TERMINAL_POD_PHASES:
          continue

        labels = (metadata.get("labels", {}) if isinstance(metadata, dict) else getattr(metadata, "labels", {})) or {}
        claim_name = labels.get("open-rl.io/assigned-claim")
        bound_model = labels.get("open-rl.io/bound-base-model")
        if claim_name and bound_model:
          locks.setdefault(claim_name, set()).add(bound_model)
    except Exception as exc:
      logger.debug("Failed to inspect active Pod claim locks: %s", exc)
    return locks

  def _discover_cluster_gpu_products(self) -> dict[str, str]:
    """Scan cluster ResourceSlices and return a mapping of memory_tier -> productName (cached for 15 mins)."""
    now = time.time()
    if now - self._resource_slice_cache_time < 900.0 and self._resource_slice_cache:
      return self._resource_slice_cache

    tier_map = {}
    tier_counts = {}
    tier_mem = {}
    try:
      custom_api = client.CustomObjectsApi(self.core_api.api_client)
      slices = custom_api.list_cluster_custom_object(group="resource.k8s.io", version="v1", plural="resourceslices")
      items = slices.get("items", []) if isinstance(slices, dict) else getattr(slices, "items", [])
      for item in items:
        if item.get("spec", {}).get("driver") == "gpu.nvidia.com":
          for dev in item.get("spec", {}).get("devices", []):
            attrs = dev.get("attributes", {})
            cap = dev.get("capacity", {})
            prod = attrs.get("productName", {}).get("string")
            mem_str = cap.get("memory", {}).get("value", "")

            if prod and mem_str:
              mem_mib = int(mem_str.rstrip("Mi")) if mem_str.endswith("Mi") else 0
              tier = "80gb" if mem_mib > 40000 else "24gb"
              tier_map[tier] = prod
              tier_mem[tier] = mem_mib
              tier_counts[tier] = tier_counts.get(tier, 0) + 1
    except Exception as exc:
      raise RuntimeError(f"Failed to auto-scan cluster ResourceSlices (resource.k8s.io/v1): {exc}") from exc

    if not tier_map:
      raise RuntimeError("No active GPU devices found in cluster ResourceSlices (resource.k8s.io/v1)")

    self._resource_slice_cache = tier_map
    self._resource_slice_cache_time = now
    logger.info(
      "Auto-discovered cluster GPU topology from ResourceSlices: products=%s, quantities=%s, memory=%s",
      tier_map,
      tier_counts,
      tier_mem,
    )
    return tier_map

  def _create_managed_claim(self, workload_type: str, role: str, memory_tier: str) -> str:
    custom_api = client.CustomObjectsApi(self.core_api.api_client)
    claim_id = uuid.uuid4().hex[:6]
    claim_name = f"open-rl-managed-{workload_type}-{role}-{claim_id}"

    discovered_products = self._discover_cluster_gpu_products()
    gpu_product = discovered_products.get(memory_tier)
    if not gpu_product:
      raise RuntimeError(
        f"No cluster ResourceSlice found matching VRAM memory tier {memory_tier!r}. Discovered tiers: {list(discovered_products.keys())}"
      )

    claim_manifest = {
      "apiVersion": "resource.k8s.io/v1",
      "kind": "ResourceClaim",
      "metadata": {
        "name": claim_name,
        "namespace": self.namespace,
        "labels": {
          "open-rl.io/managed-by": "open-rl-gateway",
          "open-rl.io/workload-type": workload_type,
          "open-rl.io/role": role,
          "open-rl.io/memory-tier": memory_tier,
        },
      },
      "spec": {
        "devices": {
          "requests": [
            {
              "name": "gpu",
              "exactly": {
                "deviceClassName": "gpu.nvidia.com",
                "selectors": [{"cel": {"expression": f"device.attributes['gpu.nvidia.com'].productName == '{gpu_product}'"}}],
              },
            }
          ]
        }
      },
    }

    custom_api.create_namespaced_custom_object(
      group="resource.k8s.io",
      version="v1",
      namespace=self.namespace,
      plural="resourceclaims",
      body=claim_manifest,
    )
    logger.info("Created dynamic managed DRA claim %s for %s/%s (%s)", claim_name, workload_type, role, memory_tier)
    return claim_name

  def reconcile_managed_claims(self) -> list[str]:
    """Delete dynamic managed claims that have 0 active worker pods referencing them."""
    deleted = []
    try:
      custom_api = client.CustomObjectsApi(self.core_api.api_client)
      res = custom_api.list_namespaced_custom_object(
        group="resource.k8s.io",
        version="v1",
        namespace=self.namespace,
        plural="resourceclaims",
        label_selector="open-rl.io/managed-by=open-rl-gateway",
      )
      items = res.get("items", []) if isinstance(res, dict) else getattr(res, "items", [])
      active_locks = self._get_claim_locks()
      for item in items:
        claim_name = item.get("metadata", {}).get("name")
        if claim_name and claim_name not in active_locks:
          try:
            custom_api.delete_namespaced_custom_object(
              group="resource.k8s.io",
              version="v1",
              namespace=self.namespace,
              plural="resourceclaims",
              name=claim_name,
            )
            deleted.append(claim_name)
            logger.info("Reconciled and deleted unused dynamic managed claim %s", claim_name)
          except Exception as del_exc:
            logger.debug("Failed to delete managed claim %s: %s", claim_name, del_exc)
    except Exception as exc:
      logger.debug("Reconciliation of managed claims failed: %s", exc)
    return deleted

  def resolve_claim(
    self,
    workload_type: str,
    role: str,
    target_id: str,
    memory_tier: str | None = None,
    meta: Any | None = None,
  ) -> str:
    """Resolve target DRA ResourceClaim using pure dynamic managed claim provisioning (No static fallbacks)."""
    if not memory_tier:
      from server.worker_manager import estimate_memory_tier

      base_model_name = (meta.base_model if meta and getattr(meta, "base_model", None) else None) or target_id
      memory_tier = estimate_memory_tier(base_model_name, fine_tuning_type=workload_type)

    claims = self._discover_eligible_claims(workload_type, role, memory_tier=memory_tier)
    locks = self._get_claim_locks()
    clean_target = sanitize_job_id(target_id)

    max_workers_per_claim = int(os.getenv("OPEN_RL_MAX_WORKERS_PER_CLAIM", "2"))

    # Rule 1: Base-Model Affinity (reuse existing dynamic claim if under capacity and already running clean_target)
    for c_name in claims:
      c_locks = locks.get(c_name, set())
      if clean_target in c_locks and len(c_locks) < max_workers_per_claim:
        return c_name

    # Rule 2: Reuse eligible dynamic claim matching workload_type, role, and memory_tier if under capacity
    eligible = []
    for c_name in claims:
      c_locks = locks.get(c_name, set())
      if len(c_locks) >= max_workers_per_claim:
        continue
      if workload_type == "lora" and c_locks and clean_target not in c_locks:
        continue
      eligible.append(c_name)

    if eligible:
      return sorted(eligible)[0]

    # Rule 3: Dynamically create a new managed claim for the requested memory tier (No fallback)
    return self._create_managed_claim(workload_type, role, memory_tier)

  @staticmethod
  def _inject_resource_claim(pod: dict[str, Any], role: str, claim_name: str) -> None:
    default_ref = "trainer-gpu" if role == "trainer" else "sampler-gpu"
    containers = pod["spec"].get("containers", [])
    claim_ref_name = default_ref
    if containers and isinstance(containers[0].get("resources"), dict):
      claims = containers[0]["resources"].get("claims", [])
      if claims and isinstance(claims[0], dict) and "name" in claims[0]:
        claim_ref_name = claims[0]["name"]
    pod["spec"]["resourceClaims"] = [{"name": claim_ref_name, "resourceClaimName": claim_name}]

  @staticmethod
  def _inject_container_resources(pod: dict[str, Any], memory_tier: str) -> None:
    containers = pod.get("spec", {}).get("containers", [])
    if not containers:
      return
    resources = containers[0].setdefault("resources", {})
    limits = resources.setdefault("limits", {})
    requests = resources.setdefault("requests", {})
    if memory_tier == "80gb":
      limits["memory"] = "110Gi"
      requests["memory"] = "90Gi"
      requests["cpu"] = "12"
    else:
      limits["memory"] = "40Gi"
      requests["memory"] = "20Gi"
      requests["cpu"] = "6"

  def render_lora_pod(
    self,
    pod_name: str,
    model_id: str,
    target_id: str,
    job_id: str,
    role: str = "trainer",
    meta: Any | None = None,
  ) -> dict[str, Any]:
    base_tmpl = self.trainer_template if role == "trainer" else self.sampler_template
    pod = copy.deepcopy(base_tmpl)
    metadata = pod.setdefault("metadata", {})
    metadata["name"] = pod_name
    app_label = "open-rl-trainer-worker" if role == "trainer" else "open-rl-sampler-worker"

    claim_name = self.resolve_claim("lora", role, target_id, meta=meta)
    labels = metadata.setdefault("labels", {})
    labels["app"] = app_label
    labels["accel-timeslicer"] = "false"
    labels["open-rl.io/workload-type"] = "lora"
    labels["open-rl.io/role"] = role
    labels["open-rl.io/assigned-claim"] = claim_name
    labels["open-rl.io/bound-base-model"] = sanitize_job_id(target_id)
    labels.pop("timeslice.io/group", None)
    labels.pop("timeslice.io/job-id", None)

    node_sel = pod["spec"].get("nodeSelector", {})
    if isinstance(node_sel, dict):
      node_sel.pop("cloud.google.com/gke-accelerator", None)
    self._inject_resource_claim(pod, role, claim_name)
    base_model_name = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL") or target_id
    from server.worker_manager import estimate_memory_tier

    memory_tier = estimate_memory_tier(base_model_name, fine_tuning_type="lora")
    self._inject_container_resources(pod, memory_tier)

    container = pod["spec"]["containers"][0]
    worker_image = os.getenv("OPEN_RL_WORKER_IMAGE")
    if worker_image:
      container["image"] = worker_image

    if role == "sampler":
      container["command"] = ["uv", "run", "python", "-u", "-m", "server.lora_sampler"]
    else:
      container["command"] = ["uv", "run", "python", "-u", "-m", "server.training_requests_processor"]

    container.setdefault("args", []).extend(["--model-id", target_id])
    if role == "trainer":
      container["args"].extend(["--active-tenant-set-id", f"{target_id}-1"])

    base_model = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL")
    if base_model:
      set_env(container, "BASE_MODEL", base_model)
      set_env(container, "OPEN_RL_BASE_MODEL", base_model)

    set_env(container, "OPEN_RL_ENABLE_FFT", "false")
    set_env(container, "OPEN_RL_FINE_TUNING_TYPE", "lora")

    remove_env(container, "OPEN_RL_TIME_SLICE_JOB_ID")
    remove_env(container, "OPEN_RL_TIME_SLICE_GROUP")
    remove_env(container, "OPEN_RL_ACCEL_TIMESLICER_HOST")
    remove_env(container, "OPEN_RL_ACCEL_TIMESLICER_PORT")

    from server.model_metadata import WeightSyncConfig

    weight_sync_cfg = meta.weight_sync_config if meta else WeightSyncConfig()
    set_env(container, "OPEN_RL_WEIGHT_SYNC_STRATEGY", weight_sync_cfg.strategy)
    if weight_sync_cfg.strategy == "delta":
      set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT", weight_sync_cfg.delta_format)
      set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD", weight_sync_cfg.delta_apply_method)
    return pod

  def render_fft_pod(
    self,
    pod_name: str,
    model_id: str,
    target_id: str,
    job_id: str,
    role: str = "trainer",
    meta: Any | None = None,
  ) -> dict[str, Any]:
    base_tmpl = self.trainer_template if role == "trainer" else self.sampler_template
    pod = copy.deepcopy(base_tmpl)
    metadata = pod.setdefault("metadata", {})
    metadata["name"] = pod_name
    app_label = "open-rl-trainer-worker" if role == "trainer" else "open-rl-sampler-worker"

    role_group = TRAINER_TIME_SLICE_GROUP if role == "trainer" else SAMPLER_TIME_SLICE_GROUP
    role_job_id = workload_job_id(role, job_id)
    claim_name = self.resolve_claim("full", role, target_id, meta=meta)
    metadata.setdefault("labels", {}).update(
      {
        "app": app_label,
        "accel-timeslicer": "true",
        "open-rl.io/workload-type": "full",
        "open-rl.io/role": role,
        "open-rl.io/assigned-claim": claim_name,
        "open-rl.io/bound-base-model": sanitize_job_id(target_id),
        "timeslice.io/group": role_group,
        "timeslice.io/job-id": role_job_id,
      }
    )

    node_sel = pod["spec"].get("nodeSelector", {})
    if isinstance(node_sel, dict):
      node_sel.pop("cloud.google.com/gke-accelerator", None)
    self._inject_resource_claim(pod, role, claim_name)
    base_model_name = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL") or target_id
    from server.worker_manager import estimate_memory_tier

    memory_tier = estimate_memory_tier(base_model_name, fine_tuning_type="full")
    self._inject_container_resources(pod, memory_tier)

    container = pod["spec"]["containers"][0]
    worker_image = os.getenv("OPEN_RL_WORKER_IMAGE")
    if worker_image:
      container["image"] = worker_image

    if role == "sampler":
      container["command"] = ["uv", "run", "python", "-u", "-m", "server.vllm_sampler"]

    container.setdefault("args", []).extend(["--model-id", target_id])

    base_model = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL")
    if base_model:
      set_env(container, "BASE_MODEL", base_model)
      set_env(container, "OPEN_RL_BASE_MODEL", base_model)
      if "gemma-4" in base_model.lower() or "gemma4" in base_model.lower():
        set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", "Gemma4ForCausalLM")
    arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
    if arch_override:
      set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", arch_override)

    set_env(container, "OPEN_RL_TIME_SLICE_JOB_ID", role_job_id)
    set_env(container, "OPEN_RL_TIME_SLICE_GROUP", role_group)
    set_env(container, "OPEN_RL_ENABLE_FFT", "true")
    set_env(container, "OPEN_RL_FINE_TUNING_TYPE", "full")

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


def remove_env(container: dict[str, Any], name: str) -> None:
  env = container.get("env", [])
  container["env"] = [item for item in env if item.get("name") != name]
