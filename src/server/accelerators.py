"""What differs between GPU and TPU workers.

A client lists the accelerators a model's trainer and sampler may run on, most
preferred first, with the openrl.trainer_accel_prefs and
openrl.sampler_accel_prefs settings. Both default to gpu. The lists are fixed
at creation and stored on the model's metadata record.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from training.types import FineTuningType

Accelerator = Literal["gpu", "tpu"]


def parse_accel_prefs(value: Any) -> Any:
  # TINKER_TAGS splits on commas, so a tag separates entries with | instead.
  if isinstance(value, str):
    value = [entry.strip().lower() for entry in value.replace("|", ",").split(",")]
  if isinstance(value, list):
    if not value:
      raise ValueError("needs at least one accelerator")
    if len(set(value)) != len(value):
      raise ValueError(f"lists an accelerator twice: {value!r}")
  return value


def check_supported(
  fine_tuning_type: FineTuningType, trainer_backend: str, trainer_prefs: list[Accelerator], sampler_prefs: list[Accelerator]
) -> None:
  """Refuse a model whose workers could land where they cannot run. A list
  with tpu in any position counts, so a later fallback cannot reach it."""
  if fine_tuning_type == "full" and ("tpu" in trainer_prefs or "tpu" in sampler_prefs):
    raise ValueError("TPU supports LoRA only; full fine-tuning needs a GPU")
  if trainer_backend != "pytorch" and "tpu" in trainer_prefs:
    raise ValueError(f"openrl.trainer_backend={trainer_backend} needs a GPU trainer")


@dataclass(frozen=True)
class PodSpec:
  """What a scheduler-mode worker pod needs for its accelerator."""

  image_env: str
  default_image: str | None  # None: the image env var must be set
  toleration_key: str
  workload_type: str | None  # None leaves the Workload CRD's default, GPU
  env_configmap_env: str  # names an optional ConfigMap of worker env vars
  env: dict[str, str] = field(default_factory=dict)
  volumes: list[dict[str, Any]] = field(default_factory=list)
  volume_mounts: list[dict[str, Any]] = field(default_factory=list)


def pod_spec(accelerator: Accelerator, role: str) -> PodSpec:
  if accelerator == "gpu":
    return PodSpec("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest", "nvidia.com/gpu", None, "OPEN_RL_GPU_WORKER_ENV_CONFIGMAP")
  # GKE taints TPU nodes with google.com/tpu.
  if role == "trainer":
    return PodSpec("OPEN_RL_TPU_TRAINER_IMAGE", None, "google.com/tpu", "TPU", "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP", env={"OPEN_RL_DEVICE": "tpu"})
  # vllm-tpu needs more shared memory than a container's default /dev/shm.
  return PodSpec(
    "OPEN_RL_TPU_SAMPLER_IMAGE",
    None,
    "google.com/tpu",
    "TPU",
    "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP",
    volumes=[{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
    volume_mounts=[{"name": "dshm", "mountPath": "/dev/shm"}],
  )
