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
  fine_tuning_type: FineTuningType, trainer_backend: str, trainer_prefs: list[Accelerator], sampler_prefs: list[Accelerator], exclusive: bool
) -> None:
  """Refuse a model whose workers could land where they cannot run. A list
  with tpu in any position counts, so a later fallback cannot reach it."""
  # Time slicing parks a worker by offloading it to host memory, which TPU workers cannot do.
  if fine_tuning_type == "full" and not exclusive and ("tpu" in trainer_prefs or "tpu" in sampler_prefs):
    raise ValueError("Full fine-tuning on TPU needs openrl.exclusive=true")
  # A job's own trainer image may run on TPU if it carries torch_tpu.
  if trainer_backend == "automodel" and "tpu" in trainer_prefs:
    raise ValueError("openrl.trainer_backend=automodel needs a GPU trainer")


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


def pod_spec(accelerator: Accelerator, role: str, is_lora: bool) -> PodSpec:
  if accelerator == "gpu":
    return PodSpec("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest", "nvidia.com/gpu", None, "OPEN_RL_GPU_WORKER_ENV_CONFIGMAP")
  # GKE taints TPU nodes with google.com/tpu.
  if role == "trainer":
    return PodSpec("OPEN_RL_TPU_TRAINER_IMAGE", None, "google.com/tpu", "TPU", "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP", env={"OPEN_RL_DEVICE": "tpu"})
  # vllm-tpu needs more shared memory than a container's default /dev/shm. An
  # FFT sampler reloads full weights each step, which needs vllm-torchtpu.
  return PodSpec(
    "OPEN_RL_TPU_SAMPLER_IMAGE" if is_lora else "OPEN_RL_TPU_FFT_SAMPLER_IMAGE",
    None,
    "google.com/tpu",
    "TPU",
    "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP",
    volumes=[{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
    volume_mounts=[{"name": "dshm", "mountPath": "/dev/shm"}],
  )
