"""What differs between GPU and TPU workers.

A client names the accelerator a model's trainer and sampler run on with the
openrl.trainer_accel_prefs and openrl.sampler_accel_prefs settings. Each takes
one accelerator and defaults to gpu. Both are fixed at creation and stored on
the model's metadata record.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from training.types import FineTuningType

Accelerator = Literal["gpu", "tpu"]


def parse_accelerator(value: Any) -> Any:
  """Case and surrounding whitespace do not matter in a tag or user_metadata value."""
  return value.strip().lower() if isinstance(value, str) else value


def check_supported(fine_tuning_type: FineTuningType, trainer_backend: str, trainer: Accelerator, sampler: Accelerator, exclusive: bool) -> None:
  """Refuse a model whose workers could land where they cannot run."""
  # Time slicing parks FFT workers with vLLM sleep mode and CPU offload, which TPU workers lack.
  if fine_tuning_type == "full" and not exclusive and "tpu" in (trainer, sampler):
    raise ValueError("Full fine-tuning on TPU needs openrl.exclusive=true")
  # A job's own trainer image may run on TPU if it carries torch_tpu.
  if trainer_backend == "automodel" and trainer == "tpu":
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


def pod_spec(accelerator: Accelerator, role: str) -> PodSpec:
  if accelerator == "gpu":
    return PodSpec("OPEN_RL_WORKER_IMAGE", "ghcr.io/openrl-dev/openrl/server:latest", "nvidia.com/gpu", None, "OPEN_RL_GPU_WORKER_ENV_CONFIGMAP")
  # GKE taints TPU nodes with google.com/tpu. Both roles outgrow a container's 64 MB /dev/shm: torch_tpu writes a
  # copy of every compiled program there (its tier-2 cache, always on with the tier-3 cache), and vLLM on TPU uses
  # it for its own shared memory too.
  shm = [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}}]
  shm_mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]
  # The trainer runs torch_tpu's fused eager mode, which runs a training step
  # about twice as fast as the default op-by-op mode.
  if role == "trainer":
    return PodSpec(
      "OPEN_RL_TPU_TRAINER_IMAGE",
      None,
      "google.com/tpu",
      "TPU",
      "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP",
      env={"OPEN_RL_DEVICE": "tpu", "TORCH_TPU_DEFER_AND_FUSE": "1"},
      volumes=shm,
      volume_mounts=shm_mounts,
    )
  return PodSpec(
    "OPEN_RL_TPU_SAMPLER_IMAGE",
    None,
    "google.com/tpu",
    "TPU",
    "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP",
    volumes=shm,
    volume_mounts=shm_mounts,
  )
