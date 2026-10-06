"""What differs between GPU and TPU workers.

A client lists the accelerators a model's trainer and sampler may run on, most
preferred first, with the openrl.trainer_accel_prefs and
openrl.sampler_accel_prefs settings. Both default to gpu. The lists are fixed
at creation and stored on the model's metadata record.
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from training.types import FineTuningType

if TYPE_CHECKING:
  from server.model_metadata import TrainingModelMetadata

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


def accelerator_for(meta: "TrainingModelMetadata", role: str) -> Accelerator:
  # The most preferred entry until the scheduler chooses among them.
  return (meta.trainer_accel_prefs if role == "trainer" else meta.sampler_accel_prefs)[0]


def check_supported(fine_tuning_type: FineTuningType, trainer_prefs: list[Accelerator], sampler_prefs: list[Accelerator]) -> None:
  if fine_tuning_type == "full" and ("gpu" not in trainer_prefs or "gpu" not in sampler_prefs):
    raise ValueError("TPU supports LoRA only; full fine-tuning needs a GPU")
  # TODO: drop once TPU workers exist.
  if "tpu" in trainer_prefs or "tpu" in sampler_prefs:
    raise ValueError("TPU workers are not supported yet")


@dataclass
class LocalLaunch:
  """How the local worker manager starts a worker: the uv extras, the uv env
  directory relative to the repo (None keeps the repo's .venv), and env vars
  added to the process."""

  extras: list[str]
  env_dir: str | None = None
  env: dict[str, str] = field(default_factory=dict)


def local_launch(accelerator: Accelerator, role: str) -> LocalLaunch:
  if accelerator == "gpu":
    return LocalLaunch(["gpu"] if role == "trainer" else ["gpu", "vllm"])
  if role != "trainer":
    raise NotImplementedError("TPU samplers are not supported yet")
  # The tpu extra conflicts with the GPU packages, so it gets its own env.
  # torch_tpu skips registering its device when backend autoload is off.
  env = {"OPEN_RL_DEVICE": "tpu", "TORCH_DEVICE_BACKEND_AUTOLOAD": "1"}
  if chips := os.getenv("TRAINER_TPU_VISIBLE_CHIPS"):
    env.update(tpu_chip_env(chips))
  return LocalLaunch(["tpu"], ".venv-tpu-trainer", env)


def tpu_chip_env(chips: str) -> dict[str, str]:
  """The libtpu env that pins a process to one chip, like CUDA_VISIBLE_DEVICES.
  libtpu gives a process one chip or the whole host and refuses subsets."""
  chip = chips.strip()
  if not chip.isdigit():
    raise ValueError(f"A TPU worker takes one chip index, got {chips!r}: libtpu refuses multi-chip subsets of a host")
  return {"TPU_VISIBLE_CHIPS": chip, "TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1"}
