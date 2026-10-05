"""What differs between GPU and TPU workers.

A client lists the accelerators a model's trainer and sampler may run on, most
preferred first, with the openrl.trainer_accel_prefs and
openrl.sampler_accel_prefs settings. Both default to gpu. The lists are fixed
at creation and stored on the model's metadata record.
"""

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
