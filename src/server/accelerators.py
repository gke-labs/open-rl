"""What differs between GPU and TPU workers.

A model's accelerator is fixed at creation and stored on its metadata record.
For now every model gets the deployment's: OPEN_RL_DEVICE=tpu on the API
server means TPU workers, anything else means GPU.
"""

import os
from typing import Literal

from training.types import FineTuningType

Accelerator = Literal["gpu", "tpu"]


def deployment_accelerator() -> Accelerator:
  return "tpu" if os.environ.get("OPEN_RL_DEVICE", "").strip().lower() == "tpu" else "gpu"


def check_supported(accelerator: Accelerator, fine_tuning_type: FineTuningType) -> None:
  if accelerator != "tpu":
    return
  if fine_tuning_type == "full":
    raise ValueError("TPU supports LoRA only; full fine-tuning needs a GPU")
  raise ValueError("TPU workers are not supported yet")
