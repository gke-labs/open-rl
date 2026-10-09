import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.store import StateStore
from training.types import TRAINER_BACKENDS, FFTConfig, FineTuningType, LoraConfig

SPARSE_DELTA_VERSION = 2


@dataclass
class WeightSyncConfig:
  """How the trainer publishes weights each step: sparse deltas or full checkpoints.
  The sampler reads the file format and applies either."""

  strategy: str = "delta"

  @classmethod
  def from_env(cls, env: Any = None) -> "WeightSyncConfig":
    """Reconstruct WeightSyncConfig dataclass from environment variables inside a worker process."""
    get_val = (env.get if hasattr(env, "get") else None) or os.getenv

    strategy = (get_val("OPEN_RL_WEIGHT_SYNC_STRATEGY") or "delta").lower()
    if strategy not in ("delta", "full"):
      strategy = "delta"
    return cls(strategy=strategy)


class TrainingModelMetadata(BaseModel):
  # Preserve fields written by other server versions when updating a record.
  model_config = ConfigDict(extra="allow")

  base_model: str
  created_at: float = 0.0
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  sampler_replicas: int = 1
  exclusive: bool = False
  trainer_backend: str = "pytorch"
  trainer_gpus: int = 1
  trainer_cp: int = 1
  sampler_router: str | None = None
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None

  def shares_gpu(self) -> bool:
    """Whether other workers may time-slice this job's GPUs. An FFT worker
    suspends between turns. A LoRA worker cannot, so its GPUs are never shared.
    A multi-GPU trainer is a torchrun group, which cannot park either."""
    return self.fine_tuning_type != "lora" and not self.exclusive and self.trainer_gpus == 1

  def shares_runtime(self) -> bool:
    """Whether this job's workers may serve other jobs too. A LoRA worker serves
    many jobs, one adapter each. An FFT worker serves one job. A multi-GPU
    trainer is sized for its job, so it serves only that job."""
    return self.fine_tuning_type == "lora" and not self.exclusive and self.trainer_gpus == 1

  def trainer_image(self) -> str | None:
    """The image trainer_backend names, when it is an image and not a trainer."""
    return None if self.trainer_backend in TRAINER_BACKENDS else self.trainer_backend

  def runtime(self, model_id: str) -> str:
    """The id of the workers that serve this job. LoRA jobs share workers only
    with jobs on the same trainer backend."""
    if not self.shares_runtime():
      return model_id
    if self.trainer_backend == "pytorch":
      return self.base_model
    if image := self.trainer_image():
      # An image ref is not label safe, so a hash of it keeps images apart.
      return f"image-{hashlib.sha256(image.encode()).hexdigest()[:10]}-{self.base_model}"
    return f"{self.trainer_backend}-{self.base_model}"


def decode_model_metadata(raw: str | None) -> TrainingModelMetadata | None:
  if raw is None:
    return None
  data = json.loads(raw)
  if not isinstance(data, dict):
    raise ValueError("Model metadata must be a JSON object")
  # Older restores used a placeholder kind and could omit the base model.
  # Normalize that persisted format here; new creates resolve the checkpoint.
  if data.get("fine_tuning_type") == "restored":
    data["fine_tuning_type"] = "lora"
    data["base_model"] = data.get("base_model") or ""
  for key in ("full_config", "lora_config", "weight_sync_config"):
    if data.get(key) is None:
      data[key] = {}
  return TrainingModelMetadata.model_validate(data)


async def get_model_metadata(state: StateStore, model_id: str) -> TrainingModelMetadata | None:
  return decode_model_metadata(await state.get_value(f"open_rl:model_meta:{model_id}"))


async def persist_model_metadata(state: StateStore, model_id: str, metadata: TrainingModelMetadata) -> None:
  await state.set_value(f"open_rl:model_meta:{model_id}", metadata.model_dump_json())
