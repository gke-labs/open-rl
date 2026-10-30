import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from server.store import StateStore
from training.types import TRAINER_BACKENDS, FFTConfig, FineTuningType, LoraConfig

K8S_NAME = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")

SPARSE_DELTA_VERSION = 2


def display_metadata(value: dict[str, Any]) -> dict[str, str]:
  """Only bounded, explicitly supplied labels become run display metadata."""
  return {
    key: value[key].strip()[:1024] for key in ("name", "run_name", "recipe_name", "git_rev") if isinstance(value.get(key), str) and value[key].strip()
  }


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
  recipe: Literal["rl", "sft"] | None = None
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None
  # Sessions that created, restored or sampled this model, in order. The model
  # view reads them to say whether the model's own session is still live; a
  # shared runtime's other sessions do not count.
  session_ids: list[str] = Field(default_factory=list)

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
    with jobs on the same trainer backend. A sampling session on a bare base-model
    name that does not share is kept apart from the base model's shared runtime."""
    if not self.shares_runtime():
      return f"base-{model_id}" if model_id == self.base_model else model_id
    if self.trainer_backend == "pytorch":
      return self.base_model
    if image := self.trainer_image():
      # An image ref is not label safe, so a hash of it keeps images apart.
      return f"image-{hashlib.sha256(image.encode()).hexdigest()[:10]}-{self.base_model}"
    return f"{self.trainer_backend}-{self.base_model}"

  def pool(self, model_id: str) -> str:
    """The sampling pool that serves this job's runtime. Base model names are not
    valid Kubernetes names, so those runtimes get a slug and a hash."""
    return pool_name(self.runtime(model_id))


def pool_name(runtime: str) -> str:
  """The sampling pool that serves a runtime."""
  return runtime_name(runtime, "sampler-")


def runtime_name(runtime: str, prefix: str = "") -> str:
  """A Kubernetes name for a runtime. A runtime that is not already a valid name
  keeps a hash of itself, so two runtimes never share workers or a pool."""
  if K8S_NAME.fullmatch(name := f"{prefix}{runtime}"):
    return name
  digest = hashlib.sha256(runtime.encode()).hexdigest()[:8]
  slug = re.sub(r"[^a-z0-9]+", "-", runtime.lower()).strip("-")[: 62 - len(prefix) - len(digest)].rstrip("-")
  return f"{prefix}{slug}-{digest}" if slug else f"{prefix}{digest}"


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
