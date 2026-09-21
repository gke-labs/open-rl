import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.store import StateStore
from training.types import FFTConfig, FineTuningType, LoraConfig


@dataclass
class WeightSyncConfig:
  strategy: str = "delta"
  delta_format: str = "vllm_fused"
  delta_apply_method: str = "patch_in_place"

  @classmethod
  def from_env(cls, env: Any = None) -> "WeightSyncConfig":
    """Reconstruct WeightSyncConfig dataclass from environment variables inside a worker process."""
    get_val = (env.get if hasattr(env, "get") else None) or os.getenv

    strategy = (get_val("OPEN_RL_WEIGHT_SYNC_STRATEGY") or "delta").lower()
    if strategy not in ("delta", "full"):
      strategy = "delta"

    delta_fmt = (get_val("OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT") or "vllm_fused").lower()
    if delta_fmt not in ("vllm_fused", "native"):
      delta_fmt = "vllm_fused"

    apply_method = (get_val("OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD") or "patch_in_place").lower()
    if apply_method not in ("patch_in_place", "full_replace"):
      apply_method = "patch_in_place"

    return cls(
      strategy=strategy,
      delta_format=delta_fmt,
      delta_apply_method=apply_method,
    )


def extract_weight_sync_config(headers: Any = None) -> WeightSyncConfig:
  """Extract and normalize WeightSyncConfig from HTTP headers with single-location defaults."""
  if not headers:
    return WeightSyncConfig()

  get_header = headers.get if hasattr(headers, "get") else (lambda k, default=None: default)

  strategy = (get_header("x-open-rl-weight-sync-strategy") or "delta").lower()
  if strategy not in ("delta", "full"):
    strategy = "delta"

  delta_fmt = (get_header("x-open-rl-weight-sync-delta-format") or get_header("x-open-rl-weight-sync-format") or "vllm_fused").lower()
  if delta_fmt not in ("vllm_fused", "native"):
    delta_fmt = "vllm_fused"

  delta_apply_method = (
    get_header("x-open-rl-weight-sync-delta-apply-method") or get_header("x-open-rl-weight-sync-apply-method") or "patch_in_place"
  ).lower()
  if delta_apply_method not in ("patch_in_place", "full_replace"):
    delta_apply_method = "patch_in_place"

  return WeightSyncConfig(
    strategy=strategy,
    delta_format=delta_fmt,
    delta_apply_method=delta_apply_method,
  )


class TrainingModelMetadata(BaseModel):
  # Preserve fields written by other server versions when updating a record.
  model_config = ConfigDict(extra="allow")

  base_model: str
  created_at: float = 0.0
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None


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
