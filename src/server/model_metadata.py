import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from server.store import StateStore
from training.types import FineTuningType


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


@dataclass
class TrainingModelMetadata:
  base_model: str
  created_at: float
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = field(default_factory=WeightSyncConfig)
  full_config: dict[str, Any] = field(default_factory=dict)
  lora_config: dict[str, Any] = field(default_factory=dict)
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None
  total_steps_completed: int = 0
  max_steps: int | None = None
  tenant_id: str = "default"

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "TrainingModelMetadata":
    raw_cfg = data.get("weight_sync_config")
    if isinstance(raw_cfg, dict):
      cfg = WeightSyncConfig(
        strategy=raw_cfg.get("strategy", "delta"),
        delta_format=raw_cfg.get("delta_format", "vllm_fused"),
        delta_apply_method=raw_cfg.get("delta_apply_method", "patch_in_place"),
      )
    elif isinstance(raw_cfg, WeightSyncConfig):
      cfg = raw_cfg
    else:
      cfg = WeightSyncConfig()

    ft_type = data.get("fine_tuning_type", "lora")
    if ft_type not in {"lora", "full"}:
      raise ValueError(f"Invalid fine_tuning_type: {ft_type!r}")
    if not isinstance(data.get("base_model"), str) or not data["base_model"]:
      raise ValueError("Model metadata must specify base_model")

    return cls(
      base_model=data["base_model"],
      created_at=data.get("created_at", 0.0),
      fine_tuning_type=ft_type,
      weight_sync_config=cfg,
      full_config=data.get("full_config") or {},
      lora_config=data.get("lora_config") or {},
      status=data.get("status", "active"),
      updated_at=data.get("updated_at", data.get("created_at", 0.0)),
      completed_at=data.get("completed_at"),
      total_steps_completed=data.get("total_steps_completed", 0),
      max_steps=data.get("max_steps"),
      tenant_id=data.get("tenant_id", "default"),
    )

  @classmethod
  def from_env(cls, env: Any = None) -> "TrainingModelMetadata":
    """Reconstruct TrainingModelMetadata dataclass from environment variables inside a worker process."""
    get_val = (env.get if hasattr(env, "get") else None) or os.getenv
    ft_type = (get_val("OPEN_RL_FINE_TUNING_TYPE") or "lora").lower()
    ft_type = "full" if ft_type == "full" else "lora"
    return cls(
      base_model=str(get_val("BASE_MODEL") or get_val("OPEN_RL_BASE_MODEL") or ""),
      created_at=0.0,
      fine_tuning_type=ft_type,
      weight_sync_config=WeightSyncConfig.from_env(env),
      status="active",
      updated_at=0.0,
    )

  def to_dict(self) -> dict[str, Any]:
    res = asdict(self)
    if isinstance(self.weight_sync_config, WeightSyncConfig):
      res["weight_sync_config"] = asdict(self.weight_sync_config)
    return res


def _decode_metadata(raw: str | None) -> dict[str, Any] | None:
  if raw is None:
    return None
  data = json.loads(raw)
  if not isinstance(data, dict):
    raise ValueError("Model metadata must be a JSON object")
  if not isinstance(data.get("base_model"), str) or not data["base_model"]:
    raise ValueError("Model metadata must specify base_model")
  if data.get("fine_tuning_type", "lora") not in {"lora", "full"}:
    raise ValueError("Model metadata must specify lora or full fine_tuning_type")
  return data


async def get_model_metadata(state: StateStore, model_id: str) -> dict[str, Any] | None:
  data = _decode_metadata(await state.get_value(f"open_rl:model_meta:{model_id}"))
  if data is not None:
    data["model_id"] = model_id
  return data


def get_model_metadata_sync(state: StateStore, model_id: str) -> dict[str, Any] | None:
  data = _decode_metadata(state.get_value_sync(f"open_rl:model_meta:{model_id}"))
  if data is not None:
    data["model_id"] = model_id
  return data


async def update_model_metadata(state: StateStore, model_id: str, updates: dict[str, Any]) -> None:
  key = f"open_rl:model_meta:{model_id}"
  data = _decode_metadata(await state.get_value(key))
  if data is None:
    raise KeyError(f"Unknown model: {model_id}")
  data.update(updates)
  data["updated_at"] = time.time()
  await state.set_value(key, json.dumps(data))


async def persist_model_metadata(state: StateStore, metadata: TrainingModelMetadata) -> str:
  model_id = str(uuid.uuid4())
  await state.set_value(f"open_rl:model_meta:{model_id}", json.dumps(metadata.to_dict()))
  return model_id
