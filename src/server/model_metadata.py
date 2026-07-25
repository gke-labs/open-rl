import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WeightSyncConfig:
  strategy: str = "delta"
  delta_format: str = "vllm_fused"
  delta_apply_method: str = "patch_in_place"
  enable_prefetching: bool = True

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

    raw_prefetch = get_val("OPEN_RL_WEIGHT_SYNC_ENABLE_PREFETCHING")
    if raw_prefetch is not None:
      enable_prefetching = str(raw_prefetch).lower() in ("true", "1", "yes")
    else:
      enable_prefetching = True

    return cls(
      strategy=strategy,
      delta_format=delta_fmt,
      delta_apply_method=apply_method,
      enable_prefetching=enable_prefetching,
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

  raw_prefetch = get_header("x-open-rl-weight-sync-enable-prefetching")
  if raw_prefetch is not None:
    enable_prefetching = str(raw_prefetch).lower() in ("true", "1", "yes")
  else:
    enable_prefetching = True

  return WeightSyncConfig(
    strategy=strategy,
    delta_format=delta_fmt,
    delta_apply_method=delta_apply_method,
    enable_prefetching=enable_prefetching,
  )


from dataclasses import dataclass, field


@dataclass
class TrainingModelMetadata:
  base_model: str | None
  created_at: float
  training_kind: str
  weight_sync_config: WeightSyncConfig = field(default_factory=WeightSyncConfig)
  full_config: dict[str, Any] | None = None
  lora_config: dict[str, Any] | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "TrainingModelMetadata":
    raw_cfg = data.get("weight_sync_config")
    if isinstance(raw_cfg, dict):
      cfg = WeightSyncConfig(
        strategy=raw_cfg.get("strategy", "delta"),
        delta_format=raw_cfg.get("delta_format", "vllm_fused"),
        delta_apply_method=raw_cfg.get("delta_apply_method", "patch_in_place"),
        enable_prefetching=raw_cfg.get("enable_prefetching", True),
      )
    elif isinstance(raw_cfg, WeightSyncConfig):
      cfg = raw_cfg
    else:
      cfg = WeightSyncConfig()

    return cls(
      base_model=data.get("base_model"),
      created_at=data.get("created_at", 0.0),
      training_kind=data.get("training_kind", "fft"),
      weight_sync_config=cfg,
      full_config=data.get("full_config"),
      lora_config=data.get("lora_config"),
    )

  @classmethod
  def from_env(cls, env: Any = None) -> "TrainingModelMetadata":
    """Reconstruct TrainingModelMetadata dataclass from environment variables inside a worker process."""
    get_val = (env.get if hasattr(env, "get") else None) or os.getenv
    return cls(
      base_model=get_val("BASE_MODEL") or get_val("OPEN_RL_BASE_MODEL"),
      created_at=0.0,
      training_kind=get_val("OPEN_RL_FINE_TUNING_TYPE", "fft"),
      weight_sync_config=WeightSyncConfig.from_env(env),
    )

  def to_dict(self) -> dict[str, Any]:
    res = asdict(self)
    if isinstance(self.weight_sync_config, WeightSyncConfig):
      res["weight_sync_config"] = asdict(self.weight_sync_config)
    return res
