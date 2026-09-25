import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class Parallelism(BaseModel):
  """How many GPUs a worker drives and how it splits them: data-parallel
  replicas, each sharded over tp tensor-parallel and cp context-parallel ranks.
  The client states it as {"dp": 1, "tp": 4, "cp": 1}; every key is optional and 1."""

  model_config = ConfigDict(extra="forbid")

  dp: int = Field(default=1, ge=1)
  tp: int = Field(default=1, ge=1)
  cp: int = Field(default=1, ge=1)

  @property
  def devices(self) -> int:
    return self.dp * self.tp * self.cp

  @property
  def spec(self) -> str:
    return f"dp={self.dp},tp={self.tp},cp={self.cp}"

  @classmethod
  def of(cls, value: Any) -> "Parallelism":
    """From the metadata mapping, or from the "dp=N,tp=N,cp=N" string the header and env still use."""
    if isinstance(value, str):
      return cls.parse(value)
    if not isinstance(value, dict):
      raise ValueError(f"parallelism must be a mapping like {{'dp': 1, 'tp': 1, 'cp': 1}}, got {type(value).__name__}")
    try:
      return cls.model_validate(value)
    except ValidationError as exc:
      raise ValueError(f"parallelism {value!r}: expected dp, tp and cp as integers >= 1") from exc

  @classmethod
  def parse(cls, spec: str) -> "Parallelism":
    values: dict[str, int] = {}
    for part in spec.replace(";", ",").split(","):
      if not part.strip():
        continue
      key, sep, value = part.partition("=")
      if not sep or key.strip() not in ("dp", "tp", "cp"):
        raise ValueError(f"parallelism {spec!r}: expected dp=N,tp=N,cp=N")
      try:
        values[key.strip()] = int(value)
      except ValueError:
        raise ValueError(f"parallelism {spec!r}: {key.strip()} must be an integer") from None
    return cls(**values)


# The user_metadata key a client sets per model or per session, holding the
# {"dp", "tp", "cp"} mapping, and the header older clients send; the API
# server reads them in that order, then the server's own default.
PARALLELISM_KEY = {"trainer": "trainer", "sampler": "sampler"}
PARALLELISM_HEADER = {"trainer": "x-open-rl-trainer-parallelism", "sampler": "x-open-rl-sampler-parallelism"}
PARALLELISM_ENV = {"trainer": "OPEN_RL_TRAINER_PARALLELISM", "sampler": "OPEN_RL_SAMPLER_PARALLELISM"}


def resolve_parallelism(role: str, *user_metadata: dict[str, Any], headers: Any = None) -> Parallelism:
  """The role's parallelism from the first source that states it: the
  user_metadata dicts in the order given (per model, then per session), the
  request header, the server env; else one device."""
  for source in user_metadata:
    if (value := (source or {}).get(PARALLELISM_KEY[role])) is not None:
      return Parallelism.of(value)
  get_header = headers.get if headers is not None and hasattr(headers, "get") else (lambda k, default=None: default)
  if spec := get_header(PARALLELISM_HEADER[role]):
    return Parallelism.parse(spec)
  if spec := os.getenv(PARALLELISM_ENV[role]):
    return Parallelism.parse(spec)
  return Parallelism()


class TrainingModelMetadata(BaseModel):
  # Preserve fields written by other server versions when updating a record.
  model_config = ConfigDict(extra="allow")

  base_model: str
  created_at: float = 0.0
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  # What the client attached to the run (tinker's user_metadata): the
  # cookbook's recipe, git revision, wandb link, renderer, plus our own keys.
  user_metadata: dict[str, Any] = Field(default_factory=dict)
  trainer_parallelism: Parallelism = Field(default_factory=Parallelism)
  sampler_parallelism: Parallelism = Field(default_factory=Parallelism)
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
