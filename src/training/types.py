"""Wire types shared by the gateway, the request queue and the trainer workers.

Nothing here imports torch, so the gateway can build and validate commands
without loading a training stack.
"""

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  """One training example: input tokens and the per-position loss inputs."""

  model_input: list[int]
  loss_fn_inputs: dict[str, TensorData]

  @model_validator(mode="before")
  @classmethod
  def flatten_wire_format(cls, raw: Any) -> Any:
    """Accept the Tinker wire datum, whose model_input is a list of token chunks
    and whose loss_fn_inputs may be bare lists."""
    if not isinstance(raw, dict) or not isinstance(raw.get("model_input"), dict):
      return raw
    tokens: list[int] = []
    for chunk in raw["model_input"].get("chunks", []):
      tokens.extend(chunk.get("tokens", []))
    loss_fn_inputs = {
      key: value if isinstance(value, dict) and "data" in value else {"data": value} for key, value in raw.get("loss_fn_inputs", {}).items()
    }
    return {"model_input": tokens, "loss_fn_inputs": loss_fn_inputs}


class LoraConfig(BaseModel):
  rank: int = 16
  seed: int | None = None
  lora_alpha: int = 16
  lora_dropout: float = 0.05
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = False


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: str | None = None


FineTuningType = Literal["lora", "full"]


class SamplerWeights(BaseModel):
  """What a trainer published for the samplers: a LoRA adapter directory the
  sampler hot-loads, or a whole checkpoint it reloads."""

  kind: Literal["adapter", "checkpoint"]
  path: str
