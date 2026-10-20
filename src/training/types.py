"""Wire types shared by the API server, the request queue and the trainer workers.

Nothing here imports torch, so the API server can build and validate commands
without loading a training stack.
"""

import os
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
# Trainers by name. Any other trainer_backend is an image.
TRAINER_BACKENDS = ("pytorch", "automodel")

# Precision a model trains and samples in. The client must name it; nothing
# falls back to another dtype when the hardware cannot run the one asked for.
Dtype = Literal["float32", "bfloat16"]
DTYPES: tuple[str, ...] = ("float32", "bfloat16")
# The dtype a static runtime (one set of workers for every model) runs in, and
# the dtype of a model stored before dtypes were recorded.
DEFAULT_DEPLOYMENT_DTYPE = "bfloat16"
# What each named trainer supports per fine-tuning type. A trainer image is not
# listed: it gets the dtype and owns the check.
#  - LoRA is bfloat16 only with the vLLM sampler: vLLM's LoRA kernels assert
#    fp16/bf16 inputs (vllm/lora/ops/triton_ops/lora_shrink_op.py). Shared
#    LoRA runtimes rely on this; allowing a second LoRA dtype there means
#    keying the runtime by dtype. The in-process torch sampler has no such
#    limit, and its one static runtime runs in OPEN_RL_DTYPE anyway.
#  - Automodel trains in bfloat16 only.
SUPPORTED_DTYPES: dict[tuple[str, str], tuple[str, ...]] = {
  ("pytorch", "lora"): ("bfloat16",),
  ("pytorch", "full"): ("float32", "bfloat16"),
  ("automodel", "lora"): ("bfloat16",),
}


def deployment_dtype(env: Any = None) -> str:
  """The deployment's dtype from OPEN_RL_DTYPE, default bfloat16. Rejects anything else."""
  value = ((env if env is not None else os.environ).get("OPEN_RL_DTYPE") or DEFAULT_DEPLOYMENT_DTYPE).lower()
  if value not in DTYPES:
    raise ValueError(f"OPEN_RL_DTYPE must be one of {', '.join(DTYPES)}, got {value!r}")
  return value


def check_dtypes(
  trainer_backend: str, fine_tuning_type: str, trainer_dtype: str | None, sampler_dtype: str | None, vllm_sampler: bool = True
) -> None:
  """Raise ValueError unless this trainer/sampler dtype pair is supported.

  The LoRA bfloat16 limit is vLLM's; a server sampling with torch in one
  process (vllm_sampler=False) lifts it for the pytorch trainer."""
  missing = [name for name, value in (("trainer_dtype", trainer_dtype), ("sampler_dtype", sampler_dtype)) if value is None]
  if missing:
    raise ValueError(" and ".join(f"openrl.{name}" for name in missing) + f" must be set ({' or '.join(DTYPES)}); there is no default")
  if trainer_dtype != sampler_dtype:
    raise ValueError(
      f"openrl.trainer_dtype={trainer_dtype} and openrl.sampler_dtype={sampler_dtype} differ: "
      "precision mismatch between trainer and sampler is not supported yet"
    )
  supported = SUPPORTED_DTYPES.get((trainer_backend, fine_tuning_type))
  if not vllm_sampler and (trainer_backend, fine_tuning_type) == ("pytorch", "lora"):
    supported = DTYPES
  if supported is None:
    if trainer_backend in TRAINER_BACKENDS:
      raise ValueError(f"trainer_backend={trainer_backend} does not support fine_tuning_type={fine_tuning_type}")
    # A trainer image checks its own dtype, but the sampler is ours whatever
    # trains, and a LoRA sampler runs in bfloat16 only.
    supported = ("bfloat16",) if fine_tuning_type == "lora" else DTYPES
  if trainer_dtype not in supported:
    reason = {
      ("automodel", "lora"): "the automodel trainer trains in bfloat16 only",
    }.get((trainer_backend, fine_tuning_type), "vLLM's LoRA kernels support fp16/bf16 only" if fine_tuning_type == "lora" else "unsupported")
    raise ValueError(
      f"openrl.trainer_dtype={trainer_dtype} is not supported for trainer_backend={trainer_backend}, "
      f"fine_tuning_type={fine_tuning_type} (supported: {', '.join(supported)}): {reason}"
    )
