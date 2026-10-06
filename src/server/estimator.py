"""Sizes a worker from its model's parameter count."""

import logging
import re
from dataclasses import dataclass

from server.accelerators import Accelerator

GIB = 1024**3

logger = logging.getLogger(__name__)

# Raw weights. gemma-4-e2b names an effective 2B but holds ~5B.
MODEL_TO_PARAM_COUNT: dict[str, int] = {
  "qwen2.5-0.5b": 494_000_000,
  "qwen3-0.6b": 596_049_920,
  "qwen2.5-1.5b": 1_540_000_000,
  "qwen3-1.7b": 1_720_000_000,
  "qwen3-4b": 4_020_000_000,
  "qwen2.5-7b": 7_620_000_000,
  "qwen3-8b": 8_190_000_000,
  "qwen3.5-9b": 9_000_000_000,
  "qwen3.5-27b": 27_000_000_000,
  "gemma-3-1b": 1_000_000_000,
  "gemma-4-e2b": 5_440_000_000,
  "gemma-4-e4b": 8_000_000_000,
}
# bf16 KV cache per token: 2 * full-attention layers * kv heads * head dim * 2
# bytes, from each model's config.json. Sliding-window and linear-attention
# layers hold a fixed per-sequence state instead and are left out. Checked
# against vLLM's "GPU KV cache size" for qwen3-0.6b and qwen3-8b.
MODEL_TO_KV_BYTES_PER_TOKEN: dict[str, int] = {
  "qwen2.5-0.5b": 12_288,  # 24 layers, 2 kv heads, head dim 64
  "qwen3-0.6b": 114_688,  # 28 layers, 8 kv heads, head dim 128
  "qwen2.5-1.5b": 28_672,  # 28 layers, 2 kv heads, head dim 128
  "qwen3-1.7b": 114_688,  # 28 layers, 8 kv heads, head dim 128
  "qwen3-4b": 147_456,  # 36 layers, 8 kv heads, head dim 128
  "qwen2.5-7b": 57_344,  # 28 layers, 4 kv heads, head dim 128
  "qwen3-8b": 147_456,  # 36 layers, 8 kv heads, head dim 128
  "qwen3.5-9b": 32_768,  # 8 of 32 layers are full attention, 4 kv heads, head dim 256
  "qwen3.5-27b": 65_536,  # 16 of 64 layers are full attention, 4 kv heads, head dim 256
  "gemma-3-1b": 4_096,  # 4 of 26 layers are global, 1 kv head, head dim 256
  "gemma-4-e2b": 7_168,  # 7 of 35 layers are full attention, 1 kv head, head dim 256
  "gemma-4-e4b": 14_336,  # 7 of 42 layers are full attention, 2 kv heads, head dim 256
}
UNKNOWN_MODEL = "qwen3-8b"  # unknown models are sized large, not small
VARIANT_SUFFIXES = ("-instruct", "-it", "-pt", "-base", "-chat")


def normalize_model_id(base_model: str) -> str:
  name = (base_model or "").strip().lower().rsplit("/", 1)[-1]
  while True:
    stripped = re.sub(r"-\d{3,}$", "", name)
    for suffix in VARIANT_SUFFIXES:
      if stripped.endswith(suffix):
        stripped = stripped[: -len(suffix)]
    if stripped == name:
      return name
    name = stripped


def parameter_count(base_model: str) -> int | None:
  return MODEL_TO_PARAM_COUNT.get(normalize_model_id(base_model))


# Trainer on the device: fft 8 B/param (bf16 weights + grads + fp32 master),
# frozen base 2 B/param; plus activations.
TRAINER_DEVICE_BYTES_PER_PARAM = {"full": 8, "lora": 2}
TRAINER_DEVICE_RESERVE_BYTES = 4 * GIB
# Sampler on the device: bf16 weights, vLLM's activation peak and CUDA graphs,
# LoRA slot buffers (max_loras 8, rank 64) for a LoRA sampler, and a KV cache
# sized for SAMPLER_KV_TOKENS. This is a placement claim: it decides which
# device the sampler may land on, not how much of it vLLM uses (see
# vllm_options.gpu_memory_utilization). Overheads measured on the live
# samplers: activation peak 0.5 GiB at 0.6B and 1.4 GiB at 8B; LoRA slots
# 1.05 GiB at 0.6B and 2.9 GiB at 8B.
SAMPLER_WEIGHT_BYTES_PER_PARAM = 2
SAMPLER_OVERHEAD_BYTES = GIB // 2
SAMPLER_OVERHEAD_BYTES_PER_PARAM = 0.125
SAMPLER_LORA_SLOT_BYTES = GIB
SAMPLER_LORA_SLOT_BYTES_PER_PARAM = 0.25
SAMPLER_KV_TOKENS = 8 * 8192  # eight max-length requests in flight
# Parked in host memory: fft trainer 12 B/param + a weight copy in flight;
# plus process overhead. Measured: 0.5B trainer 28Gi, sampler 20Gi; 8B FFT
# sampler 39Gi steady; 7B FFT trainer OOM-killed at 110Gi.
HOST_BYTES_PER_PARAM = {("full", "trainer"): 14, ("lora", "trainer"): 2, ("full", "sampler"): 2, ("lora", "sampler"): 2}
# TPU entries are placeholders near the 96Gi limit every TPU run has used.
# Each TPU worker holds a whole node, so erring high costs nothing.
# TODO: measure TPU trainer and sampler host peaks and lower these to match.
HOST_OVERHEAD_BYTES = {("gpu", "trainer"): 20 * GIB, ("gpu", "sampler"): 24 * GIB, ("tpu", "trainer"): 94 * GIB, ("tpu", "sampler"): 94 * GIB}
# Limits equal requests. Placement admits pods by request, so a pod that
# could burst past it can push a co-seated neighbour into the kernel's OOM
# killer; an 8B FFT sampler ran at 39Gi against a 34Gi request.
HOST_LIMIT_FACTOR = 1.0


def gib(n: int) -> str:
  return f"{-(-n // GIB)}Gi"


@dataclass(frozen=True)
class Footprint:
  accelerator_bytes: int
  host_request_bytes: int
  host_limit_bytes: int

  @property
  def accelerator(self) -> str:
    return gib(self.accelerator_bytes)

  @property
  def resources(self) -> dict:
    return {"requests": {"memory": gib(self.host_request_bytes)}, "limits": {"memory": gib(self.host_limit_bytes)}}


def sampler_device_bytes(params: int, kv_bytes_per_token: int, kind: str) -> int:
  device = params * SAMPLER_WEIGHT_BYTES_PER_PARAM
  device += SAMPLER_OVERHEAD_BYTES + int(params * SAMPLER_OVERHEAD_BYTES_PER_PARAM)
  if kind == "lora":
    device += SAMPLER_LORA_SLOT_BYTES + int(params * SAMPLER_LORA_SLOT_BYTES_PER_PARAM)
  return device + kv_bytes_per_token * SAMPLER_KV_TOKENS


def footprint(base_model: str, fine_tuning_type: str, role: str, accelerator: Accelerator = "gpu") -> Footprint:
  # TPU shares the device formula: it is a placement claim, and the LoRA
  # models run on TPU so far fit one 32Gi v6e chip by it.
  model = normalize_model_id(base_model)
  if model not in MODEL_TO_PARAM_COUNT:
    logger.warning("No known parameter count for %r; sizing it as %s.", base_model, UNKNOWN_MODEL)
    model = UNKNOWN_MODEL
  params = MODEL_TO_PARAM_COUNT[model]
  kind = "lora" if fine_tuning_type == "lora" else "full"
  if role == "trainer":
    device = params * TRAINER_DEVICE_BYTES_PER_PARAM[kind] + TRAINER_DEVICE_RESERVE_BYTES
  else:
    device = sampler_device_bytes(params, MODEL_TO_KV_BYTES_PER_TOKEN[model], kind)
  host = params * HOST_BYTES_PER_PARAM[(kind, role)] + HOST_OVERHEAD_BYTES[(accelerator, role)]
  return Footprint(device, host, int(host * HOST_LIMIT_FACTOR))
