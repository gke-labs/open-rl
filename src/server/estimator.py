"""Sizes a worker from its model's parameter count."""

import logging
import re
from dataclasses import dataclass

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
# Elements in the largest checkpoint tensor. A full-fine-tuning sampler applies
# each weight update through vLLM's patch loader, which stages a dense copy of
# every patched checkpoint tensor on the device, so the sampler must keep that
# much free on top of what vLLM profiles. Read from the safetensors headers for
# the models in the cache; the rest are vocab x hidden from config.json. Gemma-4
# models carry a per-layer input embedding, vocab x (256 x layers), that is far
# larger than the token embedding.
MODEL_TO_LARGEST_TENSOR_ELEMENTS: dict[str, int] = {
  "qwen2.5-0.5b": 136_134_656,  # embed_tokens [151936, 896]
  "qwen3-0.6b": 155_582_464,  # embed_tokens [151936, 1024]
  "qwen2.5-1.5b": 233_373_696,  # [151936, 1536]
  "qwen3-1.7b": 311_164_928,  # [151936, 2048]
  "qwen3-4b": 388_956_160,  # embed_tokens [151936, 2560]
  "qwen2.5-7b": 544_997_376,  # [152064, 3584]
  "qwen3-8b": 622_329_856,  # embed_tokens [151936, 4096]
  "qwen3.5-9b": 1_017_118_720,  # [248320, 4096]
  "qwen3.5-27b": 1_271_398_400,  # [248320, 5120]
  "gemma-3-1b": 301_989_888,  # [262144, 1152]
  "gemma-4-e2b": 2_348_810_240,  # embed_tokens_per_layer [262144, 8960]
  "gemma-4-e4b": 2_818_572_288,  # embed_tokens_per_layer [262144, 10752]
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
# A full-fine-tuning sampler's weight update: vLLM's patch loader stages the
# largest checkpoint tensor densely in bf16 (2 B/element), and its NaN-masked
# copy adds a bool mask (1 B) and the merged result (2 B), all live at once;
# plus the sparse delta and the loader's working set. Measured on gemma-4-e2b's
# 2.35 B-element per-layer embedding: the update died on an L4 with 1.9 and
# then 5.7 GiB free, and went through with 10.7 GiB free
# (runs/2026-10-17_demo_baselines).
SAMPLER_WEIGHT_UPDATE_BYTES_PER_ELEMENT = 5
SAMPLER_WEIGHT_UPDATE_MARGIN_BYTES = GIB
# Parked in host memory: fft trainer 12 B/param + a weight copy in flight;
# plus process overhead. Measured: 0.5B trainer 28Gi, sampler 20Gi; 8B FFT
# sampler 39Gi steady; 7B FFT trainer OOM-killed at 110Gi.
HOST_BYTES_PER_PARAM = {("full", "trainer"): 14, ("lora", "trainer"): 2, ("full", "sampler"): 2, ("lora", "sampler"): 2}
HOST_OVERHEAD_BYTES = {"trainer": 20 * GIB, "sampler": 24 * GIB}
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


def sampler_device_bytes(params: int, kv_bytes_per_token: int, kind: str, largest_tensor_elements: int = 0) -> int:
  device = params * SAMPLER_WEIGHT_BYTES_PER_PARAM
  device += SAMPLER_OVERHEAD_BYTES + int(params * SAMPLER_OVERHEAD_BYTES_PER_PARAM)
  if kind == "lora":
    device += SAMPLER_LORA_SLOT_BYTES + int(params * SAMPLER_LORA_SLOT_BYTES_PER_PARAM)
  else:
    device += weight_update_bytes(largest_tensor_elements)
  return device + kv_bytes_per_token * SAMPLER_KV_TOKENS


def weight_update_bytes(largest_tensor_elements: int) -> int:
  """What a full-fine-tuning sampler keeps free for applying a weight update."""
  if largest_tensor_elements <= 0:
    return 0
  return largest_tensor_elements * SAMPLER_WEIGHT_UPDATE_BYTES_PER_ELEMENT + SAMPLER_WEIGHT_UPDATE_MARGIN_BYTES


def footprint(base_model: str, fine_tuning_type: str, role: str) -> Footprint:
  model = normalize_model_id(base_model)
  if model not in MODEL_TO_PARAM_COUNT:
    logger.warning("No known parameter count for %r; sizing it as %s.", base_model, UNKNOWN_MODEL)
    model = UNKNOWN_MODEL
  params = MODEL_TO_PARAM_COUNT[model]
  kind = "lora" if fine_tuning_type == "lora" else "full"
  if role == "trainer":
    device = params * TRAINER_DEVICE_BYTES_PER_PARAM[kind] + TRAINER_DEVICE_RESERVE_BYTES
  else:
    device = sampler_device_bytes(params, MODEL_TO_KV_BYTES_PER_TOKEN[model], kind, MODEL_TO_LARGEST_TENSOR_ELEMENTS[model])
  host = params * HOST_BYTES_PER_PARAM[(kind, role)] + HOST_OVERHEAD_BYTES[role]
  return Footprint(device, host, int(host * HOST_LIMIT_FACTOR))
