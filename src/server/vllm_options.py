"""Engine and sampling options for the vLLM samplers."""

import json
import os
from collections.abc import Mapping, Sequence


def text_only_engine_kwargs() -> dict:
  """Stop vLLM reserving an encoder cache OpenRL can never use.

  Text checkpoints published under a `*ForConditionalGeneration` architecture
  make vLLM profile a max-resolution image and video at startup and reserve a
  multi-GiB encoder cache before the KV cache is sized, which can OOM engine
  init. OpenRL only ever passes token ids, so that capacity is unreachable.
  `VLLM_ENABLE_MULTIMODAL=1` restores stock vLLM behaviour.
  """
  if os.getenv("VLLM_ENABLE_MULTIMODAL", "0") == "1":
    return {}
  return {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def split_stop(stop: str | Sequence[str] | Sequence[int] | None) -> tuple[list[str] | None, list[int] | None]:
  """Split a sampling request's `stop` into vLLM's `stop` and `stop_token_ids`.

  The sampling API types stop as `str | Sequence[str] | Sequence[int]` and which
  one arrives depends on the client's renderer. vLLM keeps strings and token ids
  in separate arguments and rejects a string in `stop_token_ids`. Each half is
  None when empty so callers do not override a vLLM default with [].
  """
  if stop is None:
    return None, None
  if isinstance(stop, str):
    return [stop], None
  strings = [s for s in stop if isinstance(s, str)]
  # bool is an int subclass; a stray True would be read as token id 1.
  token_ids = [t for t in stop if isinstance(t, int) and not isinstance(t, bool)]
  return (strings or None), (token_ids or None)


def gpu_memory_utilization() -> float:
  """vLLM's share of the device: an explicit VLLM_GPU_MEMORY_UTILIZATION, else 0.90.

  OPEN_RL_ACCELERATOR_MEMORY is deliberately not consulted. It is the placement
  claim the scheduler sized this worker for, not a runtime cap: workers that
  share an accelerator are time-sliced, so whichever one holds the device may
  use all of it. Sizing vLLM from the claim starved the sampler of KV cache on
  large devices and made Gemma-4's first engine init fail when the cold
  torch.compile transient did not fit in the claim.
  """
  explicit = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
  if explicit:
    return float(explicit)
  return 0.90


def sampler_batch_limits(device_memory_bytes: int | None = None) -> dict[str, int]:
  """vLLM's concurrency for this device: sequences in flight and prefill tokens per step.

  A sweep's sampler serves many groups at once, and the defaults (64
  sequences, 2048 prefill tokens) leave most of an 80 GB card idle. Explicit
  VLLM_MAX_NUM_SEQS / VLLM_MAX_NUM_BATCHED_TOKENS win; otherwise size by memory.
  """
  if device_memory_bytes is None:
    device_memory_bytes = detected_device_memory_bytes()
  gib = device_memory_bytes / 2**30
  if gib >= 60:
    seqs, tokens = 256, 16384
  elif gib >= 36:
    seqs, tokens = 128, 8192
  else:
    seqs, tokens = 64, 4096
  return {
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", seqs)),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", tokens)),
  }


def detected_device_memory_bytes() -> int:
  try:
    import torch

    if torch.cuda.is_available():
      return int(torch.cuda.get_device_properties(0).total_memory)
  except Exception:
    pass
  return 0


def serve_args(base_model: str, is_lora: bool, env: Mapping[str, str | None], port: int = 8000) -> list[str]:
  """`vllm serve` flags for a sampling pool replica."""
  args = [base_model, "--port", str(port), "--api-server-count", "1", "--enable-scale-out", "--enable-prefix-caching"]
  # The queue sampler built SamplingParams itself, so a checkpoint's
  # generation_config.json defaults never applied. vllm serve would apply them.
  args += ["--generation-config", "vllm", "--max-model-len", os.getenv("VLLM_MAX_MODEL_LEN", "8192")]
  # The API server's vLLM settings, as flags. VLLM_MAX_NUM_SEQS becomes --max-num-seqs.
  for name in ("VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_BATCHED_TOKENS"):
    if os.getenv(name):
      args += ["--" + name.removeprefix("VLLM_").lower().replace("_", "-"), os.environ[name]]
  if text_only := text_only_engine_kwargs():
    args += ["--limit-mm-per-prompt", json.dumps(text_only["limit_mm_per_prompt"])]
  if architecture := env.get("VLLM_ARCHITECTURE_OVERRIDE"):
    args += ["--hf-overrides", json.dumps({"architectures": [architecture]})]
  if is_lora:
    return args + ["--enable-lora", "--max-loras", os.getenv("VLLM_MAX_LORAS", "8"), "--max-lora-rank", os.getenv("VLLM_MAX_LORA_RANK", "64")]
  return args + ["--weight-transfer-config", '{"backend":"delta_snapshot"}']
