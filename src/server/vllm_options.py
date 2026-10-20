"""Engine and sampling options shared by the two vLLM sampler workers."""

import json
import os
import struct
from collections.abc import Sequence
from typing import Any


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


DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
# vLLM must keep at least this much of the device, whatever the headroom asks.
MIN_GPU_MEMORY_UTILIZATION = 0.40
# vLLM's patch loader stages each patched checkpoint tensor densely (its bytes
# once), then its NaN-masked copy_ builds a bool mask (half) and the merged
# result (once more): 2.5x the tensor, all live at the same moment. Measured on
# gemma-4-e2b's 4.38 GiB per-layer embedding: 1x headroom still died at the
# 2.19 GiB mask, and the update went through with 10.7 GiB free.
WEIGHT_UPDATE_STAGING_FACTOR = 2.5
# The sparse delta's values and indices and the loader's working set.
WEIGHT_UPDATE_MARGIN_BYTES = 2**30
# vLLM's share must still hold the weights with room for a small KV cache.
WEIGHTS_RESIDENT_MARGIN_BYTES = 2**30
SAFETENSORS_DTYPE_BYTES = {
  "F64": 8,
  "I64": 8,
  "U64": 8,
  "F32": 4,
  "I32": 4,
  "U32": 4,
  "BF16": 2,
  "F16": 2,
  "I16": 2,
  "U16": 2,
  "I8": 1,
  "U8": 1,
  "BOOL": 1,
}


def gpu_memory_utilization(headroom_bytes: int = 0, device_memory_bytes: int | None = None, resident_bytes: int = 0) -> float:
  """vLLM's share of the device: an explicit VLLM_GPU_MEMORY_UTILIZATION, else 0.90
  less whatever headroom the caller must keep free.

  OPEN_RL_ACCELERATOR_MEMORY is deliberately not consulted. It is the placement
  claim the scheduler sized this worker for, not a runtime cap: workers that
  share an accelerator are time-sliced, so whichever one holds the device may
  use all of it. Sizing vLLM from the claim starved the sampler of KV cache on
  large devices and made Gemma-4's first engine init fail when the cold
  torch.compile transient did not fit in the claim.

  Headroom is different: it is memory a later transient needs that vLLM's own
  profiling never sees. An FFT sampler applies each weight update through
  vLLM's checkpoint patch loader, which stages dense copies of every patched
  checkpoint tensor on the device; gemma-4-e2b's per-layer embedding needs 11
  GiB of that, and on an L4 the 10% vLLM leaves free is 2.2 GiB. The sampler
  died at its first update (runs/2026-10-17_demo_baselines). The headroom comes
  off vLLM's share so the update fits, and never pushes the share below the
  floor or below what the resident weights need.
  """
  explicit = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
  if explicit:
    return float(explicit)
  if headroom_bytes <= 0:
    return DEFAULT_GPU_MEMORY_UTILIZATION
  if device_memory_bytes is None:
    device_memory_bytes = detected_device_memory_bytes()
  if device_memory_bytes <= 0:
    return DEFAULT_GPU_MEMORY_UTILIZATION
  share = DEFAULT_GPU_MEMORY_UTILIZATION - headroom_bytes / device_memory_bytes
  floor = max(MIN_GPU_MEMORY_UTILIZATION, (resident_bytes + WEIGHTS_RESIDENT_MARGIN_BYTES) / device_memory_bytes if resident_bytes > 0 else 0)
  return round(max(floor, min(DEFAULT_GPU_MEMORY_UTILIZATION, share)), 4)


def weight_update_headroom_bytes(model: str) -> int:
  """Device memory a full-fine-tuning sampler must keep free for weight updates.

  OPEN_RL_WEIGHT_UPDATE_HEADROOM_GIB overrides the estimate; "0" disables it.
  Otherwise it is the staging the patch loader allocates for the largest tensor
  in the model's checkpoint, plus a fixed margin. The checkpoint is already in
  the local Hugging Face cache because the engine loads from it.
  """
  explicit = os.getenv("OPEN_RL_WEIGHT_UPDATE_HEADROOM_GIB")
  if explicit:
    return int(float(explicit) * 2**30)
  largest, _ = checkpoint_tensor_bytes(model)
  if largest <= 0:
    return 0
  return int(largest * WEIGHT_UPDATE_STAGING_FACTOR) + WEIGHT_UPDATE_MARGIN_BYTES


def largest_checkpoint_tensor_bytes(model: str) -> int:
  return checkpoint_tensor_bytes(model)[0]


def checkpoint_tensor_bytes(model: str) -> tuple[int, int]:
  """(largest tensor, all tensors) in the model's safetensors files, from their headers only.

  Both are 0 when the checkpoint cannot be found locally or holds no safetensors.
  """
  directory = model if os.path.isdir(model) else local_snapshot(model)
  if not directory:
    return 0, 0
  largest = total = 0
  for name in sorted(os.listdir(directory)):
    if not name.endswith(".safetensors"):
      continue
    try:
      file_largest, file_total = tensor_bytes_in_header(os.path.join(directory, name))
    except (OSError, ValueError, KeyError):
      continue
    largest, total = max(largest, file_largest), total + file_total
  return largest, total


def local_snapshot(model: str) -> str | None:
  try:
    from huggingface_hub import snapshot_download

    return snapshot_download(model, local_files_only=True, allow_patterns=["*.safetensors"])
  except Exception:
    return None


def tensor_bytes_in_header(path: str) -> tuple[int, int]:
  with open(path, "rb") as f:
    (header_len,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(header_len))
  largest = total = 0
  for name, info in header.items():
    if name == "__metadata__":
      continue
    numel = 1
    for dim in info["shape"]:
      numel *= dim
    size = numel * SAFETENSORS_DTYPE_BYTES.get(info["dtype"], 2)
    largest, total = max(largest, size), total + size
  return largest, total


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


# Batch sizes vLLM records a CUDA graph for. Its default list has dozens of
# entries up to max_num_seqs and capturing them took 12-40 s on every sampler
# start (runs/2026-10-17_demo_baselines). A GRPO step sends a handful of
# shapes; powers of two up to max_num_seqs cover them with padding and cut the
# capture to a few sizes. VLLM_CUDAGRAPH_CAPTURE_SIZES: "default" keeps vLLM's
# list, otherwise a comma-separated list of sizes.
def cudagraph_capture_sizes(max_num_seqs: int) -> list[int] | None:
  raw = os.getenv("VLLM_CUDAGRAPH_CAPTURE_SIZES", "").strip().lower()
  if raw == "default":
    return None
  if raw:
    sizes = sorted({int(s) for s in raw.split(",") if s.strip()})
  else:
    sizes, size = [], 1
    while size <= max_num_seqs:
      sizes.append(size)
      size *= 2
  return [s for s in sizes if s <= max_num_seqs] or [max_num_seqs]


def compilation_kwargs(max_num_seqs: int) -> dict[str, Any]:
  sizes = cudagraph_capture_sizes(max_num_seqs)
  return {} if sizes is None else {"compilation_config": {"cudagraph_capture_sizes": sizes}}
