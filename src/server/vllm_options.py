"""Engine and sampling options for the vLLM samplers."""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

# Every setting the sampler engine reads from the environment. The API server passes each one to sampler pods, so an
# engine in the sampler's process and a vllm serve replica see the same values. VLLM_ARCHITECTURE_OVERRIDE depends on
# the model, so worker_env sets it for each pod instead.
ENGINE_ENV = (
  "VLLM_GPU_MEMORY_UTILIZATION",
  "VLLM_MAX_MODEL_LEN",
  "VLLM_MAX_NUM_SEQS",
  "VLLM_MAX_NUM_BATCHED_TOKENS",
  "VLLM_ENFORCE_EAGER",
  "VLLM_ENABLE_MULTIMODAL",
  "VLLM_MAX_LORAS",
  "VLLM_MAX_LORA_RANK",
)


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


def on_tpu_platform() -> bool:
  """Whether vLLM runs on the out-of-tree TPU platform. Its worker has no weight-transfer methods, and it compiles its
  own programs instead of capturing CUDA graphs."""
  try:
    from vllm.platforms import current_platform
  except ImportError:
    return False
  return current_platform.is_out_of_tree() and current_platform.device_type == "tpu"


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


def cudagraph_capture_sizes(max_num_seqs: int) -> list[int]:
  """Batch sizes vLLM records a CUDA graph for: powers of two up to max_num_seqs.

  vLLM's default list has dozens of sizes, and capturing them (once to profile
  memory, once for real) is a large part of a sampler's start-up. A batch runs
  on the next captured size up.
  """
  return [2**i for i in range(max_num_seqs.bit_length())]


def detected_device_memory_bytes() -> int:
  try:
    import torch

    if torch.cuda.is_available():
      return int(torch.cuda.get_device_properties(0).total_memory)
  except Exception:
    pass
  return 0


def sample_params(r: Mapping[str, Any]) -> dict[str, Any]:
  """vLLM sampling params for a pool record, asking for the logprobs every result carries."""
  return {**r["sampling_params"], "logprobs": 0, "prompt_logprobs": 0 if r["include_prompt_logprobs"] else None}


def sample_result(sequences: list[tuple[list[int], list[float], str]], prompt_logprobs: list[float | None] | None) -> dict[str, Any]:
  """The result a pool saves for a record, from each sequence's tokens, logprobs and stop reason."""
  result: dict[str, Any] = {"type": "sample", "sequences": [{"tokens": t, "logprobs": lp, "stop_reason": s} for t, lp, s in sequences]}
  if prompt_logprobs is not None:
    result["prompt_logprobs"] = prompt_logprobs
  return result


def engine_args(is_lora: bool, sleep_mode: bool = False) -> dict[str, Any]:
  """vLLM engine settings for a sampler, the same for a `vllm serve` replica and an engine in the sampler's process.
  Both compute them on the sampler's own pod, so the batch is sized for the GPU it runs on."""
  # Not for LoRA, where every save is a new adapter and no cached block outlives a step. With prefix caching on, a
  # 13-job shared pool took twice as long as main to answer its first batch.
  args: dict[str, Any] = {"enable_prefix_caching": not is_lora}
  # A checkpoint's generation_config.json could add sampling params the client never asked for, such as a repetition penalty.
  args["generation_config"] = "vllm"
  args["max_model_len"] = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
  # vLLM's own default of 0.92 has run FFT delta applies out of memory on an L4.
  args["gpu_memory_utilization"] = gpu_memory_utilization()
  args.update(sampler_batch_limits())
  if os.getenv("VLLM_ENFORCE_EAGER", "0") == "1":
    args["enforce_eager"] = True
  elif not on_tpu_platform():
    args["compilation_config"] = {"cudagraph_capture_sizes": cudagraph_capture_sizes(args["max_num_seqs"])}
  args.update(text_only_engine_kwargs())
  if architecture := os.getenv("VLLM_ARCHITECTURE_OVERRIDE"):
    args["hf_overrides"] = {"architectures": [architecture]}
  if is_lora:
    args.update(enable_lora=True, max_loras=int(os.getenv("VLLM_MAX_LORAS", "8")), max_lora_rank=int(os.getenv("VLLM_MAX_LORA_RANK", "64")))
  else:
    args["weight_transfer_config"] = {"backend": "delta_snapshot"}
    # vLLM refuses an extension that redefines a worker's methods, so only the TPU worker gets it.
    if on_tpu_platform():
      args["worker_extension_cls"] = "server.tpu_worker_extension.TPUWeightTransferExtension"
  if sleep_mode:
    args["enable_sleep_mode"] = True
  return args


def serve_args(base_model: str, is_lora: bool, port: int = 8000) -> list[str]:
  """`vllm serve` flags for a sampling pool replica, computed on the replica's pod."""
  args = [base_model, "--port", str(port), "--api-server-count", "1", "--enable-scale-out"]
  for name, value in engine_args(is_lora).items():
    flag = name.replace("_", "-")
    if isinstance(value, bool):
      # vllm serve takes booleans as --flag and --no-flag; a value after the flag is an error.
      args.append(f"--{flag}" if value else f"--no-{flag}")
    else:
      args += [f"--{flag}", json.dumps(value) if isinstance(value, dict) else str(value)]
  return args
