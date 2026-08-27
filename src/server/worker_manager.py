"""Worker managers for dedicated per-model trainer workers.

The gateway ensures a model's worker exists before enqueueing its create request:
locally by spawning a subprocess, on Kubernetes by creating a pod. There is no
separate launch queue: the subprocess table / the Kubernetes API already hold
the launched-worker state, and both launchers are idempotent per model_id.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

PROJECT_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def _py_cmd(extras: list[str], module: str, model_id: str, active_tenant_set_id: str | None = None) -> list[str]:
  if shutil.which("uv"):
    extra_args = []
    for e in extras:
      extra_args.extend(["--extra", e])
    cmd = ["uv", "run", *extra_args, "python", "-u", "-m", module, "--model-id", model_id]
  else:
    cmd = [sys.executable, "-u", "-m", module, "--model-id", model_id]
  if active_tenant_set_id:
    cmd.extend(["--active-tenant-set-id", active_tenant_set_id])
  return cmd


from server.model_metadata import TrainingModelMetadata


def _fetch_metadata_from_store(model_id: str) -> TrainingModelMetadata | None:
  """Retrieve TrainingModelMetadata dataclass from canonical open_rl:model_meta:<model_id>."""
  import json

  from server.store import get_store

  try:
    val = get_store().get_value_sync(f"open_rl:model_meta:{model_id}")
    if val:
      meta_dict = json.loads(val) if isinstance(val, str) else val
      if isinstance(meta_dict, dict):
        return TrainingModelMetadata.from_dict(meta_dict)
  except Exception:
    pass
  return None


# A bf16 full fine-tune holds 2 bytes of weights, 2 of gradients and 8 of fp32
# AdamW moments per parameter. Activations, the sampler's KV cache and reload
# buffers come on top, so only part of a tier's VRAM is budgeted here.
#
# Conservative: the FFT trainer offloads optimizer state and the weight shadow
# to pinned host DRAM, so real VRAM use is lower than 12 bytes/param implies.
# Sizing up rather than down is the safe direction -- too large a GPU wastes
# capacity, too small a one OOMs mid-run -- but it does mean a model near the
# boundary is sent to the larger tier than it strictly needs.
FFT_VRAM_BYTES_PER_PARAM = 12
# Of the 24gb tier's 23 GiB usable, leave headroom for everything above. 20 GiB
# admits FFT up to ~1.7B params, which keeps the Qwen2.5-1.5B anchor on an L4.
TIER_24GB_FFT_BUDGET_BYTES = 20 * 1024**3

# A LoRA worker holds the frozen base model in bf16; the adapter, its gradients
# and its optimizer state are negligible beside it. What actually decides the
# tier is whether the *sampler* can hold those weights plus a KV cache: vLLM is
# given a fraction of the device (VLLM_GPU_MEMORY_UTILIZATION, 0.70 in the
# shipped pod templates) and must fit the whole model inside it.
#
# 8B in bf16 is 16.4 GB against an L4's ~15.7 GB at that utilization, so it
# cannot load at all -- which is why sizing LoRA by a parameter-count threshold
# was wrong. Size it by weights against the budget the sampler actually gets.
LORA_VRAM_BYTES_PER_PARAM = 2
# Of the 24gb tier's 23 GiB, what vLLM is handed at the shipped utilization,
# minus room for the KV cache and activations it has to fit alongside.
TIER_24GB_LORA_BUDGET_BYTES = 13 * 1024**3

# Raw parameter counts for the models this project runs. Approximate on purpose:
# they only have to land on the correct side of the 24gb budget below (~1.7B).
#
# A table rather than a Hub lookup. Sizing happens on the worker-launch path,
# which should not depend on network reachability or on huggingface_hub being
# installed, and the ids that actually matter cannot be sized by parsing them:
# `gemma-4-e2b` states its *effective* size while holding roughly 5B raw
# weights, so reading "2b" out of the name puts a 5B fine-tune on a 24 GB card
# and OOMs it mid-run. Keys are normalized by _normalize_model_id.
#
# A model missing from this table is treated as large. Add an entry to let it
# run on a smaller GPU; the count is the raw weight count, not an active or
# effective parameter count.
KNOWN_PARAMETER_COUNTS: dict[str, int] = {
  # Qwen
  "qwen2.5-0.5b": 494_000_000,
  "qwen3-0.6b": 596_049_920,
  "qwen2.5-1.5b": 1_540_000_000,
  "qwen3-1.7b": 1_720_000_000,
  "qwen3-4b": 4_020_000_000,
  "qwen2.5-7b": 7_620_000_000,
  "qwen3-8b": 8_190_000_000,
  "qwen3.5-9b": 9_000_000_000,
  "qwen3.5-27b": 27_000_000_000,
  # Gemma. The e-prefixed ids are effective sizes; these are the raw weights.
  "gemma-3-1b": 1_000_000_000,
  "gemma-4-e2b": 5_440_000_000,
  "gemma-4-e4b": 8_000_000_000,
}

# Fine-tune/variant tags that do not change the weight count.
_VARIANT_SUFFIXES = ("-instruct", "-it", "-pt", "-base", "-chat")


def _normalize_model_id(base_model: str) -> str:
  """Reduce a model id to its KNOWN_PARAMETER_COUNTS key.

  Drops the org prefix, lowercases, and strips variant and release-date tags,
  so `Qwen/Qwen3-4B-Instruct-2507` and `google/gemma-4-E2B-it` both resolve.
  """
  name = (base_model or "").strip().lower().rsplit("/", 1)[-1]
  while True:
    stripped = re.sub(r"-\d{3,}$", "", name)
    for suffix in _VARIANT_SUFFIXES:
      if stripped.endswith(suffix):
        stripped = stripped[: -len(suffix)]
    if stripped == name:
      return name
    name = stripped


def known_parameter_count(base_model: str) -> int | None:
  """Raw parameter count for a known model, or None if it is not in the table."""
  return KNOWN_PARAMETER_COUNTS.get(_normalize_model_id(base_model))


def _fits_24gb_fft(params: int) -> bool:
  return params * FFT_VRAM_BYTES_PER_PARAM <= TIER_24GB_FFT_BUDGET_BYTES


def _fits_24gb_lora(params: int) -> bool:
  return params * LORA_VRAM_BYTES_PER_PARAM <= TIER_24GB_LORA_BUDGET_BYTES


def _lora_memory_tier(base_model: str) -> str:
  """Tier for a LoRA fine-tune, sized from the frozen base model's weights.

  An unknown model resolves to '80gb', for the same reason full fine-tuning
  does: the sampler failing to load is worse than an oversized GPU.
  """
  params = known_parameter_count(base_model)
  if params is None:
    logger.warning(
      "No known parameter count for %r; using the 80gb tier. Add it to KNOWN_PARAMETER_COUNTS to run it on a smaller GPU.",
      base_model,
    )
    return "80gb"
  return "24gb" if _fits_24gb_lora(params) else "80gb"


def _fft_memory_tier(base_model: str) -> str:
  """Tier for a full fine-tune, sized from the model's parameter count.

  An unknown model resolves to '80gb'. That is the direction that cannot end a
  run: too large a GPU wastes capacity, too small a one OOMs mid-training.
  """
  params = known_parameter_count(base_model)
  if params is None:
    logger.warning(
      "No known parameter count for %r; using the 80gb tier. Add it to KNOWN_PARAMETER_COUNTS to run it on a smaller GPU.",
      base_model,
    )
    return "80gb"
  return "24gb" if _fits_24gb_fft(params) else "80gb"


def estimate_memory_tier(base_model: str, fine_tuning_type: str = "lora") -> str:
  """Estimate the VRAM memory tier ('24gb' or '80gb') a workload needs.

  Both modes are sized from the model's parameter count, differing only in
  bytes per parameter: a full fine-tune carries weights, gradients and
  optimizer state, while a LoRA worker carries only the frozen base weights.

  Anchors:
    - Qwen3-0.6B / Qwen2.5-0.5B / Qwen2.5-1.5B (LoRA or FFT): '24gb' (NVIDIA L4)
    - Qwen3-4B (LoRA): '24gb'; (Full Fine-Tuning): '80gb'
    - Qwen3-8B (LoRA or Full Fine-Tuning): '80gb' (NVIDIA H100)

  An unknown model resolves to '80gb' in either mode: a workload sent to too
  small a GPU dies -- OOM mid-run for a trainer, a sampler that cannot load its
  weights at all -- while one sent to too large a GPU merely wastes it.
  """
  if fine_tuning_type == "full":
    return _fft_memory_tier(base_model)

  return _lora_memory_tier(base_model)


def get_model_target_info(model_id: str) -> tuple[TrainingModelMetadata, str, bool]:
  """Retrieve model metadata, target_id, and is_lora flag cleanly from the canonical store."""
  meta = _fetch_metadata_from_store(model_id)
  if meta is None:
    # Without metadata we cannot know the fine-tuning type (e.g. a sampling
    # session opened directly on a base-model name). Only assume FFT when this
    # deployment has FFT enabled: a LoRA deployment must never spawn FFT
    # workers — the FFT vllm_sampler drains the same per-base-model sampling
    # queue but ignores LoRA adapters, silently sampling base weights.
    fallback_type = "full" if os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true" else "lora"
    meta = TrainingModelMetadata(base_model=model_id, created_at=0.0, fine_tuning_type=fallback_type)
  is_lora = meta.fine_tuning_type == "lora"
  target_id = meta.base_model if is_lora else model_id
  return meta, target_id, is_lora


class WorkerManager(Protocol):
  def launch(self, model_id: str) -> None:
    """Ensure the model's worker exists; idempotent per model_id."""
    ...

  def launch_trainer(self, model_id: str) -> None:
    """Ensure the trainer worker exists."""
    ...

  def launch_sampler(self, model_id: str) -> None:
    """Ensure the sampler worker exists."""
    ...

  def shutdown(self, model_id: str) -> None:
    """Tear down the model's worker, if any. The idempotent launch can revive it later."""
    ...

  def shutdown_all(self) -> None: ...


class LocalWorkerManager:
  """Runs local trainer and sampler subprocesses per model."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.project_dir = project_dir
    self.train_processes: dict[str, subprocess.Popen] = {}
    self.sampler_processes: dict[str, subprocess.Popen] = {}
    self.lock = threading.Lock()

  def launch(self, model_id: str) -> None:
    self.launch_trainer(model_id)

  def launch_trainer(self, model_id: str) -> None:
    meta, target_id, is_lora = get_model_target_info(model_id)
    with self.lock:
      proc = self.train_processes.get(target_id)
      if proc is not None and proc.poll() is None:
        return

      env = {
        **os.environ,
        "OPEN_RL_ENABLE_FFT": "false" if is_lora else "true",
        "OPEN_RL_FINE_TUNING_TYPE": "lora" if is_lora else "full",
        "OPEN_RL_TIME_SLICE_JOB_ID": workload_job_id("trainer", target_id),
        "OPEN_RL_TIME_SLICE_GROUP": TRAINER_TIME_SLICE_GROUP,
      }
      if meta.base_model:
        env["BASE_MODEL"] = meta.base_model

      env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = meta.weight_sync_config.strategy
      if meta.weight_sync_config.strategy == "delta":
        env["OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT"] = meta.weight_sync_config.delta_format
        env["OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD"] = meta.weight_sync_config.delta_apply_method

      trainer_gpu = os.getenv("TRAINER_CUDA_VISIBLE_DEVICES")
      if trainer_gpu:
        env["CUDA_VISIBLE_DEVICES"] = trainer_gpu

      active_set_id = f"{target_id}-1" if is_lora else None
      tmp_dir = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp"))
      tmp_dir.mkdir(parents=True, exist_ok=True)
      clean_name = target_id.replace("/", "_")
      with open(tmp_dir / f"trainer_{clean_name}.log", "a") as train_log:
        self.train_processes[target_id] = subprocess.Popen(
          _py_cmd(["gpu"], "server.training_requests_processor", target_id, active_tenant_set_id=active_set_id),
          cwd=self.project_dir,
          env=env,
          stdout=train_log,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )

  def launch_sampler(self, model_id: str) -> None:
    meta, target_id, is_lora = get_model_target_info(model_id)
    with self.lock:
      proc = self.sampler_processes.get(target_id)
      if proc is not None and proc.poll() is None:
        return

      env = {
        **os.environ,
        "OPEN_RL_ENABLE_FFT": "false" if is_lora else "true",
        "OPEN_RL_FINE_TUNING_TYPE": "lora" if is_lora else "full",
      }
      if meta.base_model:
        env["BASE_MODEL"] = meta.base_model

      sampling_backend = os.getenv("SAMPLING_BACKEND", "vllm").lower()
      if sampling_backend == "vllm":
        sampler_env = env.copy()
        sampler_env["OPEN_RL_MODEL_ID"] = target_id
        sampler_env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id("sampler", target_id)
        sampler_env["OPEN_RL_TIME_SLICE_GROUP"] = SAMPLER_TIME_SLICE_GROUP

        sampler_env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = meta.weight_sync_config.strategy
        if meta.weight_sync_config.strategy == "delta":
          sampler_env["OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT"] = meta.weight_sync_config.delta_format
          sampler_env["OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD"] = meta.weight_sync_config.delta_apply_method
        sampler_gpu = os.getenv("SAMPLER_CUDA_VISIBLE_DEVICES")
        if sampler_gpu:
          sampler_env["CUDA_VISIBLE_DEVICES"] = sampler_gpu

        sampler_module = "server.lora_sampler" if is_lora else "server.vllm_sampler"
        tmp_dir = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        clean_name = target_id.replace("/", "_")
        with open(tmp_dir / f"sampler_{clean_name}.log", "a") as sampler_log:
          self.sampler_processes[target_id] = subprocess.Popen(
            _py_cmd(["gpu", "vllm"], sampler_module, target_id),
            cwd=self.project_dir,
            env=sampler_env,
            stdout=sampler_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
          )

  def shutdown(self, model_id: str) -> None:
    try:
      _, target_id, _ = get_model_target_info(model_id)
    except Exception:
      target_id = model_id
    for key in {model_id, target_id}:
      proc = self.train_processes.pop(key, None)
      if proc is not None and proc.poll() is None:
        proc.terminate()
      proc_s = self.sampler_processes.pop(key, None)
      if proc_s is not None and proc_s.poll() is None:
        proc_s.terminate()

  def shutdown_all(self) -> None:
    for model_id in set(list(self.train_processes) + list(self.sampler_processes)):
      self.shutdown(model_id)


def create_worker_manager() -> WorkerManager | None:
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  if mode in {"none", "disabled"}:
    # Standing worker deployments (e.g. k8s/deploy/distributed-shared) own the
    # trainer and sampler lifecycles; the gateway must not spawn its own.
    return None
  if mode in {"kubernetes", "k8s"}:
    from server.k8s_worker_manager import KubernetesWorkerManager

    return KubernetesWorkerManager()
  return LocalWorkerManager()
