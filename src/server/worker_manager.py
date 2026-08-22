"""Worker managers for dedicated per-model trainer workers.

The gateway ensures a model's worker exists before enqueueing its create request:
locally by spawning a subprocess, on Kubernetes by creating a pod. There is no
separate launch queue: the subprocess table / the Kubernetes API already hold
the launched-worker state, and both launchers are idempotent per model_id.
"""

import logging
import os
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
# buffers come on top, so only part of a tier's VRAM is budgeted here. The
# pinned-DRAM weight shadow is host memory and does not count.
FFT_VRAM_BYTES_PER_PARAM = 12
# Of the 24gb tier's 23 GiB usable, leave headroom for everything above. 20 GiB
# admits FFT up to ~1.7B params, which keeps the Qwen2.5-1.5B anchor on an L4.
TIER_24GB_FFT_BUDGET_BYTES = 20 * 1024**3

_HUB_TIMEOUT_SECONDS = 5.0
# Maps a base model to its Hub parameter count, or to None when the lookup did
# not answer. Negative results are cached too: an unreachable Hub must not cost
# a timeout on every worker launch.
_PARAM_COUNT_CACHE: dict[str, int | None] = {}


def _hub_parameter_count(base_model: str) -> int | None:
  """Parameter count from the Hub's safetensors metadata, or None if unavailable."""
  if os.getenv("HF_HUB_OFFLINE"):
    return None
  try:
    from huggingface_hub import HfApi

    info = HfApi().model_info(base_model, timeout=_HUB_TIMEOUT_SECONDS)
    total = getattr(getattr(info, "safetensors", None), "total", None)
    return int(total) if total else None
  except Exception:
    return None


def _name_parameter_count(base_model: str) -> int | None:
  """Parameter count parsed from a size token in the model id, e.g. `Qwen3-8B` -> 8e9.

  A fallback only. It reads the first size token, so an MoE id like
  `Qwen3-30B-A3B` reports the 30B total rather than the active 3B, and a
  scheme that states effective rather than raw size (`gemma-4-e2b`) understates
  the real footprint. Both err toward a larger model for `Qwen3-30B-A3B` and,
  for the effective-size schemes, are the reason the Hub is consulted first.
  """
  import re

  match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", (base_model or "").lower())
  if not match:
    return None
  return int(float(match.group(1)) * 1e9)


def _fits_24gb_fft(params: int) -> bool:
  return params * FFT_VRAM_BYTES_PER_PARAM <= TIER_24GB_FFT_BUDGET_BYTES


def _fft_memory_tier(base_model: str) -> str:
  """Tier for a full fine-tune, sized from the model's parameter count.

  The Hub is consulted only when the name claims the model is small enough for
  an L4, because that is the sole direction that can end a run: too large a GPU
  wastes capacity, too small a one OOMs mid-training. It is also the direction
  naming schemes get wrong -- an id stating an *effective* size (`gemma-4-e2b`)
  understates the weights an FFT has to hold. A name that reads large, or that
  carries no size at all, already resolves to '80gb' and needs no lookup, which
  keeps this call off the network for most launches.
  """
  named = _name_parameter_count(base_model)
  if named is None or not _fits_24gb_fft(named):
    return "80gb"

  if base_model not in _PARAM_COUNT_CACHE:
    _PARAM_COUNT_CACHE[base_model] = _hub_parameter_count(base_model)
  confirmed = _PARAM_COUNT_CACHE[base_model]

  if confirmed is None:
    logger.warning(
      "Could not confirm the parameter count of %r against the Hub; trusting the size in its name (%d params) and using the 24gb tier.",
      base_model,
      named,
    )
    return "24gb"
  return "24gb" if _fits_24gb_fft(confirmed) else "80gb"


def estimate_memory_tier(base_model: str, fine_tuning_type: str = "lora") -> str:
  """Estimate VRAM memory tier ('24gb' or '80gb') based on base model scale and fine-tuning type.

  Anchors:
    - Qwen3-0.6B / Qwen2.5-0.5B / Qwen2.5-1.5B (LoRA or FFT): '24gb' (NVIDIA L4)
    - Qwen3-8B / Qwen2.5-7B (LoRA): '24gb' (NVIDIA L4)
    - Qwen3-8B / Qwen2.5-7B (Full Fine-Tuning): '80gb' (NVIDIA H100)
    - 14B+ models (LoRA or FFT): '80gb' (NVIDIA H100)

  Full fine-tuning is sized from the model's parameter count. When that cannot
  be established the tier stays '80gb': an FFT sent to too small a GPU dies on
  an OOM mid-run, while one sent to too large a GPU merely wastes it.
  """
  model_lower = (base_model or "").lower()

  if fine_tuning_type == "full":
    return _fft_memory_tier(base_model)

  # LoRA fine-tuning memory scaling
  if any(size in model_lower for size in ["14b", "32b", "70b"]):
    return "80gb"

  return "24gb"


def get_model_target_info(model_id: str) -> tuple[TrainingModelMetadata, str, bool]:
  """Retrieve model metadata, target_id, and is_lora flag cleanly from the canonical store."""
  meta = _fetch_metadata_from_store(model_id)
  if meta is None:
    meta = TrainingModelMetadata(base_model=model_id, created_at=0.0, fine_tuning_type="full")
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

      env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
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


def create_worker_manager() -> WorkerManager:
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  if mode in {"kubernetes", "k8s"}:
    from server.k8s_worker_manager import KubernetesWorkerManager

    return KubernetesWorkerManager()
  return LocalWorkerManager()
