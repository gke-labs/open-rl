"""Worker managers for dedicated per-model trainer workers.

The gateway ensures a model's worker exists before enqueueing its create request:
locally by spawning a subprocess, on Kubernetes by creating a pod. There is no
separate launch queue: the subprocess table / the Kubernetes API already hold
the launched-worker state, and both launchers are idempotent per model_id.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

PROJECT_DIR = Path(__file__).resolve().parents[2]


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


def estimate_memory_tier(base_model: str, fine_tuning_type: str = "lora") -> str:
  """Estimate VRAM memory tier ('24gb' or '80gb') based on base model scale and fine-tuning type.

  Anchors:
    - Qwen3-0.6B / Qwen2.5-0.5B / Qwen2.5-1.5B (LoRA or FFT): '24gb' (NVIDIA L4)
    - Qwen3-8B / Qwen2.5-7B (LoRA): '24gb' (NVIDIA L4)
    - Qwen3-8B / Qwen2.5-7B (Full Fine-Tuning): '80gb' (NVIDIA H100)
    - 14B+ models (LoRA or FFT): '80gb' (NVIDIA H100)
  """
  model_lower = (base_model or "").lower()

  # Full fine-tuning always maps to the 80gb tier: FFT of any
  # multi-billion-param model (bf16 params + grads + AdamW states +
  # pinned-DRAM shadow) exceeds the 24gb tier, and name-based size
  # sniffing misses many model naming schemes (e.g. gemma-4-e2b).
  if fine_tuning_type == "full":
    return "80gb"

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
