"""Worker managers. The API server asks one to make sure a model's trainer or
sampler exists before it enqueues work. Local mode spawns subprocesses; the
scheduler mode (scheduler_worker_manager.py) creates Workloads."""

import json
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
from server.estimator import footprint
from server.model_metadata import TrainingModelMetadata

PROJECT_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


# -- what is being run ------------------------------------------------------------


def metadata_for(model_id: str) -> TrainingModelMetadata | None:
  """The metadata create_model stored for this model, or None."""
  from server.store import get_store

  try:
    raw = get_store().get_value_sync(f"open_rl:model_meta:{model_id}")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return TrainingModelMetadata.from_dict(data) if isinstance(data, dict) else None
  except Exception:
    return None


def runtime_of(model_id: str) -> tuple[TrainingModelMetadata, str, bool]:
  """Which runtime serves this model, as (metadata, runtime id, is_lora).

  An FFT job owns its runtime, so the id is the model_id. LoRA jobs on one
  base model share a runtime, so the id is the base model. A model with no
  metadata (a sampling session opened on a bare base-model name) is FFT when
  this deployment enables FFT and LoRA otherwise, because an FFT sampler in
  a LoRA deployment would serve base weights and ignore every adapter.
  """
  meta = metadata_for(model_id)
  if meta is None:
    kind = "full" if os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true" else "lora"
    meta = TrainingModelMetadata(base_model=model_id, created_at=0.0, fine_tuning_type=kind)
  is_lora = meta.fine_tuning_type == "lora"
  return meta, (meta.base_model if is_lora else model_id), is_lora


def base_model_of(meta: TrainingModelMetadata, runtime: str) -> str:
  return meta.base_model or os.getenv("BASE_MODEL") or runtime


# Owners double as label values and pod name stems.
LABEL_UNSAFE = re.compile(r"[^a-z0-9-]+")


def owner_id(runtime: str) -> str:
  """The scheduler's ownerID for a runtime, the name its trainer and sampler share."""
  cleaned = LABEL_UNSAFE.sub("-", runtime.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {runtime!r} has no label-safe characters")
  return cleaned[:63]


def owner_of(model_id: str) -> str:
  _, runtime, _ = runtime_of(model_id)
  return owner_id(runtime)


# -- the process ------------------------------------------------------------------


def worker_env(meta: TrainingModelMetadata, base_model: str, runtime: str, is_lora: bool, role: str) -> dict[str, str]:
  """The env every worker gets, whichever manager launches it. Time-slice
  identity is the manager's to add."""
  env = {
    "BASE_MODEL": base_model,
    "OPEN_RL_BASE_MODEL": base_model,
    "OPEN_RL_ENABLE_FFT": "false" if is_lora else "true",
    "OPEN_RL_FINE_TUNING_TYPE": "lora" if is_lora else "full",
    # The device budget this worker was sized for; a sampler derives its
    # vLLM fraction from it against the device it actually gets.
    "OPEN_RL_ACCELERATOR_MEMORY": str(footprint(base_model, meta.fine_tuning_type, role).accelerator_bytes),
  }
  weight_sync = getattr(meta, "weight_sync_config", None)
  if weight_sync is not None:
    env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync.strategy
    if weight_sync.strategy == "delta":
      env["OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT"] = weight_sync.delta_format
      env["OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD"] = weight_sync.delta_apply_method
  if role == "trainer":
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  else:
    env["OPEN_RL_MODEL_ID"] = runtime
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if not is_lora and ("gemma-4" in base_model.lower() or "gemma4" in base_model.lower()):
      env["VLLM_ARCHITECTURE_OVERRIDE"] = "Gemma4ForCausalLM"
  return env


def worker_module(role: str, is_lora: bool) -> str:
  if role == "trainer":
    return "server.training_requests_processor"
  return "server.lora_sampler" if is_lora else "server.vllm_sampler"


def worker_args(runtime: str, role: str, is_lora: bool) -> list[str]:
  args = ["--model-id", runtime]
  if role == "trainer" and is_lora:
    args += ["--active-tenant-set-id", f"{runtime}-1"]
  return args


# -- managers ----------------------------------------------------------------------


class WorkerManager(Protocol):
  def ensure(self, model_id: str, role: str) -> None:
    """Make sure the runtime serving this model's trainer or sampler exists. Idempotent."""
    ...

  def release(self, model_id: str) -> None:
    """Tear down the runtimes an FFT job owns. A shared LoRA runtime is left alone."""
    ...

  def release_owner(self, owner: str) -> set[str]:
    """Tear down an owner's trainer and sampler, shared or not. Returns the
    model ids they served, so the caller can clear their state."""
    ...

  def close(self) -> None:
    """The API server is exiting."""
    ...


class LocalWorkerManager:
  """Runs each runtime as a subprocess of the API server, for development."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.project_dir = project_dir
    self.processes: dict[tuple[str, str], subprocess.Popen] = {}
    self.lock = threading.Lock()

  def ensure(self, model_id: str, role: str) -> None:
    if role == "sampler" and os.getenv("SAMPLING_BACKEND", "vllm").lower() != "vllm":
      return
    meta, runtime, is_lora = runtime_of(model_id)
    with self.lock:
      proc = self.processes.get((role, runtime))
      if proc is not None and proc.poll() is None:
        return
      base_model = base_model_of(meta, runtime)
      env = {**os.environ, **worker_env(meta, base_model, runtime, is_lora, role)}
      env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id(role, runtime)
      env["OPEN_RL_TIME_SLICE_GROUP"] = TRAINER_TIME_SLICE_GROUP if role == "trainer" else SAMPLER_TIME_SLICE_GROUP
      gpus = os.getenv("TRAINER_CUDA_VISIBLE_DEVICES" if role == "trainer" else "SAMPLER_CUDA_VISIBLE_DEVICES")
      if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
      chips = os.getenv("TRAINER_TPU_VISIBLE_CHIPS" if role == "trainer" else "SAMPLER_TPU_VISIBLE_CHIPS")
      if chips:
        env.update(tpu_chip_env(chips))
      extras = ["gpu"] if role == "trainer" else ["gpu", "vllm"]
      if role == "sampler" and os.getenv("OPEN_RL_SAMPLER_DEVICE", "").lower() == "tpu":
        extras = ["tpu-sampler"]
        # tpu-inference has LoRA only on its torch/torchax model impl; models
        # with a JAX impl (e.g. Qwen2) would otherwise route around it.
        env.setdefault("MODEL_IMPL_TYPE", "vllm")
        # The sampler reaches the TPU through JAX; keep a torch_tpu install
        # from also claiming chips via torch's backend autoload.
        env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
      if role == "trainer" and os.getenv("OPEN_RL_DEVICE", "").lower() == "tpu":
        # torch_tpu skips backend registration under TORCH_DEVICE_BACKEND_AUTOLOAD=0
        # even on an explicit import, so torch.device("tpu") would fail. The gateway
        # exports 0 to keep its own torch import off the chips; undo it here.
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
      python = os.getenv("OPEN_RL_TRAINER_PYTHON" if role == "trainer" else "OPEN_RL_SAMPLER_PYTHON")
      if python:
        # An override venv (e.g. vllm-tpu for the sampler) may not have open-rl
        # installed; put the source tree on its path so worker modules resolve.
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.project_dir / "src"), env.get("PYTHONPATH")]))
      command = python_command(extras, worker_module(role, is_lora), worker_args(runtime, role, is_lora), python=python)
      log_dir = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp"))
      log_dir.mkdir(parents=True, exist_ok=True)
      with open(log_dir / f"{role}_{runtime.replace('/', '_')}.log", "a") as log:
        self.processes[(role, runtime)] = subprocess.Popen(
          command, cwd=self.project_dir, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

  def release(self, model_id: str) -> None:
    try:
      _, runtime, _ = runtime_of(model_id)
    except Exception:
      runtime = model_id
    with self.lock:
      for key in [key for key in self.processes if key[1] in {runtime, model_id}]:
        self.terminate(key)

  def release_owner(self, owner: str) -> set[str]:
    with self.lock:
      keys = [key for key in self.processes if owner_id(key[1]) == owner]
      for key in keys:
        self.terminate(key)
    return {key[1] for key in keys}

  def close(self) -> None:
    with self.lock:
      for key in list(self.processes):
        self.terminate(key)

  def terminate(self, key: tuple[str, str]) -> None:
    proc = self.processes.pop(key)
    if proc.poll() is None:
      proc.terminate()


def tpu_chip_env(chips: str) -> dict[str, str]:
  """libtpu env pinning a worker to specific chips, the TPU analogue of
  CUDA_VISIBLE_DEVICES. Visibility alone is not enough; libtpu also needs the
  process bounds. Only single-chip subsets are accepted."""
  n = len([c for c in chips.split(",") if c])
  return {
    "TPU_VISIBLE_CHIPS": chips,
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": f"{n},1,1",
  }


def python_command(extras: list[str], module: str, args: list[str], python: str | None = None) -> list[str]:
  if python:
    # A pre-built interpreter whose dependencies uv's extras cannot express —
    # e.g. a trainer venv carrying the source-built torch_tpu wheel.
    return [python, "-u", "-m", module, *args]
  if shutil.which("uv"):
    extra_args = [arg for extra in extras for arg in ("--extra", extra)]
    return ["uv", "run", *extra_args, "python", "-u", "-m", module, *args]
  return [sys.executable, "-u", "-m", module, *args]


def create_worker_manager() -> WorkerManager | None:
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  if mode in {"none", "disabled"}:
    # Standing worker deployments own the trainer and sampler lifecycles.
    return None
  if mode == "scheduler":
    from server.scheduler_worker_manager import SchedulerWorkerManager

    return SchedulerWorkerManager()
  return LocalWorkerManager()
