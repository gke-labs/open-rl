"""Worker managers. The API server asks one to make sure a model's trainer or
sampler exists before it enqueues work. Local mode spawns subprocesses; the
scheduler mode (scheduler_worker_manager.py) creates Workloads."""

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from accel_timeslicer.time_slicer import time_slicing_enabled
from accel_timeslicer.workload import TRAINER_TIME_SLICE_GROUP, workload_job_id
from server.estimator import footprint
from server.model_metadata import TrainingModelMetadata, decode_model_metadata, runtime_name
from server.sampling_dispatcher import tmp_root
from server.store import get_state_store

PROJECT_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Worker states a client may branch on (design 016, section 5.2). Every
# launcher reports the same five values; Workload phases, pod states and exit
# codes map onto them and reach the client only inside the free-text `detail`.
WAITING_FOR_PLACEMENT = "waiting_for_placement"  # no accelerator assigned yet
STARTING = "starting"  # placed; pulling the image, mounting volumes or loading the model
READY = "ready"  # serving requests
FAILED = "failed"  # exited with an error, or the pod cannot start
STOPPED = "stopped"  # exited normally, or being released
WORKER_STATES = (WAITING_FOR_PLACEMENT, STARTING, READY, FAILED, STOPPED)


# -- what is being run ------------------------------------------------------------


def metadata_for(model_id: str) -> TrainingModelMetadata | None:
  """The metadata create_model stored for this model, or None."""
  return decode_model_metadata(get_state_store().get_value_sync(f"open_rl:model_meta:{model_id}"))


def default_metadata(model_id: str) -> TrainingModelMetadata:
  """A model with no metadata is a sampling session opened on a bare base-model
  name. It is FFT when this deployment enables FFT and LoRA otherwise, because
  an FFT sampler in a LoRA deployment would serve base weights and ignore every
  adapter."""
  return TrainingModelMetadata(base_model=model_id, fine_tuning_type="full" if os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true" else "lora")


def runtime_of(model_id: str) -> tuple[TrainingModelMetadata, str, bool]:
  """Which runtime serves this model, as (metadata, runtime id, is_lora).

  A model that owns its runtime (FFT, or exclusive) has the model_id as the
  id. Other LoRA jobs on one base model share a runtime, so the id is the
  base model, prefixed with the backend or an image hash off PyTorch.
  """
  meta = metadata_for(model_id) or default_metadata(model_id)
  is_lora = meta.fine_tuning_type == "lora"
  return meta, meta.runtime(model_id), is_lora


def base_model_of(meta: TrainingModelMetadata, runtime: str) -> str:
  return meta.base_model or os.getenv("BASE_MODEL") or runtime


def owner_id(runtime: str) -> str:
  """The scheduler's ownerID for a runtime, the name its trainer and sampler share.
  Owners double as label values and pod name stems."""
  return runtime_name(runtime)


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
  env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = meta.weight_sync_config.strategy
  if role == "trainer":
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if meta.trainer_backend == "automodel":
      env["OPEN_RL_TRAINER_BACKEND"] = "automodel"
    if meta.trainer_gpus > 1:
      # FSDP2 needs NCCL; the batch broadcast between ranks goes over gloo.
      env["OPEN_RL_CONTROL_BACKEND"] = "cpu:gloo,cuda:nccl"
    if meta.trainer_cp > 1:
      env["OPEN_RL_AUTOMODEL_CP"] = str(meta.trainer_cp)
  else:
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # The text-only class skips the multimodal front end vLLM otherwise builds
    # at start-up (about 45 s); OpenRL only sends token ids. LoRA adapters
    # trained on the full model apply to it unchanged. VLLM_ENABLE_MULTIMODAL=1
    # keeps the full model.
    gemma_4 = "gemma-4" in base_model.lower() or "gemma4" in base_model.lower()
    if gemma_4 and os.getenv("VLLM_ENABLE_MULTIMODAL", "0") != "1":
      env["VLLM_ARCHITECTURE_OVERRIDE"] = "Gemma4ForCausalLM"
  return env


TRAINER_MODULE = "server.training_requests_processor"


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
    """Tear down the runtime a model owns. A shared LoRA runtime is left alone."""
    ...

  def release_owner(self, owner: str) -> set[str]:
    """Tear down an owner's trainer and sampler, shared or not. Returns the
    model ids they served, so the caller can clear their state."""
    ...

  def sampler_exists(self, model_id: str) -> bool:
    """Whether any sampler of this model's pool exists, ready or not."""
    ...

  def close(self) -> None:
    """The API server is exiting."""
    ...

  def inspect_owner(self, owner: str) -> list[dict[str, Any]]:
    """One entry per worker serving this owner, read-only: `role`, `model_id`,
    `state` (one of WORKER_STATES), `detail` (free text for a human: the
    launcher's own reason, node and pod state), and `workload`, `node` and
    `age_seconds` where the launcher knows them."""
    ...

  def workloads(self) -> list[dict[str, Any]]:
    """The scheduler Workload objects this manager created; none for a launcher without a scheduler."""
    ...


class LocalWorkerManager:
  """Runs each runtime as a subprocess of the API server, for development."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.project_dir = project_dir
    self.processes: dict[tuple[str, str], subprocess.Popen] = {}
    self.started: dict[tuple[str, str], float] = {}
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
      gpus = os.getenv("TRAINER_CUDA_VISIBLE_DEVICES" if role == "trainer" else "SAMPLER_CUDA_VISIBLE_DEVICES")
      if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
      if role == "trainer":
        env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id(role, runtime)
        env["OPEN_RL_TIME_SLICE_GROUP"] = TRAINER_TIME_SLICE_GROUP
        self.spawn(role, runtime, python_command(["gpu"], TRAINER_MODULE, worker_args(runtime, role, is_lora)), env)
        return
      env["OPEN_RL_TMP_DIR"] = tmp_root()
      same_gpu = gpus == os.getenv("TRAINER_CUDA_VISIBLE_DEVICES")
      if meta.shares_gpu() and same_gpu and time_slicing_enabled():
        # The sampler takes turns on the one GPU with its trainer, so it is in the trainer's claim.
        env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id(role, runtime)
        env["OPEN_RL_TIME_SLICE_GROUP"] = TRAINER_TIME_SLICE_GROUP
      else:
        env["OPEN_RL_TIME_SLICING"] = "off"
      self.spawn(role, runtime, python_command(["gpu", "vllm"], "server.vllm_sampler", ["--pool", meta.pool(runtime)]), env)

  def spawn(self, role: str, runtime: str, command: list[str], env: dict[str, str]) -> None:
    log_dir = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{role}_{runtime.replace('/', '_')}.log", "a") as log:
      process = subprocess.Popen(command, cwd=self.project_dir, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    self.processes[(role, runtime)] = process
    self.started[(role, runtime)] = time.time()

  def release(self, model_id: str) -> None:
    try:
      _, runtime, _ = runtime_of(model_id)
    except Exception:
      runtime = model_id
    with self.lock:
      for key in [key for key in self.processes if key[1] in {runtime, model_id}]:
        self.terminate(key)

  def sampler_exists(self, model_id: str) -> bool:
    _, runtime, _ = runtime_of(model_id)
    with self.lock:
      proc = self.processes.get(("sampler", runtime))
      return proc is not None and proc.poll() is None

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
    self.started.pop(key, None)
    if proc.poll() is None:
      proc.terminate()

  def inspect_owner(self, owner: str) -> list[dict[str, Any]]:
    now = time.time()
    with self.lock:
      workers = []
      for (role, runtime), proc in self.processes.items():
        if owner_id(runtime) != owner:
          continue
        code = proc.poll()
        if code is None:
          state, detail = READY, f"pid {proc.pid}"
        elif code:
          state, detail = FAILED, f"exited with code {code}"
        else:
          state, detail = STOPPED, "exited"
        workers.append(
          {
            "role": role,
            "model_id": runtime,
            "state": state,
            "detail": detail,
            "workload": None,
            "node": None,
            "age_seconds": now - self.started.get((role, runtime), now),
          }
        )
      return workers

  def workloads(self) -> list[dict[str, Any]]:
    return []


def python_command(extras: list[str], module: str, args: list[str]) -> list[str]:
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
