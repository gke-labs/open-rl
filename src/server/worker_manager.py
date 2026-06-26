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
from pathlib import Path
from typing import Protocol

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

PROJECT_DIR = Path(__file__).resolve().parents[2]


def _py_cmd(extras: list[str], module: str, model_id: str) -> list[str]:
  if shutil.which("uv"):
    extra_args = []
    for e in extras:
      extra_args.extend(["--extra", e])
    return ["uv", "run", *extra_args, "python", "-u", "-m", module, "--model-id", model_id]
  return [sys.executable, "-u", "-m", module, "--model-id", model_id]


class WorkerManager(Protocol):
  def launch(self, model_id: str, base_model: str | None = None) -> None:
    """Ensure the model's worker exists; idempotent per model_id."""
    ...

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> None:
    """Ensure the trainer worker exists."""
    ...

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> None:
    """Ensure the sampler worker exists."""
    ...

  def shutdown(self, model_id: str) -> None:
    """Tear down the model's worker, if any. The idempotent launch can revive it later."""
    ...

  def shutdown_all(self) -> None: ...


class FFTWorkerManager:
  """Runs local trainer and sampler subprocesses per FFT model."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.project_dir = project_dir
    self.train_processes: dict[str, subprocess.Popen] = {}
    self.sampler_processes: dict[str, subprocess.Popen] = {}

  def launch(self, model_id: str, base_model: str | None = None) -> None:
    self.launch_trainer(model_id, base_model)

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> None:
    proc = self.train_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    env = {
      **os.environ,
      "OPEN_RL_ENABLE_FFT": "true",
      "OPEN_RL_TIME_SLICE_JOB_ID": workload_job_id("trainer", model_id),
      "OPEN_RL_TIME_SLICE_GROUP": TRAINER_TIME_SLICE_GROUP,
    }
    if base_model:
      env["BASE_MODEL"] = base_model
    self.train_processes[model_id] = subprocess.Popen(
      _py_cmd(["gpu"], "server.training_requests_processor", model_id),
      cwd=self.project_dir,
      env=env,
      start_new_session=True,
    )

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> None:
    proc = self.sampler_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    if base_model:
      env["BASE_MODEL"] = base_model
    sampling_backend = os.getenv("SAMPLING_BACKEND", "vllm").lower()
    if sampling_backend == "vllm":
      sampler_env = env.copy()
      sampler_env["OPEN_RL_MODEL_ID"] = model_id
      sampler_env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id("sampler", model_id)
      sampler_env["OPEN_RL_TIME_SLICE_GROUP"] = SAMPLER_TIME_SLICE_GROUP
      sampler_gpu = os.getenv("SAMPLER_CUDA_VISIBLE_DEVICES")
      if sampler_gpu:
        sampler_env["CUDA_VISIBLE_DEVICES"] = sampler_gpu

      self.sampler_processes[model_id] = subprocess.Popen(
        _py_cmd(["gpu", "vllm"], "server.vllm_sampler", model_id),
        cwd=self.project_dir,
        env=sampler_env,
        start_new_session=True,
      )

  def shutdown(self, model_id: str) -> None:
    proc = self.train_processes.pop(model_id, None)
    if proc is not None and proc.poll() is None:
      proc.terminate()
    proc_s = self.sampler_processes.pop(model_id, None)
    if proc_s is not None and proc_s.poll() is None:
      proc_s.terminate()

  def shutdown_all(self) -> None:
    for model_id in set(list(self.train_processes) + list(self.sampler_processes)):
      self.shutdown(model_id)


def create_fft_worker_manager() -> WorkerManager:
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  if mode in {"kubernetes", "k8s"}:
    from server.k8s_worker_manager import KubernetesFFTWorkerManager

    return KubernetesFFTWorkerManager()
  return FFTWorkerManager()
