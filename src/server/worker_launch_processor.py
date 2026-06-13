import asyncio
import os
import subprocess
import sys
import traceback
from pathlib import Path

from server.store import RequestStore

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKER_LAUNCH_OPS = {"create_model", "create_model_from_state"}


class FFTWorkerManager:
  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.project_dir = project_dir
    self.processes: dict[str, list[subprocess.Popen]] = {}

  def launch(self, model_id: str) -> None:
    procs = self.processes.get(model_id)
    if procs is not None and all(p.poll() is None for p in procs):
      return

    if procs is not None:
      for p in procs:
        if p.poll() is None:
          p.terminate()

    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    self.processes[model_id] = []

    p_train = subprocess.Popen(
      [sys.executable, "-m", "server.training_requests_processor", "--model-id", model_id],
      cwd=self.project_dir,
      env=env,
    )
    self.processes[model_id].append(p_train)

    sampling_backend = os.getenv("SAMPLING_BACKEND", "vllm").lower()
    if sampling_backend == "vllm":
      sampler_env = env.copy()
      sampler_env["OPEN_RL_MODEL_ID"] = model_id
      sampler_gpu = os.getenv("SAMPLER_CUDA_VISIBLE_DEVICES")
      if sampler_gpu:
        sampler_env["CUDA_VISIBLE_DEVICES"] = sampler_gpu
      
      sampler_socket = os.getenv("OPEN_RL_SAMPLER_SNAPSHOT_AGENT_SOCKET")
      if sampler_socket:
        sampler_env["OPEN_RL_SNAPSHOT_AGENT_SOCKET"] = sampler_socket

      p_sampler = subprocess.Popen(
        [sys.executable, "-u", "-m", "server.vllm_sampler", "--model-id", model_id],
        cwd=self.project_dir,
        env=sampler_env,
      )
      self.processes[model_id].append(p_sampler)

  def shutdown_all(self) -> None:
    for procs in self.processes.values():
      for proc in procs:
        if proc.poll() is None:
          proc.terminate()


class WorkerLaunchProcessor:
  """Drain the worker launch queue and start FFT workers before enqueueing training requests."""

  def __init__(self, store: RequestStore, worker_manager: FFTWorkerManager):
    self.store = store
    self.worker_manager = worker_manager

  async def process_request(self, request: dict) -> None:
    request_id = request.get("request_id")
    try:
      op = request.get("op")
      if op not in WORKER_LAUNCH_OPS:
        raise ValueError(f"worker launch request cannot handle op {op!r}")

      model_id = request.get("model_id")
      if not model_id:
        raise ValueError("worker launch request requires model_id")

      self.worker_manager.launch(model_id)
      await self.store.put_request(request)
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      await self.store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})

  async def process_batch(self, requests: list[dict]) -> None:
    for request in requests:
      await self.process_request(request)

  async def run(self) -> None:
    while True:
      try:
        batch = await self.store.get_worker_launch_requests()
        if not batch:
          await asyncio.sleep(0.1)
          continue
        await self.process_batch(batch)
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in worker launch processor: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)
