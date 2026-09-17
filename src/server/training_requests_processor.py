"""The trainer process: drains typed training commands from the queue and runs
them against one worker.

Layers, top to bottom. The gateway turns API calls into training.commands
and enqueues them. This loop pops them (rank 0 reads the queue and fans each
batch out to the other torchrun ranks), takes the GPU lease when the
deployment needs one, and dispatches each command to the worker's Trainer
for that model. Results go back through the store's futures from rank 0.
"""

import argparse
import asyncio
import json
import os
import threading
import time
import traceback
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import NoOpTimeSlicer, TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_CLAIM, local_workload_name
from server.store import RequestStore, get_store
from training import commands
from training.commands import GPU_COMMANDS, Command, Shutdown, parse_command
from training.distributed import barrier, broadcast_object, is_distributed, is_primary, pin_executor_threads
from training.distributed import close as close_distributed
from training.distributed import initialize as initialize_distributed
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker
from training.trainer_worker import BaseTrainerWorker

tracer = trace.get_tracer(__name__)


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def describe_commands(batch: list[Command]) -> str:
  """`op:request_id` per command, matching the gateway's enqueue log line."""
  return ", ".join(f"{command.op}:{command.request_id}" for command in batch)


@dataclass(frozen=True)
class Deployment:
  """How this process is wired into the system.

  model_id: set for a dedicated per-model worker, which drains that model's
    queue; None for a shared worker draining the round-robin queue.
  active_tenant_set_id: which tenant rotation a shared worker serves.
  leased: whether GPU work runs under the time slicer's lease. A shared LoRA
    worker owns its GPU; a dedicated worker shares the card with samplers.
  """

  model_id: str | None = None
  active_tenant_set_id: str | None = None
  leased: bool = False

  @property
  def dedicated(self) -> bool:
    return self.model_id is not None


class TrainingRequestsProcessor:
  def __init__(self, store: RequestStore, worker: BaseTrainerWorker, deployment: Deployment, time_slicer: TimeSlicerClient | None = None):
    if deployment.dedicated and not os.getenv("REDIS_URL"):
      raise RuntimeError("A dedicated trainer needs REDIS_URL so it can share its queue and futures with the gateway")
    self.store = store
    self.worker = worker
    self.deployment = deployment
    self.time_slicer = time_slicer or NoOpTimeSlicer()
    self.workload = workload_from_env(os.getpid(), name=local_workload_name("trainer", deployment.model_id or "shared"), claim=TRAINER_CLAIM)
    self.registered = False

  # -- loop -----------------------------------------------------------------

  async def run(self) -> None:
    print(f"[WORKER] training requests processor started ({self.deployment}).")
    try:
      await self.time_slicer.register(self.workload)
      self.registered = True
      while True:
        try:
          await self.run_once()
        except asyncio.CancelledError:
          break
        except Exception as exc:
          print(f"Error in training requests processor: {exc}")
          traceback.print_exc()
          await asyncio.sleep(1)
    finally:
      try:
        if self.registered:
          await self.time_slicer.unregister(self.workload)
      finally:
        await self.time_slicer.close()
        close_distributed()

  async def exit_gracefully(self) -> None:
    print(f"[WORKER] Initiating immediate exit for {self.deployment.model_id or 'shared'} trainer worker...")
    if self.registered:
      with suppress(Exception):
        await self.time_slicer.unregister(self.workload)
        self.registered = False
    with suppress(Exception):
      await self.time_slicer.close()
    os._exit(0)

  async def fetch_requests(self) -> list[dict[str, Any]] | None:
    if self.deployment.dedicated:
      return await self.store.get_requests_for_model(self.deployment.model_id)
    return await self.store.get_requests(active_set_id=self.deployment.active_tenant_set_id)

  async def next_batch(self) -> list[Command]:
    """The next batch of commands, identical on every rank.

    Collectives are positional, so every rank must execute the same command
    sequence: rank 0 owns the queue and fans each batch out.
    """
    raw = await self.fetch_requests() if is_primary() else None
    if is_distributed():
      raw = await asyncio.to_thread(broadcast_object, raw)
    if not raw:
      if is_primary():
        await asyncio.sleep(0.1)
      return []
    return [parse_command(item) for item in raw]

  @asynccontextmanager
  async def gpu_lease(self):
    """Rank 0 holds the lease; the other ranks enter and leave with it."""
    async with AsyncExitStack() as stack:
      await stack.enter_async_context(self.time_slicer.acquire(self.workload))
      await asyncio.to_thread(barrier)
      try:
        yield
      finally:
        await asyncio.to_thread(barrier)

  def needs_gpu(self, command: Command) -> bool:
    return isinstance(command, GPU_COMMANDS) or self.worker.save_needs_gpu()

  async def run_once(self) -> None:
    batch = await self.next_batch()
    if not batch:
      return
    shutdown = any(isinstance(command, Shutdown) for command in batch)
    work = [command for command in batch if not isinstance(command, Shutdown)]

    with tracer.start_as_current_span("training_requests_batch") as span:
      span.set_attribute("batch_size", len(work))
      if work:
        print(f"\n[TRAINING REQUESTS] Popped {len(work)} requests: {describe_commands(work)}")
        results: list[tuple[str, dict[str, Any]]] = []
        gpu_work = [command for command in work if self.needs_gpu(command)]
        host_work = [command for command in work if not self.needs_gpu(command)]
        if gpu_work:
          async with self.gpu_lease():
            await asyncio.to_thread(self.worker.wake_up)
            try:
              for command in gpu_work:
                results.append(await self.handle(command))
            finally:
              await asyncio.to_thread(self.worker.sleep)
        for command in host_work:
          results.append(await self.handle(command))
        for request_id, result in results:
          await self.publish_result(request_id, result)

    if shutdown:
      await self.exit_gracefully()

  async def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
    # Only rank 0 writes to the store; the other ranks computed the same result.
    if is_primary():
      await self.store.set_future(request_id, result)

  # -- dispatch -------------------------------------------------------------

  async def handle(self, command: Command) -> tuple[str, dict[str, Any]]:
    """Run one command and return (request_id, result). Failures become a
    RequestFailedResponse so the client's future resolves either way."""
    ctx = propagate.extract(command.trace_context) if command.trace_context else None
    token = otel_context.attach(ctx) if ctx else None
    try:
      return command.request_id, await self.dispatch(command)
    except Exception as exc:
      traceback.print_exc()
      return command.request_id, {"type": "RequestFailedResponse", "error_message": str(exc)}
    finally:
      if token:
        otel_context.detach(token)

  async def dispatch(self, command: Command) -> dict[str, Any]:
    match command:
      case commands.CreateModel():
        await asyncio.to_thread(self.worker.create, command)
        result = {
          "base_model": command.base_model,
          "model_id": command.model_id,
          "fine_tuning_type": command.fine_tuning_type,
          "type": "model_created",
        }
        if command.fine_tuning_type == "lora":
          result["rank"] = command.lora_config.rank
        return result
      case commands.CreateModelFromState():
        await asyncio.to_thread(self.worker.restore, command)
        return {
          "base_model": self.worker.base_model_name,
          "model_id": command.model_id,
          "fine_tuning_type": command.fine_tuning_type,
          "type": "model_loaded_from_state",
        }
      case commands.ForwardBackward():
        trainer = self.worker.trainer(command.model_id)
        result = await asyncio.to_thread(trainer.forward_backward, command.data, command.loss_fn, command.loss_config)
        return {**result, "type": "forward_backward_completed"}
      case commands.OptimStep():
        trainer = self.worker.trainer(command.model_id)
        result = await asyncio.to_thread(trainer.optim_step, command.adam_params)
        await self.bump_step_count(command.model_id)
        return {**result, "type": "optim_step_completed"}
      case commands.Sample():
        trainer = self.worker.trainer(command.model_id)
        result = await asyncio.to_thread(
          trainer.generate, command.prompt_tokens, command.max_tokens, command.num_samples, command.temperature, command.prompt_logprobs
        )
        return {**result, "type": "sample_completed"}
      case commands.SaveState():
        trainer = self.worker.trainer(command.model_id)
        result = await asyncio.to_thread(trainer.save_state, command.state_path, command.include_optimizer, command.kind)
        return {"path": result.get("path", command.state_path), "type": "state_saved"}
      case commands.LoadWeights():
        trainer = self.worker.trainer(command.model_id)
        await asyncio.to_thread(trainer.load_from_state, command.state_path, command.restore_optimizer)
        return {"path": command.state_path, "type": "weights_loaded"}
      case commands.SaveWeightsForSampler():
        trainer = self.worker.trainer(command.model_id)
        published = await asyncio.to_thread(trainer.publish_sampler_weights, command)
        return {
          "path": command.path,
          "sampling_session_id": command.sampling_session_id,
          "sampler_weights": published.model_dump(),
          "type": "sampler_weights_saved",
        }
      case commands.SaveWeights():
        trainer = self.worker.trainer(command.model_id)
        await asyncio.to_thread(trainer.save_weights, command.alias)
        return {"status": "ok", "type": "weights_saved"}
      case commands.Shutdown():
        return {"status": "ok", "type": "shutdown_acknowledged"}
    raise NotImplementedError(f"Training command {type(command).__name__} is not supported")

  # -- store bookkeeping, rank 0 only ---------------------------------------

  async def bump_step_count(self, model_id: str) -> None:
    if not is_primary():
      return
    try:
      raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
      current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
      await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
    except Exception as exc:
      print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")


# -- process entry point ------------------------------------------------------


async def run_training_requests_processor(
  worker: BaseTrainerWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
  leased: bool | None = None,
) -> None:
  pin_executor_threads()
  if leased is None:
    leased = worker.full_parameter
  if leased and time_slicer is None and is_primary():
    time_slicer = time_slicer_client_from_env()
  deployment = Deployment(
    model_id=model_id,
    active_tenant_set_id=active_tenant_set_id or (f"{model_id}-1" if model_id and not worker.full_parameter else None),
    leased=leased,
  )
  await TrainingRequestsProcessor(get_store(), worker, deployment, time_slicer).run()


async def fine_tuning_type_for(model_id: str | None) -> str:
  """What kind of model this process trains: the env default, or the metadata
  of the model a dedicated worker was launched for."""
  fine_tuning_type = os.getenv("OPEN_RL_FINE_TUNING_TYPE") or ("full" if is_fft_enabled() else "lora")
  if model_id:
    try:
      raw_meta = await get_store().get_value(f"open_rl:model_meta:{model_id}")
      if raw_meta:
        fine_tuning_type = json.loads(raw_meta).get("fine_tuning_type", fine_tuning_type)
    except Exception as exc:
      print(f"[WORKER] Failed to fetch model metadata for {model_id}: {exc}")
  return fine_tuning_type


def build_worker(is_lora: bool) -> BaseTrainerWorker:
  if os.getenv("OPEN_RL_TRAINER_BACKEND", "").lower() == "automodel":
    from training.automodel_worker import AutomodelTrainingWorker

    return AutomodelTrainingWorker(full_parameter=not is_lora)
  return LoraTrainingWorker() if is_lora else FFTTrainingWorker()


def serve_health_probe() -> None:
  """A readiness endpoint for shared LoRA workers, on rank 0 only. The port
  is configurable so a trainer can share a box with a vLLM server on 8000."""
  probe_app = FastAPI()
  probe_port = int(os.getenv("OPEN_RL_WORKER_PROBE_PORT", "8000"))

  @probe_app.get("/healthz")
  def healthz():
    return {"status": "ready"}

  def run_probe_server():
    try:
      uvicorn.run(probe_app, host="0.0.0.0", port=probe_port, log_level="warning")
    except Exception as exc:
      print(f"[WORKER] Probe server on port {probe_port} skipped: {exc}")

  threading.Thread(target=run_probe_server, daemon=True).start()


async def main_async(args: argparse.Namespace) -> None:
  fine_tuning_type = await fine_tuning_type_for(args.model_id)
  is_lora = fine_tuning_type == "lora"
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora})\n")

  worker = build_worker(is_lora)
  if is_lora:
    if preload_target := os.getenv("BASE_MODEL"):
      worker.load_base_model(preload_target)
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    if is_primary():
      serve_health_probe()
  else:
    print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")

  await run_training_requests_processor(worker, args.model_id, active_tenant_set_id=args.active_tenant_set_id, leased=not is_lora)


def start_request_processing_loop() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated trainer worker drains.")
  parser.add_argument("--active-tenant-set-id", help="Active tenant rotation set ID for LoRA workers (e.g. Qwen/Qwen3-0.6B-1).")
  args = parser.parse_args()
  initialize_distributed()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")

  asyncio.run(main_async(args))


if __name__ == "__main__":
  start_request_processing_loop()
