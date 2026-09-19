# This file contains the training request processor implementation for Open-RL.

import argparse
import asyncio
import json
import os
import shutil
import threading
import time
import traceback
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_CLAIM, local_workload_name
from server.store import RequestStore, get_store
from training.commands import parse_command
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.types import Datum

tracer = trace.get_tracer(__name__)


TrainingWorker = FFTTrainingWorker | LoraTrainingWorker


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def describe_requests(batch: list[dict[str, Any]]) -> str:
  """`op:request_id` per request, matching the API server's enqueue log line."""
  return ", ".join(f"{r.get('op')}:{r.get('request_id')}" for r in batch)


class TrainingRequestsProcessor(Protocol):
  store: RequestStore

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    request_id, result = await self.handle_request(raw_request, model_id)
    if request_id is not None:
      await self.store.set_future(request_id, result)

  async def handle_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> tuple[str | None, dict[str, Any]]:
    request_id = raw_request.get("request_id")
    token = None

    try:
      command = parse_command(raw_request)
      request_id = command.request_id
      resolved_model_id = model_id or command.model_id or "default"

      ctx = propagate.extract(command.trace_context) if command.trace_context else None
      token = otel_context.attach(ctx) if ctx else None

      # The handlers read the command's fields as the payload they always took.
      payload = command.model_dump(mode="json", exclude={"op", "request_id", "model_id", "trace_context"})
      result = await self.dispatch_operation(command.op, payload, resolved_model_id)
      return request_id, result
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      return request_id, {"type": "RequestFailedResponse", "error_message": str(exc)}
    finally:
      if token:
        otel_context.detach(token)

  async def dispatch_operation(self, op: str, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    match op:
      case "create_model":
        return await self.create_model(payload, model_id)
      case "create_model_from_state":
        return await self.create_model_from_state(payload, model_id)
      case "forward_backward":
        return await self.forward_backward(payload, model_id)
      case "optim_step":
        return await self.optim_step(payload, model_id)
      case "sample":
        return await self.sample(payload, model_id)
      case "save_state":
        return await self.save_state(payload, model_id)
      case "load_weights":
        return await self.load_weights(payload, model_id)
      case "save_weights_for_sampler":
        return await self.save_weights_for_sampler(payload, model_id)
      case "save_weights":
        return await self.save_weights(payload, model_id)
      case "shutdown_workers":
        return {"status": "ok", "type": "shutdown_acknowledged"}
      case _:
        raise NotImplementedError(f"Training request op {op!r} is not supported")

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...


async def _fetch_model_meta(
  store: RequestStore,
  model_id: str,
  payload: dict[str, Any],
  default_kind: str = "full",
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
  val = None
  if hasattr(store, "get_value"):
    try:
      val = await store.get_value(f"open_rl:model_meta:{model_id}")
    except Exception:
      pass
  if val:
    try:
      meta = json.loads(val) if isinstance(val, str) else val
      if isinstance(meta, dict):
        base_model = meta.get("base_model") or payload.get("base_model") or ""
        full_config = meta.get("full_config") or payload.get("full_config") or {}
        lora_config = meta.get("lora_config") or payload.get("lora_config") or {}
        fine_tuning_type = meta.get("fine_tuning_type") or payload.get("fine_tuning_type") or ("lora" if "lora_config" in meta else default_kind)
        return base_model, full_config, lora_config, fine_tuning_type
    except Exception:
      pass
  return (
    payload.get("base_model", ""),
    payload.get("full_config") or {},
    payload.get("lora_config") or {},
    payload.get("fine_tuning_type") or ("lora" if default_kind == "lora" else "full"),
  )


class LoraTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: LoraTrainingWorker,
    model_id: str | None = None,
    active_tenant_set_id: str | None = None,
  ):
    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.active_tenant_set_id = active_tenant_set_id or (f"{model_id}-1" if model_id else None)

  async def run(self) -> None:
    print(f"[WORKER] LoRA training requests processor started (Active Set ID: {self.active_tenant_set_id}).")

    while True:
      try:
        await self.run_once()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in training requests processor: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)

  async def run_once(self) -> None:
    batch = await self.store.get_requests(active_set_id=self.active_tenant_set_id)
    if not batch:
      await asyncio.sleep(0.1)
      return

    model_id = batch[0].get("model_id", "default")

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}: {describe_requests(batch)}")
      for request in batch:
        target_model_id = request.get("adapter_id") or request.get("model_id") or model_id
        await self.process_request(request, target_model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, raw_config, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, lora_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "rank": lora_config.rank,
      "fine_tuning_type": fine_tuning_type,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "fine_tuning_type": fine_tuning_type,
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [Datum.model_validate(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
      forward_only=bool(payload.get("forward_only", False)),
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    await asyncio.to_thread(self.worker.save_adapter, model_id)
    if hasattr(self, "store") and self.store:
      try:
        raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
        current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
        await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
      except Exception as exc:
        print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")
    return result

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.generate,
      payload.get("prompt_tokens", []),
      payload.get("max_tokens", 20),
      payload.get("num_samples", 1),
      payload.get("temperature", 0.0),
      model_id,
      bool(payload.get("prompt_logprobs", False)),
    )
    result["type"] = "sample_completed"
    return result

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.save_state,
      model_id,
      payload["state_path"],
      bool(payload.get("include_optimizer", False)),
      payload.get("kind", "state"),
    )
    return {"path": result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    return {
      "path": payload.get("path"),
      "sampling_session_id": payload.get("sampling_session_id"),
      "type": "sampler_weights_saved",
    }

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    return {"status": "ok", "type": "weights_saved"}


# Sampler weight versions kept on the volume. The sampler applies each delta
# as it lands, so older versions are dead weight; an 8B run otherwise leaves
# 3 GiB per step behind.
SAMPLER_VERSIONS_KEPT = int(os.getenv("OPEN_RL_SAMPLER_VERSIONS_KEPT", "3"))


def older_versions(path: str, keep: int) -> list[str]:
  """Sibling version directories of `path` beyond the newest `keep`, oldest first."""
  parent = os.path.dirname(path)
  if not os.path.isdir(parent):
    return []
  versions = sorted((p for p in (os.path.join(parent, name) for name in os.listdir(parent)) if os.path.isdir(p)), key=os.path.getmtime)
  return versions[: max(0, len(versions) - keep)]


class FFTTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: FFTTrainingWorker,
    model_id: str | None,
    time_slicer: TimeSlicerClient,
  ):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the API server")
    if not model_id:
      raise RuntimeError("A dedicated trainer worker needs --model-id so it knows which per-model queue to drain")

    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.workload = workload_from_env(os.getpid(), name=local_workload_name("trainer", model_id), claim=TRAINER_CLAIM)
    self.time_slicer = time_slicer
    self.snapshot_registered = False

  async def exit_gracefully(self, unregister: bool = True) -> None:
    print(f"[WORKER] Initiating immediate exit for model {self.model_id} trainer worker...")
    if unregister and self.snapshot_registered:
      try:
        await self.time_slicer.unregister(self.workload)
        self.snapshot_registered = False
      except Exception as exc:
        print(f"[WORKER] Failed to unregister: {exc}")
    try:
      await self.time_slicer.close()
    except Exception:
      pass
    os._exit(0)

  async def run(self) -> None:
    print("[WORKER] Full fine-tuning training requests processor started.")

    try:
      await self.time_slicer.register(self.workload)
      self.snapshot_registered = True
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
        if self.snapshot_registered:
          await self.time_slicer.unregister(self.workload)
      finally:
        await self.time_slicer.close()

  async def run_once(self) -> None:
    batch = await self.store.get_requests_for_model(self.model_id)
    if not batch:
      await asyncio.sleep(0.1)
      return

    has_shutdown = False
    training_reqs = []
    for req in batch:
      if req.get("request_id") == "SHUTDOWN_SENTINEL" or req.get("op") in {"shutdown", "shutdown_workers"}:
        has_shutdown = True
      else:
        training_reqs.append(req)

    results: list[tuple[str | None, dict[str, Any]]] = []
    failure: Exception | None = None
    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(training_reqs))
      batch_span.set_attribute("model_id", self.model_id)
      if training_reqs:
        print(f"\n[TRAINING REQUESTS] Popped {len(training_reqs)} requests for model: {self.model_id}: {describe_requests(training_reqs)}")
        results, failure = await self.answer_batch(training_reqs)

    for request_id, result in results:
      if request_id is not None:
        await self.store.set_future(request_id, result)
    if failure is not None:
      raise failure

    if self.time_slicer.faulted:
      # This process still holds the accelerator. Exit without unregistering so
      # the grant moves on only once the memory is gone. Exit 0 keeps the pod
      # from restarting on fresh weights mid-run; the run fails on its next call.
      print(f"[WORKER] Time slicer could not park this process: {self.time_slicer.faulted}. Exiting to free the accelerator.")
      await self.exit_gracefully(unregister=False)
    if has_shutdown:
      await self.exit_gracefully()

  async def answer_batch(self, requests: list[dict[str, Any]]) -> tuple[list[tuple[str | None, dict[str, Any]]], Exception | None]:
    """Every request gets an answer: its result, or the failure that stopped the batch."""
    results: list[tuple[str | None, dict[str, Any]]] = []
    try:
      await self.handle_batch(requests, results)
    except Exception as exc:
      answered = {request_id for request_id, _ in results}
      for request in requests:
        request_id = request.get("request_id")
        if request_id and request_id not in answered:
          results.append((request_id, {"type": "RequestFailedResponse", "error_message": f"Trainer worker error: {exc}"}))
      return results, exc
    return results, None

  async def handle_batch(self, requests: list[dict[str, Any]], results: list[tuple[str | None, dict[str, Any]]]) -> None:
    """GPU work under one time-slicer turn; saves need the device only when the worker is not offloaded."""
    save_ops = {"save_state", "save_weights", "save_weights_for_sampler"}
    gpu_reqs = [r for r in requests if r.get("op") not in save_ops]
    save_reqs = [r for r in requests if r.get("op") in save_ops]

    if gpu_reqs:
      async with self.time_slicer.acquire(self.workload):
        await asyncio.to_thread(self.worker.wake_up)
        try:
          for request in gpu_reqs:
            results.append(await self.handle_request(request, self.model_id))
        finally:
          await asyncio.to_thread(self.worker.sleep)

    if not save_reqs:
      return
    if self.worker.cpu_offload:
      for request in save_reqs:
        results.append(await self.handle_request(request, self.model_id))
    else:
      async with self.time_slicer.acquire(self.workload):
        for request in save_reqs:
          results.append(await self.handle_request(request, self.model_id))

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, raw_config, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    full_config = FFTConfig(**{k: v for k, v in raw_config.items() if k in FFTConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, full_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "fine_tuning_type": fine_tuning_type,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "fine_tuning_type": fine_tuning_type,
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [Datum.model_validate(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
      forward_only=bool(payload.get("forward_only", False)),
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    if hasattr(self, "store") and self.store:
      try:
        raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
        current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
        await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
      except Exception as exc:
        print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")
    return result

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.generate,
      payload.get("prompt_tokens", []),
      payload.get("max_tokens", 20),
      payload.get("num_samples", 1),
      payload.get("temperature", 0.0),
      model_id,
      bool(payload.get("prompt_logprobs", False)),
    )
    result["type"] = "sample_completed"
    return result

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.save_state,
      model_id,
      payload["state_path"],
      bool(payload.get("include_optimizer", False)),
      payload.get("kind", "state"),
    )
    return {"path": result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    ref = payload.get("path") or payload.get("sampling_session_id")
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
    local_path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", rel_path)
    await asyncio.to_thread(self.worker.save_state, model_id, local_path, False, "sampler")
    if hasattr(self.store, "redis"):
      num_subs = await self.store.redis.publish(
        f"open_rl:weight_update:{model_id}",
        json.dumps({"weights_path": local_path}),
      )
      print(f"[Trainer] Published weight update signal to {num_subs} subscribers for version path: {local_path}")
    older = older_versions(local_path, SAMPLER_VERSIONS_KEPT)
    for path in older:
      shutil.rmtree(path, ignore_errors=True)
    if older:
      print(f"[Trainer] Removed {len(older)} sampler weight versions older than the newest {SAMPLER_VERSIONS_KEPT}")
    return {
      "path": payload.get("path"),
      "sampling_session_id": payload.get("sampling_session_id"),
      "type": "sampler_weights_saved",
    }

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_model, payload.get("alias") or model_id)
    return {"status": "ok", "type": "weights_saved"}


async def run_training_requests_processor(
  worker: TrainingWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
) -> None:
  store = get_store()
  if isinstance(worker, FFTTrainingWorker):
    time_slicer = time_slicer or time_slicer_client_from_env()
    processor = FFTTrainingRequestsProcessor(store, worker, model_id, time_slicer)
  else:
    processor = LoraTrainingRequestsProcessor(store, worker, model_id, active_tenant_set_id)
  await processor.run()


async def main_async(args: argparse.Namespace) -> None:
  fine_tuning_type = os.getenv("OPEN_RL_FINE_TUNING_TYPE") or ("full" if is_fft_enabled() else "lora")
  if args.model_id:
    try:
      store = get_store()
      raw_meta = await store.get_value(f"open_rl:model_meta:{args.model_id}")
      if raw_meta:
        meta_dict = json.loads(raw_meta)
        fine_tuning_type = meta_dict.get("fine_tuning_type", fine_tuning_type)
    except Exception as exc:
      print(f"[WORKER] Failed to fetch model metadata for {args.model_id}: {exc}")

  is_lora = fine_tuning_type == "lora"
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora})\n")

  worker: TrainingWorker = LoraTrainingWorker() if is_lora else FFTTrainingWorker()
  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and is_lora:
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if not is_lora:
      print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if is_lora:
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    def run_probe_server():
      try:
        uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")
      except Exception as exc:
        print(f"[WORKER] Probe server on port 8000 skipped: {exc}")

    threading.Thread(target=run_probe_server, daemon=True).start()

  await run_training_requests_processor(
    worker,
    args.model_id,
    active_tenant_set_id=getattr(args, "active_tenant_set_id", None),
  )


def start_request_processing_loop() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated trainer worker drains.")
  parser.add_argument("--active-tenant-set-id", help="Active tenant rotation set ID for LoRA workers (e.g. Qwen/Qwen3-0.6B-1).")
  args = parser.parse_args()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")

  asyncio.run(main_async(args))


if __name__ == "__main__":
  start_request_processing_loop()
