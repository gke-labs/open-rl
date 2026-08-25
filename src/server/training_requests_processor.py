# This file contains the training request processor implementation for Open-RL.

import argparse
import asyncio
import os
import threading
import traceback
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from server.store import RequestStore, get_store
from snapshot_agent.client import SnapshotAgentClient
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import Datum

tracer = trace.get_tracer(__name__)


TrainingWorker = FFTTrainingWorker | LoraTrainingWorker


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def create_snapshot_agent_client(socket_path: str) -> SnapshotAgentClient:
  return SnapshotAgentClient(socket_path)


def parse_datum(raw: dict[str, Any]) -> Datum:
  """Convert Tinker wire-format datum with chunks to the flat Datum type."""
  tokens: list[int] = []
  for chunk in raw.get("model_input", {}).get("chunks", []):
    tokens.extend(chunk.get("tokens", []))

  loss_fn_inputs = {
    key: value if isinstance(value, dict) and "data" in value else {"data": value} for key, value in raw.get("loss_fn_inputs", {}).items()
  }
  return Datum(model_input=tokens, loss_fn_inputs=loss_fn_inputs)


class TrainingRequestsProcessor(Protocol):
  store: RequestStore

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    request_id = raw_request.get("request_id")
    token = None

    try:
      op = raw_request["op"]
      request_id = raw_request["request_id"]
      resolved_model_id = model_id or raw_request.get("model_id") or "default"

      carrier = raw_request.get("trace_context")
      ctx = propagate.extract(carrier) if carrier else None
      token = otel_context.attach(ctx) if ctx else None

      result = await self.dispatch_operation(op, raw_request.get("payload", {}), resolved_model_id)
      await self.store.set_future(request_id, result)
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      await self.store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
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


class LoraTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(self, store: RequestStore, worker: LoraTrainingWorker):
    self.store = store
    self.worker = worker

  async def run(self) -> None:
    print("[WORKER] LoRA training requests processor started.")

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
    batch = await self.store.get_requests()
    if not batch:
      await asyncio.sleep(0.1)
      return

    model_id = batch[0].get("model_id", "default")

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}")
      for request in batch:
        await self.process_request(request, model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    raw_config = payload.get("lora_config") or {}
    lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, payload["base_model"], model_id, lora_config)
    return {
      "base_model": payload["base_model"],
      "model_id": model_id,
      "rank": lora_config.rank,
      "training_kind": "lora",
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model"),
      "model_id": result.get("model_id", model_id),
      "training_kind": "lora",
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [parse_datum(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
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


class FFTTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: FFTTrainingWorker,
    model_id: str | None,
    snapshot_client: SnapshotAgentClient,
  ):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")
    if not model_id:
      raise RuntimeError("A dedicated FFT worker needs --model-id so it knows which per-model queue to drain")

    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.pid = os.getpid()
    self.snapshot_client = snapshot_client
    self.snapshot_registered = False

  async def run(self) -> None:
    print("[WORKER] Full fine-tuning training requests processor started.")

    try:
      await self.snapshot_client.register(self.pid)
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
          await self.snapshot_client.unregister(self.pid)
      finally:
        await self.snapshot_client.close()

  async def run_once(self) -> None:
    batch = await self.store.get_requests_for_model(self.model_id)
    if not batch:
      await asyncio.sleep(0.1)
      return

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", self.model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {self.model_id}")
      async with self.snapshot_client.acquire(self.pid):
        for request in batch:
          await self.process_request(request, self.model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    raw_config = payload.get("full_config") or {}
    full_config = FFTConfig(**{k: v for k, v in raw_config.items() if k in FFTConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, payload["base_model"], model_id, full_config)
    return {
      "base_model": payload["base_model"],
      "model_id": model_id,
      "training_kind": "full",
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model"),
      "model_id": result.get("model_id", model_id),
      "training_kind": "full",
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [parse_datum(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
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
  snapshot_client: SnapshotAgentClient | None = None,
) -> None:
  store = get_store()
  if isinstance(worker, FFTTrainingWorker):
    if snapshot_client is None:
      snapshot_socket = os.getenv("OPEN_RL_SNAPSHOT_AGENT_SOCKET", "/tmp/open-rl/snapshot-agent.sock")
      snapshot_client = create_snapshot_agent_client(snapshot_socket)
    processor = FFTTrainingRequestsProcessor(store, worker, model_id, snapshot_client)
  else:
    processor = LoraTrainingRequestsProcessor(store, worker)
  await processor.run()


def start_request_processing_loop() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated FFT worker drains.")
  args = parser.parse_args()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> FFT enabled: {is_fft_enabled()}\n")

  worker: TrainingWorker = FFTTrainingWorker() if is_fft_enabled() else LoraTrainingWorker()
  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and not is_fft_enabled():
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if is_fft_enabled():
      print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if not is_fft_enabled():
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    def run_probe_server():
      uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")

    threading.Thread(target=run_probe_server, daemon=True).start()
  asyncio.run(run_training_requests_processor(worker, args.model_id))


if __name__ == "__main__":
  start_request_processing_loop()
