# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from server.store import get_store
from server.worker_launch_processor import (
  FFTWorkerManager,
  WorkerLaunchProcessor,
)

store = get_store()

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
else:
  print("OpenTelemetry: No exporter configured (ENABLE_GCP_TRACE=0)")


class FilterNoisyEndpoints(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "retrieve_future" not in msg and "session_heartbeat" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")


# *** Helpers ***


def is_single_process_mode() -> bool:
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  if sampling_backend := os.getenv("SAMPLING_BACKEND"):
    return sampling_backend.lower()
  return "torch" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  return os.getenv("BASE_MODEL")


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def sampler_session_id(model_id: str, seq_id: int | str) -> str:
  return f"tinker://{model_id}/sampler_weights/sampler-{seq_id}"


def sampler_weights_path(model_id: str, name: str) -> str:
  return f"tinker://{model_id}/sampler_weights/{name}"


def checkpoint_state_path(model_id: str, name: str) -> str:
  if os.path.isabs(name):
    return name
  return os.path.join(TMP_DIR, "checkpoints", model_id, "weights", name)


def base_model_id_from_sampling_ref(model_id: str | None) -> str | None:
  if not model_id:
    return None

  if model_id.startswith("tinker://"):
    path = model_id[len("tinker://") :]
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "sampler_weights":
      return parts[0]
    return path

  return model_id.split("-samp-")[0]


def is_sampler_weights_ref(model_id: str | None) -> bool:
  if not model_id or not model_id.startswith("tinker://"):
    return False

  path = model_id[len("tinker://") :]
  parts = path.split("/")
  return len(parts) >= 3 and parts[1] == "sampler_weights"


def make_training_request(
  op: str,
  model_id: str | None,
  payload: dict,
  request_id: str | None = None,
) -> dict:
  request = {
    "request_id": request_id or str(uuid.uuid4()),
    "op": op,
    "payload": payload,
  }
  if model_id is not None:
    request["model_id"] = model_id
  return request


async def enqueue(request: dict) -> str:
  """Create a pending future, inject trace context, push to store. Returns req_id."""
  request_id = request["request_id"]
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(request_id, {"status": "pending"})
  await store.put_request({**request, "trace_context": carrier})
  return request_id


async def enqueue_worker_launch(request: dict) -> str:
  """Create a pending future and push a create-model request to the worker launch queue."""
  request_id = request["request_id"]
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(request_id, {"status": "pending"})
  await store.put_worker_launch_request({**request, "trace_context": carrier})
  return request_id


async def preflight_vllm() -> None:
  """If SAMPLING_BACKEND=vllm, verify the vLLM worker is reachable at VLLM_URL.

  Prints a clear, actionable error instead of letting the first asample
  request fall through with a raw httpx connection refused.
  """
  if get_sampler_backend() != "vllm":
    return
  healthz = f"{VLLM_URL.rstrip('/')}/healthz"
  try:
    async with httpx.AsyncClient(timeout=3.0) as client:
      resp = await client.get(healthz)
      resp.raise_for_status()
  except Exception as exc:
    raise RuntimeError(
      f"SAMPLING_BACKEND=vllm but no vLLM worker is reachable at {VLLM_URL}.\n"
      f"Start it first with:  make vllm BASE_MODEL={os.getenv('BASE_MODEL') or '<model-id>'}"
    ) from exc


def translate_future_result(result: dict) -> dict:
  result_type = result.get("type")
  if result_type in {"model_created", "model_loaded_from_state"}:
    # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
    # even for full fine-tuning jobs.
    response = {
      "model_id": result["model_id"],
      "is_lora": True,
      "type": "create_model" if result_type == "model_created" else "create_model_from_state",
    }
    if "rank" in result:
      response["lora_rank"] = result["rank"]
    elif result.get("training_kind") == "full":
      response["lora_rank"] = 16
    if result.get("base_model"):
      response["base_model"] = result["base_model"]
    return response

  public_type_by_internal_type = {
    "forward_backward_completed": "forward_backward",
    "optim_step_completed": "optim_step",
    "sample_completed": "sample",
    "state_saved": "save_weights",
    "weights_loaded": "load_weights",
    "sampler_weights_saved": "save_weights_for_sampler",
    "weights_saved": "save_weights",
  }
  if result_type in public_type_by_internal_type:
    response = dict(result)
    response["type"] = public_type_by_internal_type[result_type]
    return response

  return result


@asynccontextmanager
async def lifespan(_: FastAPI):
  task = None
  fft_worker_manager = None
  worker_launch_task = None
  if is_fft_enabled():
    fft_worker_manager = FFTWorkerManager()
    worker_launch_processor = WorkerLaunchProcessor(store, fft_worker_manager)
    worker_launch_task = asyncio.create_task(worker_launch_processor.run())
  if is_single_process_mode():
    base_model = os.getenv("BASE_MODEL")
    print("\n" + "=" * 50)
    print(" Open-RL Single-Process Mode")
    print("=" * 50)
    print(f"-> Base model: {base_model or 'unset'}")
    print(f"-> Sampling backend: {get_sampler_backend()}")
    print(f"-> FFT enabled     : {is_fft_enabled()}")
    print("-> Server mode     : API server + worker loop in one process\n")
    await preflight_vllm()
    if not is_fft_enabled():
      from server import training_requests_processor

      worker = training_requests_processor.LoraTrainingWorker()
      if base_model:
        await asyncio.to_thread(worker.load_base_model, base_model)
      task = asyncio.create_task(training_requests_processor.run_training_requests_processor(worker))
  try:
    yield
  finally:
    if task is not None:
      task.cancel()
    if worker_launch_task is not None:
      worker_launch_task.cancel()
    if fft_worker_manager is not None:
      fft_worker_manager.shutdown_all()


app = FastAPI(title="Open-RL Server MVP", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")


# *** ServiceClient endpoints ***
@app.get("/api/v1/healthz")
async def health_check():
  return {"status": "ok"}


@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
  model_name = get_default_model_name()
  return {
    "supported_models": [{"model_name": model_name}] if model_name else [],
    "default_model": model_name,
    "single_process": is_single_process_mode(),
  }


@app.post("/api/v1/client/config")
async def client_config(_: dict):
  return {
    "pjwt_auth_enabled": False,
    "credential_default_source": "api_key",
    "sample_dispatch_bytes_semaphore_size": 10 * 1024 * 1024,
    "inflight_response_bytes_semaphore_size": 50 * 1024 * 1024,
  }


@app.post("/api/v1/create_session")
async def create_session(_: dict):
  return {"session_id": "sess-real-123", "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(_: dict):
  return {"type": "session_heartbeat"}


@app.post("/api/v1/create_model")
async def create_model(req: dict):
  """ServiceClient.create_lora_training_client_async()"""
  base_model = req.get("base_model")
  if not base_model:
    return JSONResponse(status_code=400, content={"error": "base_model is required"})
  model_id = str(uuid.uuid4())
  command = make_training_request(
    "create_model",
    model_id,
    {
      "base_model": base_model,
      "lora_config": req.get("lora_config") or {},
      "full_config": req.get("full_config") or {},
    },
    request_id=model_id,
  )
  req_id = await enqueue_worker_launch(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(req: dict):
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.get("state_path")
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "state_path is required"})
  # Resolve relative names under TMP_DIR/checkpoints, leave absolute paths alone.
  resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  model_id = str(uuid.uuid4())
  command = make_training_request(
    "create_model_from_state",
    model_id,
    {
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("restore_optimizer", False)),
    },
    request_id=model_id,
  )
  req_id = await enqueue_worker_launch(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(req: dict):
  """ServiceClient — model metadata for the training client."""
  model_name = get_default_model_name()
  if not model_name:
    return JSONResponse(status_code=404, content={"error": "No base model is configured"})
  # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
  # even when this process is running a full fine-tuning worker.
  result = {
    "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
    "model_id": req.get("model_id", "model-live-123"),
    "is_lora": True,
    "lora_rank": 16,
    "model_name": model_name,
    "type": "get_info",
  }
  return result


@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
  """ServiceClient — poll for async request results."""
  request_id = req.get("request_id")
  if not request_id:
    return JSONResponse(status_code=400, content={"error": "request_id is required"})

  result = await store.get_future(request_id, timeout=60.0)
  if result is None:
    return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
  if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
    return JSONResponse(status_code=400, content=result)
  if isinstance(result, dict):
    return translate_future_result(result)
  return result


# *** TrainingClient endpoints ***
@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
  """TrainingClient.forward_backward_async()"""
  fwd_input = req.get("forward_backward_input", {})
  req_id = await enqueue(
    make_training_request(
      "forward_backward",
      req.get("model_id"),
      {
        "data": fwd_input.get("data", []),
        "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
        "loss_config": fwd_input.get("loss_fn_config", {}),
      },
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
  """TrainingClient.optim_step_async()"""
  req_id = await enqueue(
    make_training_request(
      "optim_step",
      req.get("model_id"),
      {"adam_params": req.get("adam_params", {})},
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
  """TrainingClient.save_weights_for_sampler().

  The SDK uses this for both named sampler checkpoints and ephemeral
  save_weights_and_get_sampling_client() snapshots. Route it through the training
  queue so the sampler always sees weights saved after prior training requests.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  seq_id = req.get("sampling_session_seq_id") or int(time.time() * 1000)
  alias = req.get("name") or req.get("alias") or req.get("path")

  session_id = sampler_session_id(model_id, seq_id)
  req_id = await enqueue(
    make_training_request(
      "save_weights_for_sampler",
      model_id,
      {
        "alias": alias,
        "path": sampler_weights_path(model_id, alias) if alias else None,
        "sampling_session_id": session_id,
      },
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights")
async def save_weights(req: dict):
  """TrainingClient.save_weights() / save_state().

  This is the endpoint the tinker SDK hits for both save_weights() and save_state().
  The SDK sends save_state(name) as `path`; we resolve that checkpoint name to
  TMP_DIR/checkpoints/<model_id>/weights/<path> so separate training jobs do not
  overwrite each other's named checkpoints.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  seq_id = req.get("seq_id") or int(time.time() * 1000)
  alias = req.get("path") or f"{model_id}-samp-{seq_id}"
  state_path = checkpoint_state_path(model_id, alias)

  req_id = str(uuid.uuid4())
  await enqueue(
    make_training_request(
      "save_state",
      model_id,
      {
        "state_path": state_path,
        "include_optimizer": bool(req.get("include_optimizer", False)),
        "kind": "weights",
      },
      request_id=req_id,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/load_weights")
async def load_weights(req: dict):
  """TrainingClient.load_state() / load_state_with_optimizer()."""
  model_id = req.get("model_id")
  state_path = req.get("path")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "path is required"})

  resolved_path = checkpoint_state_path(model_id, state_path)
  req_id = await enqueue(
    make_training_request(
      "load_weights",
      model_id,
      {
        "state_path": resolved_path,
        "restore_optimizer": bool(req.get("optimizer", False)),
      },
    )
  )
  return {"request_id": req_id}


# *** SamplingClient endpoints ***
@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
  """ServiceClient.create_sampling_client()"""
  model_path = req.get("model_path")
  base_model = req.get("base_model")
  model_id = req.get("model_id")

  if model_path and model_path.startswith("tinker://"):
    sess_id = model_path
  elif base_model:
    sess_id = base_model
  else:
    sess_id = model_id or "samp-session-live-123"

  return {"sampling_session_id": sess_id, "type": "create_sampling_session"}


@app.post("/api/v1/asample")
async def asample(req: dict):
  """SamplingClient.sample_async()"""
  chunks = req.get("prompt", {}).get("chunks", [])
  prompt = []
  for chunk in chunks:
    prompt.extend(chunk.get("tokens", []))
  params = req.get("sampling_params", {})
  max_tokens = params.get("max_tokens", 20)
  temperature = params.get("temperature", 1.0)
  stop = params.get("stop")
  top_p = params.get("top_p", 1.0)
  top_k = params.get("top_k", -1)
  num_samples = req.get("num_samples", 1)
  include_prompt_logprobs = req.get("prompt_logprobs", req.get("include_prompt_logprobs", False))

  model_id = req.get("model_id") or req.get("sampling_session_id")
  base_model_id = base_model_id_from_sampling_ref(model_id)

  if get_sampler_backend() == "torch":
    req_id = await enqueue(
      make_training_request(
        "sample",
        base_model_id or model_id,
        {
          "prompt_tokens": prompt,
          "max_tokens": max_tokens,
          "temperature": temperature,
          "num_samples": num_samples,
          "prompt_logprobs": bool(include_prompt_logprobs),
        },
      )
    )
    return {"request_id": req_id}

  # vLLM backend
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  lora_path = os.path.join(TMP_DIR, "peft", base_model_id, base_model_id) if is_sampler_weights_ref(model_id) else None
  headers: dict[str, str] = {"Content-Type": "application/json"}
  propagate.inject(headers)

  try:
    async with httpx.AsyncClient(timeout=120.0) as client:
      resp = await client.post(
        f"{VLLM_URL.rstrip('/')}/generate",
        json={
          "request_id": req_id,
          "prompt_token_ids": prompt,
          "max_tokens": max_tokens,
          "temperature": temperature,
          "stop": stop,
          "top_p": top_p,
          "top_k": top_k,
          "num_samples": num_samples,
          "lora_id": model_id,
          "lora_path": lora_path,
          "include_prompt_logprobs": include_prompt_logprobs,
        },
        headers=headers,
      )
      resp.raise_for_status()
      data = resp.json()
      if data.get("type") != "RequestFailedResponse":
        data["type"] = "sample"
      await store.set_future(req_id, data)
  except Exception as e:
    traceback.print_exc()
    await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})

  return {"request_id": req_id}


# *** CLI endpoints ***


@app.get("/api/v1/list_adapters")
async def list_adapters():
  """CLI `list` — scan the peft directory for saved adapters."""
  import json

  peft_dir = os.path.join(TMP_DIR, "peft")
  adapters = []

  if os.path.exists(peft_dir):
    for entry in sorted(os.scandir(peft_dir), key=lambda e: e.stat().st_ctime, reverse=True):
      if not entry.is_dir():
        continue
      info = {"model_id": entry.name, "created_at": entry.stat().st_ctime, "timestamp": entry.stat().st_ctime, "alias": None}
      metadata_path = os.path.join(entry.path, "metadata.json")
      if os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            info.update(json.load(f))
        except Exception:
          pass
      adapters.append(info)

  return {"adapters": adapters}


# *** Internal ***


@app.post("/api/v1/telemetry")
async def telemetry(_: dict):
  return {"status": "accepted"}
