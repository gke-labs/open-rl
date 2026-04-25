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
from store import get_store

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


class _FilterNoisyEndpoints(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "retrieve_future" not in msg and "session_heartbeat" not in msg


logging.getLogger("uvicorn.access").addFilter(_FilterNoisyEndpoints())

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")


# *** Helpers ***


def is_single_process_mode() -> bool:
  explicit = os.getenv("SINGLE_PROCESS")
  if explicit is not None:
    return explicit == "1"
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  if sampler := os.getenv("SAMPLER"):
    return sampler.lower()
  return "torch" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  if is_single_process_mode():
    import clock_cycle

    if clock_cycle.engine.base_model_name:
      return clock_cycle.engine.base_model_name
  return os.getenv("BASE_MODEL")


async def _enqueue(payload: dict) -> str:
  """Create a pending future, inject trace context, push to store. Returns req_id."""
  req_id = payload.get("req_id") or str(uuid.uuid4())
  payload["req_id"] = req_id
  carrier: dict = {}
  propagate.inject(carrier)
  payload["trace_context"] = carrier
  await store.set_future(req_id, {"status": "pending"})
  await store.put_request(payload)
  return req_id


async def _preflight_vllm() -> None:
  """If SAMPLER=vllm, verify the vLLM worker is reachable at VLLM_URL.

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
      f"SAMPLER=vllm but no vLLM worker is reachable at {VLLM_URL}.\n"
      f"Start it first with:  make vllm BASE_MODEL={os.getenv('BASE_MODEL') or '<model-id>'}"
    ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
  task = None
  if is_single_process_mode():
    import clock_cycle

    base_model = os.getenv("BASE_MODEL")
    print("\n" + "=" * 50)
    print(" Open-RL Single-Process Mode")
    print("=" * 50)
    print(f"-> Base model: {base_model or 'unset'}")
    print(f"-> Sampler   : {get_sampler_backend()}")
    print("-> Backend   : gateway + worker loop in one process\n")
    await _preflight_vllm()
    if base_model:
      await asyncio.to_thread(clock_cycle.engine.load_base_model, base_model)
    task = asyncio.create_task(clock_cycle.clock_cycle_loop())
  try:
    yield
  finally:
    if task is not None:
      task.cancel()


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
  req_id = await _enqueue(
    {
      "req_id": model_id,
      "model_id": model_id,
      "type": "create_model",
      "base_model": base_model,
      "lora_config": req.get("lora_config") or {},
    }
  )
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
  req_id = await _enqueue(
    {
      "req_id": model_id,
      "model_id": model_id,
      "type": "create_model_from_state",
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("restore_optimizer", False)),
    }
  )
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(req: dict):
  """ServiceClient — model metadata for the training client."""
  model_name = get_default_model_name()
  if not model_name:
    return JSONResponse(status_code=404, content={"error": "No base model is configured"})
  return {
    "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
    "model_id": req.get("model_id", "model-live-123"),
    "is_lora": True,
    "lora_rank": 16,
    "model_name": model_name,
    "type": "get_info",
  }


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
  return result


# *** TrainingClient endpoints ***
@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
  """TrainingClient.forward_backward_async()"""
  fwd_input = req.get("forward_backward_input", {})
  req_id = await _enqueue(
    {
      "model_id": req.get("model_id"),
      "type": "forward_backward",
      "data": fwd_input.get("data", []),
      "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
      "loss_config": fwd_input.get("loss_fn_config", {}),
    }
  )
  return {"request_id": req_id}


@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
  """TrainingClient.optim_step_async()"""
  req_id = await _enqueue(
    {
      "model_id": req.get("model_id"),
      "type": "optim_step",
      "adam_params": req.get("adam_params", {}),
    }
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
  """TrainingClient.save_weights_for_sampler().

  The SDK uses this for both named sampler checkpoints and ephemeral
  save_weights_and_get_sampling_client() snapshots. Route it through the trainer
  queue so the sampler always sees weights saved after prior training requests.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  seq_id = req.get("sampling_session_seq_id") or int(time.time() * 1000)
  alias = req.get("name") or req.get("alias") or req.get("path")

  session_id = f"{model_id}-samp-{seq_id}"
  req_id = await _enqueue(
    {
      "model_id": model_id,
      "type": "save_weights_for_sampler",
      "alias": alias,
      "path": f"tinker://{session_id}" if alias else None,
      "sampling_session_id": session_id,
    }
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights")
async def save_weights(req: dict):
  """TrainingClient.save_weights() / save_state() — persists adapter to TMP_DIR/checkpoints/<alias>.

  This is the endpoint the tinker SDK hits for both save_weights() and save_state().
  When `path` is provided we treat it as a named checkpoint alias and resolve to
  TMP_DIR/checkpoints/<alias> (or leave absolute paths alone), so subsequent
  `create_training_client_from_state(alias)` calls can find the adapter.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  seq_id = req.get("seq_id") or int(time.time() * 1000)
  alias = req.get("path") or f"{model_id}-samp-{seq_id}"
  state_path = alias if os.path.isabs(alias) else os.path.join(TMP_DIR, "checkpoints", alias)

  req_id = str(uuid.uuid4())
  await _enqueue(
    {
      "req_id": req_id,
      "model_id": model_id,
      "type": "save_state",
      "state_path": state_path,
      "include_optimizer": bool(req.get("include_optimizer", False)),
      "kind": "weights",
    }
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

  resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  req_id = await _enqueue(
    {
      "model_id": model_id,
      "type": "load_weights",
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("optimizer", False)),
    }
  )
  return {"request_id": req_id}


# *** SamplingClient endpoints ***
@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
  """ServiceClient.create_sampling_client()"""
  model_path = req.get("model_path")
  model_id = req.get("model_id")

  if model_path and model_path.startswith("tinker://"):
    sess_id = model_path[len("tinker://") :]
  else:
    sess_id = model_id or "samp-session-live-123"

  return {"sampling_session_id": sess_id, "type": "create_sampling_session"}


@app.post("/api/v1/asample")
async def asample(req: dict):
  """SamplingClient.sample_async()"""
  prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
  params = req.get("sampling_params", {})
  max_tokens = params.get("max_tokens", 20)
  temperature = params.get("temperature", 1.0)
  stop = params.get("stop")
  top_p = params.get("top_p", 1.0)
  top_k = params.get("top_k", -1)
  num_samples = req.get("num_samples", 1)

  model_id = req.get("model_id") or req.get("sampling_session_id")
  if model_id and model_id.startswith("tinker://"):
    model_id = model_id[len("tinker://") :]
  base_model_id = model_id.split("-samp-")[0] if model_id else None

  if get_sampler_backend() == "torch":
    req_id = await _enqueue(
      {
        "model_id": base_model_id or model_id,
        "type": "sample",
        "prompt_tokens": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "top_p": top_p,
        "top_k": top_k,
        "num_samples": num_samples,
      }
    )
    return {"request_id": req_id}

  # vLLM backend
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  lora_path = os.path.join(TMP_DIR, "peft", base_model_id, base_model_id) if base_model_id else None
  headers: dict[str, str] = {"Content-Type": "application/json"}
  propagate.inject(headers)

  try:
    async with httpx.AsyncClient(timeout=60.0) as client:
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
