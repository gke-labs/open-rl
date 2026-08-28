# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from server.model_metadata import TrainingModelMetadata, extract_weight_sync_config
from server.store import get_store
from server.worker_manager import WorkerManager, create_worker_manager

store = get_store()
worker_manager: WorkerManager | None = None

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


def resolve_sampler_weights_path(model_id: str) -> str:
  """Resolves a model_id or tinker session reference to a fully-qualified step-specific weights path on disk."""
  rel_path = model_id[len("tinker://") :] if model_id.startswith("tinker://") else model_id.lstrip("/")
  local_path = os.path.join(TMP_DIR, "sampler_full", rel_path)
  weights_path = local_path
  if not os.path.basename(weights_path).startswith("sampler-"):
    sampler_weights_dir = os.path.join(weights_path, "sampler_weights")
    if os.path.exists(sampler_weights_dir):
      try:
        steps = [int(d.split("-")[1]) for d in os.listdir(sampler_weights_dir) if d.startswith("sampler-")]
        if steps:
          weights_path = os.path.join(sampler_weights_dir, f"sampler-{max(steps)}")
      except Exception as e:
        print(f"[GATEWAY] Warning: Failed parsing step subdirectories in {sampler_weights_dir}: {e}")
  return weights_path


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


async def _extract_and_persist_model_metadata(
  req: dict[str, Any],
  request: Request | None = None,
  default_fine_tuning_type: str = "lora",
) -> str:
  """Extract and normalize model configuration from headers and payload, persisting TrainingModelMetadata exactly once."""
  base_model = req.get("base_model")
  if not base_model and default_fine_tuning_type != "restored":
    raise ValueError("base_model is required in request payload")

  full_config = dict(req.get("full_config") or {})
  lora_config = dict(req.get("lora_config") or {})

  headers = request.headers if (request and hasattr(request, "headers")) else {}
  weight_sync_cfg = extract_weight_sync_config(headers)

  fine_tuning_type = default_fine_tuning_type
  if request and hasattr(request, "headers") and "x-open-rl-fine-tuning-type" in request.headers:
    h_val = (request.headers.get("x-open-rl-fine-tuning-type") or "").lower()
    if h_val == "full":
      fine_tuning_type = "full"
    elif h_val == "lora":
      fine_tuning_type = "lora"

  if fine_tuning_type == "full" and not is_fft_enabled():
    raise ValueError("Full Fine-Tuning (FFT) is disabled on this Open-RL Gateway instance")

  if fine_tuning_type != "full" and default_fine_tuning_type != "restored":
    fine_tuning_type = "lora"

  full_config["weight_sync_strategy"] = weight_sync_cfg.strategy

  model_id = str(uuid.uuid4())
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    fine_tuning_type=fine_tuning_type,
    weight_sync_config=weight_sync_cfg,
    full_config=full_config,
    lora_config=lora_config,
  )
  await store.set_value(f"open_rl:model_meta:{model_id}", json.dumps(meta_obj.to_dict()))

  return model_id


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


async def _resolve_active_set_id(model_id: str | None) -> str | None:
  if not model_id or not hasattr(store, "get_model_metadata"):
    return None
  meta = await store.get_model_metadata(model_id)
  if meta and meta.get("fine_tuning_type") == "lora" and meta.get("base_model"):
    return f"{meta['base_model']}-1"
  return None


async def enqueue(request: dict) -> str:
  """Create a pending future, inject trace context, push to store. Returns req_id."""
  request_id = request["request_id"]
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(request_id, {"status": "pending"})

  active_set_id = await _resolve_active_set_id(request.get("model_id"))
  await store.put_request({**request, "trace_context": carrier}, active_set_id=active_set_id)
  return request_id


async def launch_worker_and_enqueue(request: dict) -> str:
  """Ensure the model's dedicated trainer worker exists, then enqueue onto its queue.

  The launcher is idempotent per model_id, and Kubernetes (or the local process
  table) owns the worker's lifecycle from here; there is no separate launch
  queue. Launch failures resolve the future immediately so clients don't long-poll
  a request that can never be served.
  """
  assert worker_manager is not None, "Worker manager is initialized by the app lifespan"
  request_id = request["request_id"]
  await store.set_future(request_id, {"status": "pending"})
  try:
    await asyncio.to_thread(worker_manager.launch_trainer, request["model_id"])
  except Exception as exc:
    traceback.print_exc()
    await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    return request_id
  return await enqueue(request)


async def ensure_sampler_launched(model_id: str) -> None:
  if worker_manager is not None and get_sampler_backend() == "vllm":
    try:
      await asyncio.to_thread(worker_manager.launch_sampler, model_id)
    except Exception:
      traceback.print_exc()


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
    elif result.get("fine_tuning_type") == "full":
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


async def run_claim_reconciler(manager: WorkerManager, interval: float) -> None:
  """Periodically reclaim dynamic DRA claims left behind by finished workers.

  Nothing else deletes them: the scheduler provisions a claim whenever no
  eligible one is free, so without this loop every completed job strands a GPU
  claim until an operator removes it by hand.
  """
  while True:
    await asyncio.sleep(interval)
    try:
      deleted = await asyncio.to_thread(manager.reconcile_managed_claims)
      if deleted:
        print(f"[GATEWAY] Reclaimed {len(deleted)} unused DRA claim(s): {', '.join(deleted)}")
    except asyncio.CancelledError:
      raise
    except Exception:
      traceback.print_exc()


def start_claim_reconciler(manager: WorkerManager | None) -> asyncio.Task | None:
  """Start the reconcile loop when the worker manager provisions claims (Kubernetes mode only)."""
  if manager is None or not hasattr(manager, "reconcile_managed_claims"):
    return None
  interval = float(os.getenv("OPEN_RL_CLAIM_RECONCILE_INTERVAL_SECONDS", "300"))
  if interval <= 0:
    print("[GATEWAY] DRA claim reconciliation disabled (OPEN_RL_CLAIM_RECONCILE_INTERVAL_SECONDS <= 0)")
    return None
  print(f"[GATEWAY] DRA claim reconciliation every {interval:.0f}s")
  return asyncio.create_task(run_claim_reconciler(manager, interval))


@asynccontextmanager
async def lifespan(_: FastAPI):
  global worker_manager
  task = None
  reconcile_task = None
  if is_fft_enabled() or os.getenv("REDIS_URL") or os.getenv("OPEN_RL_WORKER_MANAGER"):
    worker_manager = create_worker_manager()
    reconcile_task = start_claim_reconciler(worker_manager)
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
    if reconcile_task is not None:
      reconcile_task.cancel()
    if worker_manager is not None:
      worker_manager.shutdown_all()
      worker_manager = None


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


def _get_request(request: Request) -> Request:
  return request


@app.post("/api/v1/create_model")
async def create_model(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_lora_training_client_async()"""
  try:
    model_id = await _extract_and_persist_model_metadata(req, request, default_fine_tuning_type="lora")
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model",
    model_id,
    {},
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if worker_manager is not None else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(req: dict):
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  meta_dict = None
  try:
    raw_meta = await store.get_value(f"open_rl:model_meta:{model_id}")
    if raw_meta:
      meta_dict = json.loads(raw_meta)
  except Exception:
    pass
  is_lora = meta_dict and meta_dict.get("fine_tuning_type") == "lora"
  if is_fft_enabled() and not is_lora:
    print(f"[GATEWAY] Requesting shutdown of workers for model {model_id}...")
    await store.put_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id, "op": "shutdown_workers"})
    await store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
  now = time.time()
  await store.update_job_metadata(model_id, {"status": "completed", "completed_at": now, "updated_at": now})
  return {"status": "ok"}


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.get("state_path")
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "state_path is required"})
  # Resolve relative names under TMP_DIR/checkpoints, leave absolute paths alone.
  resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  try:
    model_id = await _extract_and_persist_model_metadata(req, request, default_fine_tuning_type="restored")
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model_from_state",
    model_id,
    {
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("restore_optimizer", False)),
    },
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if worker_manager is not None else await enqueue(command)
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
@app.post("/api/v1/forward")
async def forward(req: dict):
  """TrainingClient.forward_async()"""
  fwd_input = req.get("forward_input") or req.get("forward_backward_input") or {}
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

  await ensure_sampler_launched(model_id)
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
    path = model_path[len("tinker://") :]
    parts = path.split("/")
    target_model_id = parts[0]
  elif base_model:
    sess_id = base_model
    target_model_id = base_model
  else:
    sess_id = model_id or "samp-session-live-123"
    target_model_id = sess_id

  model_meta = await store.get_model_metadata(target_model_id) if target_model_id else None
  fine_tuning_type = model_meta.get("fine_tuning_type", "lora") if model_meta else "lora"
  ready_check_id = (model_meta.get("base_model") or target_model_id) if (fine_tuning_type == "lora" and model_meta) else target_model_id

  if get_sampler_backend() == "vllm" and ready_check_id:
    await ensure_sampler_launched(ready_check_id)
    s = get_store()
    if hasattr(s, "redis"):
      print(f"[GATEWAY] Waiting for dynamic vLLM sampler worker to be ready for model {ready_check_id}...")
      start_time = time.monotonic()
      while True:
        is_ready = await s.redis.get(f"open_rl:sampler_ready:{ready_check_id}")
        if is_ready == "1" or is_ready == b"1":
          print(f"[GATEWAY] Dynamic vLLM sampler worker is ready! (took {time.monotonic() - start_time:.2f}s)")
          break
        if time.monotonic() - start_time > 300:
          raise TimeoutError("Timed out waiting for dynamic vLLM sampler worker to be ready")
        await asyncio.sleep(1)

  return {"sampling_session_id": sess_id, "type": "create_sampling_session"}


@app.get("/api/v1/samplers/{sampler_id:path}")
async def get_sampler(sampler_id: str):
  """SamplingClient.get_tokenizer() and .get_base_model().

  The sampler id is whatever create_sampling_session handed back, so it is
  either a base model name or a `tinker://<model_id>/sampler_weights/...` path;
  `:path` on the route is what lets the slash in either form through. Both
  resolve to the base model, which is all the client wants -- it loads the
  tokenizer from the Hub itself.
  """
  base_model_id = base_model_id_from_sampling_ref(sampler_id)
  model_meta = await store.get_model_metadata(base_model_id) if base_model_id else None
  base_model = (model_meta or {}).get("base_model") or base_model_id or get_default_model_name()
  if not base_model:
    return JSONResponse(status_code=404, content={"error": f"Unknown sampler {sampler_id}"})
  return {
    "sampler_id": sampler_id,
    "base_model": base_model,
    "model_path": sampler_id if sampler_id.startswith("tinker://") else None,
  }


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
  lookup_id = base_model_id or model_id

  if get_sampler_backend() == "torch":
    req_id = await enqueue(
      make_training_request(
        "sample",
        lookup_id,
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
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(req_id, {"status": "pending"})

  model_meta = await store.get_model_metadata(lookup_id)
  fine_tuning_type = model_meta.get("fine_tuning_type", "lora") if model_meta else "lora"

  if fine_tuning_type == "lora":
    weights_path = None
    lora_id = model_id
    peft_dir = os.path.join(TMP_DIR, "peft", lookup_id, lookup_id)
    lora_path = peft_dir if os.path.exists(peft_dir) else None
    queue_id = (model_meta.get("base_model") if model_meta else None) or lookup_id
  else:
    resolved_path = resolve_sampler_weights_path(model_id) if is_sampler_weights_ref(model_id) or is_fft_enabled() else None
    weights_path = resolved_path
    lora_id = None
    lora_path = None
    queue_id = lookup_id

  sampling_req = {
    "request_id": req_id,
    "prompt_token_ids": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "stop": stop,
    "top_p": top_p,
    "top_k": top_k,
    "num_samples": num_samples,
    "lora_id": lora_id,
    "lora_path": lora_path,
    "weights_path": weights_path,
    "include_prompt_logprobs": include_prompt_logprobs,
    "model_id": queue_id,
    "trace_context": carrier,
  }

  await store.put_sampling_request(sampling_req)
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
