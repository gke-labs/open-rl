# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import AliasChoices, BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from server import proto_codec
from server.api_runtime import ApiRuntime
from server.model_metadata import TrainingModelMetadata, extract_weight_sync_config
from server.store import get_store
from server.worker_manager import create_worker_manager
from training import commands


def get_runtime(request: Request) -> ApiRuntime:
  return request.app.state.runtime


Runtime = Annotated[ApiRuntime, Depends(get_runtime)]


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


# *** Request bodies ***
# Only the fields a handler reads. Unknown fields the SDK sends are ignored; a
# missing required field is a 422 that names it.


class SessionHeartbeatRequest(BaseModel):
  session_id: str | None = None


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str | None = None
  lora_config: dict[str, Any] | None = None
  full_config: dict[str, Any] | None = None


class CreateModelFromStateRequest(BaseModel):
  state_path: str
  restore_optimizer: bool = False
  # The checkpoint's metadata names the base model when the client does not.
  base_model: str | None = None
  session_id: str | None = None
  lora_config: dict[str, Any] | None = None
  full_config: dict[str, Any] | None = None


class ModelRequest(BaseModel):
  model_id: str


class GetInfoRequest(BaseModel):
  # Without an id the answer describes the configured default model.
  model_id: str | None = None


class ForwardBackwardInput(BaseModel):
  data: list[dict[str, Any]] = []
  loss_fn: str = "cross_entropy"
  loss_fn_config: dict[str, Any] | None = None


class ForwardBackwardRequest(ModelRequest):
  forward_backward_input: ForwardBackwardInput | None = None
  # The pre-0.25 forward route sent the batch as forward_input.
  forward_input: ForwardBackwardInput | None = None
  forward_only: bool = False


class RetrieveFutureRequest(BaseModel):
  request_id: str


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = {}


class SaveWeightsForSamplerRequest(ModelRequest):
  sampling_session_seq_id: int | str | None = None
  name: str | None = None
  alias: str | None = None
  path: str | None = None


class SaveWeightsRequest(ModelRequest):
  seq_id: int | str | None = None
  path: str | None = None


class LoadWeightsRequest(ModelRequest):
  path: str
  optimizer: bool = False


class WeightsInfoRequest(BaseModel):
  tinker_path: str


class CreateSamplingSessionRequest(BaseModel):
  model_path: str | None = None
  base_model: str | None = None
  model_id: str | None = None
  session_id: str | None = None


class SamplingParams(BaseModel):
  max_tokens: int | None = 20
  temperature: float | None = 1.0
  stop: Any = None
  top_p: float | None = 1.0
  top_k: int | None = -1


class AsampleRequest(BaseModel):
  prompt: dict[str, Any] = {}
  sampling_params: SamplingParams = SamplingParams()
  num_samples: int = 1
  # The SDK has sent this under both names.
  prompt_logprobs: bool = Field(default=False, validation_alias=AliasChoices("prompt_logprobs", "include_prompt_logprobs"))
  model_id: str | None = None
  sampling_session_id: str | None = None


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
        print(f"[API_SERVER] Warning: Failed parsing step subdirectories in {sampler_weights_dir}: {e}")
  return weights_path


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


def build_model_metadata(
  req: CreateModelRequest | CreateModelFromStateRequest,
  headers: Mapping[str, str],
) -> TrainingModelMetadata:
  """Normalize model settings without writing to the store or changing the request."""
  restoring = isinstance(req, CreateModelFromStateRequest)
  if not req.base_model and not restoring:
    raise ValueError("base_model is required in request payload")

  fine_tuning_type = (headers.get("x-open-rl-fine-tuning-type") or "").lower()
  if fine_tuning_type not in {"full", "lora"}:
    fine_tuning_type = "restored" if restoring else "lora"
  if fine_tuning_type == "full" and not is_fft_enabled():
    raise ValueError("Full Fine-Tuning (FFT) is disabled on this Open-RL API server instance")

  weight_sync_config = extract_weight_sync_config(headers)
  return TrainingModelMetadata(
    base_model=req.base_model,
    created_at=time.time(),
    fine_tuning_type=fine_tuning_type,
    weight_sync_config=weight_sync_config,
    full_config={**(req.full_config or {}), "weight_sync_strategy": weight_sync_config.strategy},
    lora_config=dict(req.lora_config or {}),
  )


def new_request_id() -> str:
  return str(uuid.uuid4())


async def enqueue_sampling(runtime: ApiRuntime, request: dict[str, Any]) -> str:
  """Inject the active trace at the sampling queue boundary."""
  request_id = request["request_id"]
  carrier: dict[str, str] = {}
  propagate.inject(carrier)
  await runtime.store.put_sampling_request({**request, "trace_context": carrier})
  return request_id


async def ensure_sampler_launched(runtime: ApiRuntime, model_id: str) -> None:
  if runtime.worker_manager is not None and get_sampler_backend() == "vllm":
    try:
      await asyncio.to_thread(runtime.worker_manager.ensure, model_id, "sampler")
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


def checkpoint_from_uri(root: str, path: str) -> str | None:
  if not path.startswith("tinker://"):
    return None
  owner, sep, rest = path[len("tinker://") :].partition("/weights/")
  if not (owner and sep):
    return None
  return os.path.join(root, owner, "weights", rest)


def checkpoint_path(root: str, model_id: str, name: str) -> str:
  """A tinker reference retains its original owner when another model loads it."""
  if name.startswith("tinker://"):
    path = checkpoint_from_uri(root, name)
    if path is None:
      raise ValueError(f"{name} is not a tinker://<model>/weights/<name> path")
    return path
  if os.path.isabs(name):
    return name
  return os.path.join(root, model_id, "weights", name)


def checkpoint_uri(root: str, state_path: str) -> str:
  prefix = root + os.sep
  if state_path.startswith(prefix):
    model_id, sep, rest = state_path[len(prefix) :].partition("/weights/")
    if model_id and sep:
      return f"tinker://{model_id}/weights/{rest}"
  return state_path


def checkpoint_info(root: str, path: str) -> dict[str, Any] | None:
  state_dir = checkpoint_from_uri(root, path)
  metadata_path = os.path.join(state_dir, "metadata.json") if state_dir else None
  if not metadata_path or not os.path.exists(metadata_path):
    return None
  with open(metadata_path) as f:
    saved = json.load(f)
  adapter_config_path = os.path.join(state_dir, saved.get("model_id", ""), "adapter_config.json")
  is_lora = os.path.exists(adapter_config_path)
  rank = None
  if is_lora:
    with open(adapter_config_path) as f:
      rank = json.load(f).get("r")
  return {"base_model": saved["base_model"], "is_lora": is_lora, "lora_rank": rank, "type": "weights_info"}


def translate_future_result(result: dict, checkpoint_root: str) -> dict:
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
    if result_type == "state_saved" and isinstance(response.get("path"), str):
      response["path"] = checkpoint_uri(checkpoint_root, response["path"])
    return response

  return result


@asynccontextmanager
async def lifespan(app: FastAPI):
  store = get_store()
  manager = create_worker_manager() if is_fft_enabled() or os.getenv("REDIS_URL") or os.getenv("OPEN_RL_WORKER_MANAGER") else None
  runtime = ApiRuntime(store, manager, TMP_DIR)
  app.state.runtime = runtime
  try:
    if is_single_process_mode():
      base_model = os.getenv("BASE_MODEL")
      print(f"[API_SERVER] Single-process mode: base_model={base_model or 'unset'} sampling_backend={get_sampler_backend()} fft={is_fft_enabled()}")
      await preflight_vllm()
      if not is_fft_enabled():
        await runtime.start_local_training(base_model)
    runtime.start_reaper()
    yield
  finally:
    try:
      await runtime.close()
    finally:
      del app.state.runtime


app = FastAPI(title="Open-RL Server MVP", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
  # FastAPI's default handler 500s when the offending input is bytes, e.g. a protobuf body on a JSON route.
  errors = jsonable_encoder(exc.errors(), custom_encoder={bytes: lambda b: f"<{len(b)} bytes>"})
  return JSONResponse(status_code=422, content={"error": "invalid request", "detail": errors})


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> Response:
  """Every refused request answers {"error": ...}; handlers raise instead of building responses."""
  if not is_body_allowed_for_status_code(exc.status_code):
    return Response(status_code=exc.status_code, headers=exc.headers)
  return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


def content_type(request: Request) -> str:
  return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def forward_backward_body(request: Request) -> ForwardBackwardRequest:
  """Tinker SDK 0.25.0 and later send forward_backward as protobuf; older SDKs
  send JSON. Both validate into the same request model."""
  encoding = request.headers.get("content-encoding", "identity").strip().lower()
  if encoding not in ("", "identity"):
    raise HTTPException(status_code=415, detail=f"Content-Encoding {encoding!r} is not supported; this server does not enable request compression")
  body = await request.body()
  media = content_type(request)
  if media == proto_codec.PROTO_CONTENT_TYPE:
    try:
      return ForwardBackwardRequest.model_validate(proto_codec.decode_forward_backward(body))
    except proto_codec.ProtoDecodeError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
  if media not in ("application/json", ""):
    raise HTTPException(status_code=415, detail=f"unsupported Content-Type {media!r}; send application/json or {proto_codec.PROTO_CONTENT_TYPE}")
  try:
    return ForwardBackwardRequest.model_validate_json(body or b"{}")
  except ValidationError as exc:
    raise RequestValidationError(exc.errors()) from exc


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
async def create_session(runtime: Runtime, _: dict):
  session_id = f"sess-{uuid.uuid4().hex[:12]}"
  await runtime.sessions.heartbeat(session_id)
  return {"session_id": session_id, "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(runtime: Runtime, req: SessionHeartbeatRequest):
  if req.session_id:
    await runtime.sessions.heartbeat(req.session_id)
  return {"type": "session_heartbeat"}


@app.post("/api/v1/create_model")
async def create_model(runtime: Runtime, req: CreateModelRequest, request: Request) -> dict[str, Any]:
  """ServiceClient.create_lora_training_client_async()"""
  try:
    meta = build_model_metadata(req, request.headers)
    model_id = await runtime.persist_model_metadata(meta)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  await runtime.bind_session(req.session_id, model_id)
  command = commands.CreateModel(
    request_id=model_id,
    model_id=model_id,
    base_model=meta.base_model,
    fine_tuning_type=meta.fine_tuning_type,
    lora_config=meta.lora_config or {},
    full_config=meta.full_config or {},
  )
  req_id = await runtime.submit(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(runtime: Runtime, req: ModelRequest):
  model_id = req.model_id
  meta = await runtime.store.get_model_metadata(model_id)
  is_lora = bool(meta and meta.get("fine_tuning_type") == "lora")
  if is_fft_enabled() and not is_lora:
    print(f"[API_SERVER] Requesting shutdown of workers for model {model_id}...")
    await runtime.store.put_request(commands.wire(commands.Shutdown(model_id=model_id)))
    await runtime.store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
    if runtime.worker_manager is not None:
      await asyncio.to_thread(runtime.worker_manager.release, model_id)
  now = time.time()
  await runtime.store.update_job_metadata(model_id, {"status": "completed", "completed_at": now, "updated_at": now})
  return {"status": "ok"}


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(runtime: Runtime, req: CreateModelFromStateRequest, request: Request) -> dict[str, Any]:
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.state_path
  # Legacy restore names are relative to the checkpoint root, not a new model.
  resolved_path = checkpoint_from_uri(runtime.checkpoint_root, state_path) or os.path.join(runtime.checkpoint_root, state_path)
  try:
    meta = build_model_metadata(req, request.headers)
    model_id = await runtime.persist_model_metadata(meta)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  await runtime.bind_session(req.session_id, model_id)
  command = commands.CreateModelFromState(
    request_id=model_id,
    model_id=model_id,
    state_path=resolved_path,
    restore_optimizer=req.restore_optimizer,
    fine_tuning_type="full" if meta.fine_tuning_type == "full" else "lora",
  )
  req_id = await runtime.submit(command)
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(runtime: Runtime, req: GetInfoRequest):
  """ServiceClient — model metadata for the training client.

  TrainingClient.get_tokenizer() loads whatever tokenizer this names, so it
  has to be the model's own base model; BASE_MODEL is only the fallback for
  an id we have no metadata for. Answering with the API server default sent a
  Gemma job Qwen's tokenizer and every sample came back as token soup.
  """
  model_id = req.model_id
  meta = await runtime.store.get_model_metadata(base_model_id_from_sampling_ref(model_id) or model_id) if model_id else None
  model_name = (meta or {}).get("base_model") or get_default_model_name()
  if not model_name:
    raise HTTPException(status_code=404, detail="No base model is configured")
  # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
  # even when this process is running a full fine-tuning worker.
  result = {
    "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
    "model_id": model_id or "model-live-123",
    "is_lora": True,
    "lora_rank": 16,
    "model_name": model_name,
    "type": "get_info",
  }
  return result


@app.post("/api/v1/retrieve_future")
async def retrieve_future(runtime: Runtime, req: RetrieveFutureRequest, accept: str = Header(default="")):
  """ServiceClient — poll for async request results.

  Clients that send ``Accept: application/x-protobuf`` get protobuf for the
  result types the SDK only reads as protobuf (forward_backward and sample);
  pending, failed, and every other result stay JSON.
  """
  request_id = req.request_id
  result = await runtime.store.get_future(request_id, timeout=60.0)
  if result is None:
    return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
  if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
    return JSONResponse(status_code=400, content=result)
  if isinstance(result, dict):
    if proto_codec.PROTO_CONTENT_TYPE in accept:
      encoded = proto_codec.encode_future_result(result)
      if encoded is not None:
        return Response(content=encoded, media_type=proto_codec.PROTO_CONTENT_TYPE)
    return translate_future_result(result, runtime.checkpoint_root)
  return result


# *** TrainingClient endpoints ***
async def enqueue_forward_backward(runtime: ApiRuntime, req: ForwardBackwardRequest, forward_only: bool) -> dict[str, str]:
  fwd_input = req.forward_backward_input or req.forward_input or ForwardBackwardInput()
  req_id = await runtime.submit(
    commands.ForwardBackward(
      request_id=new_request_id(),
      model_id=req.model_id,
      data=fwd_input.data,
      loss_fn=fwd_input.loss_fn,
      loss_config=fwd_input.loss_fn_config or {},
      forward_only=forward_only,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/forward")
async def forward(runtime: Runtime, req: Annotated[ForwardBackwardRequest, Depends(forward_backward_body)]):
  """TrainingClient.forward_async() on SDKs before 0.25; newer SDKs send
  forward() to /api/v1/forward_backward with forward_only=true."""
  return await enqueue_forward_backward(runtime, req, forward_only=True)


@app.post("/api/v1/forward_backward")
async def forward_backward(runtime: Runtime, req: Annotated[ForwardBackwardRequest, Depends(forward_backward_body)]):
  """TrainingClient.forward_backward_async(), and forward_async() when the
  body carries forward_only=true (no gradient is accumulated)."""
  return await enqueue_forward_backward(runtime, req, forward_only=req.forward_only)


@app.post("/api/v1/optim_step")
async def optim_step(runtime: Runtime, req: OptimStepRequest):
  """TrainingClient.optim_step_async()"""
  req_id = await runtime.submit(commands.OptimStep(request_id=new_request_id(), model_id=req.model_id, adam_params=req.adam_params))
  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(runtime: Runtime, req: SaveWeightsForSamplerRequest):
  """TrainingClient.save_weights_for_sampler().

  The SDK uses this for both named sampler checkpoints and ephemeral
  save_weights_and_get_sampling_client() snapshots. Route it through the training
  queue so the sampler always sees weights saved after prior training requests.
  """
  model_id = req.model_id
  await ensure_sampler_launched(runtime, model_id)
  # The client's counter is 0-based; `or` would treat the first save's seq_id
  # of 0 as missing and mint a timestamp id instead.
  seq_id = req.sampling_session_seq_id
  if seq_id is None:
    seq_id = int(time.time() * 1000)
  alias = req.name or req.alias or req.path

  session_id = sampler_session_id(model_id, seq_id)
  req_id = await runtime.submit(
    commands.SaveWeightsForSampler(
      request_id=new_request_id(),
      model_id=model_id,
      alias=alias,
      path=sampler_weights_path(model_id, alias) if alias else None,
      sampling_session_id=session_id,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights")
async def save_weights(runtime: Runtime, req: SaveWeightsRequest):
  """TrainingClient.save_weights() / save_state().

  This is the endpoint the tinker SDK hits for both save_weights() and save_state().
  The SDK sends save_state(name) as `path`; we resolve that checkpoint name to
  TMP_DIR/checkpoints/<model_id>/weights/<path> so separate training jobs do not
  overwrite each other's named checkpoints.
  """
  model_id = req.model_id
  # 0 is a valid seq_id; only fall back when the field is absent.
  seq_id = req.seq_id
  if seq_id is None:
    seq_id = int(time.time() * 1000)
  alias = req.path or f"{model_id}-samp-{seq_id}"
  try:
    state_path = checkpoint_path(runtime.checkpoint_root, model_id, alias)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  # save_state is the whole training state in tinker's API. The client
  # chooses on load whether the optimizer comes back.
  req_id = await runtime.submit(
    commands.SaveState(request_id=new_request_id(), model_id=model_id, state_path=state_path, include_optimizer=True, kind="weights")
  )
  return {"request_id": req_id}


@app.post("/api/v1/load_weights")
async def load_weights(runtime: Runtime, req: LoadWeightsRequest):
  """TrainingClient.load_state() / load_state_with_optimizer()."""
  model_id = req.model_id
  state_path = req.path
  try:
    resolved_path = checkpoint_path(runtime.checkpoint_root, model_id, state_path)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  req_id = await runtime.submit(
    commands.LoadWeights(request_id=new_request_id(), model_id=model_id, state_path=resolved_path, restore_optimizer=req.optimizer)
  )
  return {"request_id": req_id}


@app.post("/api/v1/weights_info")
async def weights_info(runtime: Runtime, req: WeightsInfoRequest):
  """RestClient.get_weights_info_by_tinker_path(). What a checkpoint was
  trained from, so create_training_client_from_state can open a matching
  client and load_state into it. Answered from the checkpoint directory, so
  it survives an API server or Redis restart."""
  info = await asyncio.to_thread(checkpoint_info, runtime.checkpoint_root, req.tinker_path)
  if info is None:
    raise HTTPException(status_code=404, detail=f"No checkpoint at {req.tinker_path}")
  return info


# *** SamplingClient endpoints ***
@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(runtime: Runtime, req: CreateSamplingSessionRequest):
  """ServiceClient.create_sampling_client()"""
  if req.model_path and req.model_path.startswith("tinker://"):
    sess_id = req.model_path
    target_model_id = req.model_path[len("tinker://") :].split("/")[0]
  elif req.base_model:
    sess_id = req.base_model
    target_model_id = req.base_model
  else:
    sess_id = req.model_id or "samp-session-live-123"
    target_model_id = sess_id

  model_meta = await runtime.store.get_model_metadata(target_model_id) if target_model_id else None
  fine_tuning_type = model_meta.get("fine_tuning_type", "lora") if model_meta else "lora"
  ready_check_id = (model_meta.get("base_model") or target_model_id) if (fine_tuning_type == "lora" and model_meta) else target_model_id

  await runtime.bind_session(req.session_id, target_model_id)

  if get_sampler_backend() == "vllm" and ready_check_id:
    # Launch by model ID so the worker manager retains the training kind.
    # LoRA readiness is still reported under the shared base-model runtime.
    await ensure_sampler_launched(runtime, target_model_id)
    s = runtime.store
    if hasattr(s, "redis"):
      print(f"[API_SERVER] Waiting for dynamic vLLM sampler worker to be ready for model {ready_check_id}...")
      start_time = time.monotonic()
      while True:
        is_ready = await s.redis.get(f"open_rl:sampler_ready:{ready_check_id}")
        if is_ready == "1" or is_ready == b"1":
          print(f"[API_SERVER] Dynamic vLLM sampler worker is ready! (took {time.monotonic() - start_time:.2f}s)")
          break
        if time.monotonic() - start_time > 300:
          raise TimeoutError("Timed out waiting for dynamic vLLM sampler worker to be ready")
        await asyncio.sleep(1)

  return {"sampling_session_id": sess_id, "type": "create_sampling_session"}


@app.get("/api/v1/samplers/{sampler_id:path}")
async def get_sampler(runtime: Runtime, sampler_id: str):
  """SamplingClient.get_tokenizer() and .get_base_model().

  The sampler id is whatever create_sampling_session handed back, so it is
  either a base model name or a `tinker://<model_id>/sampler_weights/...` path;
  `:path` on the route is what lets the slash in either form through. Both
  resolve to the base model, which is all the client wants -- it loads the
  tokenizer from the Hub itself.
  """
  base_model_id = base_model_id_from_sampling_ref(sampler_id)
  model_meta = await runtime.store.get_model_metadata(base_model_id) if base_model_id else None
  base_model = (model_meta or {}).get("base_model") or base_model_id or get_default_model_name()
  if not base_model:
    raise HTTPException(status_code=404, detail=f"Unknown sampler {sampler_id}")
  return {
    "sampler_id": sampler_id,
    "base_model": base_model,
    "model_path": sampler_id if sampler_id.startswith("tinker://") else None,
  }


def sample_sequence_ids(request_id: str, num_samples: int) -> list[str]:
  """One id per requested sample, in response order.

  Tinker SDK 0.25+ requires them on the asample promise and stamps each onto
  the matching returned sequence; the final SampleResponse does not repeat them.
  """
  return [f"{request_id}:{index}" for index in range(max(1, int(num_samples)))]


@app.post("/api/v1/asample")
async def asample(runtime: Runtime, req: AsampleRequest):
  """SamplingClient.sample_async()"""
  prompt = [token for chunk in req.prompt.get("chunks", []) for token in chunk.get("tokens", [])]
  params = req.sampling_params
  num_samples = req.num_samples

  model_id = req.model_id or req.sampling_session_id
  base_model_id = base_model_id_from_sampling_ref(model_id)
  lookup_id = base_model_id or model_id

  if get_sampler_backend() == "torch":
    req_id = await runtime.submit(
      commands.Sample(
        request_id=new_request_id(),
        model_id=lookup_id,
        prompt_tokens=prompt,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        num_samples=num_samples,
        prompt_logprobs=req.prompt_logprobs,
      )
    )
    return {"request_id": req_id, "sample_sequence_ids": sample_sequence_ids(req_id, num_samples)}

  # vLLM backend
  req_id = str(uuid.uuid4())

  model_meta = await runtime.store.get_model_metadata(lookup_id)
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
    "max_tokens": params.max_tokens,
    "temperature": params.temperature,
    "stop": params.stop,
    "top_p": params.top_p,
    "top_k": params.top_k,
    "num_samples": num_samples,
    "lora_id": lora_id,
    "lora_path": lora_path,
    "weights_path": weights_path,
    "include_prompt_logprobs": req.prompt_logprobs,
    "model_id": queue_id,
  }

  await enqueue_sampling(runtime, sampling_req)
  return {"request_id": req_id, "sample_sequence_ids": sample_sequence_ids(req_id, num_samples)}


# *** CLI endpoints ***


@app.get("/api/v1/list_adapters")
async def list_adapters():
  """CLI `list` — scan the peft directory for saved adapters."""
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
