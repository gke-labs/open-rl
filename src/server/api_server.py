# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict
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
from pydantic import AliasChoices, BaseModel, Field, ValidationError, ValidationInfo, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from server import proto_codec
from server.model_metadata import TrainingModelMetadata, extract_weight_sync_config, get_model_metadata, persist_model_metadata, resolve_parallelism
from server.session_registry import SessionRegistry
from server.store import RedisStateStore, get_state_store, get_store
from server.worker_manager import WorkerManager, create_worker_manager, owner_of
from training import commands
from training.commands import Command
from training.types import Datum, FFTConfig, LoraConfig

store = get_store()
state = get_state_store()
worker_manager: WorkerManager | None = None

session_registry = SessionRegistry(state)
SESSION_REAP_INTERVAL_SEC = 30
# Attaching a session to an owner and reaping that owner take turns, so a
# session cannot attach between the reaper deciding an owner is unused and
# deleting its workers. In-process, which is why there is one API server replica.
owner_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def bind_session(session_id: str | None, model_id: str) -> None:
  if worker_manager is not None and session_id:
    owner = await asyncio.to_thread(owner_of, model_id)
    async with owner_locks[owner]:
      await session_registry.attach(session_id, owner)


async def reap_owner(owner: str) -> None:
  async with owner_locks[owner]:
    if await session_registry.in_use(owner):
      return
    print(f"[API_SERVER] No live session uses {owner}; tearing its workers down")
    for model in await asyncio.to_thread(worker_manager.release_owner, owner):
      await state.delete_values(f"open_rl:sampler_ready:{model}")
    await session_registry.forget(owner)


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


class CreateSessionRequest(BaseModel):
  user_metadata: dict[str, Any] | None = None


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = None
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)

  @field_validator("lora_config", "full_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


class CreateModelFromStateRequest(BaseModel):
  state_path: str
  restore_optimizer: bool = False
  # The checkpoint's metadata names the base model when the client does not.
  base_model: str | None = None
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = None
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)

  @field_validator("lora_config", "full_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


class ModelRequest(BaseModel):
  model_id: str


class GetInfoRequest(BaseModel):
  # Without an id the answer describes the configured default model.
  model_id: str | None = None


class ForwardBackwardInput(BaseModel):
  data: list[Datum] = Field(default_factory=list)
  loss_fn: str = "cross_entropy"
  loss_fn_config: dict[str, Any] = Field(default_factory=dict)

  @field_validator("loss_fn_config", mode="before")
  @classmethod
  def default_loss_config(cls, value):
    return {} if value is None else value


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
  max_tokens: int = 20
  temperature: float = 1.0
  stop: Any = None
  top_p: float = 1.0
  top_k: int = -1

  @field_validator("max_tokens", "temperature", "top_p", "top_k", mode="before")
  @classmethod
  def default_sampling_param(cls, value, info: ValidationInfo):
    return cls.model_fields[info.field_name].default if value is None else value


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


def tinker_checkpoint_dir(path: str) -> str | None:
  """Disk directory for tinker://<model>/weights/<name>, else None."""
  if not path.startswith("tinker://"):
    return None
  owner, sep, rest = path[len("tinker://") :].partition("/weights/")
  if not (owner and sep):
    return None
  return os.path.join(TMP_DIR, "checkpoints", owner, "weights", rest)


def checkpoint_state_path(model_id: str, name: str) -> str:
  """Where a named checkpoint lives. Names are scoped under the model that
  saved them, so two jobs calling save_state("final") never collide. A tinker
  path names its own model, which is how a resumed job reaches the
  checkpoint of the one that died."""
  if (state_dir := tinker_checkpoint_dir(name)) is not None:
    return state_dir
  if os.path.isabs(name):
    return name
  return os.path.join(TMP_DIR, "checkpoints", model_id, "weights", name)


def tinker_state_path(state_path: str) -> str:
  """The tinker path for a checkpoint directory under TMP_DIR/checkpoints,
  which is the form the client hands back to weights_info and load_state.
  Anything else is returned unchanged."""
  root = os.path.join(TMP_DIR, "checkpoints") + os.sep
  if state_path.startswith(root):
    model_id, sep, rest = state_path[len(root) :].partition("/weights/")
    if model_id and sep:
      return f"tinker://{model_id}/weights/{rest}"
  return state_path


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
  req: CreateModelRequest | CreateModelFromStateRequest,
  request: Request | None = None,
  default_fine_tuning_type: str = "lora",
) -> tuple[str, TrainingModelMetadata]:
  """Extract and normalize model configuration from headers and payload, persisting TrainingModelMetadata exactly once."""
  base_model = req.base_model
  if not base_model and default_fine_tuning_type != "restored":
    raise ValueError("base_model is required in request payload")

  full_config = req.full_config.model_dump()
  lora_config = req.lora_config.model_dump()

  headers = request.headers if request is not None else {}
  weight_sync_cfg = extract_weight_sync_config(headers)

  fine_tuning_type = default_fine_tuning_type
  h_val = (headers.get("x-open-rl-fine-tuning-type") or "").lower()
  if h_val == "full":
    fine_tuning_type = "full"
  elif h_val == "lora":
    fine_tuning_type = "lora"

  if fine_tuning_type == "full" and not is_fft_enabled():
    raise ValueError("Full Fine-Tuning (FFT) is disabled on this Open-RL API server instance")

  if fine_tuning_type != "full" and default_fine_tuning_type != "restored":
    fine_tuning_type = "lora"

  full_config["weight_sync_strategy"] = weight_sync_cfg.strategy

  # The model's own user_metadata wins over what the session was opened with.
  user_metadata = dict(req.user_metadata or {})
  session_metadata = await session_registry.user_metadata(req.session_id)

  model_id = str(uuid.uuid4())
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    fine_tuning_type=fine_tuning_type,
    weight_sync_config=weight_sync_cfg,
    full_config=full_config,
    lora_config=lora_config,
    user_metadata=user_metadata,
    trainer_parallelism=resolve_parallelism("trainer", user_metadata, session_metadata, headers=headers),
    sampler_parallelism=resolve_parallelism("sampler", user_metadata, session_metadata, headers=headers),
  )
  # Kept on the model so a multi-GPU trainer can be launched from it later.
  if meta_obj.trainer_parallelism.devices > 1:
    raise ValueError("trainer parallelism beyond one GPU is not supported yet")
  sampler = meta_obj.sampler_parallelism
  if sampler.tp > 1 or sampler.cp > 1:
    raise ValueError("sampler parallelism supports dp only for now (one single-GPU sampler per replica)")
  await persist_model_metadata(state, model_id, meta_obj)

  return model_id, meta_obj


def new_request_id() -> str:
  return str(uuid.uuid4())


async def _resolve_active_set_id(model_id: str | None) -> str | None:
  if not model_id:
    return None
  meta = await get_model_metadata(state, model_id)
  if meta and meta.fine_tuning_type == "lora" and meta.base_model:
    return f"{meta.base_model}-1"
  return None


async def open_future(request_id: str) -> dict[str, str]:
  """Return the trace carrier to send with a request."""
  carrier: dict[str, str] = {}
  propagate.inject(carrier)
  return carrier


async def enqueue(command: Command) -> str:
  """Inject trace context and enqueue the command. Returns its request_id."""
  carrier = await open_future(command.request_id)

  active_set_id = await _resolve_active_set_id(command.model_id)
  await store.put_request(commands.wire(command.model_copy(update={"trace_context": carrier})), active_set_id=active_set_id)
  # One line per training request so a request that never reaches a worker can
  # be traced end to end (the workers log the same id when they pop it).
  print(f"[API_SERVER] enqueued op={command.op} request_id={command.request_id} model_id={command.model_id} active_set={active_set_id}")
  return command.request_id


async def launch_worker_and_enqueue(command: Command) -> str:
  """Ensure the model's dedicated trainer worker exists, then enqueue onto its queue.

  The launcher is idempotent per model_id, and Kubernetes (or the local process
  table) owns the worker's lifecycle from here; there is no separate launch
  queue. Launch failures resolve the future immediately so clients don't long-poll
  a request that can never be served.
  """
  assert worker_manager is not None, "Worker manager is initialized by the app lifespan"
  request_id = command.request_id
  try:
    await asyncio.to_thread(worker_manager.ensure, command.model_id, "trainer")
  except Exception as exc:
    traceback.print_exc()
    await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    return request_id
  return await enqueue(command)


async def ensure_sampler_launched(model_id: str) -> None:
  if worker_manager is not None and get_sampler_backend() == "vllm":
    try:
      await asyncio.to_thread(worker_manager.ensure, model_id, "sampler")
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
    if result_type == "state_saved" and isinstance(response.get("path"), str):
      response["path"] = tinker_state_path(response["path"])
    return response

  return result


async def reap_dead_sessions():
  while True:
    await asyncio.sleep(SESSION_REAP_INTERVAL_SEC)
    # Nothing in a sweep may end the loop. A failed call is retried next sweep.
    try:
      owners = await session_registry.owners()
    except Exception:
      traceback.print_exc()
      continue
    for owner in owners:
      try:
        await reap_owner(owner)
      except Exception:
        traceback.print_exc()


@asynccontextmanager
async def lifespan(_: FastAPI):
  global worker_manager
  task = None
  if is_fft_enabled() or os.getenv("REDIS_URL") or os.getenv("OPEN_RL_WORKER_MANAGER"):
    worker_manager = create_worker_manager()
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
  reap_task = asyncio.create_task(reap_dead_sessions()) if worker_manager is not None else None
  try:
    yield
  finally:
    if reap_task is not None:
      reap_task.cancel()
    if task is not None:
      task.cancel()
    if worker_manager is not None:
      worker_manager.close()
      worker_manager = None


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
    except ValidationError as exc:
      raise RequestValidationError(exc.errors()) from exc
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
async def create_session(req: CreateSessionRequest):
  session_id = f"sess-{uuid.uuid4().hex[:12]}"
  await session_registry.heartbeat(session_id)
  await session_registry.remember(session_id, req.user_metadata or {})
  return {"session_id": session_id, "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: SessionHeartbeatRequest):
  if req.session_id:
    await session_registry.heartbeat(req.session_id)
  return {"type": "session_heartbeat"}


@app.post("/api/v1/create_model")
async def create_model(req: CreateModelRequest, request: Request) -> dict[str, Any]:
  """ServiceClient.create_lora_training_client_async()"""
  try:
    model_id, meta = await _extract_and_persist_model_metadata(req, request, default_fine_tuning_type="lora")
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  await bind_session(req.session_id, model_id)
  command = commands.CreateModel(
    request_id=model_id,
    model_id=model_id,
    base_model=meta.base_model,
    fine_tuning_type=meta.fine_tuning_type,
    lora_config=meta.lora_config or {},
    full_config=meta.full_config or {},
  )
  req_id = await launch_worker_and_enqueue(command) if worker_manager is not None else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(req: ModelRequest):
  model_id = req.model_id
  meta = await get_model_metadata(state, model_id)
  if meta is None:
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
  is_lora = meta.fine_tuning_type == "lora"
  if is_fft_enabled() and not is_lora:
    print(f"[API_SERVER] Requesting shutdown of workers for model {model_id}...")
    await store.put_request(commands.wire(commands.Shutdown(model_id=model_id)))
    await store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
    if worker_manager is not None:
      await asyncio.to_thread(worker_manager.release, model_id)
  now = time.time()
  meta.status = "completed"
  meta.completed_at = now
  meta.updated_at = now
  await persist_model_metadata(state, model_id, meta)
  return {"status": "ok"}


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(req: CreateModelFromStateRequest, request: Request) -> dict[str, Any]:
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.state_path
  # Resolve relative names under TMP_DIR/checkpoints, leave absolute paths alone.
  resolved_path = tinker_checkpoint_dir(state_path)
  if resolved_path is None and state_path.startswith("tinker://"):
    raise HTTPException(status_code=400, detail="Invalid checkpoint URI")
  if resolved_path is None:
    resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  try:
    checkpoint = await asyncio.to_thread(checkpoint_info, resolved_path)
    kind = "lora" if checkpoint["is_lora"] else "full"
    if req.base_model is not None and req.base_model != checkpoint["base_model"]:
      raise ValueError("base_model does not match the checkpoint")
    requested_kind = request.headers.get("x-open-rl-fine-tuning-type", "").lower()
    if requested_kind in {"lora", "full"} and requested_kind != kind:
      raise ValueError("fine-tuning type does not match the checkpoint")
    req = req.model_copy(update={"base_model": checkpoint["base_model"]})
    model_id, meta = await _extract_and_persist_model_metadata(req, request, default_fine_tuning_type=kind)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  await bind_session(req.session_id, model_id)
  command = commands.CreateModelFromState(
    request_id=model_id,
    model_id=model_id,
    state_path=resolved_path,
    restore_optimizer=req.restore_optimizer,
    fine_tuning_type="full" if meta.fine_tuning_type == "full" else "lora",
  )
  req_id = await launch_worker_and_enqueue(command) if worker_manager is not None else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(req: GetInfoRequest):
  """ServiceClient — model metadata for the training client.

  TrainingClient.get_tokenizer() loads whatever tokenizer this names, so it
  has to be the model's own base model; BASE_MODEL is only the fallback for
  an id we have no metadata for. Answering with the API server default sent a
  Gemma job Qwen's tokenizer and every sample came back as token soup.
  """
  model_id = req.model_id
  meta = await get_model_metadata(state, base_model_id_from_sampling_ref(model_id) or model_id) if model_id else None
  model_name = (meta.base_model if meta is not None else None) or get_default_model_name()
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
async def retrieve_future(req: RetrieveFutureRequest, accept: str = Header(default="")):
  """ServiceClient — poll for async request results.

  Clients that send ``Accept: application/x-protobuf`` get protobuf for the
  result types the SDK only reads as protobuf (forward_backward and sample);
  pending, failed, and every other result stay JSON.
  """
  request_id = req.request_id
  result = await store.get_future(request_id, timeout=60.0)
  if result is None:
    return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
  if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
    return JSONResponse(status_code=400, content=result)
  if isinstance(result, dict):
    if proto_codec.PROTO_CONTENT_TYPE in accept:
      encoded = proto_codec.encode_future_result(result)
      if encoded is not None:
        return Response(content=encoded, media_type=proto_codec.PROTO_CONTENT_TYPE)
    return translate_future_result(result)
  return result


# *** TrainingClient endpoints ***
async def enqueue_forward_backward(req: ForwardBackwardRequest, forward_only: bool) -> dict[str, str]:
  fwd_input = req.forward_backward_input or req.forward_input or ForwardBackwardInput()
  req_id = await enqueue(
    commands.ForwardBackward(
      request_id=new_request_id(),
      model_id=req.model_id,
      data=fwd_input.data,
      loss_fn=fwd_input.loss_fn,
      loss_config=fwd_input.loss_fn_config,
      forward_only=forward_only,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/forward")
async def forward(req: Annotated[ForwardBackwardRequest, Depends(forward_backward_body)]):
  """TrainingClient.forward_async() on SDKs before 0.25; newer SDKs send
  forward() to /api/v1/forward_backward with forward_only=true."""
  return await enqueue_forward_backward(req, forward_only=True)


@app.post("/api/v1/forward_backward")
async def forward_backward(req: Annotated[ForwardBackwardRequest, Depends(forward_backward_body)]):
  """TrainingClient.forward_backward_async(), and forward_async() when the
  body carries forward_only=true (no gradient is accumulated)."""
  return await enqueue_forward_backward(req, forward_only=req.forward_only)


@app.post("/api/v1/optim_step")
async def optim_step(req: OptimStepRequest):
  """TrainingClient.optim_step_async()"""
  req_id = await enqueue(commands.OptimStep(request_id=new_request_id(), model_id=req.model_id, adam_params=req.adam_params))
  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: SaveWeightsForSamplerRequest):
  """TrainingClient.save_weights_for_sampler().

  The SDK uses this for both named sampler checkpoints and ephemeral
  save_weights_and_get_sampling_client() snapshots. Route it through the training
  queue so the sampler always sees weights saved after prior training requests.
  """
  model_id = req.model_id
  await ensure_sampler_launched(model_id)
  # The client's counter is 0-based; `or` would treat the first save's seq_id
  # of 0 as missing and mint a timestamp id instead.
  seq_id = req.sampling_session_seq_id
  if seq_id is None:
    seq_id = int(time.time() * 1000)
  alias = req.name or req.alias or req.path

  session_id = sampler_session_id(model_id, seq_id)
  req_id = await enqueue(
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
async def save_weights(req: SaveWeightsRequest):
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
  state_path = checkpoint_state_path(model_id, alias)

  # save_state is the whole training state in tinker's API. The client
  # chooses on load whether the optimizer comes back.
  req_id = await enqueue(
    commands.SaveState(request_id=new_request_id(), model_id=model_id, state_path=state_path, include_optimizer=True, kind="weights")
  )
  return {"request_id": req_id}


@app.post("/api/v1/load_weights")
async def load_weights(req: LoadWeightsRequest):
  """TrainingClient.load_state() / load_state_with_optimizer()."""
  model_id = req.model_id
  state_path = req.path
  if state_path.startswith("tinker://") and tinker_checkpoint_dir(state_path) is None:
    raise HTTPException(status_code=400, detail=f"{state_path} is not a tinker://<model>/weights/<name> path")

  resolved_path = checkpoint_state_path(model_id, state_path)
  req_id = await enqueue(
    commands.LoadWeights(request_id=new_request_id(), model_id=model_id, state_path=resolved_path, restore_optimizer=req.optimizer)
  )
  return {"request_id": req_id}


@app.post("/api/v1/weights_info")
async def weights_info(req: WeightsInfoRequest):
  """RestClient.get_weights_info_by_tinker_path(). What a checkpoint was
  trained from, so create_training_client_from_state can open a matching
  client and load_state into it. Answered from the checkpoint directory, so
  it survives an API server or Redis restart."""
  return await asyncio.to_thread(checkpoint_info, req.tinker_path)


def checkpoint_info(path: str) -> dict[str, Any]:
  state_dir = tinker_checkpoint_dir(path) or (path if os.path.isabs(path) else None)
  metadata_path = os.path.join(state_dir, "metadata.json") if state_dir else None
  if not metadata_path or not os.path.exists(metadata_path):
    raise HTTPException(status_code=404, detail=f"No checkpoint at {path}")
  with open(metadata_path) as f:
    saved = json.load(f)
  if not isinstance(saved, dict) or not isinstance(saved.get("base_model"), str) or not saved["base_model"]:
    raise ValueError("Checkpoint metadata must specify base_model")
  saved_model_id = saved.get("model_id", "")
  if not isinstance(saved_model_id, str):
    raise ValueError("Checkpoint model_id must be a string")
  adapter_config_path = os.path.join(state_dir, saved_model_id, "adapter_config.json")
  if not os.path.exists(adapter_config_path):
    adapter_config_path = os.path.join(state_dir, "adapter_config.json")
  is_lora = os.path.exists(adapter_config_path)
  rank = None
  if is_lora:
    with open(adapter_config_path) as f:
      adapter_config = json.load(f)
    if not isinstance(adapter_config, dict):
      raise ValueError("Checkpoint adapter config must be an object")
    rank = adapter_config.get("r")
  return {"base_model": saved["base_model"], "is_lora": is_lora, "lora_rank": rank, "type": "weights_info"}


# *** SamplingClient endpoints ***
@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: CreateSamplingSessionRequest):
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

  model_meta = await get_model_metadata(state, target_model_id) if target_model_id else None
  fine_tuning_type = model_meta.fine_tuning_type if model_meta else "lora"
  ready_check_id = (model_meta.base_model or target_model_id) if (fine_tuning_type == "lora" and model_meta) else target_model_id

  await bind_session(req.session_id, target_model_id)

  if get_sampler_backend() == "vllm" and ready_check_id:
    # Launch by model ID so the worker manager retains the training kind.
    # LoRA readiness is still reported under the shared base-model runtime.
    await ensure_sampler_launched(target_model_id)
    if isinstance(state, RedisStateStore):
      print(f"[API_SERVER] Waiting for dynamic vLLM sampler worker to be ready for model {ready_check_id}...")
      start_time = time.monotonic()
      while True:
        is_ready = await state.get_value(f"open_rl:sampler_ready:{ready_check_id}")
        if is_ready == "1" or is_ready == b"1":
          print(f"[API_SERVER] Dynamic vLLM sampler worker is ready! (took {time.monotonic() - start_time:.2f}s)")
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
  model_meta = await get_model_metadata(state, base_model_id) if base_model_id else None
  base_model = (model_meta.base_model if model_meta is not None else None) or base_model_id or get_default_model_name()
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
async def asample(req: AsampleRequest):
  """SamplingClient.sample_async()"""
  prompt = [token for chunk in req.prompt.get("chunks", []) for token in chunk.get("tokens", [])]
  params = req.sampling_params
  num_samples = req.num_samples

  model_id = req.model_id or req.sampling_session_id
  base_model_id = base_model_id_from_sampling_ref(model_id)
  lookup_id = base_model_id or model_id

  if get_sampler_backend() == "torch":
    req_id = await enqueue(
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
  carrier = await open_future(req_id)

  model_meta = await get_model_metadata(state, lookup_id)
  fine_tuning_type = model_meta.fine_tuning_type if model_meta else "lora"

  if fine_tuning_type == "lora":
    weights_path = None
    lora_id = model_id
    peft_dir = os.path.join(TMP_DIR, "peft", lookup_id, lookup_id)
    lora_path = peft_dir if os.path.exists(peft_dir) else None
    queue_id = (model_meta.base_model if model_meta else None) or lookup_id
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
    "trace_context": carrier,
  }

  await store.put_sampling_request(sampling_req)
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
