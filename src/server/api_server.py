# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import AfterValidator, AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, ValidationInfo, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from server import openrl_status, proto_codec
from server.accelerators import Accelerator, check_supported, parse_accel_prefs
from server.lora_snapshots import snapshot_name, snapshot_root
from server.model_metadata import (
  TrainingModelMetadata,
  WeightSyncConfig,
  get_model_metadata,
  persist_model_metadata,
  pool_name,
)
from server.observability.router import router
from server.sampling_dispatcher import SAMPLE_DEADLINE_SECONDS
from server.sampling_store import SamplingStore
from server.session_registry import SessionRegistry
from server.store import get_state_store, get_store
from server.vllm_options import split_stop
from server.worker_manager import LocalWorkerManager, WorkerManager, create_worker_manager, default_metadata, owner_id, owner_of, runtime_of
from training import commands
from training.commands import Command
from training.types import TRAINER_BACKENDS, Datum, FFTConfig, FineTuningType, LoraConfig

store = get_store()
state = get_state_store()
worker_manager: WorkerManager | None = None

session_registry = SessionRegistry(state)
SESSION_REAP_INTERVAL_SEC = 30
# Jobs whose pools the reconcile may ensure at once.
RECONCILE_CONCURRENCY = 4
# Attaching a session to an owner and reaping that owner take turns, so a
# session cannot attach between the reaper deciding an owner is unused and
# deleting its workers. In-process, which is why there is one API server replica.
owner_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def bind_session(session_id: str | None, model_id: str) -> None:
  if not session_id:
    return
  owner = await asyncio.to_thread(owner_of, model_id)
  async with owner_locks[owner]:
    # On the model, for the model view: which sessions are this model's own.
    meta = await get_model_metadata(state, model_id)
    if meta is not None and session_id not in meta.session_ids:
      meta.session_ids.append(session_id)
      await persist_model_metadata(state, model_id, meta)
    # On the owner, for the reaper: workers stay while any attached session lives.
    if worker_manager is not None:
      await session_registry.attach(session_id, owner)


async def reap_owner(owner: str) -> None:
  async with owner_locks[owner]:
    if await session_registry.in_use(owner):
      return
    print(f"[API_SERVER] No live session uses {owner}; tearing its workers down")
    for model in await asyncio.to_thread(worker_manager.release_owner, owner):
      await state.delete_values(f"open_rl:trainer_ready:{model}")
      if get_sampler_backend() == "vllm":
        await SamplingStore(store.redis).delete_if_empty(pool_name(model))
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


# *** Request bodies ***
# Only the fields a handler reads. Unknown fields the SDK sends are ignored; a
# missing required field is a 422 that names it.


class SessionHeartbeatRequest(BaseModel):
  session_id: str | None = None


USER_METADATA_DOC = (
  "Free-form client metadata. Keys that start with openrl. are OpenRL settings: "
  "see components.schemas.OpenRLSettings in this document. "
  "Values are strings on the wire."
)


class CreateSessionRequest(BaseModel):
  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  # Removed; FFT options are openrl. settings now. Kept to reject it loudly.
  full_config: Any = None

  @field_validator("lora_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


class CreateModelFromStateRequest(BaseModel):
  state_path: str
  restore_optimizer: bool = False
  # The checkpoint's metadata names the base model when the client does not.
  base_model: str | None = None
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  # Removed; FFT options are openrl. settings now. Kept to reject it loudly.
  full_config: Any = None

  @field_validator("lora_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


# The settings a client chooses for its models. Highest precedence first, they
# come from the model's user_metadata, the session's user_metadata, and the
# session's openrl. tags (TINKER_TAGS), for scripts that cannot pass
# user_metadata. Only openrl. keys are ours; the client owns the rest.
SETTINGS_PREFIX = "openrl."


def parse_bool(value: Any) -> bool:
  # The SDK types user_metadata values as strings, and tags are always strings.
  if isinstance(value, str) and value.lower() in ("true", "false"):
    return value.lower() == "true"
  if not isinstance(value, bool):
    raise ValueError(f"must be 'true' or 'false', got {value!r}")
  return value


def parse_trainer_backend(value: str) -> str:
  # An image ref always has a registry or repository path.
  if value not in TRAINER_BACKENDS and "/" not in value:
    raise ValueError(f"must be one of {', '.join(TRAINER_BACKENDS)} or an image like ghcr.io/org/trainer:tag, got {value!r}")
  return value


class Settings(BaseModel):
  """The descriptions are what get_server_capabilities and the skill show, so
  they are written for the client choosing a value. Each field's wire name is
  its alias, openrl.<field>: pydantic validates the user_metadata keys as the
  client wrote them, names them that way in errors, and puts them in the JSON
  Schema that capabilities and /openapi.json publish."""

  model_config = ConfigDict(extra="forbid", alias_generator=lambda field: f"{SETTINGS_PREFIX}{field}", populate_by_name=True)

  sampler_replicas: int = Field(default=1, ge=1, le=8, description="Single-GPU samplers that drain the model's sampling queue together.")
  exclusive: Annotated[bool, BeforeValidator(parse_bool)] = Field(
    default=False,
    description="A workload no other workload shares. The model gets its own trainer and sampler, and nothing time-slices their GPUs.",
  )
  trainer_backend: Annotated[str, AfterValidator(parse_trainer_backend)] = Field(
    default="pytorch",
    description="The trainer the model runs on: pytorch, automodel (LoRA only), or an image like ghcr.io/org/trainer:tag that runs its own trainer.",
  )
  trainer_accel_prefs: Annotated[list[Accelerator], BeforeValidator(parse_accel_prefs)] = Field(
    default=["gpu"],
    description="The accelerators the model's trainer may run on, most preferred first: gpu, tpu or tpu,gpu (tpu|gpu in a tag).",
  )
  sampler_accel_prefs: Annotated[list[Accelerator], BeforeValidator(parse_accel_prefs)] = Field(
    default=["gpu"],
    description="The accelerators the model's sampler may run on, most preferred first, like openrl.trainer_accel_prefs.",
  )
  trainer_gpus: int = Field(
    default=1,
    ge=1,
    le=8,
    description="GPUs the trainer drives as one data-parallel torchrun group. More than one is Automodel only and makes the model exclusive.",
  )
  trainer_cp: int = Field(
    default=1,
    ge=1,
    le=8,
    description="Trainer GPUs that share one sequence, as context parallelism. The rest of trainer_gpus is data parallel.",
  )
  fine_tuning_type: FineTuningType | None = Field(
    default=None,
    description="lora unless set. A model restored from a checkpoint takes the checkpoint's type.",
  )
  recipe: Literal["rl", "sft"] | None = Field(
    default=None,
    description="What the job runs. rl starts the sampler together with the trainer, so its start-up overlaps the trainer's. "
    "sft or unset starts it at the first sample or save for sampling.",
  )
  client_log_id: str | None = Field(
    default=None,
    description="The id on the client's log lines, as the label openrl_client=<id> in the server's log store. "
    "The inspection API reads the run's client logs by it.",
  )
  # The rest apply to full fine-tuning only.
  weight_sync: Literal["delta", "full"] | None = Field(
    default=None,
    description="Full fine-tuning only. How the trainer hands each step's weights to the sampler: delta (changed weights only, the default) or full.",
  )
  fft_seed: int | None = Field(default=None, description="Full fine-tuning only. The trainer's random seed.")
  fft_cpu_offload: Annotated[bool | None, BeforeValidator(parse_bool)] = Field(
    default=None,
    description="Full fine-tuning only. Park trainer state in host memory between turns. Exclusive models default to off.",
  )


def settings_schema() -> dict[str, Any]:
  """The openrl.* settings as JSON Schema, straight from pydantic, published
  once: components.schemas.OpenRLSettings in /openapi.json. Capabilities
  point there; the skill sends agents there."""
  return Settings.model_json_schema(ref_template="#/components/schemas/{model}")


def tag_metadata(tags: list[str]) -> dict[str, str]:
  """The openrl. tags as user_metadata entries. The SDK splits TINKER_TAGS on
  commas, so a value cannot hold one."""
  entries = {}
  for tag in tags:
    if tag.startswith(SETTINGS_PREFIX):
      key, sep, value = tag.partition("=")
      if not sep:
        raise ValueError(f"tag {tag!r} needs a value, like {key}=true")
      entries[key] = value
  return entries


def resolve_settings(*user_metadata: dict[str, Any]) -> Settings:
  """The settings from user_metadata layers, highest precedence first. An
  unknown openrl. key or a bad value is an error, not a silent default."""
  merged = {}
  for layer in reversed(user_metadata):
    # The user's own metadata keys stay out; the openrl. keys go in as written and
    # pydantic matches them to the fields by alias.
    merged.update({key: value for key, value in (layer or {}).items() if key.startswith(SETTINGS_PREFIX)})
  try:
    return Settings.model_validate(merged)
  except ValidationError as exc:
    problems = []
    for error in exc.errors():
      message = "unknown setting" if error["type"] == "extra_forbidden" else error["msg"].removeprefix("Value error, ")
      problems.append(f"{error['loc'][0]}: {message}")
    raise ValueError("; ".join(problems)) from None


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
  seed: int | None = None

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
  seq_id: int | None = None


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


def sampling_ref(sampling_session_id: str) -> str:
  """The model reference a sampling session was created for, without its per-session suffix."""
  return re.sub(r"@[0-9a-f]{32}$", "", sampling_session_id)


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


async def _extract_and_persist_model_metadata(
  req: CreateModelRequest | CreateModelFromStateRequest,
  request: Request | None = None,
  checkpoint_kind: FineTuningType | None = None,
) -> tuple[str, TrainingModelMetadata]:
  """Resolve the model's configuration from its openrl. settings and persist TrainingModelMetadata once.
  checkpoint_kind is the fine-tuning type of the checkpoint a restored model loads."""
  base_model = req.base_model
  if not base_model:
    raise ValueError("base_model is required in request payload")
  headers = request.headers if request is not None else {}
  for header, setting in (("x-open-rl-fine-tuning-type", "fine_tuning_type"), ("x-open-rl-weight-sync-strategy", "weight_sync")):
    if header in headers:
      raise ValueError(f"The {header} header was removed; set {SETTINGS_PREFIX}{setting} in TINKER_TAGS or user_metadata")
  if req.full_config is not None:
    raise ValueError(f"full_config was removed; set {SETTINGS_PREFIX}fft_seed and {SETTINGS_PREFIX}fft_cpu_offload in TINKER_TAGS or user_metadata")
  lora_config = req.lora_config.model_dump()

  # The model's own user_metadata wins over what the session was opened with.
  settings = resolve_settings(req.user_metadata or {}, await session_registry.user_metadata(req.session_id))
  if checkpoint_kind is not None and settings.fine_tuning_type not in (None, checkpoint_kind):
    raise ValueError(f"{SETTINGS_PREFIX}fine_tuning_type={settings.fine_tuning_type} does not match the {checkpoint_kind} checkpoint")
  fine_tuning_type = settings.fine_tuning_type or checkpoint_kind or "lora"
  if fine_tuning_type == "full" and not is_fft_enabled():
    raise ValueError("Full Fine-Tuning (FFT) is disabled on this Open-RL API server instance")
  fft_settings = {"weight_sync": settings.weight_sync, "fft_seed": settings.fft_seed, "fft_cpu_offload": settings.fft_cpu_offload}
  if fine_tuning_type != "full" and (given := [name for name, value in fft_settings.items() if value is not None]):
    raise ValueError(", ".join(f"{SETTINGS_PREFIX}{name}" for name in given) + " applies to fine_tuning_type=full only")
  weight_sync = settings.weight_sync or "delta"
  # A TPU sampler holds its linear weights transposed, so sparse patches in
  # checkpoint coordinates cannot apply; it reloads full checkpoints instead.
  if fine_tuning_type == "full" and "tpu" in settings.sampler_accel_prefs and weight_sync != "full":
    print(f"[API_SERVER] weight sync {weight_sync} -> full: a TPU sampler reloads full checkpoints")
    weight_sync = "full"
  weight_sync_cfg = WeightSyncConfig(strategy=weight_sync)
  # Nothing parks an exclusive trainer, so it stays on the GPU.
  cpu_offload = settings.fft_cpu_offload if settings.fft_cpu_offload is not None else not settings.exclusive
  full_config = FFTConfig(seed=settings.fft_seed, cpu_offload=cpu_offload, weight_sync_strategy=weight_sync_cfg.strategy).model_dump()
  # Without a worker manager one static runtime serves every model.
  if settings.exclusive and worker_manager is None:
    raise ValueError("openrl.exclusive needs a server that launches workers per model")
  if settings.sampler_replicas > 1 and isinstance(worker_manager, LocalWorkerManager):
    raise ValueError("openrl.sampler_replicas above 1 needs a server that launches workers as pods")
  if settings.trainer_backend != "pytorch" and worker_manager is None:
    raise ValueError(f"openrl.trainer_backend={settings.trainer_backend} needs a server that launches workers per model")
  if settings.trainer_backend not in TRAINER_BACKENDS and isinstance(worker_manager, LocalWorkerManager):
    raise ValueError("A trainer image needs a server that launches workers as pods")
  if settings.trainer_backend == "automodel" and fine_tuning_type != "lora":
    raise ValueError("The automodel trainer supports LoRA only")
  check_supported(fine_tuning_type, settings.trainer_backend, settings.trainer_accel_prefs, settings.sampler_accel_prefs, settings.exclusive)
  if settings.trainer_gpus > 1 and settings.trainer_backend != "automodel":
    raise ValueError("openrl.trainer_gpus above 1 needs openrl.trainer_backend=automodel")
  if settings.trainer_gpus > 1 and isinstance(worker_manager, LocalWorkerManager):
    raise ValueError("openrl.trainer_gpus above 1 needs a server that launches workers as pods")
  if settings.trainer_gpus % settings.trainer_cp:
    raise ValueError(f"openrl.trainer_cp={settings.trainer_cp} must divide openrl.trainer_gpus={settings.trainer_gpus}")
  model_id = str(uuid.uuid4())
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    fine_tuning_type=fine_tuning_type,
    weight_sync_config=weight_sync_cfg,
    full_config=full_config,
    lora_config=lora_config,
    sampler_replicas=settings.sampler_replicas,
    exclusive=settings.exclusive,
    trainer_backend=settings.trainer_backend,
    trainer_accel_prefs=settings.trainer_accel_prefs,
    sampler_accel_prefs=settings.sampler_accel_prefs,
    trainer_gpus=settings.trainer_gpus,
    trainer_cp=settings.trainer_cp,
    recipe=settings.recipe,
    client_log_id=settings.client_log_id,
  )
  if meta_obj.sampler_replicas > 1 and meta_obj.shares_gpu():
    raise ValueError("openrl.sampler_replicas above 1 needs samplers on GPUs of their own; set openrl.exclusive=true")
  await persist_model_metadata(state, model_id, meta_obj)

  return model_id, meta_obj


def new_request_id() -> str:
  return str(uuid.uuid4())


async def _resolve_active_set_id(model_id: str | None) -> str | None:
  if not model_id:
    return None
  meta = await get_model_metadata(state, model_id)
  if meta and meta.fine_tuning_type == "lora" and meta.base_model:
    return f"{meta.runtime(model_id)}-1"
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


async def launch_worker_and_enqueue(command: Command, start_sampler: bool = False) -> str:
  """Ensure the model's trainer worker exists, then enqueue onto its queue.

  The launcher is idempotent per model_id, and Kubernetes (or the local process
  table) owns the worker's lifecycle from here; there is no separate launch
  queue. Launch failures resolve the future immediately so clients don't long-poll
  a request that can never be served.

  With start_sampler (an RL job), the sampler starts here too rather than at
  the client's first save or sampling call, so its engine build overlaps the
  trainer's start-up and the client's dataset loading. An SFT job may never
  sample, so by default it gets no sampler until it asks for one.
  """
  assert worker_manager is not None, "Worker manager is initialized by the app lifespan"
  request_id = command.request_id
  try:
    await asyncio.to_thread(worker_manager.ensure, command.model_id, "trainer")
  except Exception as exc:
    traceback.print_exc()
    await store.set_result(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    return request_id
  if start_sampler:
    try:
      await ensure_sampler_launched(command.model_id)
    except HTTPException as exc:
      await store.set_result(request_id, {"type": "RequestFailedResponse", "error_message": str(exc.detail)})
      return request_id
  return await enqueue(command)


async def ensure_sampler_launched(model_id: str) -> None:
  # Fail the request with the reason, e.g. a missing image env var, instead of
  # letting create_sampling_session wait out the ready timeout.
  if worker_manager is not None and get_sampler_backend() == "vllm":
    try:
      await asyncio.to_thread(worker_manager.ensure, model_id, "sampler")
    except Exception as exc:
      traceback.print_exc()
      raise HTTPException(status_code=500, detail=f"Could not start the sampler: {exc}") from exc


def check_sampler_backend() -> None:
  if get_sampler_backend() == "vllm" and not os.getenv("REDIS_URL"):
    raise RuntimeError(
      "Sampling pools need REDIS_URL, because vLLM sampling requests wait in a Redis stream for each pool's dispatcher. "
      "Set REDIS_URL (for example redis://127.0.0.1:6379/0), or set BASE_MODEL and unset SAMPLING_BACKEND to sample with torch in one process."
    )


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


async def every_interval(step) -> None:
  """Runs `step` every SESSION_REAP_INTERVAL_SEC. A failed step is retried next time."""
  while True:
    await asyncio.sleep(SESSION_REAP_INTERVAL_SEC)
    try:
      await step()
    except Exception:
      traceback.print_exc()


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


async def reconcile_pools() -> None:
  """For each job with pending requests whose pool is gone, recreates the pool when the job has a live session and fails
  the requests when it has none, so no request waits on a pool nothing will bring back. Standing worker deployments
  run without a worker manager and own their pools."""
  if worker_manager is None:
    return
  sampling = SamplingStore(store.redis)
  pairs = []
  for pool in await sampling.pools():
    try:
      jobs = await sampling.pending_jobs(pool)
      # Whichever path answered a pool's last request, its empty stream stays until something deletes it.
      if not jobs:
        await sampling.delete_if_empty(pool)
      pairs += [(pool, job) for job in jobs]
    except Exception:
      traceback.print_exc()
  # Concurrently, since an ensure can wait minutes for an old pool to terminate, but a few at a time so stuck ensures
  # cannot take every thread the API server runs blocking calls on.
  slots = asyncio.Semaphore(RECONCILE_CONCURRENCY)

  async def reconcile(pool: str, job: str) -> None:
    async with slots:
      await reconcile_job(sampling, pool, job)

  for outcome in await asyncio.gather(*(reconcile(pool, job) for pool, job in pairs), return_exceptions=True):
    if isinstance(outcome, Exception):
      traceback.print_exception(outcome)


async def reconcile_job(sampling: SamplingStore, pool: str, job: str) -> None:
  meta, runtime, _ = await asyncio.to_thread(runtime_of, job)
  if meta.pool(job) != pool:
    return  # a record from before records named their job, whose model is no job on this pool
  owner = owner_id(runtime)
  # Under the owner lock, so a reaper release cannot land between the checks and the ensure and leave a pool no one removes.
  async with owner_locks[owner]:
    # Read again under the lock, since delete_model may have completed the job meanwhile.
    if (current := await get_model_metadata(state, job)) is not None and current.status == "completed":
      if failed := await sampling.fail_job(pool, job, {"type": "RequestFailedResponse", "error_message": f"{job} was deleted"}):
        print(f"[API_SERVER] Failed {failed} sampling requests of {job}, which was deleted")
      return
    # A pool that is up serves its requests, since a client may sample without a session.
    if await asyncio.to_thread(worker_manager.sampler_exists, job):
      return
    if await session_registry.in_use(owner):
      await asyncio.to_thread(worker_manager.ensure, job, "sampler")
      return
    message = f"The pool of {job} is gone and its session ended before a sampler read the request"
    if failed := await sampling.fail_job(pool, job, {"type": "RequestFailedResponse", "error_message": message}):
      print(f"[API_SERVER] Failed {failed} sampling requests of {job}, whose pool is gone and whose session ended")


async def sweep_sample_deadlines() -> None:
  """Fails pool requests past the deadline, which no dispatcher does while the pool is gone."""
  result = {"type": "RequestFailedResponse", "error_message": f"No sampler read the request within {SAMPLE_DEADLINE_SECONDS}s"}
  if failed := await SamplingStore(store.redis).fail_older_than(SAMPLE_DEADLINE_SECONDS, result):
    print(f"[API_SERVER] Failed {failed} sampling requests no sampler read in time")


@asynccontextmanager
async def lifespan(_: FastAPI):
  global worker_manager
  task = None
  check_sampler_backend()
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
    if not is_fft_enabled():
      from server import training_requests_processor

      worker = training_requests_processor.LoraTrainingWorker()
      if base_model:
        await asyncio.to_thread(worker.load_base_model, base_model)
      task = asyncio.create_task(training_requests_processor.run_training_requests_processor(worker))
  reap_task = asyncio.create_task(reap_dead_sessions()) if worker_manager is not None else None
  pool_tasks = []
  if get_sampler_backend() == "vllm":
    # Separate tasks, so a reconcile stuck in an ensure never holds up the deadline sweep.
    pool_tasks = [asyncio.create_task(every_interval(sweep_sample_deadlines)), asyncio.create_task(every_interval(reconcile_pools))]
  try:
    yield
  finally:
    for background in (reap_task, *pool_tasks):
      if background is not None:
        background.cancel()
    if task is not None:
      task.cancel()
    if worker_manager is not None:
      worker_manager.close()
      worker_manager = None


# The release tag, baked in at image build (OPEN_RL_VERSION build arg); the
# VERSION file is an internal counter and does not name a release.
OPEN_RL_VERSION = os.getenv("OPEN_RL_VERSION") or "dev"  # an empty build arg must not become the version
RELEASES_URL = "https://github.com/gke-labs/open-rl/releases"
EXAMPLE_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "google/gemma-4-e2b", "Qwen/Qwen3-8B"]

app = FastAPI(
  title="OpenRL API server",
  version=OPEN_RL_VERSION,
  description=(
    "A self-hosted, Kubernetes-native post-training API that serves the Tinker API. "
    "Tinker SDK clients use the /api/v1 paths. GET /api/v1/openrl/models/{model_id} is OpenRL's own: "
    "what the server knows about one model, its settings, its workers and whether its session is live. "
    "Repository and docs: https://github.com/gke-labs/open-rl"
  ),
  lifespan=lifespan,
)


def openapi_with_settings() -> dict[str, Any]:
  """FastAPI publishes the request models on its own. The openrl.* settings
  travel inside user_metadata, so their schema is added under components by
  hand. It is the one place the settings are described."""
  if app.openapi_schema:
    return app.openapi_schema
  schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
  settings = settings_schema()
  schemas = schema.setdefault("components", {}).setdefault("schemas", {})
  schemas.update(settings.pop("$defs", {}))
  schemas["OpenRLSettings"] = settings
  app.openapi_schema = schema
  return schema


app.openapi = openapi_with_settings
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")
app.include_router(router)


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


def worker_launcher_mode() -> str:
  """How this server runs workers: single_process (trainer in this process),
  process (subprocesses), scheduler (Workloads placed on a cluster), or none
  (standing worker deployments)."""
  if is_single_process_mode():
    return "single_process"
  if worker_manager is None:
    return "none"
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  return "scheduler" if mode == "scheduler" else "process"


def env_int(name: str) -> int | None:
  value = os.getenv(name)
  return int(value) if value and value.isdigit() else None


def openrl_info() -> dict[str, Any]:
  """What an agent needs to know about this deployment before its first
  request (design 016, section 5.1). Nested under one key so the SDK's
  response models ignore it."""
  release = OPEN_RL_VERSION.startswith("v")
  return {
    "version": OPEN_RL_VERSION,
    "worker_launcher": worker_launcher_mode(),
    "sampler_backend": get_sampler_backend(),
    "fft_enabled": is_fft_enabled(),
    "models": {
      # Outside single-process mode the base model is whatever create_model names.
      "any_huggingface_model": not is_single_process_mode(),
      "default": get_default_model_name(),
      "examples": EXAMPLE_MODELS,
    },
    # The settings themselves live in one place, the OpenAPI document.
    "settings_schema_url": "/openapi.json#/components/schemas/OpenRLSettings",
    "settings_howto": "Pass as user_metadata on create_session or create_model, or as TINKER_TAGS=openrl.key=value,...",
    "limits": {
      "vllm_max_model_len": env_int("VLLM_MAX_MODEL_LEN"),
      "train_token_budget": env_int("OPEN_RL_TRAIN_TOKEN_BUDGET"),
      "session_ttl_seconds": session_registry.ttl_seconds,
      "session_reap_interval_seconds": SESSION_REAP_INTERVAL_SEC,
    },
    "skill_url": f"{RELEASES_URL}/download/{OPEN_RL_VERSION}/openrl-skill.tar.gz" if release else None,
    "getting_started_url": "https://github.com/gke-labs/open-rl/blob/main/docs/getting-started.md",
  }


@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
  model_name = get_default_model_name()
  return {
    "supported_models": [{"model_name": model_name}] if model_name else [],
    "default_model": model_name,
    "single_process": is_single_process_mode(),
    "openrl": openrl_info(),
  }


# *** The OpenRL model view: read-only, not part of the Tinker API ***
@app.get("/api/v1/openrl/models/{model_id}")
async def openrl_model_status(model_id: str):
  """One model: its settings, its workers with their state, whether its session is live, and a one-line summary."""
  ctx = openrl_status.StatusContext(state=state, registry=session_registry, manager=worker_manager, worker_launcher=worker_launcher_mode())
  result = await openrl_status.model_status(ctx, model_id)
  if result is None:
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
  return result


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
  # Tags are the session's defaults when the script cannot pass user_metadata.
  try:
    user_metadata = {**tag_metadata(req.tags), **(req.user_metadata or {})}
    resolve_settings(user_metadata)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  session_id = f"sess-{uuid.uuid4().hex[:12]}"
  await session_registry.update_metadata(session_id, user_metadata)
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
    model_id, meta = await _extract_and_persist_model_metadata(req, request)
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
  req_id = await launch_worker_and_enqueue(command, start_sampler=meta.recipe == "rl") if worker_manager is not None else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(req: ModelRequest):
  model_id = req.model_id
  meta = await get_model_metadata(state, model_id)
  if meta is None:
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
  if meta.shares_runtime():
    # Other jobs keep the trainer, so it only frees this job's adapter, after
    # the job's queued work.
    delete = commands.DeleteModel(request_id=str(uuid.uuid4()), model_id=model_id)
    await store.put_request(commands.wire(delete), active_set_id=await _resolve_active_set_id(model_id))
  elif worker_manager is not None:
    print(f"[API_SERVER] Requesting shutdown of workers for model {model_id}...")
    await store.put_request(commands.wire(commands.Shutdown(model_id=model_id)), active_set_id=await _resolve_active_set_id(model_id))
  # Under the owner lock the reconcile takes, so it cannot bring back a pool for the job between these steps.
  async with owner_locks[await asyncio.to_thread(owner_of, model_id)]:
    now = time.time()
    meta.status = "completed"
    meta.completed_at = now
    meta.updated_at = now
    await persist_model_metadata(state, model_id, meta)
    if get_sampler_backend() == "vllm":
      result = {"type": "RequestFailedResponse", "error_message": f"{model_id} was deleted"}
      await SamplingStore(store.redis).fail_job(meta.pool(model_id), model_id, result)
    if worker_manager is not None and not meta.shares_runtime():
      await asyncio.to_thread(worker_manager.release, model_id)
      if get_sampler_backend() == "vllm":
        await SamplingStore(store.redis).delete_if_empty(meta.pool(model_id))
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
    req = req.model_copy(update={"base_model": checkpoint["base_model"]})
    model_id, meta = await _extract_and_persist_model_metadata(req, request, checkpoint_kind=kind)
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
  req_id = await launch_worker_and_enqueue(command, start_sampler=meta.recipe == "rl") if worker_manager is not None else await enqueue(command)
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
  result = await store.get_result(request_id, timeout=60.0)
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

  await bind_session(req.session_id, target_model_id)

  if get_sampler_backend() == "vllm":
    # Requests wait in the pool's stream, so there is no sampler to wait for. The SDK
    # numbers seq_id from 0 in every sampling client, so each session needs its own id.
    await ensure_sampler_launched(target_model_id)
    return {"sampling_session_id": f"{sess_id}@{uuid.uuid4().hex}", "type": "create_sampling_session"}

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
  ref = sampling_ref(sampler_id)
  base_model_id = base_model_id_from_sampling_ref(ref)
  model_meta = await get_model_metadata(state, base_model_id) if base_model_id else None
  base_model = (model_meta.base_model if model_meta is not None else None) or base_model_id or get_default_model_name()
  if not base_model:
    raise HTTPException(status_code=404, detail=f"Unknown sampler {sampler_id}")
  return {
    "sampler_id": sampler_id,
    "base_model": base_model,
    "model_path": ref if ref.startswith("tinker://") else None,
  }


def sample_sequence_ids(request_id: str, num_samples: int) -> list[str]:
  """One id per requested sample, in response order.

  Tinker SDK 0.25+ requires them on the asample promise and stamps each onto
  the matching returned sequence; the final SampleResponse does not repeat them.
  """
  return [f"{request_id}:{index}" for index in range(max(1, int(num_samples)))]


# How long a sample waits for weights another node just saved to show up here.
SAVED_WEIGHTS_WAIT_SECONDS = 2.0


def visible(path: Path) -> bool:
  """Whether a file another node saved shows here. An NFS client keeps a failed
  lookup until it revalidates the directory, and listing a directory makes it
  revalidate, so on a miss this lists the nearest directory above the file that
  exists and looks again."""
  if path.is_file():
    return True
  for directory in path.parents:
    try:
      os.listdir(directory)
      break
    except OSError:
      continue
  return path.is_file()


async def wait_until_visible(path: Path) -> bool:
  deadline = time.monotonic() + SAVED_WEIGHTS_WAIT_SECONDS
  while not await asyncio.to_thread(visible, path):
    if time.monotonic() >= deadline:
      return False
    await asyncio.sleep(0.2)
  return True


async def accept_pool_sample(req: AsampleRequest, prompt: list[int]) -> str:
  """Missing weights are a 400, never a fallback to other weights."""
  ref = sampling_ref(req.model_id or req.sampling_session_id or "")
  lookup_id = base_model_id_from_sampling_ref(ref) or ref
  meta = await get_model_metadata(state, lookup_id)
  if meta is None:
    meta, revision, model = default_metadata(lookup_id), "", lookup_id
  elif meta.fine_tuning_type == "full":
    revision, model = resolve_sampler_weights_path(ref), meta.base_model
    # The trainer writes metadata.json last, so a save in progress is missing too.
    if not await wait_until_visible(Path(revision, "metadata.json")):
      raise HTTPException(status_code=400, detail=f"No saved sampler weights for {ref}")
  else:
    revision, model = "", snapshot_name(ref)
    if not await wait_until_visible(snapshot_root() / model / "adapter_config.json"):
      raise HTTPException(status_code=400, detail=f"No saved LoRA sampler weights for {ref}")
  pool = meta.pool(lookup_id)

  params = req.sampling_params
  stop, stop_token_ids = split_stop(params.stop)
  return await SamplingStore(store.redis).accept(
    {
      "request_id": new_request_id(),
      "pool": pool,
      "job": lookup_id,
      "revision": revision,
      "model": model,
      "idempotency_key": f"{req.sampling_session_id}:{req.seq_id}" if req.sampling_session_id and req.seq_id is not None else None,
      "prompt_token_ids": prompt,
      "sampling_params": {"n": req.num_samples, **params.model_dump(exclude={"stop"}), "stop": stop or [], "stop_token_ids": stop_token_ids or []},
      "include_prompt_logprobs": req.prompt_logprobs,
    }
  )


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

  req_id = await accept_pool_sample(req, prompt)
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
