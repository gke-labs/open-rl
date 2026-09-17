"""Typed training commands.

The gateway turns each API call into one of these and puts it on the queue;
the worker loop parses it back and dispatches on its type. The wire form is
the model's dump, with `op` naming the type.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from training.types import Datum, FFTConfig, FineTuningType, LoraConfig

SHUTDOWN_REQUEST_ID = "SHUTDOWN_SENTINEL"


class Command(BaseModel):
  request_id: str
  model_id: str
  trace_context: dict[str, str] | None = None


class CreateModel(Command):
  op: Literal["create_model"] = "create_model"
  base_model: str
  fine_tuning_type: FineTuningType = "lora"
  lora_config: LoraConfig = LoraConfig()
  full_config: FFTConfig = FFTConfig()


class CreateModelFromState(Command):
  op: Literal["create_model_from_state"] = "create_model_from_state"
  state_path: str
  restore_optimizer: bool = False
  fine_tuning_type: FineTuningType = "lora"


class ForwardBackward(Command):
  op: Literal["forward_backward"] = "forward_backward"
  data: list[Datum]
  loss_fn: str = "cross_entropy"
  loss_config: dict[str, Any] = {}


class OptimStep(Command):
  op: Literal["optim_step"] = "optim_step"
  adam_params: dict[str, Any] = {}


class Sample(Command):
  op: Literal["sample"] = "sample"
  prompt_tokens: list[int]
  max_tokens: int = 20
  num_samples: int = 1
  temperature: float = 0.0
  prompt_logprobs: bool = False


class SaveState(Command):
  op: Literal["save_state"] = "save_state"
  state_path: str
  include_optimizer: bool = False
  kind: str = "state"


class LoadWeights(Command):
  op: Literal["load_weights"] = "load_weights"
  state_path: str
  restore_optimizer: bool = False


class SaveWeightsForSampler(Command):
  op: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"
  alias: str | None = None
  path: str | None = None
  sampling_session_id: str | None = None


class SaveWeights(Command):
  op: Literal["save_weights"] = "save_weights"
  alias: str | None = None


class Shutdown(Command):
  op: Literal["shutdown_workers"] = "shutdown_workers"
  request_id: str = SHUTDOWN_REQUEST_ID


TrainingCommand = Annotated[
  CreateModel
  | CreateModelFromState
  | ForwardBackward
  | OptimStep
  | Sample
  | SaveState
  | LoadWeights
  | SaveWeightsForSampler
  | SaveWeights
  | Shutdown,
  Field(discriminator="op"),
]

# Commands whose work touches the model on the GPU. The rest are saves that a
# worker may serve from the host; see BaseTrainerWorker.save_needs_gpu.
GPU_COMMANDS = (CreateModel, CreateModelFromState, ForwardBackward, OptimStep, Sample, LoadWeights)

command_adapter: TypeAdapter[Any] = TypeAdapter(TrainingCommand)


def parse_command(raw: dict[str, Any]) -> Any:
  if raw.get("request_id") == SHUTDOWN_REQUEST_ID or raw.get("op") in {"shutdown", "shutdown_workers"}:
    return Shutdown(model_id=raw.get("model_id", "default"))
  return command_adapter.validate_python(raw)


def wire(command: Command) -> dict[str, Any]:
  return command.model_dump(mode="json")
