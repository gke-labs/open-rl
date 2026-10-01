"""Typed training commands.

The API server turns each API call into one of these and puts it on the queue;
the worker loop parses it back and dispatches on its type. The wire form is
the existing queue envelope, with `op` naming the type and `payload` holding its fields.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from training.types import Datum, FFTConfig, FineTuningType, LoraConfig

SHUTDOWN_REQUEST_ID = "SHUTDOWN_SENTINEL"


class Command(BaseModel):
  # Reject misspelled fields instead of silently using defaults.
  model_config = ConfigDict(extra="forbid")

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
  forward_only: bool = False


class OptimStep(Command):
  op: Literal["optim_step"] = "optim_step"
  adam_params: dict[str, Any] = {}


class Sample(Command):
  op: Literal["sample"] = "sample"
  prompt_tokens: list[int]
  max_tokens: int = 20
  num_samples: int = 1
  temperature: float = 1.0
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


class Shutdown(Command):
  op: Literal["shutdown_workers"] = "shutdown_workers"
  request_id: str = SHUTDOWN_REQUEST_ID


# fmt: off
TrainingCommand = Annotated[
  CreateModel
  | CreateModelFromState
  | ForwardBackward
  | OptimStep
  | Sample
  | SaveState
  | LoadWeights
  | SaveWeightsForSampler
  | Shutdown,
  Field(discriminator="op"),
]
# fmt: on

command_adapter: TypeAdapter[TrainingCommand] = TypeAdapter(TrainingCommand)


def parse_command(raw: dict[str, Any]) -> TrainingCommand:
  # Accept the flat commands already emitted by this branch as well as the
  # established envelope used by deployed gateways and persisted Redis queues.
  if "payload" in raw:
    envelope = dict(raw)
    payload = envelope.pop("payload")
    if not isinstance(payload, dict) or payload.keys() & envelope.keys():
      raise ValueError("Invalid command payload")
    raw = {**payload, **envelope}
  return command_adapter.validate_python(raw)


def wire(command: Command) -> dict[str, Any]:
  raw = command.model_dump(mode="json")
  envelope = {key: raw.pop(key) for key in ("op", "request_id", "model_id", "trace_context")}
  if isinstance(command, ForwardBackward):
    # Older workers flatten Tinker chunks themselves.
    for datum in raw["data"]:
      datum["model_input"] = {"chunks": [{"tokens": datum["model_input"]}]}
  return {**envelope, "payload": raw}
