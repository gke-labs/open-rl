from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import importlib.metadata
import inspect
import io
import os
import pkgutil
import signal
import tempfile
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker.lib import public_interfaces as tinker_public_interfaces
from tinker.lib.public_interfaces import rest_client as rest_client_module
from tinker.lib.public_interfaces.sampling_client import QueueState

from tests._server_fixture import openrl_server

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs" / "tinker-client-compatibility.md"

MODEL_ID = "compat-training-client"
SAMPLING_ID = "compat-sampling-session"
STATE_NAME = "compat-state"
PROBE_TIMEOUT = 5.0
STARTUP_TIMEOUT = 60.0

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_DIR = tempfile.TemporaryDirectory(prefix="open-rl-compat-model-")


@functools.cache
def compat_model_path() -> str:
  from tokenizers import Tokenizer
  from tokenizers.models import WordLevel
  from tokenizers.pre_tokenizers import Whitespace
  from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

  path = Path(MODEL_DIR.name)
  vocab = {str(token): token for token in range(32)}
  vocab.update({"[UNK]": 32, "[PAD]": 33, "[BOS]": 34, "[EOS]": 35})
  tokenizer_model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
  tokenizer_model.pre_tokenizer = Whitespace()
  tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_model, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
  tokenizer.save_pretrained(path)
  config = LlamaConfig(
    vocab_size=36,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=32,
    pad_token_id=33,
    bos_token_id=34,
    eos_token_id=35,
  )
  LlamaForCausalLM(config).save_pretrained(path)
  return str(path)


def discover_methods() -> list[str]:
  methods = []
  prefix = f"{tinker_public_interfaces.__name__}."
  for module_info in pkgutil.iter_modules(tinker_public_interfaces.__path__, prefix):
    module = importlib.import_module(module_info.name)
    module_name = module_info.name.rsplit(".", 1)[-1]
    client_name = "".join(part.capitalize() for part in module_name.split("_"))
    cls = getattr(module, client_name, None)
    if not (inspect.isclass(cls) and client_name.endswith("Client")):
      continue
    for method_name in dir(cls):
      if not method_name.startswith("_") and callable(getattr(cls, method_name, None)):
        methods.append(f"{client_name}.{method_name}")
  return sorted(methods)


def make_clients(base_url: str):
  service = tinker.ServiceClient(api_key="tml-dummy-key", base_url=base_url)
  training_client = service.create_lora_training_client(base_model=compat_model_path(), rank=2, train_unembed=False)
  state_path = training_client.save_state(STATE_NAME).result(timeout=PROBE_TIMEOUT).path
  sampling_client = training_client.save_weights_and_get_sampling_client(name=SAMPLING_ID)
  return (
    service,
    state_path,
    {
      "RestClient": rest_client_module.RestClient(service.holder),
      "SamplingClient": sampling_client,
      "ServiceClient": service,
      "TrainingClient": training_client,
    },
  )


def probe_datum() -> types.Datum:
  return types.Datum(model_input=types.ModelInput.from_ints(tokens=[1, 2]), loss_fn_inputs={"target_tokens": [2], "weights": [1.0]})


def probe_arg_values(state_path: str) -> dict[str, Any]:
  return {
    "adam_params": types.AdamParams(learning_rate=1e-4),
    "base_model": compat_model_path(),
    "checkpoint_id": "compat-checkpoint",
    "data": [probe_datum()],
    "loss_fn": "cross_entropy",
    "model_path": f"tinker://{MODEL_ID}/sampler_weights/probe",
    "name": "compat-checkpoint",
    "num_samples": 1,
    "path": state_path,
    "prompt": types.ModelInput.from_ints(tokens=[1, 2]),
    "queue_state": QueueState.ACTIVE,
    "queue_state_reason": None,
    "rank": 2,
    "sampler_id": SAMPLING_ID,
    "sampling_params": types.SamplingParams(max_tokens=1),
    "seed": 0,
    "session_id": "compat-session",
    "tinker_path": f"tinker://{MODEL_ID}/weights/compat-checkpoint",
    "train_attn": True,
    "train_mlp": True,
    "train_unembed": False,
    "training_run_id": MODEL_ID,
    "ttl_seconds": 60,
    "user_metadata": {},
  }


def method_call_args(method: Any, state_path: str) -> tuple[list[Any], dict[str, Any]]:
  values = probe_arg_values(state_path)
  args: list[Any] = []
  kwargs: dict[str, Any] = {}
  params = inspect.signature(method).parameters
  for name, param in params.items():
    if name == "self" or param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
      continue
    if name not in values:
      if param.default is inspect.Parameter.empty:
        raise KeyError(name)
      continue
    if param.default is not inspect.Parameter.empty and (
      name not in {"base_model", "model_path", "rank", "seed", "train_attn", "train_mlp", "train_unembed", "user_metadata"}
      or (name == "model_path" and "base_model" in params)
    ):
      continue
    if param.kind is param.POSITIONAL_ONLY:
      args.append(values[name])
    else:
      kwargs[name] = values[name]
  return args, kwargs


def wait_for_result(value: Any, timeout: float = PROBE_TIMEOUT) -> Any:
  async def wait_for_awaitable(awaitable: Any) -> Any:
    return await asyncio.wait_for(awaitable, timeout=timeout)

  if inspect.isawaitable(value):
    value = asyncio.run(wait_for_awaitable(value))
  result = getattr(value, "result", None)
  if callable(result):
    value = result(timeout=timeout)
  return asyncio.run(wait_for_awaitable(value)) if inspect.isawaitable(value) else value


@contextlib.contextmanager
def method_timeout():
  def raise_timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError(f"Probe exceeded {PROBE_TIMEOUT}s")

  previous_handler = signal.getsignal(signal.SIGALRM)
  signal.signal(signal.SIGALRM, raise_timeout)
  previous_timer = signal.setitimer(signal.ITIMER_REAL, PROBE_TIMEOUT)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    signal.signal(signal.SIGALRM, previous_handler)


def probe_method(clients: dict[str, Any], state_path: str, method_id: str) -> str:
  client_name, method_name = method_id.split(".", 1)
  method = getattr(clients[client_name], method_name)
  args, kwargs = method_call_args(method, state_path)
  with method_timeout():
    wait_for_result(method(*args, **kwargs))
  return "supported"


def probe_server(methods: list[str]) -> dict[str, str]:
  with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
    openrl_server(
      compat_model_path(),
      single_process=True,
      startup_timeout=STARTUP_TIMEOUT,
      extra_env={"OPEN_RL_TARGET_MODULES": "all-linear"},
    ) as base_url,
  ):
    service, state_path, clients = make_clients(base_url)
    try:
      statuses = {}
      methods = [method_id for method_id in methods if "load_" not in method_id] + [method_id for method_id in methods if "load_" in method_id]
      for method_id in methods:
        try:
          statuses[method_id] = probe_method(clients, state_path, method_id)
        except Exception:
          statuses[method_id] = "unsupported"
      return statuses
    finally:
      service.holder.close()


@functools.cache
def compatibility_statuses() -> dict[str, str]:
  return probe_server(discover_methods())


def append_methods(lines: list[str], title: str, methods: list[str]) -> None:
  lines.extend([f"## {title}", ""])
  if not methods:
    lines.extend(["None.", ""])
    return
  clients: dict[str, list[str]] = {}
  for method_id in sorted(methods):
    client, method = method_id.split(".", 1)
    clients.setdefault(client, []).append(method)
  for client, client_methods in clients.items():
    lines.extend([f"### {client}", ""])
    lines.extend(f"- `{method}`" for method in client_methods)
    lines.append("")


def render_report(statuses: dict[str, str]) -> str:
  supported = sorted(key for key, value in statuses.items() if value == "supported")
  unsupported = sorted(key for key, value in statuses.items() if value == "unsupported")
  lines = [
    "# Tinker Client Compatibility",
    "",
    f"Generated from `tinker=={importlib.metadata.version('tinker')}` by",
    "`tests/tinker_client_compat.py`.",
    "",
    "The test discovers public Tinker client methods with `dir()` and `inspect`,",
    "starts the real Open-RL FastAPI gateway in single-process mode with a tiny",
    "local model fixture, lets the SDK fetch server bootstrap config, calls each",
    "discovered method",
    "with small fixture arguments, and records whether the call succeeds before",
    "the probe timeout.",
    "",
    f"- Supported methods: {len(supported)}",
    f"- Unsupported methods: {len(unsupported)}",
    "",
  ]
  append_methods(lines, "Supported Methods", supported)
  append_methods(lines, "Unsupported Methods", unsupported)
  return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
  statuses = compatibility_statuses()
  if set(statuses) != set(discover_methods()):
    raise RuntimeError("Compatibility probe did not cover every discovered method.")
  REPORT_PATH.write_text(render_report(statuses))
  print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
