"""Tiny SFT smoke test: overfit a single prompt/completion pair and require the
training loss to drop.

  uv --project examples run python examples/tiny/tiny_sft.py base_url=http://127.0.0.1:9003
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from tinker import types

BASE_URL = "http://127.0.0.1:9003"

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
  base_model: str = "Qwen/Qwen2.5-0.5B"
  base_url: str = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", BASE_URL))
  log_dir: str = str(Path(__file__).with_name("artifacts") / "tiny_sft")
  prompt: str = "Question: What is 2 + 2?\nAnswer:"
  completion: str = " 4"
  steps: int = 10
  rank: int = 16
  learning_rate: float = 1e-3
  grad_clip_norm: float = 1.0
  min_loss_drop: float = 0.02
  seed: int = 0
  behavior_if_log_dir_exists: str = "delete"
  sample_after_train: bool = False


def reset_log_dir(path: Path, behavior: str) -> None:
  if not path.exists():
    path.mkdir(parents=True)
    return
  if behavior == "delete":
    shutil.rmtree(path)
    path.mkdir(parents=True)
    return
  if behavior == "error":
    raise RuntimeError(f"Log directory already exists: {path}")
  raise ValueError(f"Unsupported behavior_if_log_dir_exists={behavior!r}")


def write_metric(log_dir: Path, row: dict[str, Any]) -> None:
  with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")


def build_datum(tokenizer: Any, prompt: str, completion: str) -> tuple[types.Datum, int]:
  prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
  completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
  if tokenizer.eos_token_id is not None:
    completion_tokens.append(tokenizer.eos_token_id)
  if not prompt_tokens or not completion_tokens:
    raise RuntimeError("Tiny SFT prompt and completion must both produce tokens")

  tokens = prompt_tokens + completion_tokens
  weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
  datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
    loss_fn_inputs=cast(Any, {"target_tokens": tokens[1:], "weights": weights[1:]}),
  )
  return datum, len(completion_tokens)


def measure_loss(trainer: tinker.TrainingClient, datum: types.Datum, active_tokens: int, grad_clip_norm: float) -> float:
  fwdbwd = trainer.forward_backward([datum], "cross_entropy").result()
  trainer.optim_step(types.AdamParams(learning_rate=0.0, grad_clip_norm=grad_clip_norm)).result()
  loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / active_tokens
  if not math.isfinite(loss):
    raise RuntimeError(f"Loss must be finite, got {loss!r}")
  return loss


def main(config: Config) -> None:
  log_dir = Path(config.log_dir)
  reset_log_dir(log_dir, config.behavior_if_log_dir_exists)

  client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)
  trainer = client.create_lora_training_client(
    base_model=config.base_model,
    rank=config.rank,
    seed=config.seed,
    train_attn=True,
    train_mlp=True,
    # Qwen2.5-0.5B ties lm_head to embed_tokens; LoRA on the tied head trips a
    # PEFT warning and vLLM cannot load lm_head adapter weights at all.
    train_unembed=False,
  )
  # Progress and diagnosis live on the server. While a call below blocks, this
  # tells you what the cluster is doing for this model.
  print(f"[tiny-sft] model_id={trainer.model_id}")
  print(f"[tiny-sft] status: curl -s {config.base_url}/api/v1/openrl/models/{trainer.model_id} | jq .summary")
  tokenizer = trainer.get_tokenizer()
  datum, active_tokens = build_datum(tokenizer, config.prompt, config.completion)

  initial_loss = measure_loss(trainer, datum, active_tokens, config.grad_clip_norm)
  write_metric(log_dir, {"phase": "initial", "loss": initial_loss, "step": 0})
  print(f"[tiny-sft] initial_loss={initial_loss:.6f}")

  for step in range(1, config.steps + 1):
    fwdbwd = trainer.forward_backward([datum], "cross_entropy").result()
    trainer.optim_step(types.AdamParams(learning_rate=config.learning_rate, grad_clip_norm=config.grad_clip_norm)).result()
    loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / active_tokens
    write_metric(log_dir, {"phase": "train", "loss": loss, "step": step})
    print(f"[tiny-sft] step={step:02d}/{config.steps} loss={loss:.6f}")

  final_loss = measure_loss(trainer, datum, active_tokens, config.grad_clip_norm)
  loss_drop = (initial_loss - final_loss) / (abs(initial_loss) or 1.0)
  final_state_path = trainer.save_state("tiny-sft-final").result().path
  write_metric(
    log_dir,
    {
      "final_state_path": final_state_path,
      "loss": final_loss,
      "loss_drop": loss_drop,
      "phase": "final",
      "step": config.steps,
    },
  )

  print(f"[tiny-sft] final_loss={final_loss:.6f}")
  print(f"[tiny-sft] loss_drop={loss_drop:.1%}")
  print(f"final_state_path={final_state_path}")

  if loss_drop < config.min_loss_drop:
    raise RuntimeError(f"Expected loss_drop >= {config.min_loss_drop:.1%}, got {loss_drop:.1%}")

  if config.sample_after_train:
    sampler = trainer.save_weights_and_get_sampling_client()
    result = sampler.sample(
      prompt=types.ModelInput.from_ints(tokenizer.encode(config.prompt, add_special_tokens=False)),
      num_samples=1,
      sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0),
    ).result()
    if len(result.sequences) != 1 or not result.sequences[0].tokens:
      raise RuntimeError("The sampler did not generate tokens from the saved adapter")
    tokens = result.sequences[0].tokens
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    write_metric(log_dir, {"phase": "sample", "tokens": tokens, "text": text})
    print(f"[tiny-sft] sampled_saved_adapter={text!r}")


if __name__ == "__main__":
  chz.nested_entrypoint(main, allow_hyphens=True)
