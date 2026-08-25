"""Tiny RL smoke test: sample from the current policy, reward completions that
contain the target answer, and run a few importance-sampling policy-gradient steps.

  uv --project examples run python examples/tiny/tiny_rl.py base_url=http://127.0.0.1:9003
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
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
  log_dir: str = str(Path(__file__).with_name("artifacts") / "tiny_rl")
  prompt: str = "Question: What is 2 + 2?\nAnswer:"
  target: str = "4"
  steps: int = 2
  samples_per_prompt: int = 8
  max_tokens: int = 16
  temperature: float = 1.0
  learning_rate: float = 1e-5
  grad_clip_norm: float = 1.0
  loss_fn: str = "importance_sampling"
  rank: int = 16
  seed: int = 0
  behavior_if_log_dir_exists: str = "delete"


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


def build_datum(prompt_tokens: list[int], completion_tokens: list[int], logprobs: list[float], advantage: float) -> types.Datum:
  tokens = prompt_tokens + completion_tokens
  prompt_pad = [0.0] * (len(prompt_tokens) - 1)
  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
    loss_fn_inputs=cast(
      Any,
      {
        "target_tokens": tokens[1:],
        "weights": prompt_pad + [1.0] * len(completion_tokens),
        "logprobs": prompt_pad + logprobs,
        "advantages": prompt_pad + [advantage] * len(completion_tokens),
      },
    ),
  )


def main(config: Config) -> None:
  if config.steps < 1:
    raise ValueError("Tiny RL needs steps >= 1")
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
  tokenizer = trainer.get_tokenizer()
  prompt_tokens = tokenizer.encode(config.prompt, add_special_tokens=False)
  prompt = types.ModelInput.from_ints(tokens=prompt_tokens)
  sampling_params = types.SamplingParams(max_tokens=config.max_tokens, temperature=config.temperature)

  mean_reward = 0.0
  for step in range(1, config.steps + 1):
    sampler = trainer.save_weights_and_get_sampling_client()
    sequences = sampler.sample(prompt=prompt, num_samples=config.samples_per_prompt, sampling_params=sampling_params).result().sequences

    rewards = []
    for sequence in sequences:
      tokens, logprobs = list(sequence.tokens), list(sequence.logprobs or [])
      if not tokens or len(tokens) != len(logprobs):
        raise RuntimeError(f"Sampler must return aligned tokens and logprobs, got {len(tokens)} tokens and {len(logprobs)} logprobs")
      rewards.append(1.0 if config.target in tokenizer.decode(tokens) else 0.0)

    # Group-centered advantages; when every reward ties, fall back to a uniform
    # positive advantage so the update still exercises a nonzero gradient.
    mean_reward = statistics.fmean(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    if all(abs(advantage) < 1e-8 for advantage in advantages):
      advantages = [1.0] * len(rewards)

    datums = [
      build_datum(prompt_tokens, list(sequence.tokens), list(sequence.logprobs or []), advantage)
      for sequence, advantage in zip(sequences, advantages)
    ]
    fwdbwd = trainer.forward_backward(datums, config.loss_fn).result()
    trainer.optim_step(types.AdamParams(learning_rate=config.learning_rate, grad_clip_norm=config.grad_clip_norm)).result()

    loss = float(fwdbwd.metrics.get("loss:mean", 0.0))
    if not math.isfinite(loss):
      raise RuntimeError(f"Loss must be finite, got {loss!r}")
    write_metric(log_dir, {"phase": "train", "step": step, "loss": loss, "mean_reward": mean_reward, "num_datums": len(datums)})
    print(f"[tiny-rl] step={step:02d}/{config.steps} loss={loss:.6f} mean_reward={mean_reward:.2f} datums={len(datums)}")

  final_state_path = trainer.save_state("tiny-rl-final").result().path
  write_metric(log_dir, {"phase": "final", "step": config.steps, "final_state_path": final_state_path, "mean_reward": mean_reward})
  print(f"[tiny-rl] mean_reward={mean_reward:.2f}")
  print(f"final_state_path={final_state_path}")


if __name__ == "__main__":
  chz.nested_entrypoint(main, allow_hyphens=True)
