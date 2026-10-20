"""The demo as a client sees it: one LoRA job, with a timestamp per phase.

  uv --project examples run python scripts/cold_start_client.py base_model=Qwen/Qwen2.5-0.5B out=/path/client.jsonl

Writes one JSON line per phase boundary, with the laptop's UTC clock, so the
timeline (scripts/cold_start_timeline.py) can line the client's view up with
the cluster's records. Phases: create_model, first forward_backward (the
trainer is serving), save_weights_and_get_sampling_client, first sample (the
sampler is serving). Nothing here is tuned; it is the smallest job that
touches both workers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any, cast

import chz
import tinker
from tinker import types

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
sys.stdout.reconfigure(line_buffering=True)


@chz.chz
class Config:
  base_model: str = "Qwen/Qwen2.5-0.5B"
  base_url: str = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:9003"))
  out: str = "client.jsonl"
  prompt: str = "Question: What is 2 + 2?\nAnswer:"
  completion: str = " 4"
  rank: int = 16
  max_tokens: int = 8
  # Stop after the trainer answers; skip the sampler phases.
  trainer_only: bool = False


class Log:
  def __init__(self, path: str):
    self.path = path
    self.t0 = time.time()

  def mark(self, phase: str, **extra: Any) -> None:
    row = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "t": round(time.time() - self.t0, 3), "phase": phase, **extra}
    with open(self.path, "a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")
    print(f"[cold-start] {row['t']:8.3f}s {phase} {extra if extra else ''}")


def build_datum(tokenizer: Any, prompt: str, completion: str) -> types.Datum:
  prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
  completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
  tokens = prompt_tokens + completion_tokens
  weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
    loss_fn_inputs=cast(Any, {"target_tokens": tokens[1:], "weights": weights[1:]}),
  )


def main(config: Config) -> None:
  log = Log(config.out)
  log.mark("start", base_model=config.base_model, base_url=config.base_url)
  client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)

  log.mark("create_model.start")
  trainer = client.create_lora_training_client(base_model=config.base_model, rank=config.rank, train_attn=True, train_mlp=True, train_unembed=False)
  log.mark("create_model.done", model_id=trainer._guaranteed_model_id())

  tokenizer = trainer.get_tokenizer()
  datum = build_datum(tokenizer, config.prompt, config.completion)
  log.mark("forward_backward.start")
  trainer.forward_backward([datum], "cross_entropy").result()
  log.mark("forward_backward.done")
  trainer.optim_step(types.AdamParams(learning_rate=1e-4)).result()
  log.mark("optim_step.done")

  if config.trainer_only:
    log.mark("end")
    return

  log.mark("save_weights_and_get_sampling_client.start")
  sampler = trainer.save_weights_and_get_sampling_client()
  log.mark("save_weights_and_get_sampling_client.done")
  log.mark("sample.start")
  result = sampler.sample(
    prompt=types.ModelInput.from_ints(tokenizer.encode(config.prompt, add_special_tokens=False)),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=config.max_tokens, temperature=0.0),
  ).result()
  text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
  log.mark("sample.done", text=text)
  log.mark("end")


if __name__ == "__main__":
  chz.nested_entrypoint(main, allow_hyphens=True)
