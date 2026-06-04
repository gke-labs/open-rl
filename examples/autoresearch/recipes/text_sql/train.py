"""Editable and runnable Text-SQL autoresearch attempt."""

from __future__ import annotations

import json
import os
import statistics
import time
from difflib import SequenceMatcher
from itertools import count
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from tinker import types
from tinker_cookbook.utils import ml_log
from tinker_utils import force_rich_log_colors, resolve_base_url
from transformers import AutoTokenizer

from recipes.text_sql import prepare

TRAIN_STEPS = 40
DEFAULT_BASE_MODEL = "google/gemma-4-e2b"


@chz.chz
class RunConfig:
  run_dir: Path = chz.field(doc="Attempt artifact directory written by run_attempt.")
  data_dir: Path = Path("artifacts/autoresearch/text_sql")
  attempt_timeout_minutes: float = 30.0
  base_url: str | None = None
  base_model: str = DEFAULT_BASE_MODEL
  tokenizer_name: str | None = None
  rank: int = 8
  batch_size: int = 4
  samples_per_prompt: int = 4
  learning_rate: float = 0.0002
  grad_clip_norm: float = 0.3
  temperature: float = 0.8
  clip_range: float = 0.2
  kl_coeff: float = 0.1
  eval_max_tokens: int = 96
  seed: int = prepare.SEED


def prompt_for(example: dict[str, Any]) -> str:
  return (
    "Return only one SQLite query. Do not include markdown or explanation.\n\n"
    f"Schema:\n{example['context']}\n\n"
    f"Question:\n{example['question']}\n\n"
    "SQL:\n"
  )


def make_datum(
  tokens: list[int],
  weights: list[float],
  logprobs: list[float],
  advantages: list[float],
) -> types.Datum:
  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
    loss_fn_inputs=cast(
      Any,
      {
        "target_tokens": tokens[1:],
        "weights": weights,
        "logprobs": logprobs,
        "advantages": advantages,
      },
    ),
  )


def score_sql(example: dict[str, Any], predicted: str) -> tuple[float, bool, str]:
  target = prepare.clean_sql(example["target"])
  predicted = prepare.clean_sql(predicted)
  correct = predicted == target
  if correct:
    return 1.0, True, ""
  similarity = SequenceMatcher(None, predicted.lower(), target.lower()).ratio()
  return 0.25 * similarity, False, ""


def build_rollout(tokenizer: Any, example: dict[str, Any], sequence: Any) -> dict[str, Any]:
  decoded = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
  decoded = decoded.split(";")[0].split("```")[0].split("Example:")[0].split("Question:")[0].split("Schema:")[0].strip()
  predicted = prepare.clean_sql(decoded)
  reward, correct, execution_error = score_sql(example, predicted)
  return {
    "correct": correct,
    "execution_error": execution_error,
    "predicted": predicted,
    "prompt_tokens": tokenizer.encode(prompt_for(example), add_special_tokens=False),
    "completion_tokens": list(sequence.tokens),
    "completion_logprobs": [float(value) for value in (sequence.logprobs or [])],
    "reward": reward,
  }


def create_service_client(args: RunConfig) -> Any:
  base_url = resolve_base_url(args.base_url)
  return tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=base_url)


def create_base_sampler_and_tokenizer(args: RunConfig) -> tuple[Any, Any]:
  client = create_service_client(args)
  sampler = client.create_sampling_client(base_model=args.base_model)
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.base_model)
  return sampler, tokenizer


def create_trainer_and_tokenizer(args: RunConfig) -> tuple[Any, Any]:
  client = create_service_client(args)
  trainer = client.create_lora_training_client(
    base_model=args.base_model,
    rank=args.rank,
    seed=args.seed,
    train_mlp=True,
    train_attn=True,
    train_unembed=False,
  )
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.base_model)
  return trainer, tokenizer


def sample_sql(sampler: Any, tokenizer: Any, example: dict[str, Any], max_tokens: int, seed: int) -> str:
  prompt_tokens = tokenizer.encode(prompt_for(example), add_special_tokens=False)
  response = sampler.sample(
    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=seed),
  ).result()
  completion = response.sequences[0].tokens if response.sequences else []
  decoded = tokenizer.decode(completion, skip_special_tokens=True)
  decoded = decoded.split(";")[0].split("```")[0].split("Example:")[0].split("Question:")[0].split("Schema:")[0].strip()
  return prepare.clean_sql(decoded)


def train(examples: list[dict[str, Any]], args: RunConfig, metrics_logger) -> tuple[Any, Any, dict[str, float]]:
  trainer, tokenizer = create_trainer_and_tokenizer(args)
  prompts_per_step = max(1, min(args.batch_size, len(examples)))
  train_rows = examples[:]
  deadline = time.monotonic() + args.attempt_timeout_minutes * 60
  start = time.monotonic()
  losses: list[float] = []
  rewards: list[float] = []
  for step in range(1, TRAIN_STEPS + 1):
    step_start = time.monotonic()
    if time.monotonic() >= deadline:
      break

    sampler = trainer.save_weights_and_get_sampling_client(name=f"text_sql_rollout_{step}")
    offset = ((step - 1) * prompts_per_step) % len(train_rows)
    batch = train_rows[offset : offset + prompts_per_step]
    if len(batch) < prompts_per_step:
      batch += train_rows[: prompts_per_step - len(batch)]

    futures = []
    for prompt_index, example in enumerate(batch):
      prompt_tokens = tokenizer.encode(prompt_for(example), add_special_tokens=False)
      rollout_seed = args.seed + step * 1000 + prompt_index
      sampling_params = types.SamplingParams(
        max_tokens=args.eval_max_tokens,
        temperature=args.temperature,
        seed=rollout_seed,
      )
      future = sampler.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=args.samples_per_prompt,
        sampling_params=sampling_params,
      )
      futures.append(future)

    responses = []
    for future in futures:
      response = future.result()
      responses.append(response)

    datums = []
    rollouts = []
    for example, response in zip(batch, responses, strict=True):
      group = [build_rollout(tokenizer, example, sequence) for sequence in response.sequences]
      group_rewards = [rollout["reward"] for rollout in group]
      reward_mean = statistics.fmean(group_rewards)
      reward_std = statistics.pstdev(group_rewards)

      if len(group_rewards) < 2 or reward_std < 1e-8:
        advantages = [0.0] * len(group_rewards)
      else:
        advantages = []
        for reward in group_rewards:
          advantages.append((reward - reward_mean) / reward_std)

      for rollout, advantage in zip(group, advantages, strict=True):
        completion_tokens = rollout["completion_tokens"]
        completion_logprobs = rollout["completion_logprobs"]

        if abs(advantage) < 1e-8:
          continue
        if not completion_tokens or len(completion_tokens) != len(completion_logprobs):
          continue

        prompt_pad = [0.0] * (len(rollout["prompt_tokens"]) - 1)
        datums.append(
          make_datum(
            rollout["prompt_tokens"] + completion_tokens,
            prompt_pad + [1.0] * len(completion_tokens),
            prompt_pad + completion_logprobs,
            prompt_pad + [advantage] * len(completion_tokens),
          )
        )
        rollouts.append(rollout)

    if datums:
      fwdbwd = trainer.forward_backward(datums, "ppo", loss_fn_config={"clip_range": args.clip_range, "kl_coeff": args.kl_coeff}).result()
      trainer.optim_step(types.AdamParams(learning_rate=args.learning_rate, grad_clip_norm=args.grad_clip_norm)).result()
      loss = float(fwdbwd.metrics.get("loss:mean", 0.0))
    else:
      loss = 0.0

    reward = statistics.fmean(float(rollout["reward"]) for rollout in rollouts) if rollouts else 0.0
    accuracy = statistics.fmean(float(rollout["correct"]) for rollout in rollouts) if rollouts else 0.0
    error_rate = statistics.fmean(float(bool(rollout["execution_error"])) for rollout in rollouts) if rollouts else 0.0
    step_seconds = time.monotonic() - step_start
    losses.append(loss)
    rewards.append(reward)

    train_step_metrics = {
      "phase": "train",
      "train/loss": loss,
      "train/reward": reward,
      "train/rollout_accuracy": accuracy,
      "train/rollout_errors": error_rate,
      "train/rollouts": float(len(rollouts)),
      "train/step_seconds": step_seconds,
      "train/steps": float(step),
      "progress/done_frac": step / TRAIN_STEPS,
    }

    metrics_logger.log_metrics(
      train_step_metrics,
      step=step,
    )

  final_sampler = trainer.save_weights_and_get_sampling_client(name="text_sql_final")
  seconds = time.monotonic() - start
  train_metrics = {
    "train/steps": float(len(losses)),
    "train/seconds": seconds,
    "train/loss": losses[-1] if losses else 0.0,
    "train/reward": rewards[-1] if rewards else 0.0,
  }
  return final_sampler, tokenizer, train_metrics


def print_eval_examples(results: list[dict[str, Any]]) -> None:
  print("\nText-SQL eval examples")
  for index, result in enumerate(results, start=1):
    status = "PASS" if result["correct"] else "FAIL"
    print(f"\n[{index:02d}] {status}")
    print(f"Question: {result['question']}")
    print("Schema:")
    print(result["context"])
    print(f"Target: {result['target']}")
    print(f"Predicted: {result['predicted']}")
    if result["execution_error"]:
      print(f"Error: {result['execution_error']}")


def evaluate(examples: list[dict[str, str]], sampler: Any, tokenizer: Any, args: RunConfig) -> tuple[dict[str, float], int]:
  predictions = []
  _, test_examples = prepare.split_examples(examples)
  for seed, example in zip(count(args.seed), test_examples, strict=False):
    predictions.append(sample_sql(sampler, tokenizer, example, args.eval_max_tokens, seed))

  results, metrics = prepare.score_test_predictions(examples, predictions)
  print_eval_examples(results)
  return metrics, int(metrics["dataset/test_size"])


def run(args: RunConfig) -> Path:
  args.run_dir.mkdir(parents=True, exist_ok=True)
  metrics_logger = ml_log.setup_logging(log_dir=str(args.run_dir), config=args, do_configure_logging_module=True)
  try:
    examples = prepare.load_examples(args.data_dir)
    train_examples, _ = prepare.split_examples(examples)
    sampler, tokenizer, train_metrics = train(train_examples, args, metrics_logger)
    metrics, samples = evaluate(examples, sampler, tokenizer, args)
    training_state = {
      "loss": train_metrics["train/loss"],
      "samples": samples,
      "steps": train_metrics["train/steps"],
    }
    training_state_path = args.run_dir / "training_state.json"
    training_state_path.write_text(
      json.dumps(training_state, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    metrics.update(train_metrics)
    metrics["step"] = metrics["train/steps"]
    metrics_logger.log_metrics({"phase": "eval", **metrics}, step=int(metrics["step"]))
  finally:
    metrics_logger.close()
  return args.run_dir


def main() -> None:
  force_rich_log_colors()
  run(chz.entrypoint(RunConfig, allow_hyphens=True))


if __name__ == "__main__":
  main()
