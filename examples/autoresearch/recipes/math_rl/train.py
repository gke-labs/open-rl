"""Training client for the math-RL autoresearch recipe."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import chz
import tinker
import tomllib
from tinker_cookbook import model_info
from tinker_cookbook.eval.benchmarks.gsm8k import check_gsm8k
from tinker_cookbook.recipes.math_rl import math_env as cookbook_math_env
from tinker_cookbook.recipes.math_rl.train import get_dataset_builder as get_math_dataset_builder
from tinker_cookbook.rl import train as rl_train
from tinker_utils import LimitedDatasetBuilder, force_rich_log_colors, resolve_base_url

DEFAULT_CONFIG = Path(__file__).with_name("config.toml")
FIXED_ENV = "gsm8k"
LORA_RANK = 8
MAX_EVAL_BATCHES = 1
NUM_SUBSTEPS = 1
NUM_GROUPS_TO_LOG = 0
RENDERER_OVERRIDES = {
  "Qwen/Qwen2.5-0.5B-Instruct": "qwen3_instruct",
}


@chz.chz
class RunConfig:
  config: Path = DEFAULT_CONFIG
  run_dir: Path = chz.field(doc="Attempt artifact directory written by run_attempt.")
  attempt_name: str = chz.field(doc="Attempt name used for W&B and logs.")
  base_url: str | None = None
  wandb_project: str | None = None
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))


@chz.chz
class TrainConfig:
  max_steps: int = 1
  seed: int = 0
  batch_size: int = 2
  rollouts_per_example: int = 2
  max_tokens: int = 512
  temperature: float = 1.0
  lr: float = 3e-6
  loss: str = "importance_sampling"
  eval_enabled: bool = True
  eval_interval: int = 20


def load_config(path: Path) -> dict[str, Any]:
  with path.open("rb") as f:
    return tomllib.load(f)


def config_view(raw: dict[str, Any]) -> TrainConfig:
  return chz.Blueprint(TrainConfig).apply(raw).make()


def validate(config: TrainConfig) -> None:
  if config.max_steps < 1:
    raise ValueError("max_steps must be >= 1")
  if config.rollouts_per_example < 1 or config.batch_size < 1:
    raise ValueError("rollouts_per_example and batch_size must be >= 1")
  if config.lr <= 0:
    raise ValueError("lr must be positive")


def grade_gsm8k(given_answer: str | None, ground_truth: str, *_: Any, **__: Any) -> bool:
  return bool(given_answer and check_gsm8k(given_answer, ground_truth))


def rollout_metrics(row: dict[str, Any]) -> dict[str, float]:
  steps = row.get("steps", [])

  def avg_step_metric(key: str) -> float:
    return sum(float(step.get("metrics", {}).get(key, 0.0)) for step in steps) / max(len(steps), 1)

  return {
    "step": float(row.get("iteration", 0)),
    "env/all/reward/total": float(row.get("total_reward", 0.0)),
    "accuracy": avg_step_metric("correct"),
    "env/all/format": avg_step_metric("format"),
  }


def summarize_rollouts(run_dir: Path) -> list[dict[str, float]]:
  rows = []
  for path in sorted(run_dir.glob("iteration_*/train_rollout_summaries.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
  by_iteration: dict[float, list[dict[str, float]]] = {}
  for row in rows:
    metrics = rollout_metrics(row)
    by_iteration.setdefault(metrics["step"], []).append(metrics)

  summarized = []
  for step, iteration_rows in sorted(by_iteration.items()):
    summarized.append(
      {
        "step": step,
        "env/all/reward/total": sum(row["env/all/reward/total"] for row in iteration_rows) / len(iteration_rows),
        "accuracy": sum(row["accuracy"] for row in iteration_rows) / len(iteration_rows),
        "env/all/format": sum(row["env/all/format"] for row in iteration_rows) / len(iteration_rows),
      }
    )
  return summarized


def write_metric_summary(run_dir: Path) -> dict[str, float]:
  rows = summarize_rollouts(run_dir)
  if rows:
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
  return rows[-1] if rows else {}


async def backend_model(base_url: str | None) -> str:
  client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=base_url)
  capabilities = await client.get_server_capabilities_async()
  model = next((row.model_name for row in capabilities.supported_models if row.model_name), None)
  if not model:
    raise RuntimeError(f"No model reported by OpenRL backend at {base_url or 'default Tinker endpoint'}")
  return model


def renderer_for(model: str) -> str:
  if model in RENDERER_OVERRIDES:
    return RENDERER_OVERRIDES[model]
  try:
    return model_info.get_recommended_renderer_name(model)
  except KeyError as exc:
    raise RuntimeError(f"No renderer known for backend model {model!r}; add it to RENDERER_OVERRIDES in train.py") from exc


def dataset_builder(config: TrainConfig, model: str, renderer_name: str):
  # GSM8K answers are numeric; reuse the cookbook eval grader instead of the
  # math-RL env's symbolic grader, which routes through signal-based timeouts.
  cookbook_math_env.safe_grade = grade_gsm8k
  return get_math_dataset_builder(
    env=FIXED_ENV,
    batch_size=config.batch_size,
    model_name=model,
    renderer_name=renderer_name,
    group_size=config.rollouts_per_example,
    seed=config.seed,
  )


async def run_training(args: RunConfig) -> None:
  raw = load_config(args.config)
  config = config_view(raw)
  validate(config)
  base_url = resolve_base_url(args.base_url)
  os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
  print("Open-RL Math-RL autoresearch", flush=True)
  print(f"attempt={args.attempt_name}", flush=True)
  print(f"attempt_timeout_minutes={args.attempt_timeout_minutes}", flush=True)
  print(f"resolving_backend={base_url or 'default Tinker endpoint'}", flush=True)
  model = await backend_model(base_url)
  renderer_name = renderer_for(model)
  builder = LimitedDatasetBuilder(dataset_builder(config, model, renderer_name), max_batches=None, max_eval_batches=MAX_EVAL_BATCHES)
  train_config = rl_train.Config(
    learning_rate=config.lr,
    dataset_builder=builder,
    model_name=model,
    renderer_name=renderer_name,
    max_tokens=config.max_tokens,
    temperature=config.temperature,
    lora_rank=LORA_RANK,
    log_path=str(args.run_dir),
    wandb_project=args.wandb_project,
    wandb_name=args.attempt_name,
    base_url=base_url,
    eval_every=config.eval_interval if config.eval_enabled else 0,
    save_every=max(1, config.max_steps),
    max_steps=config.max_steps,
    num_substeps=NUM_SUBSTEPS,
    num_groups_to_log=NUM_GROUPS_TO_LOG,
    loss_fn=config.loss,
    loss_fn_config=None,
    kl_penalty_coef=0.0,
    kl_discount_factor=0.0,
    remove_constant_reward_groups=False,
  )
  print(f"backend_model={model}")
  print(f"renderer={renderer_name}")
  print(f"env={FIXED_ENV}")
  print(f"max_steps={config.max_steps}")
  print(f"log_path={args.run_dir}")
  await rl_train.main(train_config)
  write_metric_summary(args.run_dir)


def main() -> None:
  force_rich_log_colors()
  args = chz.entrypoint(RunConfig, allow_hyphens=True)
  try:
    asyncio.run(asyncio.wait_for(run_training(args), timeout=args.attempt_timeout_minutes * 60))
  except TimeoutError as exc:
    write_metric_summary(args.run_dir)
    print(f"Timed out after {args.attempt_timeout_minutes} minutes; partial metrics remain in {args.run_dir}")
    raise SystemExit(124) from exc


if __name__ == "__main__":
  main()
