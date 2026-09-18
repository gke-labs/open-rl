"""Text-to-SQL RL with tinker-cookbook's ``rl_train`` and sandboxed reward execution.

    uv --project examples run python examples/text-to-sql/cookbook/train.py \\
        model_name=Qwen/Qwen3-1.7B base_url=http://open-rl-gateway-service:8000 \\
        sandbox=agent_sandbox log_path=/mnt/shared/open-rl/runs/textsql-cookbook

Each prompt group claims its own gVisor sandbox from the warm pool
(``examples/text-to-sql/k8s/agent-sandbox``) in ``make_envs`` and releases it in
``cleanup``; ``kubectl get sandboxclaims -w`` shows one claim per group per
step. ``sandbox=local`` runs the model's SQL in-process for laptop smoke tests.
The training loop is the cookbook's, unmodified: OpenRL is only the Tinker
endpoint behind ``base_url``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make `common` (examples/) and `utils` (examples/text-to-sql/) importable when
# run as a script from any working directory.
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE.parents[1])]

import chz  # noqa: E402
from tinker_cookbook import cli_utils  # noqa: E402
from tinker_cookbook.rl.rollout_strategy import MinViableGroup  # noqa: E402
from tinker_cookbook.rl.train import Config, main  # noqa: E402

from cookbook.data import TextToSqlDatasetBuilder  # noqa: E402

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")


@chz.chz
class CLIConfig:
  model_name: str = "Qwen/Qwen3-1.7B"
  renderer_name: str = "qwen3_disable_thinking"
  lora_rank: int = 32
  load_checkpoint_path: str | None = None

  # Data
  dataset_limit: int = 12_500
  train_limit: int = 5_000
  eval_limit: int = 100
  seed: int = 42

  # Rollouts / optimisation
  group_size: int = 8
  groups_per_batch: int = 8
  max_tokens: int = 96
  temperature: float = 1.0
  learning_rate: float = 1e-5
  loss_fn: str = "importance_sampling"
  kl_penalty_coef: float = 0.0
  max_steps: int | None = 40
  eval_every: int = 10
  # OpenRL has no durable checkpoint store for cookbook saves yet (issue #83).
  save_every: int = 0

  # Reward execution: "agent_sandbox" claims one sandbox per training prompt
  # group and leases eval sandboxes from a pool of eval_pool_size; "local"
  # runs the model's SQL in-process.
  sandbox: str = "agent_sandbox"
  warm_pool: str = "text-to-sql-executor"
  namespace: str = "openrl-system"
  sandbox_ready_timeout: int = 180
  exec_timeout: int = 30
  eval_pool_size: int = 4

  # Service / logging
  base_url: str | None = os.getenv("TINKER_BASE_URL")
  log_path: str | None = None
  behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"
  wandb_project: str | None = None
  wandb_name: str | None = None


async def cli_main(cli: CLIConfig) -> None:
  model_slug = cli.model_name.replace("/", "-")
  stamp = f"{datetime.now():%Y-%m-%d-%H-%M}"
  run_name = f"textsql-{model_slug}-{cli.lora_rank}rank-{cli.learning_rate}lr-{cli.group_size}x{cli.groups_per_batch}-{cli.sandbox}-{stamp}"
  log_path = cli.log_path or f"/tmp/tinker-examples/text_to_sql/{run_name}"

  dataset_builder = TextToSqlDatasetBuilder(
    model_name_for_tokenizer=cli.model_name,
    renderer_name=cli.renderer_name,
    group_size=cli.group_size,
    groups_per_batch=cli.groups_per_batch,
    dataset_limit=cli.dataset_limit,
    train_limit=cli.train_limit,
    eval_limit=cli.eval_limit,
    seed=cli.seed,
    exec_timeout=cli.exec_timeout,
    sandbox=cli.sandbox,
    warm_pool=cli.warm_pool,
    namespace=cli.namespace,
    sandbox_ready_timeout=cli.sandbox_ready_timeout,
    eval_pool_size=cli.eval_pool_size,
  )
  config = Config(
    learning_rate=cli.learning_rate,
    dataset_builder=dataset_builder,
    model_name=cli.model_name,
    recipe_name="recipe_text_to_sql_sandboxed",
    renderer_name=cli.renderer_name,
    lora_rank=cli.lora_rank,
    max_tokens=cli.max_tokens,
    temperature=cli.temperature,
    loss_fn=cli.loss_fn,  # type: ignore[arg-type]
    kl_penalty_coef=cli.kl_penalty_coef,
    eval_every=cli.eval_every,
    save_every=cli.save_every,
    log_path=log_path,
    base_url=cli.base_url,
    load_checkpoint_path=cli.load_checkpoint_path,
    wandb_project=cli.wandb_project,
    wandb_name=cli.wandb_name or run_name,
    max_steps=cli.max_steps,
    # One failed sandbox claim should cost a rollout, not the whole prompt group.
    rollout_error_tolerance=MinViableGroup(),
  )
  cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
  logging.getLogger("tinker").setLevel(logging.WARNING)
  await main(config)


if __name__ == "__main__":
  asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
