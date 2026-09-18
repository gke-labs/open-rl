"""RL dataset for the cookbook Text-to-SQL recipe, built on ``utils.rewards.load_dataset_splits``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chz
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from utils.rewards import DEFAULT_DATASET, load_dataset_splits

from .env import TextToSqlGroupBuilder


class TextToSqlDataset(RLDataset):
  def __init__(self, builders: list[TextToSqlGroupBuilder], batch_size: int):
    self.builders = builders
    self.batch_size = batch_size

  def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    start = index * self.batch_size
    return self.builders[start : start + self.batch_size]

  def __len__(self) -> int:
    return max(1, len(self.builders) // self.batch_size)


@chz.chz
class TextToSqlDatasetBuilder(RLDatasetBuilder):
  model_name_for_tokenizer: str
  renderer_name: str
  group_size: int
  groups_per_batch: int
  dataset_name: str = DEFAULT_DATASET
  dataset_limit: int = 12_500
  train_limit: int = 5_000
  eval_limit: int = 100
  seed: int = 42
  exec_timeout: int = 30
  # "agent_sandbox": training groups claim one sandbox each (make_envs/cleanup),
  # the eval leases from a shared pool of eval_pool_size. "local": in-process.
  sandbox: str = "agent_sandbox"
  warm_pool: str = "text-to-sql-executor"
  namespace: str = "openrl-system"
  sandbox_ready_timeout: int = 180
  eval_pool_size: int = 4

  def _builders(self, rows: list[dict[str, Any]], group_size: int, *, factory: Any, pool: Any) -> list[TextToSqlGroupBuilder]:
    return [
      TextToSqlGroupBuilder(
        row=row,
        model_name_for_tokenizer=self.model_name_for_tokenizer,
        renderer_name=self.renderer_name,
        group_size=group_size,
        sandbox_factory=factory,
        pool=pool,
        exec_timeout=self.exec_timeout,
      )
      for row in rows
    ]

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    train_rows, eval_rows = load_dataset_splits(
      dataset_name=self.dataset_name,
      dataset_limit=self.dataset_limit,
      train_limit=self.train_limit,
      eval_limit=self.eval_limit,
      seed=self.seed,
    )
    factory = pool = None
    if self.sandbox == "agent_sandbox":
      from common.agent_sandbox import AgentSandboxPool, make_client, make_sandbox_factory

      factory = make_sandbox_factory(warm_pool=self.warm_pool, namespace=self.namespace, ready_timeout=self.sandbox_ready_timeout)
      # The eval pool lives for the whole run; the SDK's atexit sweep deletes its claims.
      pool = AgentSandboxPool(
        size=self.eval_pool_size,
        warm_pool=self.warm_pool,
        namespace=self.namespace,
        ready_timeout=self.sandbox_ready_timeout,
        client=make_client(cleanup_at_exit=True),
      )
      await pool.__aenter__()
    elif self.sandbox != "local":
      raise ValueError(f"sandbox must be 'agent_sandbox' or 'local', got {self.sandbox!r}")

    train = TextToSqlDataset(self._builders(train_rows, self.group_size, factory=factory, pool=None), self.groups_per_batch)
    # Eval: one sample per prompt, whole eval split in one batch, sandboxes leased from the pool.
    test = TextToSqlDataset(self._builders(eval_rows, 1, factory=None, pool=pool), max(1, len(eval_rows)))
    return train, test
