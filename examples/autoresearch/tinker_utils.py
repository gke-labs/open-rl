"""Shared helpers for Open-RL/Tinker autoresearch recipes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from io import StringIO
from typing import Any

from rich.console import Console
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset
from tinker_cookbook.utils import ml_log

DEFAULT_BASE_URL = "http://localhost:9003"


def force_rich_log_colors() -> None:
  """Keep Rich ANSI colors when cookbook console output is captured into attempt.log."""

  def init(self: ml_log.PrettyPrintLogger) -> None:
    self.console = Console(
      color_system="truecolor",
      force_terminal=True,
      file=StringIO(),
      record=True,
      width=int(os.getenv("COLUMNS", "120")),
    )
    self._last_step = None

  @contextmanager
  def rich_console_use_logger(console: Console):
    yield
    text = console.export_text(styles=True).rstrip()
    if text:
      ml_log.logger.info("\n" + text)

  ml_log.PrettyPrintLogger.__init__ = init
  ml_log._rich_console_use_logger = rich_console_use_logger


def resolve_base_url(cli_base_url: str | None) -> str:
  base_url = cli_base_url or os.getenv("TINKER_BASE_URL") or os.getenv("BASE_URL") or DEFAULT_BASE_URL
  os.environ["TINKER_BASE_URL"] = base_url
  return base_url


class LimitedDataset(RLDataset):
  def __init__(self, dataset: RLDataset, max_batches: int):
    self.dataset = dataset
    self.max_batches = max_batches

  def __len__(self) -> int:
    return min(len(self.dataset), self.max_batches)

  def get_batch(self, index: int) -> list[EnvGroupBuilder]:
    if index >= len(self):
      raise IndexError(index)
    return list(self.dataset.get_batch(index))


class LimitedDatasetBuilder:
  def __init__(self, builder: Any, max_batches: int | None, max_eval_batches: int | None = None):
    self.builder = builder
    self.max_batches = max_batches
    self.max_eval_batches = max_eval_batches

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    train_dataset, eval_dataset = await self.builder()
    if self.max_batches is not None:
      train_dataset = LimitedDataset(train_dataset, self.max_batches)
    if eval_dataset is not None and self.max_eval_batches is not None:
      eval_dataset = LimitedDataset(eval_dataset, self.max_eval_batches)
    return train_dataset, eval_dataset
