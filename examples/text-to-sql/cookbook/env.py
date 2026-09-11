"""Text-to-SQL as a tinker-cookbook ``Env``: one prompt, one SQL answer, one sandbox per group.

``TextToSqlGroupBuilder.make_envs`` claims a sandbox through the injected
``sandbox_factory`` and hands it to ``group_size`` envs; ``cleanup`` deletes the
claim after the group's rollouts and rewards are done. A builder given a
``pool`` instead leases a sandbox per call and never claims: that is what the
held-out eval uses, where one claim per example would mean a hundred cold
starts per eval. The model's SQL is the
only thing that runs in the sandbox; the target query's rows come precomputed
from the dataset filter, which executes trusted dataset content locally.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker
from common.agent_sandbox import run_sql_in_sandbox
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Action, ActionExtra, Env, EnvGroupBuilder, Observation, StepResult, StopCondition
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.rewards import clean_sql_for_execution, local_executor, score_from_execution

logger = logging.getLogger(__name__)

SandboxFactory = Callable[[], Awaitable[SandboxInterface]]
Executor = Callable[[str, str], Awaitable[tuple[list[tuple[Any, ...]] | None, str | None]]]

METRIC_NAMES = ("compile", "execution_match", "exact_match", "similarity")


class TextToSqlEnv(Env):
  """Single-turn env: render the plain prompt as one user message, score the SQL that comes back."""

  def __init__(self, row: dict[str, Any], renderer: renderers.Renderer, executor: Executor, exec_timeout: int):
    self.row = row
    self.renderer = renderer
    self.executor = executor
    self.exec_timeout = exec_timeout

  async def initial_observation(self) -> tuple[Observation, StopCondition]:
    convo = [{"role": "user", "content": self.row["prompt_text"]}]
    return self.renderer.build_generation_prompt(convo), self.renderer.get_stop_sequences()

  async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
    message, _termination = self.renderer.parse_response(action)
    predicted_sql = clean_sql_for_execution(renderers.get_text_content(message))
    started = time.monotonic()
    rows, error = await self.executor(self.row["context"], predicted_sql)
    exec_ms = (time.monotonic() - started) * 1000.0
    score = score_from_execution(
      predicted_sql=predicted_sql,
      target_sql=self.row["target"],
      context=self.row["context"],
      target_rows=self.row.get("target_rows"),
      question=self.row["question"],
      predicted_rows=rows,
      predicted_error=error,
    )
    metrics = {name: float(score[name]) for name in METRIC_NAMES}
    metrics["sandbox_exec_ms"] = exec_ms
    metrics["sandbox_error"] = float(bool(error and error.startswith("sandbox:")))
    return StepResult(
      reward=float(score["reward"]),
      episode_done=True,
      next_observation=tinker.ModelInput.from_ints([]),
      next_stop_condition=[],
      metrics=metrics,
      logs={"predicted_sql": predicted_sql, "target_sql": score["target"], "sqlite_error": score["sqlite_error"]},
    )


@dataclass(frozen=True)
class TextToSqlGroupBuilder(EnvGroupBuilder):
  """``group_size`` envs for one prompt sharing one sandbox claimed in ``make_envs``.

  Holds only the row, names, and the factory reference so it stays cheap; the
  renderer and the sandbox are constructed lazily. With ``sandbox_factory=None``
  the model's SQL runs in-process (laptop / smoke-test mode).
  """

  row: dict[str, Any]
  model_name_for_tokenizer: str
  renderer_name: str
  group_size: int
  sandbox_factory: SandboxFactory | None = None
  # An entered AgentSandboxPool; when set, envs lease per call instead of claiming.
  pool: Any = None
  exec_timeout: int = 30
  _sandbox: list[SandboxInterface] = field(default_factory=list, compare=False, repr=False)

  async def make_envs(self) -> Sequence[Env]:
    renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
    executor: Executor = local_executor
    if self.pool is not None:
      pool = self.pool

      async def executor(context: str, query: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
        async with pool.lease() as sandbox:
          return await run_sql_in_sandbox(sandbox, context, query, timeout=self.exec_timeout)

    elif self.sandbox_factory is not None:
      started = time.monotonic()
      sandbox = await self.sandbox_factory()
      self._sandbox.append(sandbox)
      logger.debug("group sandbox %s claimed in %.1fs", sandbox.sandbox_id, time.monotonic() - started)

      async def executor(context: str, query: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
        return await run_sql_in_sandbox(sandbox, context, query, timeout=self.exec_timeout)

    return [TextToSqlEnv(self.row, renderer, executor, self.exec_timeout) for _ in range(self.group_size)]

  async def cleanup(self) -> None:
    while self._sandbox:
      sandbox = self._sandbox.pop()
      try:
        await sandbox.cleanup()
      except Exception as exc:  # cleanup must not raise into do_group_rollout
        logger.warning("sandbox %s cleanup failed: %s", sandbox.sandbox_id, exc)

  def logging_tags(self) -> list[str]:
    return ["text_to_sql"]
