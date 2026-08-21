"""LAB task environments for tinker-cookbook RL training."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import chz
from prompts import copy_skill_scripts, default_skills, initial_messages, lab_renderer, lab_system_prompt
from reward import LabRubricReward
from tasks import LabTask, load_lab_tasks, task_slug
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tool_use import build_agent_tool_env
from tools import LabTool

logger = logging.getLogger(__name__)


def add_lab_to_path(lab_root: Path) -> None:
  resolved = str(lab_root.resolve())
  if resolved not in sys.path:
    sys.path.insert(0, resolved)


class CountedCriteriaEnv(Env):
  """Guarantees every episode reports criterion counts, so step-level means
  over `lab/*` keys cover all episodes — including ones that fail before
  grading (parse failure, context overflow)."""

  def __init__(self, env: Env, criteria_count: int):
    self.env = env
    self.criteria_count = criteria_count

  def __getattr__(self, name):
    return getattr(self.env, name)

  async def initial_observation(self):
    return await self.env.initial_observation()

  async def step(self, action, *, extra=None) -> StepResult:
    result = await self.env.step(action, extra=extra)
    metrics = result.metrics or {}
    if result.episode_done and "lab/criteria_total" not in metrics:
      metrics = dict(metrics)
      metrics.update(
        {
          "lab/criteria_passed": 0.0,
          "lab/criteria_total": float(self.criteria_count),
          "lab/criteria_pass_fraction": 0.0,
          "lab/graded": 0.0,
          "lab/reward_error": 0.0,
          "lab/failed_before_grading": 1.0,
        }
      )
      result = dataclass_replace(result, metrics=metrics)
    return result


@dataclass
class LabEnvGroupBuilder(EnvGroupBuilder):
  task: LabTask
  lab_root: Path
  model_name: str
  renderer_name: str | None
  group_size: int
  max_turns: int
  command_timeout: int
  judge_model: str
  judge_parallel: int
  max_reward_criteria: int | None
  max_trajectory_tokens: int
  max_generation_tokens: int | None
  max_tool_result_tokens: int
  sandboxes: list[Any] = field(default_factory=list, init=False, repr=False)

  async def make_envs(self) -> Sequence[Env]:
    add_lab_to_path(self.lab_root)
    from harness.tools import ToolExecutor, get_all_tool_definitions
    from sandbox.sandbox import DEFAULT_IMAGE, Sandbox

    renderer = lab_renderer(self.model_name, self.renderer_name)
    system_prompt = lab_system_prompt(self.lab_root)
    # Keep Harvey LAB's canonical tool names and schemas: its system prompt and
    # teacher traces teach `read`; renaming only the live schema makes valid
    # calls fail.
    lab_tool_definitions = get_all_tool_definitions()
    skills = default_skills(self.lab_root)
    prefix_messages = initial_messages(self.task, renderer, system_prompt, lab_tool_definitions)

    def start_sandbox() -> tuple[str, Any]:
      run_id = f"open-rl-harvey-labs/{task_slug(self.task.name)}/{uuid.uuid4().hex[:12]}"
      run_dir = self.lab_root / "results" / run_id
      output_dir = run_dir / "output"
      workspace_dir = run_dir / "workspace"
      output_dir.mkdir(parents=True, exist_ok=True)
      workspace_dir.mkdir(parents=True, exist_ok=True)
      copy_skill_scripts(self.lab_root, workspace_dir)
      sandbox = Sandbox(
        documents_dir=self.task.documents_dir,
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        image=DEFAULT_IMAGE,
        default_timeout=self.command_timeout,
      )
      sandbox.start()
      return run_id, sandbox

    # Containers are independent; start the group concurrently off the event loop.
    started = await asyncio.gather(*(asyncio.to_thread(start_sandbox) for _ in range(self.group_size)))

    criteria_count = self.task.criteria_count
    if self.max_reward_criteria is not None:
      criteria_count = min(criteria_count, self.max_reward_criteria)

    envs: list[Env] = []
    for run_id, sandbox in started:
      self.sandboxes.append(sandbox)
      executor = ToolExecutor(sandbox=sandbox, shell_timeout=self.command_timeout)
      tools = [
        LabTool(spec=dict(spec), executor=executor, tokenizer=renderer.tokenizer, max_result_tokens=self.max_tool_result_tokens)
        for spec in lab_tool_definitions
      ]
      reward = LabRubricReward(
        lab_root=self.lab_root,
        run_id=run_id,
        task_name=self.task.name,
        judge_model=self.judge_model,
        task_instructions=self.task.instructions,
        judge_parallel=self.judge_parallel,
        max_criteria=self.max_reward_criteria,
        criteria_count=criteria_count,
        tool_metrics=executor.get_metrics,
        config={
          "model": self.model_name,
          "renderer": self.renderer_name,
          "max_turns": self.max_turns,
          "skills": skills,
        },
      )
      envs.append(
        CountedCriteriaEnv(
          build_agent_tool_env(
            renderer=renderer,
            tools=tools,
            initial_messages=prefix_messages,
            reward_fn=reward,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
          ),
          criteria_count,
        )
      )
    return envs

  async def cleanup(self) -> None:
    for sandbox in self.sandboxes:
      try:
        sandbox.stop()
      except Exception as exc:
        logger.warning("LAB sandbox cleanup failed: %s", exc)
    self.sandboxes.clear()

  def logging_tags(self) -> list[str]:
    return ["harvey-labs"]


@dataclass(frozen=True)
class LabDataset(RLDataset):
  groups: list[LabEnvGroupBuilder]
  batch_size: int

  def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    start = index * self.batch_size
    return self.groups[start : start + self.batch_size]

  def __len__(self) -> int:
    return (len(self.groups) + self.batch_size - 1) // self.batch_size


@chz.chz
class LabDatasetBuilder(RLDatasetBuilder):
  """All knobs are required; defaults live in train.py's RunConfig."""

  lab_root: Path
  task_names: list[str]
  eval_task_names: list[str]
  train_limit: int | None
  eval_limit: int | None
  batch_size: int
  group_size: int
  model_name: str
  renderer_name: str | None
  max_turns: int
  command_timeout: int
  judge_model: str
  judge_parallel: int
  max_reward_criteria: int | None
  max_trajectory_tokens: int
  max_generation_tokens: int | None
  max_tool_result_tokens: int

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    lab_root = self.lab_root.resolve()
    train_tasks = load_lab_tasks(lab_root, self.task_names, limit=self.train_limit)
    if not train_tasks:
      raise ValueError("No LAB train tasks selected")

    train = LabDataset([self._env_group(task, lab_root, self.group_size) for task in train_tasks], self.batch_size)
    if not self.eval_limit or not self.eval_task_names:
      return train, None
    # Held-out progress evals: single rollouts on the dedicated eval tasks,
    # graded with the same rubric settings as training.
    eval_tasks = load_lab_tasks(lab_root, self.eval_task_names, limit=self.eval_limit)
    eval_dataset = LabDataset([self._env_group(task, lab_root, 1) for task in eval_tasks], self.batch_size)
    return train, eval_dataset

  def _env_group(self, task: LabTask, lab_root: Path, group_size: int) -> LabEnvGroupBuilder:
    return LabEnvGroupBuilder(
      task=task,
      lab_root=lab_root,
      model_name=self.model_name,
      renderer_name=self.renderer_name,
      group_size=group_size,
      max_turns=self.max_turns,
      command_timeout=self.command_timeout,
      judge_model=self.judge_model,
      judge_parallel=self.judge_parallel,
      max_reward_criteria=self.max_reward_criteria,
      max_trajectory_tokens=self.max_trajectory_tokens,
      max_generation_tokens=self.max_generation_tokens,
      max_tool_result_tokens=self.max_tool_result_tokens,
    )
