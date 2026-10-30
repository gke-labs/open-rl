"""LAB task environments for tinker-cookbook RL training."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from .config import RunConfig
from .episode import LabEpisodeEnv
from .prompts import copy_skill_scripts, default_skills, initial_messages, lab_renderer, lab_system_prompt
from .reward import LabRubricReward
from .sandbox import LabSandbox, SandboxFactory, SandboxRequest, podman_sandbox_factory
from .tasks import LabTask, load_lab_tasks, random_task_split, task_slug
from .tools import LabTool

logger = logging.getLogger(__name__)


@chz.chz
class LabEnvGroupBuilder(EnvGroupBuilder):
  task: LabTask
  config: RunConfig
  group_size: int
  sandbox_factory: SandboxFactory = podman_sandbox_factory

  @chz.init_property
  def sandboxes(self) -> list[LabSandbox]:
    return []

  async def make_envs(self) -> Sequence[Env]:
    renderer = lab_renderer(self.config.model_name, self.config.resolved_renderer)
    system_prompt = lab_system_prompt(self.config.lab_root)
    skills = default_skills(self.config.lab_root)

    async def start_sandbox() -> tuple[str, LabSandbox]:
      run_id = f"open-rl-harvey-labs/{task_slug(self.task.name)}/{uuid.uuid4().hex[:12]}"
      run_dir = self.config.lab_root / "results" / run_id
      output_dir = run_dir / "output"
      workspace_dir = run_dir / "workspace"
      output_dir.mkdir(parents=True, exist_ok=True)
      workspace_dir.mkdir(parents=True, exist_ok=True)
      await asyncio.to_thread(copy_skill_scripts, self.config.lab_root, workspace_dir)
      sandbox = await self.sandbox_factory(
        SandboxRequest(
          lab_root=self.config.lab_root,
          run_id=run_id,
          documents_dir=self.task.documents_dir,
          workspace_dir=workspace_dir,
          output_dir=output_dir,
          command_timeout=self.config.command_timeout,
        )
      )
      self.sandboxes.append(sandbox)
      return run_id, sandbox

    # TaskGroup waits for siblings to settle on failure/cancellation. Register
    # each returned sandbox immediately, so partial startup can be cleaned up.
    try:
      async with asyncio.TaskGroup() as group:
        starts = [group.create_task(start_sandbox()) for _ in range(self.group_size)]
      return self._build_envs([task.result() for task in starts], renderer, system_prompt, skills)
    except BaseException:
      await self.cleanup()
      raise

  def _build_envs(self, started, renderer, system_prompt, skills) -> Sequence[Env]:
    criteria_count = self.task.criteria_count

    # Rendering the prefix runs the full chat template over the system prompt,
    # skills, and tool schemas; every sandbox of a group normally shares them.
    prefixes: dict[str, list] = {}
    envs: list[Env] = []
    for run_id, sandbox in started:
      tool_definitions = sandbox.tool_definitions
      schema_key = json.dumps(tool_definitions, sort_keys=True)
      if schema_key not in prefixes:
        prefixes[schema_key] = initial_messages(self.task, renderer, system_prompt, tool_definitions)
      prefix_messages = prefixes[schema_key]
      tools = [
        LabTool(spec=dict(spec), sandbox=sandbox, tokenizer=renderer.tokenizer, max_result_tokens=self.config.max_tool_result_tokens)
        for spec in tool_definitions
      ]
      reward = LabRubricReward(
        lab_root=self.config.lab_root,
        run_id=run_id,
        task_name=self.task.name,
        judge_model=self.config.judge_model,
        judge_parallel=self.config.judge_parallel or (16 if "glm" in self.config.judge_model else 1),
        criteria_count=criteria_count,
        tool_metrics=sandbox.tool_metrics,
        collect_outputs=sandbox.collect_outputs,
        config={
          "model": self.config.model_name,
          "renderer": self.config.resolved_renderer,
          "max_turns": self.config.max_turns,
          "skills": skills,
        },
      )
      envs.append(
        LabEpisodeEnv(
          build_agent_tool_env(
            renderer=renderer,
            tools=tools,
            initial_messages=prefix_messages,
            reward_fn=reward,
            max_turns=self.config.max_turns,
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            max_generation_tokens=self.config.max_tokens,
            terminate_on_length=not self.config.continue_after_truncation,
          ),
          criteria_count,
        )
      )
    return envs

  async def cleanup(self) -> None:
    for sandbox in self.sandboxes:
      try:
        await sandbox.cleanup()
      except Exception as exc:
        logger.warning("LAB sandbox cleanup failed: %s", exc)
    self.sandboxes.clear()

  def logging_tags(self) -> list[str]:
    return ["harvey-labs"]


@chz.chz
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
  config: RunConfig
  sandbox_factory: SandboxFactory = podman_sandbox_factory

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    config = self.config
    if config.task:
      train_names, eval_names = [config.task], []
    else:
      train_names, eval_names = random_task_split(
        config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed, train_seed=config.train_split_seed
      )

    def dataset(names: list[str], group_size: int) -> LabDataset:
      return LabDataset(
        groups=[
          LabEnvGroupBuilder(task=task, config=config, group_size=group_size, sandbox_factory=self.sandbox_factory)
          for task in load_lab_tasks(config.lab_root, names)
        ],
        batch_size=config.batch_size,
      )

    if not train_names:
      raise ValueError("No LAB train tasks selected")
    return dataset(train_names, config.rollouts_per_example), dataset(eval_names, config.eval_rollouts_per_task) if eval_names else None
