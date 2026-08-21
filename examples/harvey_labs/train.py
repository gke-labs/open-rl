"""Train on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
from pathlib import Path

import chz
import tinker
from env import LabDatasetBuilder
from tasks import BOOTSTRAP_TASKS, EVAL_TASKS, random_task_split
from tinker.lib.public_interfaces import training_client as tinker_training_client
from tinker_cookbook import checkpoint_utils

# 5MB chunks put one long-context datum per request, collapsing DP sharding
# to rank 0; ~30MB keeps real shards without stalling the HTTP/JSON path.
tinker_training_client.MAX_CHUNK_BYTES_COUNT = 30_000_000

from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore
from tinker_utils import force_rich_log_colors, resolve_base_url

MODEL_NAME = "google/gemma-4-E4B-it"
COMMAND_TIMEOUT = 60


def default_judge_parallel(judge_model: str) -> int:
  # Self-hosted GLM absorbs concurrent grading; Gemini rate limits force 1.
  return 16 if "glm" in judge_model else 1


NUM_GROUPS_TO_LOG = 1


def print_group_summary(traj_group, tokenizer) -> None:
  rewards = traj_group.get_total_rewards()
  buf = io.StringIO()
  buf.write("====== Trajectory Group ======\n")
  for idx, traj in enumerate(traj_group.trajectories_G):
    metrics = traj_group.metrics_G[idx] or {}
    ac_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
    last_ac = len(traj.transitions[-1].ac.tokens) if traj.transitions else 0
    extras = "".join(
      f" {key.rsplit('/', 1)[-1]}={metrics[key]:.3g}"
      for key in ("lab/criteria_pass_fraction", "lab/document_coverage", "lab/reward_error")
      if isinstance(metrics.get(key), (int, float))
    )
    buf.write(f"  rollout {idx}: reward={rewards[idx]:.3f} turns={len(traj.transitions)} ac_tokens={ac_tokens} last_ac={last_ac}{extras}\n")
  buf.write("====== End Trajectory Group ======")
  rl_train.logger.info(buf.getvalue())


def print_group_responses(traj_group, tokenizer) -> None:
  rewards = traj_group.get_total_rewards()
  buf = io.StringIO()
  buf.write("\n====== Trajectory Group (model responses only) ======\n")
  for idx, traj in enumerate(traj_group.trajectories_G):
    buf.write(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******\n")
    for key, value in (traj_group.metrics_G[idx] or {}).items():
      buf.write(f"  {key}: {value}\n")
    for turn, transition in enumerate(traj.transitions):
      buf.write(f"---- turn {turn} | ob_len={transition.ob.length} ac_len={len(transition.ac.tokens)} reward={transition.reward:.3f} ----\n")
      buf.write(tokenizer.decode(transition.ac.tokens).rstrip() + "\n")
  buf.write("====== End Trajectory Group ======")
  rl_train.logger.info(buf.getvalue())


_print_group_responses = print_group_responses


@chz.chz
class RunConfig:
  """Small set of knobs for the LAB RL experiment."""

  base_url: str | None = None
  model_name: str = MODEL_NAME
  renderer_name: str | None = None
  learning_rate: float = 3e-6
  lora_rank: int = 32
  lab_root: Path = Path(__file__).resolve().parent / "harvey-labs"
  # Single-task override; otherwise task_set picks the pools ("random" =
  # seeded split, "bootstrap" = curated lists).
  task: str | None = None
  task_set: str = "random"
  train_tasks: int = 300
  eval_tasks: int = 50
  task_split_seed: int = 0
  eval_every: int = 20
  batch_size: int = 1
  rollouts_per_example: int = 4
  max_steps: int = 40
  max_turns: int = 40
  max_tokens: int = 3072
  max_trajectory_tokens: int = 128 * 1024
  max_tool_result_tokens: int = 8 * 1024
  judge_model: str = "gemini-3.5-flash"
  # Criteria graded concurrently within one episode. 0 = auto by judge model.
  judge_parallel: int = 0
  max_reward_criteria: int | None = None
  # Full-state checkpoint cadence (weights + optimizer, resumable). 0 = off.
  save_every: int = 5
  # Overlap training with sampling: forward_backward per finished trajectory
  # group. Gradient math unchanged at num_substeps=1.
  stream_minibatches: bool = False
  num_substeps: int = 1
  # Warm-start from a sampler snapshot (fresh optimizer, batch counter
  # restarts at 0).
  load_checkpoint_path: str | None = None
  log_path: str = "artifacts/harvey-labs"
  log_full_rollouts: bool = False
  # In-loop evals measure the model before each optim step; this one runs
  # after training, on the final checkpoint.
  final_eval: bool = True


def resolve_renderer_name(config: RunConfig) -> str:
  if config.renderer_name:
    return config.renderer_name
  name = config.model_name.lower()
  if "qwen" in name:
    return "qwen3_5"
  if "gemma" in name:
    return "gemma4"
  raise ValueError(f"Cannot infer a renderer for model {config.model_name!r}; pass renderer_name explicitly.")


def preflight_grading(config: RunConfig) -> None:
  """Fail before step 0 on grading-environment rot (missing LAB venv, stale
  judge) instead of silently losing gradings mid-run."""
  lab_python = config.lab_root / ".venv" / "bin" / "python"
  if not lab_python.exists():
    raise RuntimeError(
      f"LAB venv not found at {lab_python}. Run setup_lab.sh so grading uses the "
      "LAB environment; without it every reward silently falls back to the recipe venv."
    )
  probe = subprocess.run(
    [str(lab_python), "-c", "from evaluation.judge import Judge; Judge._salvage_verdict"],
    cwd=str(config.lab_root),
    capture_output=True,
    text=True,
  )
  if probe.returncode != 0:
    raise RuntimeError(
      "LAB grading preflight failed — every episode would score 0 as reward_error. "
      "Missing deps mean setup_lab.sh didn't finish; a missing Judge._salvage_verdict "
      "means the LAB checkout predates the judge fix (git pull in the LAB checkout).\n"
      f"{probe.stderr.strip()}"
    )


def build_dataset_builder(config: RunConfig) -> LabDatasetBuilder:
  if config.task:
    train_names, eval_names = [config.task], []
  elif config.task_set == "bootstrap":
    train_names, eval_names = list(BOOTSTRAP_TASKS), list(EVAL_TASKS)
  elif config.task_set == "random":
    train_names, eval_names = random_task_split(config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed)
  else:
    raise ValueError(f"Unknown task_set {config.task_set!r} (use 'random' or 'bootstrap').")
  return LabDatasetBuilder(
    lab_root=config.lab_root,
    task_names=train_names,
    eval_task_names=eval_names,
    train_limit=None,
    eval_limit=len(eval_names) or None,
    batch_size=config.batch_size,
    group_size=config.rollouts_per_example,
    model_name=config.model_name,
    renderer_name=resolve_renderer_name(config),
    max_turns=config.max_turns,
    command_timeout=COMMAND_TIMEOUT,
    judge_model=config.judge_model,
    judge_parallel=config.judge_parallel or default_judge_parallel(config.judge_model),
    max_reward_criteria=config.max_reward_criteria,
    max_trajectory_tokens=config.max_trajectory_tokens,
    max_generation_tokens=config.max_tokens,
    max_tool_result_tokens=config.max_tool_result_tokens,
  )


async def run_final_eval(train_config: rl_train.Config) -> None:
  record = checkpoint_utils.get_last_checkpoint(train_config.log_path, required_key="sampler_path")
  if record is None:
    raise RuntimeError(f"No sampler checkpoint in {train_config.log_path}/checkpoints.jsonl; cannot run the final eval.")
  _, test_dataset = await train_config.dataset_builder()
  if test_dataset is None:
    return
  batch = record.batch if record.batch is not None else train_config.max_steps or 0
  service_client = tinker.ServiceClient(base_url=train_config.base_url)
  sampling_client = service_client.create_sampling_client(model_path=record.sampler_path)
  evaluator = RLTestSetEvaluator(test_dataset, max_tokens=train_config.max_tokens)
  store = TrainingRunStore(LocalStorage(Path(train_config.log_path)))
  metrics = await rl_train.run_single_evaluation(evaluator, train_config, batch, sampling_client, "test", store=store)
  with open(Path(train_config.log_path) / "metrics.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"progress/batch": batch, **metrics}) + "\n")
  passed = metrics.get("test/env/harvey-labs/lab/criteria_passed")
  total = metrics.get("test/env/harvey-labs/lab/criteria_total")
  episodes = metrics.get("test/env/harvey-labs/total_episodes")
  if passed is not None and total and episodes:
    rl_train.logger.info(f"Final eval after {batch} steps: pooled criteria {passed * episodes:.0f}/{total * episodes:.0f} ({passed / total:.1%})")


async def run(config: RunConfig) -> None:
  preflight_grading(config)
  rl_train.print_group = print_group_responses if config.log_full_rollouts else print_group_summary
  train_config = rl_train.Config(
    learning_rate=config.learning_rate,
    lora_rank=config.lora_rank,
    dataset_builder=build_dataset_builder(config),
    model_name=config.model_name,
    recipe_name="harvey_labs",
    renderer_name=resolve_renderer_name(config),
    max_tokens=config.max_tokens,
    log_path=config.log_path,
    base_url=resolve_base_url(config.base_url),
    eval_every=config.eval_every,
    save_every=config.save_every,
    max_steps=config.max_steps,
    num_groups_to_log=NUM_GROUPS_TO_LOG,
    load_checkpoint_path=config.load_checkpoint_path,
    num_substeps=config.num_substeps,
    stream_minibatch_config=(
      rl_train.StreamMinibatchConfig(
        groups_per_batch=config.batch_size,
        num_minibatches=config.batch_size // config.num_substeps,
      )
      if config.stream_minibatches
      else None
    ),
  )
  await rl_train.main(train_config)
  if config.final_eval:
    await run_final_eval(train_config)


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
