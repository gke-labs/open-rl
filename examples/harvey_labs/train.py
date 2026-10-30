"""Train on Harvey LAB with live tool-use rollouts."""

import asyncio
import os
import shlex
from pathlib import Path

import chz
import tinker
from common.tinker_utils import force_rich_log_colors, resolve_base_url
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore

from .config import RunConfig
from .env import LabDatasetBuilder
from .plot_run import write_report
from .results import eval_result, format_eval, format_summary, read_results
from .reward import preflight_grading
from .sandbox import SandboxFactory, configured_sandbox_factory, podman_sandbox_factory


def build_train_config(config: RunConfig, sandbox_factory: SandboxFactory = podman_sandbox_factory) -> rl_train.Config:
  return rl_train.Config(
    learning_rate=config.learning_rate,
    lora_rank=config.lora_rank,
    dataset_builder=LabDatasetBuilder(config=config, sandbox_factory=sandbox_factory),
    model_name=config.model_name,
    recipe_name="harvey_labs",
    renderer_name=config.resolved_renderer,
    max_tokens=config.max_tokens,
    log_path=config.log_path,
    base_url=resolve_base_url(config.base_url),
    eval_every=config.eval_every,
    save_every=config.save_every,
    max_steps=config.max_steps,
    num_groups_to_log=config.log_groups,
    load_checkpoint_path=config.load_checkpoint_path,
    num_substeps=config.num_substeps,
    kl_penalty_coef=config.kl_penalty_coef,
    kl_discount_factor=config.kl_discount_factor,
    kl_reference_config=(rl_train.KLReferenceConfig(base_model=config.model_name) if config.kl_penalty_coef > 0 else None),
    stream_minibatch_config=(
      rl_train.StreamMinibatchConfig(
        groups_per_batch=config.batch_size,
        num_minibatches=config.batch_size // config.num_substeps,
      )
      if config.stream_minibatches
      else None
    ),
  )


async def evaluate(config: rl_train.Config, model_path: str | None = None, batch: int = 0) -> dict:
  """Use cookbook's evaluator and rollout exports for standalone and final eval."""
  _, test_dataset = await config.dataset_builder()
  if test_dataset is None:
    raise ValueError("No eval tasks configured; set eval_tasks > 0 and omit task=.")
  client = tinker.ServiceClient(base_url=config.base_url)
  sampling_client = (
    client.create_sampling_client(model_path=model_path) if model_path else client.create_sampling_client(base_model=config.model_name)
  )
  evaluator = RLTestSetEvaluator(test_dataset, max_tokens=config.max_tokens)
  store = TrainingRunStore(LocalStorage(Path(config.log_path)))
  return await rl_train.run_single_evaluation(evaluator, config, batch, sampling_client, "test", store=store)


async def run_final_eval(config: rl_train.Config) -> None:
  record = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
  if record is None:
    raise RuntimeError(f"No sampler checkpoint in {config.log_path}/checkpoints.jsonl; cannot run the final eval.")
  batch = record.batch if record.batch is not None else config.max_steps or 0
  metrics = await evaluate(config, record.sampler_path, batch)
  store = TrainingRunStore(LocalStorage(Path(config.log_path)))
  store.write_metrics({**metrics, "progress/batch": batch, "eval_phase": "final"}, step=batch)
  rl_train.logger.info(format_eval(eval_result(metrics), f"Final eval after {batch} batches"))


async def run(config: RunConfig, *, sandbox_factory: SandboxFactory = podman_sandbox_factory) -> None:
  if config.log_groups == 0:
    # The cookbook prints up to two trajectory groups per step regardless of
    # num_groups_to_log; a group is a whole multi-turn transcript.
    rl_train.print_group = lambda traj_group, tokenizer: None
  preflight_grading(config.lab_root)
  project = os.path.relpath(Path(__file__).resolve().parents[1])
  project_arg = "" if project == "." else f" --project {shlex.quote(project)}"
  print(f"Read this run: uv run{project_arg} harvey-results log_dir={shlex.quote(config.log_path)} json=True plot=True", flush=True)
  train_config = build_train_config(config, sandbox_factory)
  try:
    await rl_train.main(train_config)
    if config.final_eval and not config.task and config.eval_tasks > 0:
      await run_final_eval(train_config)
  finally:
    # Keep partial runs inspectable, and never hide a training failure behind
    # an error while creating its report.
    if (Path(config.log_path) / "metrics.jsonl").exists():
      try:
        print(format_summary(write_report(Path(config.log_path), read_results(Path(config.log_path)))))
        print(f"Saved {config.log_path}/results.json and {config.log_path}/run_plot.png")
      except Exception:
        rl_train.logger.exception("Could not generate the run report; read metrics with harvey-results")


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config, sandbox_factory=configured_sandbox_factory(config.sandbox_warmpool, config.sandbox_namespace)))


if __name__ == "__main__":
  main()
