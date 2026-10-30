"""Evaluate a base model or a tinker:// sampler checkpoint on the held-out tasks."""

import asyncio
import json

import chz

from .config import RunConfig
from .results import eval_result, format_eval
from .reward import preflight_grading
from .sandbox import SandboxFactory, configured_sandbox_factory, podman_sandbox_factory
from .train import build_train_config, evaluate


@chz.chz
class EvalConfig(RunConfig):
  checkpoint: str = ""  # Empty selects the untrained base model.
  log_path: str = "artifacts/harvey-labs/eval"


async def run(config: EvalConfig, *, sandbox_factory: SandboxFactory = podman_sandbox_factory) -> None:
  if config.checkpoint and not config.checkpoint.startswith("tinker://"):
    raise ValueError("Use the tinker:// sampler_path from checkpoints.jsonl for checkpoint=.")
  preflight_grading(config.lab_root)
  print(f"Evaluating {config.checkpoint or config.model_name}...")
  metrics = await evaluate(build_train_config(config, sandbox_factory), config.checkpoint or None)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  print(format_eval(eval_result(metrics)))


def main() -> None:
  config = chz.entrypoint(EvalConfig, allow_hyphens=True)
  asyncio.run(run(config, sandbox_factory=configured_sandbox_factory(config.sandbox_warmpool, config.sandbox_namespace)))


if __name__ == "__main__":
  main()
