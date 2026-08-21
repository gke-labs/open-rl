"""Evaluate a trained adapter checkpoint from its on-disk directory.

Points the gateway's sampler at an existing adapter dir (e.g.
/tmp/open-rl/peft/<model-id>/sampler-12 or any copied checkpoint folder) and
runs the held-out eval task set against it:

  uv --project examples run python examples/harvey_labs/eval_checkpoint.py \
    checkpoint=/tmp/open-rl/peft/<model-id>/final \
    model_name=Qwen/Qwen3.5-27B base_url=http://127.0.0.1:9003

A directory outside the gateway's peft tree needs model_id= of a live model
with the same base; the script symlinks the checkpoint into that model's
adapter dir so the sampler can resolve it.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import chz
import tinker
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_utils import resolve_base_url
from train import RunConfig, build_dataset_builder, preflight_grading

TMP_DIR = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"))


@chz.chz
class EvalConfig(RunConfig):
  checkpoint: str = ""
  # Required only when `checkpoint` lives outside <TMP_DIR>/peft/.
  model_id: str | None = None
  label: str = "eval-checkpoint"


def sampler_ref(config: EvalConfig) -> str:
  path = Path(config.checkpoint).expanduser().resolve()
  if not path.is_dir():
    raise RuntimeError(f"Checkpoint dir does not exist: {path}")
  if not (path / "adapter_config.json").exists():
    raise RuntimeError(f"{path} does not look like an adapter checkpoint (no adapter_config.json). FFT full checkpoints are not supported here.")

  peft_root = (TMP_DIR / "peft").resolve()
  if path.parent.parent == peft_root:
    model_id, label = path.parent.name, path.name
    return f"tinker://{model_id}/sampler_weights/{label}"

  if not config.model_id:
    raise RuntimeError(f"{path} is outside {peft_root}; pass model_id=<live model id> so the checkpoint can be linked into that model's adapter dir.")
  link = peft_root / config.model_id / config.label
  link.parent.mkdir(parents=True, exist_ok=True)
  if link.resolve() != path:
    if link.is_symlink() or link.exists():
      raise RuntimeError(f"{link} already exists and points elsewhere; remove it or pass a different label=.")
    os.symlink(path, link)
    print(f"Linked {link} -> {path}")
  return f"tinker://{config.model_id}/sampler_weights/{config.label}"


async def run(config: EvalConfig) -> None:
  preflight_grading(config)
  ref = sampler_ref(config)
  _, test_dataset = await build_dataset_builder(config)()
  if test_dataset is None:
    raise RuntimeError("No eval tasks configured (eval_tasks=0, or a single-task run via task=?).")

  print(f"Evaluating {ref} on {len(test_dataset)} eval batches...")
  service_client = tinker.ServiceClient(base_url=resolve_base_url(config.base_url))
  sampling_client = service_client.create_sampling_client(base_model=config.model_name, model_path=ref)
  evaluator = RLTestSetEvaluator(test_dataset, max_tokens=config.max_tokens)
  metrics = await evaluator(sampling_client)

  print(json.dumps(metrics, indent=2, sort_keys=True))
  passed = metrics.get("test/env/harvey-labs/lab/criteria_passed")
  total = metrics.get("test/env/harvey-labs/lab/criteria_total")
  episodes = metrics.get("test/env/harvey-labs/total_episodes")
  if passed is not None and total and episodes:
    print(f"Pooled criteria: {passed * episodes:.0f}/{total * episodes:.0f} ({passed / total:.1%}) over {episodes:.0f} episodes")


def main() -> None:
  config = chz.entrypoint(EvalConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
