"""Patch vLLM's weight transfer config, engine factory, and GPU worker for Open-RL delta sync.

Upstream vLLM currently restricts WeightTransferConfig.backend to Literal["nccl", "ipc"]
and dispatches update_weights to WeightTransferEngine requiring start_weight_update.
This script patches vllm/v1/worker/gpu_worker.py to support Open-RL's DeltaSnapshotWeightTransferEngine.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CONFIG_BAD = 'backend: Literal["nccl", "ipc"] = "nccl"'
CONFIG_GOOD = 'backend: Literal["nccl", "ipc", "delta_snapshot"] = "nccl"'

FACTORY_ADD = """
WeightTransferEngineFactory.register_engine(
    "delta_snapshot",
    "server.delta_weight_transfer_engine",
    "DeltaSnapshotWeightTransferEngine",
)
"""

OPEN_RL_UPDATE_WEIGHTS_MARKER = "# Open-RL delta_snapshot update_weights patch"

GPU_WORKER_UPDATE_WEIGHTS = """    def update_weights(self, update_info: dict = None, **kwargs) -> None:
        # Open-RL delta_snapshot update_weights patch
        if isinstance(update_info, str):
            update_info = {"target_weights_path": update_info}
        elif update_info is None:
            update_info = kwargs

        if hasattr(self, "weight_transfer_engine") and self.weight_transfer_engine is not None and getattr(self, "_weight_update_active", False):
            self.weight_transfer_engine.update_weights(update_info)
            return

        if not hasattr(self, "_delta_transfer_engine"):
            from server.delta_weight_transfer_engine import DeltaSnapshotWeightTransferEngine
            self._delta_transfer_engine = DeltaSnapshotWeightTransferEngine()

        if isinstance(update_info, dict):
            target_weights_path = update_info.get("target_weights_path", "")
        else:
            target_weights_path = getattr(update_info, "target_weights_path", "")

        model = None
        if hasattr(self, "get_model"):
            model = self.get_model()
        else:
            runner = getattr(self, "model_runner", self)
            if hasattr(runner, "get_model"):
                model = runner.get_model()

        def load_weights_fn(weights):
            if model is not None:
                import torch
                named_params = dict(model.named_parameters())
                with torch.no_grad():
                    for name, tensor in weights:
                        if name in named_params:
                            named_params[name].copy_(tensor)

        load_weights_fn.__self__ = model

        self._delta_transfer_engine.receive_weights(
            target_path=target_weights_path,
            load_weights=load_weights_fn,
            update_info=update_info,
        )"""


def find_module_path(module_name: str, venv: str | None = None, subpath: str = "") -> Path:
  if venv:
    candidates = list(Path(venv).rglob(subpath))
    if candidates:
      return candidates[0]
  spec = importlib.util.find_spec(module_name)
  if spec and spec.origin:
    return Path(spec.origin)
  default_path = Path(f"/app/.venv/lib/python3.12/site-packages/{subpath}")
  if default_path.exists():
    return default_path
  raise FileNotFoundError(f"Cannot find {module_name} ({subpath})")


def patch_config(venv: str | None = None, check_only: bool = False) -> int:
  try:
    path = find_module_path("vllm.config.weight_transfer", venv, "vllm/config/weight_transfer.py")
  except FileNotFoundError:
    print("WARN: vllm/config/weight_transfer.py not found (bypassing patch)")
    return 0

  source = path.read_text()

  if CONFIG_GOOD in source or "delta_snapshot" in source:
    print(f"OK: {path} is already patched")
    return 0

  if CONFIG_BAD not in source:
    print(f"WARN: {path} does not match expected WeightTransferConfig pattern (bypassing config patch)")
    return 0

  if check_only:
    print(f"NEEDS_PATCH: {path} restricts backend to nccl/ipc")
    return 2

  path.write_text(source.replace(CONFIG_BAD, CONFIG_GOOD, 1))
  print(f"PATCHED: {path}")
  return 0


def patch_factory(venv: str | None = None, check_only: bool = False) -> int:
  try:
    path = find_module_path("vllm.distributed.weight_transfer.factory", venv, "vllm/distributed/weight_transfer/factory.py")
  except FileNotFoundError:
    print("WARN: vllm/distributed/weight_transfer/factory.py not found (bypassing patch)")
    return 0

  source = path.read_text()

  if "delta_snapshot" in source:
    print(f"OK: {path} is already patched")
    return 0

  if check_only:
    print(f"NEEDS_PATCH: {path} missing delta_snapshot registration")
    return 2

  path.write_text(source + FACTORY_ADD)
  print(f"PATCHED: {path}")
  return 0


def patch_gpu_worker(venv: str | None = None, check_only: bool = False) -> int:
  try:
    path = find_module_path("vllm.v1.worker.gpu_worker", venv, "vllm/v1/worker/gpu_worker.py")
  except FileNotFoundError:
    print("WARN: vllm/v1/worker/gpu_worker.py not found (bypassing patch)")
    return 0

  source = path.read_text()

  if OPEN_RL_UPDATE_WEIGHTS_MARKER in source:
    print(f"OK: {path} is already patched with Open-RL update_weights")
    return 0

  target_def = "    def update_weights(self, update_info: dict) -> None:"
  if target_def not in source:
    print(f"WARN: {path} missing target update_weights signature")
    return 1

  if check_only:
    print(f"NEEDS_PATCH: {path} missing Open-RL update_weights RPC handler")
    return 2

  start_idx = source.find(target_def)
  end_idx = source.find("    def finish_weight_update", start_idx)
  if start_idx != -1 and end_idx != -1:
    new_source = source[:start_idx] + GPU_WORKER_UPDATE_WEIGHTS + "\n\n" + source[end_idx:]
  else:
    new_source = source.replace(target_def, GPU_WORKER_UPDATE_WEIGHTS, 1)

  path.write_text(new_source)
  print(f"PATCHED: {path}")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--check", action="store_true", help="Check only, do not patch")
  parser.add_argument("--venv", type=str, help="Optional venv path to patch explicitly")
  args = parser.parse_args()

  ret1 = patch_config(args.venv, args.check)
  ret2 = patch_factory(args.venv, args.check)
  ret3 = patch_gpu_worker(args.venv, args.check)
  return max(ret1, ret2, ret3)


if __name__ == "__main__":
  sys.exit(main())
