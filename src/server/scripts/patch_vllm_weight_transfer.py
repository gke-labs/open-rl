"""Patch vLLM's weight transfer config and engine factory for Open-RL delta sync.

Upstream vLLM currently restricts WeightTransferConfig.backend to Literal["nccl", "ipc"]
and only registers the nccl and ipc weight transfer engines inside WeightTransferEngineFactory.

For Open-RL's sparse coordinate delta weight synchronization, we register our native
DeltaSnapshotWeightTransferEngine ("delta_snapshot") directly into vLLM's config and engine
factory so that both the parent Sampler/Trainer process and any child multiprocessing
EngineCore worker processes can cleanly instantiate and invoke the delta_snapshot engine.
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


def find_module_path(module_name: str, venv: str | None = None, subpath: str = "") -> Path:
  if venv:
    candidates = list(Path(venv).rglob(subpath))
    if candidates:
      return candidates[0]
  spec = importlib.util.find_spec(module_name)
  if spec and spec.origin:
    return Path(spec.origin)
  raise FileNotFoundError(f"Cannot find {module_name} ({subpath})")


def patch_config(venv: str | None = None, check_only: bool = False) -> int:
  path = find_module_path("vllm.config.weight_transfer", venv, "vllm/config/weight_transfer.py")
  source = path.read_text()

  if CONFIG_GOOD in source or "delta_snapshot" in source:
    print(f"OK: {path} is already patched")
    return 0

  if CONFIG_BAD not in source:
    print(f"WARN: {path} does not match expected WeightTransferConfig pattern")
    return 1

  if check_only:
    print(f"NEEDS_PATCH: {path} restricts backend to nccl/ipc")
    return 2

  path.write_text(source.replace(CONFIG_BAD, CONFIG_GOOD, 1))
  print(f"PATCHED: {path}")
  return 0


def patch_factory(venv: str | None = None, check_only: bool = False) -> int:
  path = find_module_path("vllm.distributed.weight_transfer.factory", venv, "vllm/distributed/weight_transfer/factory.py")
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


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--check", action="store_true", help="Check only, do not patch")
  parser.add_argument("--venv", type=str, help="Optional venv path to patch explicitly")
  args = parser.parse_args()

  ret1 = patch_config(args.venv, args.check)
  ret2 = patch_factory(args.venv, args.check)
  return max(ret1, ret2)


if __name__ == "__main__":
  sys.exit(main())
