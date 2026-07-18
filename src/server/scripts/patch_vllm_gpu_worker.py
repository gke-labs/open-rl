"""Patch vLLM 0.25.1 GPU worker memory profiling assertions for Open-RL time-slicing.

In multi-tenant time-slicing environments (accel_timeslicer), GPU memory snapshots
fluctuate as the PyTorch trainer yields and re-acquires GPU contexts. This script
replaces strict runtime memory assertions in vLLM's v1 gpu_worker.py with safe
upper-bound adjustments to prevent crashes during sleep/wake cycles.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def find_gpu_worker(venv: str | None = None) -> Path:
  if venv:
    candidates = list(Path(venv).rglob("vllm/v1/worker/gpu_worker.py"))
    if candidates:
      return candidates[0]
  spec = importlib.util.find_spec("vllm.v1.worker.gpu_worker")
  if spec and spec.origin:
    return Path(spec.origin)
  # Fallback to standard venv location if spec discovery fails before vLLM import
  default_path = Path("/app/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py")
  if default_path.exists():
    return default_path
  raise FileNotFoundError("Cannot find vllm/v1/worker/gpu_worker.py")


def patch_gpu_worker(venv: str | None = None, check_only: bool = False) -> int:
  try:
    path = find_gpu_worker(venv)
  except FileNotFoundError:
    print("WARN: vllm/v1/worker/gpu_worker.py not found (bypassing patch)")
    return 0

  source = path.read_text()

  target_assert = "assert self.init_snapshot.free_memory >= free_gpu_memory"
  safe_replacement = "if free_gpu_memory > self.init_snapshot.free_memory:\n            self.init_snapshot.free_memory = free_gpu_memory"

  if safe_replacement in source or "if free_gpu_memory > self.init_snapshot.free_memory:" in source:
    print(f"OK: {path} is already patched")
    return 0

  if target_assert not in source:
    print(f"WARN: {path} does not contain target memory profiling assertion (upstream may have updated v1 worker)")
    return 0

  if check_only:
    print(f"NEEDS_PATCH: {path} contains strict GPU memory profiling assertion")
    return 2

  new_source = source.replace(target_assert, safe_replacement, 1)
  path.write_text(new_source)
  print(f"PATCHED: {path}")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--check", action="store_true", help="Check only, do not patch")
  parser.add_argument("--venv", type=str, help="Optional venv path to patch explicitly")
  args = parser.parse_args()

  return patch_gpu_worker(args.venv, args.check)


if __name__ == "__main__":
  sys.exit(main())
