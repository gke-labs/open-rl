"""Patch vLLM's LoRA module registration to avoid duplicate Gemma4 modules.

Gemma4's YOCO decoder split introduces shared module references. Older vLLM
builds register those duplicates separately with
`named_modules(remove_duplicate=False)`, which causes LoRA weights to be set on
one path and then immediately zeroed out on the duplicate path.

This helper makes the tiny local patch that switches back to the default
deduping `named_modules()` behavior. Run it after installing or upgrading vLLM
until the upstream fix lands everywhere we care about.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

BAD = "self.model.named_modules(remove_duplicate=False)"
GOOD = "self.model.named_modules()"


def find_model_manager(venv: str | None = None) -> Path:
  if venv:
    candidates = list(Path(venv).rglob("vllm/lora/model_manager.py"))
    if candidates:
      return candidates[0]
  spec = importlib.util.find_spec("vllm.lora.model_manager")
  if spec and spec.origin:
    return Path(spec.origin)
  raise FileNotFoundError("Cannot find vllm/lora/model_manager.py")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--check", action="store_true", help="Check only, do not patch")
  parser.add_argument("--venv", type=str, help="Optional venv path to patch explicitly")
  args = parser.parse_args()

  path = find_model_manager(args.venv)
  source = path.read_text()

  if BAD not in source:
    if GOOD in source:
      print(f"OK: {path} is already patched")
      return 0
    print(f"WARN: {path} has neither the buggy nor fixed pattern")
    return 1

  if args.check:
    print(f"NEEDS_PATCH: {path} has the buggy remove_duplicate=False pattern")
    return 2

  path.write_text(source.replace(BAD, GOOD, 1))
  print(f"PATCHED: {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
