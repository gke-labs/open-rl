"""Training curves from the recipe's own metrics files on the shared volume.

The cookbook writes one metrics.jsonl per run under its log_path, and every
gateway deployment mounts that volume. This reads those files back, bounded,
so reward and correctness are visible without a metrics service.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

# Metric keys worth charting, and the short names the UI shows.
SERIES = {
  "env/all/reward/total": "reward",
  "env/all/correct": "correct",
  "env/all/format": "format",
  "optim/entropy": "entropy",
  "optim/kl_sample_train_v1": "kl",
  "grad_norm:mean": "grad_norm",
  "optim/lr": "lr",
  "progress/done_frac": "done_frac",
}
CONFIG_KEYS = ("model_name", "lora_rank", "learning_rate", "max_steps", "recipe_name", "renderer_name", "temperature", "max_tokens", "loss_fn")
MAX_RUNS = 200
MAX_ROWS = 5000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_DEPTH = 4
FRESH_FOR = 15.0

_cache: dict[str, Any] = {"at": 0.0, "data": None}


def runs_root() -> Path:
  return Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")) / "runs"


def read_run(directory: Path, root: Path) -> dict[str, Any] | None:
  metrics_path = directory / "metrics.jsonl"
  try:
    size = metrics_path.stat().st_size
  except OSError:
    return None
  config: dict[str, Any] = {}
  try:
    with open(directory / "config.json") as f:
      raw = json.load(f)
    config = {key: raw.get(key) for key in CONFIG_KEYS if raw.get(key) is not None}
  except (OSError, ValueError):
    pass
  series: dict[str, list[list[float]]] = {short: [] for short in SERIES.values()}
  last: dict[str, float] = {}
  rows = 0
  step = 0
  truncated = size > MAX_FILE_BYTES
  try:
    with open(metrics_path) as f:
      for line in f:
        if rows >= MAX_ROWS:
          truncated = True
          break
        try:
          row = json.loads(line)
        except ValueError:
          continue
        rows += 1
        step = int(row.get("step", rows - 1))
        for key, short in SERIES.items():
          value = row.get(key)
          if isinstance(value, (int, float)) and value == value:
            series[short].append([step, float(value)])
            last[short] = float(value)
  except OSError:
    return None
  relative = directory.relative_to(root)
  return {
    "path": str(relative),
    "sweep": relative.parts[0] if len(relative.parts) > 1 else "",
    "name": relative.parts[-1],
    "config": config,
    "rows": rows,
    "step": step,
    "updated_at": metrics_path.stat().st_mtime,
    "truncated": truncated,
    "series": {short: points for short, points in series.items() if points},
    "last": last,
  }


def scan(root: Path) -> list[dict[str, Any]]:
  found: list[dict[str, Any]] = []
  if not root.is_dir():
    return found
  base_depth = len(root.parts)
  for current, dirs, files in os.walk(root):
    depth = len(Path(current).parts) - base_depth
    if depth >= MAX_DEPTH:
      dirs[:] = []
    dirs[:] = sorted(d for d in dirs if not d.startswith(("iteration_", ".")))
    if "metrics.jsonl" not in files:
      continue
    run = read_run(Path(current), root)
    if run is not None:
      found.append(run)
    if len(found) >= MAX_RUNS:
      break
  found.sort(key=lambda r: r["updated_at"], reverse=True)
  return found


async def experiments() -> dict[str, Any]:
  now = time.monotonic()
  if _cache["data"] is not None and now - _cache["at"] < FRESH_FOR:
    return _cache["data"]
  root = runs_root()
  try:
    runs = await asyncio.to_thread(scan, root)
    data = {"available": True, "root": str(root), "runs": runs, "scanned_at": time.time()}
  except Exception as exc:
    data = {"available": False, "root": str(root), "runs": [], "error": f"Cannot read run metrics: {type(exc).__name__}"}
  _cache["at"], _cache["data"] = now, data
  return data
