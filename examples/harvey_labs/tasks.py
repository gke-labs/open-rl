"""LAB task selection and loading."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import chz


@chz.chz
class LabTask:
  name: str
  instructions: str
  documents_dir: Path
  criteria_count: int


def task_slug(task_name: str) -> str:
  return task_name.replace("/", "__")


def load_task(lab_root: Path, task_name: str) -> LabTask:
  task_parts = task_name.split("/")
  if not task_name or any(part in {"", ".", ".."} for part in task_parts):
    raise ValueError(f"Invalid LAB task name: {task_name!r}")

  tasks_root = (lab_root / "tasks").resolve()
  task_dir = (tasks_root / Path(*task_parts)).resolve()
  if not task_dir.is_relative_to(tasks_root):
    raise ValueError(f"LAB task directory escapes tasks root: {task_dir}")

  config_path = task_dir / "task.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"LAB task config not found: {config_path}")
  config = json.loads(config_path.read_text(encoding="utf-8"))
  instructions = config.get("instructions")
  if not instructions:
    instructions = (task_dir / "instructions.md").read_text(encoding="utf-8")
  documents_dir = (task_dir / "documents").resolve()
  if not documents_dir.is_relative_to(task_dir):
    raise ValueError(f"LAB documents directory escapes task directory: {documents_dir}")
  if not documents_dir.is_dir():
    raise FileNotFoundError(f"LAB documents directory not found: {documents_dir}")
  if not any(path.is_file() for path in documents_dir.rglob("*")):
    raise ValueError(f"LAB documents directory is empty: {documents_dir}")
  return LabTask(
    name=task_name,
    instructions=instructions,
    documents_dir=documents_dir,
    criteria_count=len(config.get("criteria") or []),
  )


def load_lab_tasks(lab_root: Path, task_names: list[str]) -> list[LabTask]:
  lab_root = lab_root.resolve()
  return [load_task(lab_root, task_name) for task_name in task_names]


def discover_lab_tasks(lab_root: Path) -> tuple[list[str], int]:
  """Every runnable task under <lab_root>/tasks, sorted, plus the skipped count.

  A task dir is any directory holding a task.json. Runnable means load_task
  would succeed and the task has grading criteria: instructions present,
  criteria non-empty, and a non-empty documents dir.
  """
  tasks_root = (lab_root / "tasks").resolve()
  if not tasks_root.is_dir():
    raise FileNotFoundError(f"LAB tasks root not found: {tasks_root}")
  names: list[str] = []
  skipped = 0
  for config_path in sorted(tasks_root.rglob("task.json")):
    name = config_path.parent.relative_to(tasks_root).as_posix()
    try:
      task = load_task(lab_root, name)
      if not task.criteria_count:
        raise ValueError("Task has no grading criteria")
    except (OSError, ValueError):
      skipped += 1
      continue
    names.append(name)
  return names, skipped


# The held-out eval slice is taken after the first EVAL_SLICE_OFFSET names of the
# seeded shuffle. Freezing the offset (rather than keying it on the train count)
# keeps the benchmark byte-identical when the train pool grows or is redrawn.
# Runs 19-47 took this slice at seed 0, so keep both pinned across compared runs.
EVAL_SLICE_OFFSET = 300


def task_family(name: str) -> str:
  return re.sub(r"/scenario-\d+$", "", name)


def random_task_split(
  lab_root: Path,
  num_train: int,
  num_eval: int,
  seed: int,
  eval_slice_offset: int = EVAL_SLICE_OFFSET,
  train_seed: int | None = None,
) -> tuple[list[str], list[str]]:
  """Seeded train/eval split whose eval set depends only on the seed.

  Eval is shuffle(seed)[eval_slice_offset : eval_slice_offset + num_eval], so it
  never moves with num_train. Train draws the family-disjoint remainder, so a
  scenario sibling of an eval task never leaks into training. A train_seed
  redraws that remainder in its own order, as the reference runs did.
  """
  names, skipped = discover_lab_tasks(lab_root)
  shuffled = list(names)
  random.Random(seed).shuffle(shuffled)
  if eval_slice_offset + num_eval > len(shuffled):
    raise ValueError(
      f"Requested eval slice [{eval_slice_offset}:{eval_slice_offset + num_eval}] but only "
      f"{len(shuffled)} runnable tasks exist under {lab_root / 'tasks'}"
    )
  eval_names = shuffled[eval_slice_offset : eval_slice_offset + num_eval]
  eval_families = {task_family(name) for name in eval_names}
  train_pool = [name for name in shuffled if task_family(name) not in eval_families]
  if train_seed is not None:
    train_pool = [name for name in names if task_family(name) not in eval_families]
    random.Random(train_seed).shuffle(train_pool)
  if num_train > len(train_pool):
    raise ValueError(f"Requested {num_train} train tasks but only {len(train_pool)} sit outside eval's {len(eval_families)} scenario families")
  train_names = train_pool[:num_train]
  print(
    f"[tasks] split seed={seed} train_seed={train_seed}: {len(train_names)} train / {num_eval} eval from {len(names)} "
    f"runnable tasks, eval slice [{eval_slice_offset}:{eval_slice_offset + num_eval}] "
    f"({skipped} skipped as broken)"
  )
  return train_names, eval_names
