"""Collect deliverables and use LAB's full-rubric pass fraction as reward."""

import asyncio
import json
import os
import subprocess
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook.renderers.base import Message, message_to_jsonable

# Bound grading across episodes to avoid bursts of judge API rate limits.
GRADING_CONCURRENCY = threading.Semaphore(int(os.getenv("OPEN_RL_GRADING_CONCURRENCY", "6")))


def preflight_grading(lab_root: Path) -> None:
  lab_python = lab_root / ".venv" / "bin" / "python"
  if not lab_python.exists():
    raise RuntimeError(f"LAB venv not found at {lab_python}. Run harvey_labs/setup_lab.sh before grading.")
  probe = subprocess.run(
    [str(lab_python), "-c", "from evaluation.judge import Judge; Judge._salvage_verdict"],
    cwd=lab_root,
    capture_output=True,
    text=True,
  )
  if probe.returncode != 0:
    raise RuntimeError(f"LAB grading preflight failed. Update the LAB checkout and run setup_lab.sh.\n{probe.stderr.strip()}")


@chz.chz
class LabRubricReward:
  lab_root: Path
  run_id: str
  task_name: str
  judge_model: str
  judge_parallel: int
  criteria_count: int
  tool_metrics: Callable[[], dict[str, Any]]
  collect_outputs: Callable[[Path], Awaitable[None]]
  config: dict[str, Any] = chz.field(default_factory=dict)
  timeout_seconds: int = 3600

  @property
  def run_dir(self) -> Path:
    return self.lab_root / "results" / self.run_id

  async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    try:
      await self.collect_outputs(self.run_dir / "output")
    except Exception as exc:
      # Raising here fails the whole rollout group, and a streamed step then
      # waits forever for it. Fail just this episode.
      print(f"[reward] could not collect outputs for {self.run_id}: {exc!r}")
      return 0.0, {**self.base_metrics(), "lab/reward_error": 1.0}
    return await asyncio.to_thread(self.score, history)

  def base_metrics(self) -> dict[str, float]:
    return {
      "lab/criteria_passed": 0.0,
      "lab/criteria_total": float(self.criteria_count),
      "lab/criteria_pass_fraction": 0.0,
      "lab/all_pass": 0.0,
      "lab/graded": 0.0,
      "lab/failed_before_grading": 0.0,
      "lab/no_output": 0.0,
      "lab/reward_error": 0.0,
    }

  def score(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    self.write_metadata(history)
    metrics = self.base_metrics()
    if not any(path.is_file() and path.stat().st_size > 0 for path in (self.run_dir / "output").rglob("*")):
      return 0.0, {**metrics, "lab/no_output": 1.0}

    cmd = [
      str(self.lab_root / ".venv" / "bin" / "python"),
      "-m",
      "evaluation.run_eval",
      "--run-id",
      self.run_id,
      "--task",
      self.task_name,
      "--judge-model",
      self.judge_model,
      "--parallel",
      str(self.judge_parallel),
    ]
    # Stream rubric progress to grading.log, including failures and timeouts.
    with GRADING_CONCURRENCY, (self.run_dir / "grading.log").open("w", encoding="utf-8") as log:
      try:
        subprocess.run(
          cmd,
          cwd=self.lab_root,
          stdout=log,
          stderr=subprocess.STDOUT,
          env={**os.environ, "PYTHONUNBUFFERED": "1"},
          timeout=self.timeout_seconds,
          check=True,
        )
      except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        log.write(f"\nGrading failed: {exc}\n")
        return 0.0, {**metrics, "lab/reward_error": 1.0}

    scores = json.loads((self.run_dir / "scores.json").read_text(encoding="utf-8"))
    reward, rubric_metrics = reward_from_scores(scores)
    return reward, {**metrics, **rubric_metrics, "lab/graded": 1.0}

  def write_metadata(self, history: list[Message]) -> None:
    self.run_dir.mkdir(parents=True, exist_ok=True)
    with (self.run_dir / "tinker_history.jsonl").open("w", encoding="utf-8") as f:
      for message in history:
        f.write(json.dumps(message_to_jsonable(message), sort_keys=True) + "\n")
    config = {"task": self.task_name, "run_id": self.run_id, "judge_model": self.judge_model, **self.config}
    (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (self.run_dir / "metrics.json").write_text(json.dumps(self.tool_metrics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reward_from_scores(scores: dict[str, Any]) -> tuple[float, dict[str, float]]:
  n_criteria = int(scores.get("n_criteria", 0) or 0)
  n_passed = int(scores.get("n_passed", 0) or 0)
  pass_fraction = n_passed / n_criteria if n_criteria else 0.0
  all_pass = float(bool(scores.get("all_pass")))
  return pass_fraction, {
    "lab/criteria_total": float(n_criteria),
    "lab/criteria_passed": float(n_passed),
    "lab/criteria_pass_fraction": pass_fraction,
    "lab/all_pass": all_pass,
  }
