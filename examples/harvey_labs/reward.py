"""LAB rubric reward for terminal tool-use episodes."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinker_cookbook.renderers.base import Message, message_to_jsonable

ARTIFACT_EXTENSIONS = ("docx", "xlsx", "pptx", "pdf", "md", "txt")

# Bound cross-episode grading so end-of-batch bursts don't rate-limit the
# judge into failures.
GRADING_CONCURRENCY = threading.Semaphore(int(os.getenv("OPEN_RL_GRADING_CONCURRENCY", "6")))
_EXPECTED_EXTENSION_RE = re.compile(rf"\.({'|'.join(ARTIFACT_EXTENSIONS)})\b", re.IGNORECASE)


@dataclass
class LabRubricReward:
  lab_root: Path
  run_id: str
  task_name: str
  judge_model: str
  task_instructions: str
  judge_parallel: int
  max_criteria: int | None
  criteria_count: int
  tool_metrics: Callable[[], dict[str, Any]]
  config: dict[str, Any] = field(default_factory=dict)
  timeout_seconds: int = 3600
  # Reward is pure rubric score by default; process metrics are still computed
  # and logged, and the no-output gate still skips pointless judge calls.
  process_reward_weight: float = 0.0

  @property
  def run_dir(self) -> Path:
    return self.lab_root / "results" / self.run_id

  async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    return await asyncio.to_thread(self.score, history)

  def failure_metrics(self) -> dict[str, float]:
    return {
      "lab/criteria_passed": 0.0,
      "lab/criteria_total": float(self.criteria_count),
      "lab/criteria_pass_fraction": 0.0,
      "lab/graded": 0.0,
      "lab/failed_before_grading": 0.0,
    }

  def score(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    self.write_metadata(history)
    process_reward, process_metrics = self.process_reward()
    if not process_metrics["lab/has_output"]:
      return self.combine_rewards(0.0, process_reward), {
        **process_metrics,
        **self.failure_metrics(),
        "lab/no_output": 1.0,
        "lab/reward_error": 0.0,
      }

    cmd = [
      str(self.lab_python()),
      str(Path(__file__).with_name("score_lab_run.py")),
      "--lab-root",
      str(self.lab_root),
      "--run-id",
      self.run_id,
      "--task",
      self.task_name,
      "--judge-model",
      self.judge_model,
      "--parallel",
      str(self.judge_parallel),
    ]
    if self.max_criteria is not None:
      cmd += ["--max-criteria", str(self.max_criteria)]

    # Stream judge output to grading.log so progress is visible live.
    grading_log = self.run_dir / "grading.log"
    with GRADING_CONCURRENCY:
      return self.run_grading(cmd, grading_log, process_reward, process_metrics)

  def run_grading(self, cmd, grading_log, process_reward, process_metrics):
    try:
      with open(grading_log, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(
          cmd,
          cwd=str(self.lab_root),
          stdout=log,
          stderr=subprocess.STDOUT,
          env={**os.environ, "PYTHONUNBUFFERED": "1"},
          timeout=self.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
      # Contain timeouts like any other grading failure.
      (self.run_dir / "reward_error.log").write_text(
        f"TIMEOUT after {self.timeout_seconds}s\n\n" + grading_log.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
      )
      return self.combine_rewards(0.0, process_reward), {
        **process_metrics,
        **self.failure_metrics(),
        "lab/reward_error": 1.0,
      }
    if result.returncode != 0:
      (self.run_dir / "reward_error.log").write_text(
        grading_log.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
      )
      return self.combine_rewards(0.0, process_reward), {
        **process_metrics,
        **self.failure_metrics(),
        "lab/reward_error": 1.0,
      }

    scores = json.loads((self.run_dir / "scores.json").read_text(encoding="utf-8"))
    rubric_reward, rubric_metrics = reward_from_scores(scores)
    return self.combine_rewards(rubric_reward, process_reward), {
      **process_metrics,
      **rubric_metrics,
      "lab/rubric_reward": rubric_reward,
      "lab/graded": 1.0,
      "lab/reward_error": 0.0,
      "lab/failed_before_grading": 0.0,
    }

  def combine_rewards(self, rubric_reward: float, process_reward: float) -> float:
    weight = self.process_reward_weight
    return (1.0 - weight) * rubric_reward + weight * process_reward

  def process_reward(self) -> tuple[float, dict[str, float]]:
    """Give bounded credit for grounded progress without rewarding tool loops."""
    # Reuse LAB's read tracking so reward and metrics share one definition.
    metrics = self.tool_metrics()
    total_documents = int(metrics.get("total_documents") or 0)
    coverage = int(metrics.get("documents_read") or 0) / total_documents if total_documents else 0.0

    output_files = [path for path in (self.run_dir / "output").rglob("*") if path.is_file() and path.stat().st_size > 0]
    expected_extensions = {f".{extension.lower()}" for extension in _EXPECTED_EXTENSION_RE.findall(self.task_instructions)}
    has_output = bool(output_files)
    has_valid_expected_output = any(path.suffix.lower() in expected_extensions and valid_artifact(path) for path in output_files)
    process_reward = 0.5 * coverage + 0.25 * float(has_output) + 0.25 * float(has_valid_expected_output)
    return process_reward, {
      "lab/document_coverage": coverage,
      "lab/has_output": float(has_output),
      "lab/has_valid_expected_output": float(has_valid_expected_output),
      "lab/process_reward": process_reward,
    }

  def lab_python(self) -> Path:
    candidate = self.lab_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)

  def write_metadata(self, history: list[Message]) -> None:
    self.run_dir.mkdir(parents=True, exist_ok=True)
    with (self.run_dir / "tinker_history.jsonl").open("w", encoding="utf-8") as f:
      for message in history:
        f.write(json.dumps(message_to_jsonable(message), sort_keys=True) + "\n")

    config = {
      "task": self.task_name,
      "run_id": self.run_id,
      "judge_model": self.judge_model,
      **self.config,
    }
    (self.run_dir / "config.json").write_text(
      json.dumps(config, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    (self.run_dir / "metrics.json").write_text(
      json.dumps(self.tool_metrics(), indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )


def reward_from_scores(scores: dict[str, Any]) -> tuple[float, dict[str, float]]:
  n_criteria = int(scores.get("n_criteria", 0) or 0)
  n_passed = int(scores.get("n_passed", 0) or 0)
  reward = (n_passed / n_criteria) if n_criteria else 0.0
  return (
    float(reward),
    {
      "lab/criteria_total": float(n_criteria),
      "lab/criteria_passed": float(n_passed),
      "lab/criteria_pass_fraction": float(reward),
      "lab/all_pass": float(bool(scores.get("all_pass"))),
    },
  )


def valid_artifact(path: Path) -> bool:
  suffix = path.suffix.lower()
  if suffix in {".md", ".txt"}:
    return path.stat().st_size > 0
  if suffix == ".pdf":
    return path.read_bytes()[:5] == b"%PDF-"
  office_roots = {
    ".docx": "word/document.xml",
    ".xlsx": "xl/workbook.xml",
    ".pptx": "ppt/presentation.xml",
  }
  if root := office_roots.get(suffix):
    try:
      with zipfile.ZipFile(path) as archive:
        return root in archive.namelist()
    except (OSError, zipfile.BadZipFile):
      return False
  return False
