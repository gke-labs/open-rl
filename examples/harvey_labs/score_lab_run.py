#!/usr/bin/env python3
"""Score one LAB run from the LAB virtual environment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Score a Harvey LAB run for Open-RL.")
  parser.add_argument("--lab-root", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--judge-model", required=True)
  parser.add_argument("--parallel", type=int, default=1)
  parser.add_argument("--max-criteria", type=int, default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  lab_root = Path(args.lab_root).resolve()
  sys.path.insert(0, str(lab_root))

  from evaluation.judge import Judge
  from evaluation.run_eval import _load_env, evaluate_run
  from evaluation.scoring import score_rubric

  _load_env()
  judge = Judge(model=args.judge_model)

  if args.max_criteria is None:
    # LAB's own scorer: validates task.json, grades every criterion, and
    # writes results/<run-id>/scores.json with the schema reward.py parses.
    evaluate_run(run_id=args.run_id, task=args.task, judge=judge, parallel=args.parallel)
    return

  # Criteria-subset path for cheap reward passes; evaluate_run always grades
  # the full rubric, so mirror only its scoring core here.
  task_dir = lab_root / "tasks" / Path(*args.task.split("/"))
  task_config = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
  criteria = list(task_config["criteria"])[: args.max_criteria]

  run_dir = lab_root / "results" / args.run_id
  result = score_rubric(
    criteria=criteria,
    run_dir=run_dir,
    judge=judge,
    task_desc=task_config["title"],
    parallel=args.parallel,
  )
  n_criteria = len(result.criteria_results)
  n_passed = sum(1 for criterion in result.criteria_results if criterion["verdict"] == "pass")
  scores = {
    "score": result.score,
    "max_score": result.max_score,
    "summary": f"{n_passed}/{n_criteria} criteria passed.",
    "all_pass": n_criteria > 0 and n_passed == n_criteria,
    "n_criteria": n_criteria,
    "n_passed": n_passed,
    "criteria_results": result.criteria_results,
    "run_id": args.run_id,
    "task": args.task,
    "judge_model": args.judge_model,
    "scored_at": datetime.now(UTC).isoformat(),
  }
  (run_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
