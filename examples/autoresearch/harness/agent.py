"""Launch one autoresearch agent and maintain its manifest."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import chz

from harness.utils import AgentPaths, agent_id, close_attempt_agent_logs, recipe_spec_hash, run_id, run_logged, write_json_atomic

IGNORED_RECIPE_DIRS = {".git", "__pycache__", "gke"}
IGNORED_RECIPE_FILES = {".DS_Store", "kustomization.yaml"}


@chz.chz
class AgentConfig:
  agent_id: str = os.getenv("AGENT_ID", "")
  project_dir: Path = Path(os.getenv("PROJECT_DIR", Path(__file__).parents[2]))
  image_dir: Path = Path(os.getenv("IMAGE_DIR", Path(__file__).parents[1]))
  recipe_dir: Path | None = Path(value) if (value := os.getenv("RECIPE_DIR", "")) else None
  log_root: Path = Path(os.getenv("LOG_ROOT", "artifacts/autoresearch"))
  run_name: str = os.getenv("RUN_NAME", "")
  recipe: Path = Path(os.getenv("RECIPE", "autoresearch.toml"))
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))
  agent_timeout_minutes: float = float(os.getenv("AGENT_TIMEOUT_MINUTES", "10"))
  agent_model: str = os.getenv("AGENT_MODEL", "gemini-3.1-pro-preview")
  agent_flags: str = os.getenv("AGENT_FLAGS", "--yolo --output-format stream-json")


def log(paths: AgentPaths, message: str) -> None:
  print(message)
  paths.agent_log.parent.mkdir(parents=True, exist_ok=True)
  with paths.agent_log.open("a", encoding="utf-8") as f:
    print(message, file=f)


def write_agent_manifest(
  paths: AgentPaths,
  args: AgentConfig,
  status: str,
  spec_hash: str = "",
  exit_code: int | None = None,
) -> None:
  paths.root.mkdir(parents=True, exist_ok=True)
  paths.agent_log.touch(exist_ok=True)
  paths.notes.touch(exist_ok=True)
  write_json_atomic(
    paths.metadata,
    {
      "spec_hash": spec_hash,
      "status": status,
      "started_at": paths.root.stat().st_ctime,
      "finished_at": None if status == "running" else time.time(),
      "exit_code": exit_code,
      "agent_model": args.agent_model,
      "agent_timeout_minutes": args.agent_timeout_minutes,
    },
  )


def copy_recipe_files(source: Path, target: Path) -> None:
  target.mkdir(parents=True, exist_ok=True)
  for path in sorted(source.iterdir()):
    if path.name.startswith("..") or path.name in IGNORED_RECIPE_FILES:
      continue
    if path.is_dir():
      if path.name in IGNORED_RECIPE_DIRS:
        continue
      raise SystemExit(f"recipe directories are not supported yet: {path}")
    if path.is_file():
      shutil.copy2(path, target / path.name)


def prepare_workspace(args: AgentConfig, paths: AgentPaths) -> None:
  paths.root.mkdir(parents=True, exist_ok=True)
  if paths.workspace.exists():
    shutil.rmtree(paths.workspace)
  paths.workspace.mkdir(parents=True)
  source = args.recipe_dir or args.image_dir / args.recipe.parent
  target = paths.workspace / args.recipe.parent
  copy_recipe_files(source, paths.workspace if args.recipe.parent == Path(".") else target)
  subprocess.run(["git", "init", "-b", "main"], cwd=paths.workspace, check=True)
  subprocess.run(["git", "config", "user.email", "agent@open-rl.local"], cwd=paths.workspace, check=True)
  subprocess.run(["git", "config", "user.name", "Autoresearch Agent"], cwd=paths.workspace, check=True)
  with (paths.workspace / ".git/info/exclude").open("a", encoding="utf-8") as f:
    f.write("__pycache__/\n*.pyc\n")
  subprocess.run(["git", "add", "."], cwd=paths.workspace, check=True)
  subprocess.run(["git", "commit", "-m", "Autoresearch run recipe"], cwd=paths.workspace, check=True)
  log(paths, f"Created fresh workspace at {paths.workspace} from recipe source {source}.")


def attempt_command(args: AgentConfig, baseline: bool = False) -> str:
  # Everything else (recipe, agent id, log root, run name, timeout) is read from the
  # inherited environment, so the attempt command stays a bare module invocation.
  parts = ["uv", "run", "--project", str(args.project_dir), "--no-sync", "python", "-m", "harness.attempt", "echo=False"]
  if baseline:
    parts.append("baseline=True")
  return shlex.join(parts)


def attempt_env(args: AgentConfig, paths: AgentPaths) -> dict[str, str]:
  return {
    "AGENT_ID": args.agent_id,
    "RUN_NAME": args.run_name,
    "LOG_ROOT": str(args.log_root),
    "RECIPE": str(args.recipe),
    "WORK_DIR": str(paths.root),
    "ATTEMPT_TIMEOUT_MINUTES": str(args.attempt_timeout_minutes),
    "PYTHONPATH": str(args.image_dir),
    "RUN_ATTEMPT_COMMAND": attempt_command(args),
    "GEMINI_CLI_TRUST_WORKSPACE": "true",
    "PYTHONDONTWRITEBYTECODE": "1",
  }


def run_agent(args: AgentConfig, paths: AgentPaths) -> tuple[int, str]:
  prepare_workspace(args, paths)
  spec_hash = recipe_spec_hash(paths.workspace / args.recipe.parent)
  program_file = paths.workspace / args.recipe.parent / "program.md"
  if not program_file.is_file():
    raise SystemExit(f"program file does not exist next to RECIPE: {program_file}")
  paths.program.write_text(program_file.read_text(encoding="utf-8"), encoding="utf-8")
  write_agent_manifest(paths, args, "running", spec_hash)
  env = {**os.environ, **attempt_env(args, paths)}
  baseline = shlex.split(attempt_command(args, baseline=True))
  if run_logged(baseline, paths.agent_log, args.attempt_timeout_minutes * 60, cwd=paths.workspace, env=env) != 0:
    raise SystemExit("baseline failed; not starting the research agent.")
  if not shutil.which("gemini"):
    raise SystemExit("gemini not found; install @google/gemini-cli in the image or local environment.")
  log(paths, f"Starting {args.agent_timeout_minutes} minute agent timeout.")
  gemini_command = [
    "gemini",
    "--model",
    args.agent_model,
    *shlex.split(args.agent_flags),
    "--prompt",
    paths.program.read_text(encoding="utf-8"),
  ]
  code = run_logged(gemini_command, paths.agent_log, args.agent_timeout_minutes * 60, cwd=paths.workspace, env=env)
  if code in {124, 137}:
    log(paths, f"Agent timed out after {args.agent_timeout_minutes} minutes.")
    return 0, "timed_out"
  return code, "completed" if code == 0 else "failed"


def main(argv: list[str] | None = None) -> None:
  args = chz.entrypoint(AgentConfig, argv=argv, allow_hyphens=True)
  if not args.agent_id:
    raise SystemExit("agent_id is required; set AGENT_ID or pass agent_id=...")
  args = chz.replace(
    args,
    agent_id=agent_id(args.agent_id),
    project_dir=Path(args.project_dir).resolve(),
    image_dir=Path(args.image_dir).resolve(),
    recipe_dir=Path(args.recipe_dir).resolve() if args.recipe_dir else None,
    log_root=Path(args.log_root).resolve(),
    run_name=run_id(args.run_name),
    recipe=Path(args.recipe),
  )
  paths = AgentPaths.from_run(args.log_root, args.run_name, args.agent_id)
  code, status = 1, "failed"
  try:
    code, status = run_agent(args, paths)
  except KeyboardInterrupt:
    code, status = 143, "stopped"
  finally:
    manifest = paths.metadata
    spec_hash = str(json.loads(manifest.read_text(encoding="utf-8"))["spec_hash"]) if manifest.exists() else ""
    write_agent_manifest(paths, args, status, spec_hash, code)
    log(paths, f"Autoresearch session exiting with code {code}.")
    if paths.agent_log.exists():
      close_attempt_agent_logs(paths.root, paths.agent_log.stat().st_size)
  raise SystemExit(code)


if __name__ == "__main__":
  main()
