from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEC_HASH_IGNORED_DIRS = {".git", "__pycache__"}
SPEC_HASH_IGNORED_FILES = {".DS_Store", "kustomization.yaml"}
SPEC_HASH_IGNORED_PREFIXES = ("..",)

# Single source of truth for the on-disk layout. The keys are the artifact names
# used in UI/HTTP requests; the values are the filenames the harness writes.
METADATA = "metadata.json"
AGENT_ARTIFACTS = {"agent": "agent.log", "notes": "notes.md"}
ATTEMPT_ARTIFACTS = {"logs": "attempt.log", "metrics": "metrics.jsonl", "diff": "diff.patch"}


@dataclass(frozen=True)
class AgentPaths:
  root: Path
  run_root: Path
  attempts: Path
  workspace: Path
  metadata: Path
  agent_log: Path
  notes: Path
  program: Path

  @classmethod
  def from_run(cls, log_root: Path, run_name: str, agent: str) -> AgentPaths:
    root = (log_root / run_name / "researchers" / agent).resolve()
    return cls(
      root,
      root.parent.parent,
      root / "attempts",
      root / "workspace",
      root / METADATA,
      root / AGENT_ARTIFACTS["agent"],
      root / AGENT_ARTIFACTS["notes"],
      root / "program.md",
    )

  def attempt_dir(self, name: str) -> Path:
    return self.attempts / name


def close_attempt_agent_logs(agent_dir: Path, end: int) -> None:
  for path in sorted((agent_dir / "attempts").glob(f"*/{METADATA}")):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("agent_log_end") is not None:
      continue
    start = int(data.get("agent_log_start", 0) or 0)
    data["agent_log_start"] = start
    data["agent_log_end"] = max(start, end)
    write_json_atomic(path, data)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def agent_id(value: str) -> str:
  normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40]
  if not normalized:
    raise ValueError("agent_id is required")
  return normalized


def run_id(value: str) -> str:
  normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
  if not normalized:
    raise ValueError("run_name is required")
  return normalized


def recipe_spec_hash(root: Path) -> str:
  root = root.resolve()
  digest = hashlib.sha256()
  for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if path.is_dir() or path.name in SPEC_HASH_IGNORED_FILES:
      continue
    if any(part in SPEC_HASH_IGNORED_DIRS for part in rel.parts):
      continue
    if any(part.startswith(SPEC_HASH_IGNORED_PREFIXES) for part in rel.parts):
      continue
    digest.update(rel.as_posix().encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
  return "sha256:" + digest.hexdigest()


def git_text(*args: str, cwd: Path | None = None) -> str:
  result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
  return result.stdout if result.returncode == 0 else ""


def stop_process(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    return


def run_logged(
  command: list[str],
  log_path: Path,
  timeout_seconds: float | None = None,
  cwd: Path | None = None,
  env: dict[str, str] | None = None,
  echo: bool = True,
) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("ab") as file:
    lock = threading.Lock()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace") if echo else None

    def echo_child(data: bytes) -> None:
      if not echo or decoder is None:
        return
      text = decoder.decode(data)
      if text:
        sys.stdout.write(text)
        sys.stdout.flush()

    def write_control(text: str) -> None:
      with lock:
        file.write(text.encode())
        file.flush()
        if echo:
          sys.stdout.write(text)
          sys.stdout.flush()

    def write_child(data: bytes) -> None:
      with lock:
        file.write(data)
        file.flush()
        echo_child(data)

    def flush_echo_unlocked() -> None:
      if decoder is None:
        return
      tail = decoder.decode(b"", final=True)
      if tail:
        sys.stdout.write(tail)
        sys.stdout.flush()

    def flush_echo() -> None:
      with lock:
        flush_echo_unlocked()

    write_control(f"$ {shlex.join(command)}\n")
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    if process.stdout is None:
      raise RuntimeError("subprocess stdout pipe was not created")

    def copy_output() -> None:
      while chunk := process.stdout.read1(65536):
        write_child(chunk)
      flush_echo()

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    try:
      try:
        process.wait(timeout=timeout)
      except subprocess.TimeoutExpired:
        stop_process(process)
        try:
          process.wait(timeout=10)
        except subprocess.TimeoutExpired:
          stop_process(process, signal.SIGKILL)
          process.wait()
        reader.join(timeout=1)
        return 124
      reader.join(timeout=1)
      return process.returncode
    finally:
      if process.poll() is None:
        stop_process(process)
      reader.join(timeout=1)
      file.flush()
