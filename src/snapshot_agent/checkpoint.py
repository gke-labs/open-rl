import logging
import os
import shlex
import subprocess
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class CheckpointRestorer(Protocol):
  def checkpoint(self, pid: int) -> None:
    pass

  def restore(self, pid: int) -> None:
    pass


class CudaCheckpointRestorer:
  def __init__(self, cuda_checkpoint_bin: str | None = None, timeout_ms: int | None = None):
    self.cuda_checkpoint_bin = cuda_checkpoint_bin or os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint")
    self.timeout_ms = timeout_ms

  def checkpoint(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("checkpoint pid=%s", pid)
    lock_args = ["--action", "lock", "--pid", str(pid)]
    if self.timeout_ms is not None:
      lock_args.extend(["--timeout", str(self.timeout_ms)])

    self.run_cuda_checkpoint(lock_args)
    self.run_cuda_checkpoint(["--action", "checkpoint", "--pid", str(pid)])
    logger.info("checkpoint pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def restore(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("restore pid=%s", pid)
    self.run_cuda_checkpoint(["--action", "restore", "--pid", str(pid)])
    self.run_cuda_checkpoint(["--action", "unlock", "--pid", str(pid)])
    logger.info("restore pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def run_cuda_checkpoint(self, args: list[str]) -> None:
    full_argv = [self.cuda_checkpoint_bin, *args]
    result = subprocess.run(full_argv, capture_output=True, check=False, text=True)
    if result.returncode != 0:
      stderr = result.stderr.strip()
      stdout = result.stdout.strip()
      detail = stderr or stdout or f"exit code {result.returncode}"
      rendered_argv = " ".join(shlex.quote(arg) for arg in full_argv)
      raise RuntimeError(f"{rendered_argv} failed: {detail}")
