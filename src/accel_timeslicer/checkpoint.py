import logging
import os
import shlex
import subprocess
import time
from typing import Protocol

from .process_discovery import discover_workload_gpu_pids
from .workload import WorkloadRef

logger = logging.getLogger(__name__)


class CheckpointRestorer(Protocol):
  def checkpoint(self, workload: WorkloadRef) -> bool | None:
    pass

  def restore(self, workload: WorkloadRef) -> None:
    pass


class CudaCheckpointRestorer:
  def __init__(self, cuda_checkpoint_bin: str | None = None, timeout_ms: int | None = None):
    self.cuda_checkpoint_bin = cuda_checkpoint_bin or os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint")
    self.timeout_ms = timeout_ms
    # Checkpointed processes no longer show up in nvidia-smi, so restore must
    # use the PID set captured before checkpoint.
    self.checkpointed_pids: dict[str, list[int]] = {}

  def checkpoint(self, workload: WorkloadRef) -> bool:
    pids = self.discover_pids(workload)
    if not pids:
      self.checkpointed_pids.pop(workload.key, None)
      logger.info("checkpoint skipped for workload=%s: no GPU PIDs found", workload.key)
      return False
    start = time.perf_counter()
    logger.info("checkpoint workload=%s pids=%s", workload.key, pids)
    for pid in pids:
      lock_args = ["--action", "lock", "--pid", str(pid)]
      if self.timeout_ms is not None:
        lock_args.extend(["--timeout", str(self.timeout_ms)])

      self.run_cuda_checkpoint(lock_args)
    for pid in pids:
      self.run_cuda_checkpoint(["--action", "checkpoint", "--pid", str(pid)])
    self.checkpointed_pids[workload.key] = pids
    logger.info("checkpoint workload=%s took %.0f ms", workload.key, (time.perf_counter() - start) * 1000)
    return True

  def restore(self, workload: WorkloadRef) -> None:
    pids = self.checkpointed_pids.get(workload.key)
    if not pids:
      raise RuntimeError(f"no checkpointed PIDs found for workload {workload.key}")
    start = time.perf_counter()
    logger.info("restore workload=%s pids=%s", workload.key, pids)
    for pid in pids:
      self.run_cuda_checkpoint(["--action", "restore", "--pid", str(pid)])
    for pid in pids:
      self.run_cuda_checkpoint(["--action", "unlock", "--pid", str(pid)])
    self.checkpointed_pids.pop(workload.key, None)
    logger.info("restore workload=%s took %.0f ms", workload.key, (time.perf_counter() - start) * 1000)

  def run_cuda_checkpoint(self, args: list[str]) -> None:
    full_argv = [self.cuda_checkpoint_bin, *args]
    result = subprocess.run(full_argv, capture_output=True, check=False, text=True)
    if result.returncode != 0:
      stderr = result.stderr.strip()
      stdout = result.stdout.strip()
      detail = stderr or stdout or f"exit code {result.returncode}"
      rendered_argv = " ".join(shlex.quote(arg) for arg in full_argv)
      raise RuntimeError(f"{rendered_argv} failed: {detail}")

  def discover_pids(self, workload: WorkloadRef) -> list[int]:
    return discover_workload_gpu_pids(workload)
