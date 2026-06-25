from typing import Protocol

from .checkpoint import CheckpointRestorer
from .workload import WorkloadRef


class LlmDOperationResult(Protocol):
  status: str
  error: str | None


class LlmDClient(Protocol):
  def snapshot_and_wait(self, job_id: str, group: str, backend: str, poll_interval_sec: float = 1.0) -> LlmDOperationResult: ...

  def restore_and_wait(self, job_id: str, group: str, backend: str, poll_interval_sec: float = 1.0) -> LlmDOperationResult: ...

  def close(self) -> None: ...


class LlmDCheckpointRestorer(CheckpointRestorer):
  """CheckpointRestorer backed by llm-d's snapshot-agent Python client."""

  def __init__(
    self,
    client: LlmDClient,
    backend: str = "CUDA",
    poll_interval_sec: float = 1.0,
  ):
    self.backend = backend
    self.poll_interval_sec = poll_interval_sec
    self.client = client

  def checkpoint(self, workload: WorkloadRef) -> bool:
    result = self.client.snapshot_and_wait(workload.job_id, workload.group, self.backend, self.poll_interval_sec)
    ensure_complete("snapshot", workload.job_id, result)
    return True

  def restore(self, workload: WorkloadRef) -> None:
    result = self.client.restore_and_wait(workload.job_id, workload.group, self.backend, self.poll_interval_sec)
    ensure_complete("restore", workload.job_id, result)


def ensure_complete(op: str, job_id: str, result: LlmDOperationResult) -> None:
  if result.status == "OPERATION_STATUS_COMPLETE":
    return
  detail = f": {result.error}" if result.error else ""
  raise RuntimeError(f"llm-d {op} for job {job_id} finished with {result.status}{detail}")
