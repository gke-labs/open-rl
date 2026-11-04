from typing import Any, Protocol

from .checkpoint import CheckpointRestorer
from .workload import WorkloadRef


class LlmDOperationResult(Protocol):
  status: str
  error: str | None


class LlmDClient(Protocol):
  def snapshot_and_wait(self, job_id: str, group: str = "", poll_interval_sec: float = 1.0, backend_config: Any = None) -> LlmDOperationResult: ...

  def restore_and_wait(self, job_id: str, group: str = "", poll_interval_sec: float = 1.0, backend_config: Any = None) -> LlmDOperationResult: ...

  def close(self) -> None: ...


class LlmDCheckpointRestorer(CheckpointRestorer):
  """CheckpointRestorer backed by llm-d's snapshot-agent Python client.
  llm-d calls the workload name job_id and the claim group."""

  def __init__(
    self,
    client: LlmDClient,
    backend_config: Any = None,
    poll_interval_sec: float = 1.0,
  ):
    self.backend_config = backend_config
    self.poll_interval_sec = poll_interval_sec
    self.client = client
    # The agent's state for each job that is not RUNNING. The agent keeps a job's record for the life of the agent, snapshots
    # only a RUNNING job, and has no call to reset one, so a job left SAVED or FAULTED by a process that exited can never be
    # parked again under that name.
    self.job_states: dict[str, str] = {}

  def checkpoint(self, workload: WorkloadRef) -> bool:
    result = self.client.snapshot_and_wait(
      workload.name, group=workload.claim, poll_interval_sec=self.poll_interval_sec, backend_config=self.backend_config
    )
    self.job_states[workload.name] = "SAVED" if result.status == "OPERATION_STATUS_COMPLETE" else "FAULTED"
    ensure_complete("snapshot", workload.name, result)
    return True

  def restore(self, workload: WorkloadRef) -> None:
    result = self.client.restore_and_wait(
      workload.name, group=workload.claim, poll_interval_sec=self.poll_interval_sec, backend_config=self.backend_config
    )
    if result.status == "OPERATION_STATUS_COMPLETE":
      self.job_states.pop(workload.name, None)
    else:
      self.job_states[workload.name] = "FAULTED"
    ensure_complete("restore", workload.name, result)

  def forget(self, workload: WorkloadRef) -> str | None:
    state = self.job_states.get(workload.name)
    if state is None:
      return None
    return (
      f"the snapshot agent holds job {workload.name} as {state} for a process that exited and cannot reset it, "
      "so no new process can be parked under this name until the agent restarts"
    )


def ensure_complete(op: str, job_id: str, result: LlmDOperationResult) -> None:
  if result.status == "OPERATION_STATUS_COMPLETE":
    return
  detail = f": {result.error}" if result.error else ""
  raise RuntimeError(f"llm-d {op} for job {job_id} finished with {result.status}{detail}")
