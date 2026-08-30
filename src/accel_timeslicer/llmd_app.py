# llm-d time-slicing platform integration ("llmd-app" mode).
#
# In this mode OpenRL stops running its own accel-timeslicer daemonset and
# instead delegates coordination to the llm-d TimeSlice Orchestrator
# (cluster-scoped lock queues, one per time-slice group) and delegates
# suspend/resume mechanics to the node-local llm-d Snapshot Agent via the
# app_channel backend: workers register once at startup with
# timeslice.snapshot_agent.register_workload and the agent PUSHES
# snapshot/restore commands over the stream when the orchestrator swaps jobs.
#
# Mode selection and addressing are environment driven:
#   OPEN_RL_TIME_SLICE_MODE       "llmd-app" enables this mode (default: legacy)
#   OPEN_RL_TIME_SLICE_ORCH_ADDR  TimeSlice Orchestrator gRPC target
#   OPEN_RL_SNAPSHOT_AGENT_ADDR  node-local Snapshot Agent gRPC target
#                                (falls back to LLMD_SNAPSHOT_AGENT_ENDPOINT, then NODE_IP:9001)
#   OPEN_RL_TIME_SLICE_ACQUIRE_TIMEOUT_SEC  optional acquire RPC timeout
#
# Identity: the same job_id/group pair is used for the orchestrator lock, the
# app_channel registration, and the timeslice.io/job-id / timeslice.io/group
# pod labels (which the snapshot agent's k8s watcher uses to track job state).

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from .workload import WorkloadRef

LLMD_APP_MODE = "llmd-app"
DEFAULT_ORCHESTRATOR_ADDR = "timeslice-timesliceorchestrator.timeslice-system.svc.cluster.local:50051"
DEFAULT_SNAPSHOT_AGENT_PORT = 9001


def timeslice_mode() -> str:
  return os.getenv("OPEN_RL_TIME_SLICE_MODE", "legacy").strip().lower()


def is_llmd_app_mode() -> bool:
  return timeslice_mode() == LLMD_APP_MODE


def orchestrator_addr() -> str:
  return os.getenv("OPEN_RL_TIME_SLICE_ORCH_ADDR", DEFAULT_ORCHESTRATOR_ADDR)


def snapshot_agent_addr() -> str:
  addr = os.getenv("OPEN_RL_SNAPSHOT_AGENT_ADDR") or os.getenv("LLMD_SNAPSHOT_AGENT_ENDPOINT")
  if addr:
    return addr
  node_ip = os.getenv("NODE_IP")
  if node_ip:
    return f"{node_ip}:{DEFAULT_SNAPSHOT_AGENT_PORT}"
  return f"127.0.0.1:{DEFAULT_SNAPSHOT_AGENT_PORT}"


def acquire_timeout_sec() -> float | None:
  raw = os.getenv("OPEN_RL_TIME_SLICE_ACQUIRE_TIMEOUT_SEC")
  if not raw:
    return None
  return float(raw)


def register_app_channel_workload(
  workload: WorkloadRef,
  *,
  engine: Any = None,
  on_snapshot: Callable[..., Any] | None = None,
  on_restore: Callable[..., Any] | None = None,
  supported_modes: list[str] | None = None,
  default_mode: str | None = None,
  tags: list[str] | None = None,
  agent_addr: str | None = None,
) -> Any:
  """Register this process with the node-local Snapshot Agent (app_channel).

  Pass a recognized engine object (e.g. vLLM AsyncLLMEngine) via engine=, or
  on_snapshot(mode, tags)/on_restore(tags) callbacks for custom workloads such
  as the FFT trainer (supported_modes=["offload"]). Returns a WorkloadHandle;
  call close() on clean shutdown. The library owns the stream lifecycle and
  reconnects with backoff, re-registering after agent restarts.
  """
  from timeslice.snapshot_agent import register_workload

  return register_workload(
    agent_addr or snapshot_agent_addr(),
    job_id=workload.job_id,
    group=workload.group,
    workload=engine,
    on_snapshot=on_snapshot,
    on_restore=on_restore,
    supported_modes=supported_modes,
    default_mode=default_mode,
    tags=tags,
  )


class OrchestratorTimeSlicerClient:
  """TimeSlicerClient-compatible adapter over the llm-d TimeSlice Orchestrator.

  Semantics differ from the legacy accel-timeslicer:
  - acquire() blocks until the group lock is granted; if this job's context
    was snapshotted, the orchestrator restores it (via the Snapshot Agents on
    the group's nodes) BEFORE the call returns. The yielded AcquireResult
    carries context_restored=False on the zero-overhead path (the context was
    never evicted because nobody else wanted the lock).
  - release() returns immediately; the snapshot of this job's context is
    DEFERRED until another job actually acquires the lock (the agent then
    pushes the snapshot command over the app_channel). Do not offload
    manually around release in this mode.
  - register()/unregister() are no-ops: workload identity registration
    happens via register_app_channel_workload (the app_channel stream), and
    queue membership is implicit in acquire().
  """

  def __init__(self, target: str | None = None):
    from timeslice import TimeSliceOrchestratorClient

    self.target = target or orchestrator_addr()
    self.client = TimeSliceOrchestratorClient(target=self.target)
    self.acquire_timeout_sec = acquire_timeout_sec()

  async def register(self, workload: WorkloadRef) -> dict[str, Any]:
    return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, workload: WorkloadRef) -> AsyncIterator[Any]:
    result = await asyncio.to_thread(
      self.client.acquire,
      workload.job_id,
      workload.group,
      self.acquire_timeout_sec,
    )
    if not result.success:
      raise RuntimeError(f"orchestrator acquire failed for {workload.key}")
    try:
      yield result
    finally:
      await asyncio.to_thread(self.client.release, workload.job_id, workload.group)

  async def close(self) -> None:
    await asyncio.to_thread(self.client.close)
