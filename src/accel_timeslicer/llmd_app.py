# llm-d time-slicing platform integration ("llmd-app" mode).
#
# In this mode OpenRL delegates suspend/resume mechanics to the node-local
# llm-d Snapshot Agent via the app_channel backend: workers register once at
# startup with timeslice.snapshot_agent.register_workload and the agent PUSHES
# snapshot/restore commands over the stream when jobs are swapped.
#
# Mode selection and addressing are environment driven:
#   OPEN_RL_TIME_SLICE_MODE       "llmd-app" enables this mode (default: legacy)
#   OPEN_RL_SNAPSHOT_AGENT_ADDR  node-local Snapshot Agent gRPC target
#                                (falls back to LLMD_SNAPSHOT_AGENT_ENDPOINT, then NODE_IP:9001)
#
# Identity: the same job_id/group pair is used for the app_channel
# registration and the timeslice.io/job-id / timeslice.io/group pod labels
# (which the snapshot agent's k8s watcher uses to track job state).

import os
from collections.abc import Callable
from typing import Any

from .workload import WorkloadRef

LLMD_APP_MODE = "llmd-app"
DEFAULT_SNAPSHOT_AGENT_PORT = 9001


def timeslice_mode() -> str:
  return os.getenv("OPEN_RL_TIME_SLICE_MODE", "legacy").strip().lower()


def is_llmd_app_mode() -> bool:
  return timeslice_mode() == LLMD_APP_MODE


def snapshot_agent_addr() -> str:
  addr = os.getenv("OPEN_RL_SNAPSHOT_AGENT_ADDR") or os.getenv("LLMD_SNAPSHOT_AGENT_ENDPOINT")
  if addr:
    return addr
  node_ip = os.getenv("NODE_IP")
  if node_ip:
    return f"{node_ip}:{DEFAULT_SNAPSHOT_AGENT_PORT}"
  return f"127.0.0.1:{DEFAULT_SNAPSHOT_AGENT_PORT}"


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
