import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointRestorer
from .time_slicer import TimeSlicer
from .workload import WorkloadRef

logger = logging.getLogger(__name__)


@dataclass
class WorkloadState:
  connection_id: int | None
  workload: WorkloadRef
  checkpointed: bool = False
  failed: bool = False
  # Why no process can run under this name again, once one left state behind that the restorer cannot drop.
  unservable: str | None = None


class SingleNodeTimeSlicer(TimeSlicer):
  """Grants accelerator turns. Claims are independent; within a claim one
  workload holds the grant at a time. A release checkpoints the workload off
  the devices and its next acquire restores it."""

  def __init__(self, restorer: CheckpointRestorer, scheduling_policy: str = "lrs"):
    self.restorer = restorer
    self.scheduling_policy = scheduling_policy.lower()
    self.workloads: dict[str, WorkloadState] = {}
    self.waiting_workloads: deque[str] = deque()
    # Per claim, the workload holding the grant (acquire..release).
    self.running: dict[str, str] = {}
    self.condition = asyncio.Condition()
    self.last_release_time: dict[str, float] = {}

  def next_waiter(self, claim: str) -> str | None:
    """The earliest waiter under fifo, the least recently served otherwise."""
    waiting = [self.workloads[name].workload for name in self.waiting_workloads if self.workloads[name].workload.claim == claim]
    if not waiting:
      return None
    if self.scheduling_policy == "fifo" or len(waiting) == 1:
      return waiting[0].name
    return min(waiting, key=lambda w: self.last_release_time.get(w.name, 0.0)).name

  async def register(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      state = self.workloads.get(workload.name)
      if state is None:
        self.workloads[workload.name] = WorkloadState(connection_id=connection_id, workload=workload)
      elif state.unservable:
        # Refused up front, so the new process exits with the reason instead of booting inside a turn it can never leave.
        return {"ok": False, "error": f"workload {workload.name} cannot run on this node again: {state.unservable}"}
      else:
        # A registration is a live process announcing itself, so a failure
        # recorded against an earlier process under this name is cleared.
        state.connection_id = connection_id
        state.workload = workload
        state.failed = False
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      name, claim = workload.name, workload.claim
      state = self.workloads.get(name)
      if state is None:
        state = WorkloadState(connection_id=None, workload=workload)
        self.workloads[name] = state
      if state.unservable:
        return {"ok": False, "error": f"workload {name} cannot run on this node again: {state.unservable}"}
      if state.failed:
        return {"ok": False, "error": f"workload {name} is failed"}
      if name in self.waiting_workloads or self.running.get(claim) == name:
        return {"ok": False, "error": f"workload {name} already has a pending or active acquire"}

      self.waiting_workloads.append(name)
      try:
        while name in self.waiting_workloads and (claim in self.running or self.next_waiter(claim) != name):
          await self.condition.wait()
      except BaseException:
        if name in self.waiting_workloads:
          self.waiting_workloads.remove(name)
        self.condition.notify_all()
        raise

      state = self.workloads.get(name)
      if state is None or state.failed or name not in self.waiting_workloads:
        self.clear_workload(name)
        self.condition.notify_all()
        return {"ok": False, "error": f"workload {name} is not available"}

      self.waiting_workloads.remove(name)
      self.running[claim] = name
      self.condition.notify_all()

    if state.checkpointed:
      await self.run_restore(state)
      async with self.condition:
        state = self.workloads.get(name)
        if state is not None:
          state.checkpointed = False
        self.condition.notify_all()
    return {"ok": True}

  async def release(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      name, claim = workload.name, workload.claim
      state = self.workloads.get(name)
      if state is None or self.running.get(claim) != name:
        return {"ok": False, "error": f"workload {name} does not hold an active acquire"}

    # The grant is held through the checkpoint, so the next waiter is not
    # granted until this workload is off the devices.
    try:
      checkpointed = await self.run_checkpoint(state)
    except Exception as exc:
      # A workload that could not be parked is still on the devices. It keeps
      # the grant so nobody is restored on top of its memory; the seat frees
      # when its process exits and the connection closes.
      logger.error("checkpoint failed for workload %s claim %s; it keeps the grant until its process exits: %s", name, claim, exc)
      async with self.condition:
        state = self.workloads.get(name)
        if state is not None:
          state.failed = True
          state.checkpointed = False
        self.condition.notify_all()
      return {"ok": False, "faulted": True, "error": f"checkpoint failed for workload {name}; it still holds the accelerator and must exit: {exc}"}

    async with self.condition:
      state = self.workloads.get(name)
      if state is not None:
        state.checkpointed = checkpointed is not False
      self.last_release_time[workload.name] = time.time()
      self.clear_workload(name)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      name = workload.name
      if name not in self.workloads:
        return {"ok": False, "error": f"workload {name} is not registered"}

      self.clear_workload(name)
      if not self.process_gone(self.workloads[name]):
        del self.workloads[name]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for name, state in self.workloads.items():
        if state.connection_id != connection_id:
          continue
        self.clear_workload(name)
        state.failed = True
        state.checkpointed = False
        state.connection_id = None
        self.process_gone(state)
      self.condition.notify_all()

  def process_gone(self, state: WorkloadState) -> bool:
    """The workload's process has left. Whether its name must stay on record because the restorer can no longer park
    a process under it, so the next one is refused with the reason."""
    reason = self.restorer.forget(state.workload)
    if reason:
      logger.error("workload %s claim %s cannot run on this node again: %s", state.workload.name, state.workload.claim, reason)
      state.unservable = reason
    return bool(reason)

  def clear_workload(self, name: str) -> None:
    if name in self.waiting_workloads:
      self.waiting_workloads.remove(name)
    for claim, holder in list(self.running.items()):
      if holder == name:
        del self.running[claim]

  async def run_checkpoint(self, state: WorkloadState) -> bool | None:
    """Parks the workload. Raises when the restorer could not, since the workload then still holds the devices."""
    workload = state.workload
    start = time.monotonic()
    checkpointed = await asyncio.to_thread(self.restorer.checkpoint, workload)
    if checkpointed is False:
      logger.info("released workload %s claim %s without checkpoint in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    else:
      logger.info("checkpointed workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    return checkpointed

  async def run_restore(self, state: WorkloadState) -> None:
    workload = state.workload
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.restore, workload)
      logger.info("restored workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    except Exception as exc:
      logger.warning("restore failed for workload %s claim %s: %s", workload.name, workload.claim, exc)
