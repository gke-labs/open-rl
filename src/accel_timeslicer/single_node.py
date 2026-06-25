import asyncio
import logging
import os
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


class SingleNodeTimeSlicer(TimeSlicer):
  def __init__(self, restorer: CheckpointRestorer):
    self.restorer = restorer
    self.workloads: dict[str, WorkloadState] = {}
    self.waiting_workloads: deque[str] = deque()
    self.active_workload: str | None = None
    self.condition = asyncio.Condition()

  async def register(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      key = workload.key
      if key in self.workloads:
        return {"ok": False, "error": f"workload {key} is already registered"}

      self.workloads[key] = WorkloadState(connection_id=connection_id, workload=workload)
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key = workload.key
      state = self.workloads.get(key)
      if state is None:
        return {"ok": False, "error": f"workload {key} is not registered"}
      if state.failed:
        return {"ok": False, "error": f"workload {key} is failed"}
      if key in self.waiting_workloads or self.active_workload == key:
        return {"ok": False, "error": f"workload {key} already has a pending or active acquire"}

      self.waiting_workloads.append(key)
      try:
        while self.active_workload is not None or (key in self.waiting_workloads and self.waiting_workloads[0] != key):
          await self.condition.wait()
      except BaseException:
        if key in self.waiting_workloads:
          self.waiting_workloads.remove(key)
        self.condition.notify_all()
        raise

      state = self.workloads.get(key)
      if state is None or state.failed or key not in self.waiting_workloads:
        self.clear_workload(key)
        self.condition.notify_all()
        return {"ok": False, "error": f"workload {key} is not available"}

      self.waiting_workloads.popleft()
      self.active_workload = key
      self.condition.notify_all()

    if state.checkpointed:
      await self.run_restore(state)
      async with self.condition:
        state = self.workloads.get(workload.key)
        if state is not None:
          state.checkpointed = False
        self.condition.notify_all()
    return {"ok": True}

  async def release(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key = workload.key
      state = self.workloads.get(key)
      if state is None:
        return {"ok": False, "error": f"workload {key} is not registered"}
      if self.active_workload != key:
        return {"ok": False, "error": f"workload {key} does not hold an active acquire"}

    checkpointed = await self.run_checkpoint(state)

    async with self.condition:
      state = self.workloads.get(workload.key)
      if state is not None:
        state.checkpointed = checkpointed is not False
      self.clear_workload(workload.key)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key = workload.key
      if key not in self.workloads:
        return {"ok": False, "error": f"workload {key} is not registered"}

      self.clear_workload(key)
      del self.workloads[key]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for key, state in self.workloads.items():
        if state.connection_id != connection_id:
          continue
        self.clear_workload(key)
        state.failed = True
        state.checkpointed = False
        state.connection_id = None
      self.condition.notify_all()

  def clear_workload(self, key: str) -> None:
    if key in self.waiting_workloads:
      self.waiting_workloads.remove(key)
    if self.active_workload == key:
      self.active_workload = None

  async def run_checkpoint(self, state: WorkloadState) -> bool | None:
    workload = state.workload
    start = time.monotonic()
    try:
      checkpointed = await asyncio.to_thread(self.restorer.checkpoint, workload)
      if checkpointed is False:
        logger.info("released workload %s group %s without checkpoint in %.2fs", workload.key, workload.group, time.monotonic() - start)
      else:
        logger.info("checkpointed workload %s group %s in %.2fs", workload.key, workload.group, time.monotonic() - start)
      return checkpointed
    except Exception:
      logger.critical(
        "checkpoint failed for workload %s group %s after %.2fs; GPU state is unknown, killing time slicer",
        workload.key,
        workload.group,
        time.monotonic() - start,
        exc_info=True,
      )
      os._exit(1)

  async def run_restore(self, state: WorkloadState) -> None:
    workload = state.workload
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.restore, workload)
      logger.info("restored workload %s group %s in %.2fs", workload.key, workload.group, time.monotonic() - start)
    except Exception:
      logger.critical(
        "restore failed for workload %s group %s after %.2fs; GPU state is unknown, killing time slicer",
        workload.key,
        workload.group,
        time.monotonic() - start,
        exc_info=True,
      )
      os._exit(1)
