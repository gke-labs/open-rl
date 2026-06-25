import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from .workload import DEFAULT_TIME_SLICE_GROUP, WorkloadRef

DEFAULT_SOCKET_PATH = "/tmp/open-rl/accel-timeslicer.sock"
DEFAULT_TCP_PORT = 9753


class TimeSlicerClient(Protocol):
  async def register(self, workload: WorkloadRef) -> dict[str, Any]: ...

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]: ...

  def acquire(self, workload: WorkloadRef) -> AbstractAsyncContextManager[None]: ...

  async def close(self) -> None: ...


class TimeSlicer(Protocol):
  async def register(self, workload: WorkloadRef) -> dict[str, Any]: ...

  async def acquire(self, workload: WorkloadRef) -> dict[str, Any]: ...

  async def release(self, workload: WorkloadRef) -> dict[str, Any]: ...

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]: ...


class SocketTimeSlicerClient:
  def __init__(
    self,
    socket_path: str | None = None,
    host: str | None = None,
    port: int = DEFAULT_TCP_PORT,
  ):
    self.socket_path = socket_path or DEFAULT_SOCKET_PATH
    self.host = host
    self.port = port
    self.reader: asyncio.StreamReader | None = None
    self.writer: asyncio.StreamWriter | None = None

  async def connect(self) -> None:
    if self.writer is not None and not self.writer.is_closing():
      return
    if self.host:
      self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
    else:
      self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)

  async def close(self) -> None:
    if self.writer is None:
      return
    self.writer.close()
    await self.writer.wait_closed()
    self.reader = None
    self.writer = None

  async def register(self, workload: WorkloadRef) -> dict[str, Any]:
    return await self.request({"command": "REGISTER", **workload.as_payload()})

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    return await self.request({"command": "UNREGISTER", **workload.as_payload()})

  @asynccontextmanager
  async def acquire(self, workload: WorkloadRef) -> AsyncIterator[None]:
    payload = workload.as_payload()
    await self.request({"command": "ACQUIRE", **payload})
    try:
      yield
    finally:
      await self.request({"command": "RELEASE", **payload})

  async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
    await self.connect()
    assert self.reader is not None
    assert self.writer is not None

    self.writer.write(json.dumps(payload).encode("utf-8") + b"\n")
    await self.writer.drain()
    line = await self.reader.readline()
    if not line:
      raise RuntimeError("time slicer connection closed")

    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
      raise RuntimeError(response.get("error", "time slicer command failed"))
    return response


def workload_from_env(pid: int | None = None, job_id: str | None = None, group: str = DEFAULT_TIME_SLICE_GROUP) -> WorkloadRef:
  env_job_id = os.getenv("OPEN_RL_TIME_SLICE_JOB_ID")
  if env_job_id:
    return WorkloadRef(job_id=env_job_id, group=group)
  if job_id:
    return WorkloadRef(job_id=job_id, group=group)
  if pid is None:
    raise ValueError("workload requires job_id")
  return WorkloadRef(job_id=str(pid), group=group)


def time_slicer_client_from_env() -> TimeSlicerClient:
  host = os.getenv("OPEN_RL_ACCEL_TIMESLICER_HOST")
  if host:
    return SocketTimeSlicerClient(host=host, port=int(os.getenv("OPEN_RL_ACCEL_TIMESLICER_PORT", str(DEFAULT_TCP_PORT))))

  return SocketTimeSlicerClient(socket_path=os.getenv("OPEN_RL_ACCEL_TIMESLICER_SOCKET", DEFAULT_SOCKET_PATH))
