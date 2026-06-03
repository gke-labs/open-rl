import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class SnapshotAgentClient:
  def __init__(self, socket_path: str):
    self.socket_path = socket_path
    self.reader: asyncio.StreamReader | None = None
    self.writer: asyncio.StreamWriter | None = None

  async def connect(self) -> None:
    if self.writer is not None and not self.writer.is_closing():
      return
    self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)

  async def close(self) -> None:
    if self.writer is None:
      return
    self.writer.close()
    await self.writer.wait_closed()
    self.reader = None
    self.writer = None

  async def register(self, run_id: str, pid: int) -> dict[str, Any]:
    return await self.request({"command": "REGISTER", "run_id": run_id, "pid": pid})

  async def unregister(self, run_id: str) -> dict[str, Any]:
    return await self.request({"command": "UNREGISTER", "run_id": run_id})

  @asynccontextmanager
  async def acquire(self, run_id: str) -> AsyncIterator[None]:
    await self.request({"command": "ACQUIRE", "run_id": run_id})
    try:
      yield
    finally:
      await self.request({"command": "RELEASE", "run_id": run_id})

  async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
    await self.connect()
    assert self.reader is not None
    assert self.writer is not None

    self.writer.write(json.dumps(payload).encode("utf-8") + b"\n")
    await self.writer.drain()
    line = await self.reader.readline()
    if not line:
      raise RuntimeError("snapshot agent connection closed")

    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
      raise RuntimeError(response.get("error", "snapshot agent command failed"))
    return response
