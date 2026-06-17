import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

DEFAULT_SOCKET_PATH = "/tmp/open-rl/snapshot-agent.sock"
DEFAULT_TCP_PORT = 9753


class SnapshotClient(Protocol):
  async def register(self, pid: int) -> dict[str, Any]: ...

  async def unregister(self, pid: int) -> dict[str, Any]: ...

  def acquire(self, pid: int) -> AbstractAsyncContextManager[None]: ...

  async def close(self) -> None: ...


class SnapshotAgentClient:
  def __init__(self, socket_path: str | None = None, host: str | None = None, port: int = DEFAULT_TCP_PORT):
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

  async def register(self, pid: int) -> dict[str, Any]:
    return await self.request({"command": "REGISTER", "pid": pid})

  async def unregister(self, pid: int) -> dict[str, Any]:
    return await self.request({"command": "UNREGISTER", "pid": pid})

  @asynccontextmanager
  async def acquire(self, pid: int) -> AsyncIterator[None]:
    await self.request({"command": "ACQUIRE", "pid": pid})
    try:
      yield
    finally:
      await self.request({"command": "RELEASE", "pid": pid})

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


class NoopSnapshotAgentClient:
  """Snapshot-agent-compatible client for unsafe oversubscription experiments."""

  async def close(self) -> None:
    pass

  async def register(self, pid: int) -> dict[str, Any]:
    return {"ok": True, "pid": pid}

  async def unregister(self, pid: int) -> dict[str, Any]:
    return {"ok": True, "pid": pid}

  @asynccontextmanager
  async def acquire(self, pid: int) -> AsyncIterator[None]:
    yield


def snapshot_client_from_env() -> SnapshotClient:
  if os.getenv("OPEN_RL_SNAPSHOT_AGENT_MODE", "").lower() == "noop":
    return NoopSnapshotAgentClient()

  host = os.getenv("OPEN_RL_SNAPSHOT_AGENT_HOST")
  if host:
    return SnapshotAgentClient(host=host, port=int(os.getenv("OPEN_RL_SNAPSHOT_AGENT_PORT", str(DEFAULT_TCP_PORT))))

  return SnapshotAgentClient(socket_path=os.getenv("OPEN_RL_SNAPSHOT_AGENT_SOCKET", DEFAULT_SOCKET_PATH))
