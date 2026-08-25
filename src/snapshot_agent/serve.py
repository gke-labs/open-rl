import argparse
import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointRestorer, CudaCheckpointRestorer

logger = logging.getLogger(__name__)


@dataclass
class ProcessRegistration:
  connection_id: int | None
  checkpointed: bool = False
  failed: bool = False


class SnapshotAgent:
  def __init__(self, restorer: CheckpointRestorer):
    self.restorer = restorer
    self.processes: dict[int, ProcessRegistration] = {}
    self.waiting_pids: deque[int] = deque()
    self.active_pid: int | None = None
    self.condition = asyncio.Condition()

  def clear_process(self, pid: int) -> None:
    if pid in self.waiting_pids:
      self.waiting_pids.remove(pid)
    if self.active_pid == pid:
      self.active_pid = None

  async def register(self, pid: int, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(pid)

      if process is not None:
        return {"ok": False, "error": f"pid {pid} is already registered"}

      self.processes[pid] = ProcessRegistration(connection_id=connection_id)
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, pid: int) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(pid)
      if process is None:
        return {"ok": False, "error": f"pid {pid} is not registered"}
      if process.failed:
        return {"ok": False, "error": f"pid {pid} is failed"}
      if pid in self.waiting_pids or self.active_pid == pid:
        return {"ok": False, "error": f"pid {pid} already has a pending or active acquire"}

      self.waiting_pids.append(pid)
      try:
        while self.active_pid is not None or (pid in self.waiting_pids and self.waiting_pids[0] != pid):
          await self.condition.wait()
      except BaseException:
        if pid in self.waiting_pids:
          self.waiting_pids.remove(pid)
        self.condition.notify_all()
        raise

      process = self.processes.get(pid)
      if process is None or process.failed or pid not in self.waiting_pids:
        self.clear_process(pid)
        self.condition.notify_all()
        return {"ok": False, "error": f"pid {pid} is not available"}

      self.waiting_pids.popleft()
      self.active_pid = pid
      if process.checkpointed:
        await self.run_restore(pid)
        process.checkpointed = False

      self.condition.notify_all()
      return {"ok": True}

  async def release(self, pid: int) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(pid)
      if process is None:
        return {"ok": False, "error": f"pid {pid} is not registered"}
      if self.active_pid != pid:
        return {"ok": False, "error": f"pid {pid} does not hold an active acquire"}

      await self.run_checkpoint(pid)
      process.checkpointed = True
      self.clear_process(pid)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, pid: int) -> dict[str, Any]:
    async with self.condition:
      if pid not in self.processes:
        return {"ok": False, "error": f"pid {pid} is not registered"}

      self.clear_process(pid)
      del self.processes[pid]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for pid, process in self.processes.items():
        if process.connection_id != connection_id:
          continue
        self.clear_process(pid)
        process.failed = True
        process.checkpointed = False
        process.connection_id = None
      self.condition.notify_all()

  async def run_checkpoint(self, pid: int) -> None:
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.checkpoint, pid)
      logger.info("checkpointed pid %s in %.2fs", pid, time.monotonic() - start)
    except Exception:
      logger.critical(
        "checkpoint failed for pid %s after %.2fs; GPU state is unknown, killing snapshot agent", pid, time.monotonic() - start, exc_info=True
      )
      os._exit(1)

  async def run_restore(self, pid: int) -> None:
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.restore, pid)
      logger.info("restored pid %s in %.2fs", pid, time.monotonic() - start)
    except Exception:
      logger.critical(
        "restore failed for pid %s after %.2fs; GPU state is unknown, killing snapshot agent", pid, time.monotonic() - start, exc_info=True
      )
      os._exit(1)


async def start_snapshot_agent(agent: SnapshotAgent, socket_path: str) -> asyncio.Server:
  socket = Path(socket_path)
  socket.parent.mkdir(parents=True, exist_ok=True)
  socket.unlink(missing_ok=True)

  async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    connection_id = id(writer)
    try:
      while line := await reader.readline():
        response = await dispatch(agent, line, connection_id)
        writer.write(json.dumps(response).encode("utf-8") + b"\n")
        await writer.drain()
    finally:
      await agent.connection_closed(connection_id)
      writer.close()
      await writer.wait_closed()

  return await asyncio.start_unix_server(handle_connection, path=socket_path)


async def dispatch(agent: SnapshotAgent, line: bytes, connection_id: int) -> dict[str, Any]:
  payload = json.loads(line.decode("utf-8"))

  command = payload.get("command", "").upper()
  pid = payload.get("pid")

  assert pid is not None, "pid is required"
  pid = int(pid)

  match command:
    case "REGISTER":
      return await agent.register(pid, connection_id=connection_id)
    case "ACQUIRE":
      return await agent.acquire(pid)
    case "RELEASE":
      return await agent.release(pid)
    case "UNREGISTER":
      return await agent.unregister(pid)
    case _:
      return {"ok": False, "error": f"unknown command '{command}'"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run the OpenRL snapshot agent.")
  parser.add_argument("--socket", default=os.getenv("OPEN_RL_SNAPSHOT_AGENT_SOCKET", "/tmp/open-rl/snapshot-agent.sock"))
  parser.add_argument("--cuda-checkpoint-bin", default=os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint"))
  parser.add_argument("--cuda-checkpoint-timeout-ms", type=int, default=None)
  return parser.parse_args()


async def main_async() -> None:
  args = parse_args()
  restorer = CudaCheckpointRestorer(args.cuda_checkpoint_bin, args.cuda_checkpoint_timeout_ms)
  agent = SnapshotAgent(restorer)
  server = await start_snapshot_agent(agent, args.socket)
  logger.info("listening on %s", args.socket)
  async with server:
    await server.serve_forever()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="[SNAPSHOT_AGENT] %(levelname)s %(message)s")
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
