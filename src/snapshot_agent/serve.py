import argparse
import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointRestorer, CudaCheckpointRestorer

logger = logging.getLogger(__name__)


@dataclass
class ProcessRegistration:
  pid: int
  connection_id: int | None
  checkpointed: bool = False
  failed: bool = False


class SnapshotAgent:
  def __init__(self, restorer: CheckpointRestorer):
    self.restorer = restorer
    self.processes: dict[str, ProcessRegistration] = {}
    self.waiting_run_ids: deque[str] = deque()
    self.active_run_id: str | None = None
    self.condition = asyncio.Condition()

  def clear_run(self, run_id: str) -> None:
    if run_id in self.waiting_run_ids:
      self.waiting_run_ids.remove(run_id)
    if self.active_run_id == run_id:
      self.active_run_id = None

  async def register(self, run_id: str, pid: int, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(run_id)

      if process is not None:
        return {"ok": False, "error": f"run '{run_id}' is already registered"}

      self.processes[run_id] = ProcessRegistration(pid=pid, connection_id=connection_id)
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(run_id)
      if process is None:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}
      if process.failed:
        return {"ok": False, "error": f"run '{run_id}' is failed"}
      if run_id in self.waiting_run_ids or self.active_run_id == run_id:
        return {"ok": False, "error": f"run '{run_id}' already has a pending or active acquire"}

      self.waiting_run_ids.append(run_id)
      try:
        while self.active_run_id is not None or (run_id in self.waiting_run_ids and self.waiting_run_ids[0] != run_id):
          await self.condition.wait()
      except BaseException:
        if run_id in self.waiting_run_ids:
          self.waiting_run_ids.remove(run_id)
        self.condition.notify_all()
        raise

      process = self.processes.get(run_id)
      if process is None or process.failed or run_id not in self.waiting_run_ids:
        self.clear_run(run_id)
        self.condition.notify_all()
        return {"ok": False, "error": f"run '{run_id}' is not available"}

      self.waiting_run_ids.popleft()
      self.active_run_id = run_id
      if process.checkpointed:
        await self.run_restore(process.pid)
        process.checkpointed = False

      self.condition.notify_all()
      return {"ok": True}

  async def release(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      process = self.processes.get(run_id)
      if process is None:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}
      if self.active_run_id != run_id:
        return {"ok": False, "error": f"run '{run_id}' does not hold an active acquire"}

      await self.run_checkpoint(process.pid)
      process.checkpointed = True
      self.clear_run(run_id)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      if run_id not in self.processes:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}

      self.clear_run(run_id)
      del self.processes[run_id]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for run_id, process in self.processes.items():
        if process.connection_id != connection_id:
          continue
        self.clear_run(run_id)
        process.failed = True
        process.checkpointed = False
        process.connection_id = None
      self.condition.notify_all()

  async def run_checkpoint(self, pid: int) -> None:
    try:
      await asyncio.to_thread(self.restorer.checkpoint, pid)
    except Exception:
      logger.critical("checkpoint failed for pid %s; GPU state is unknown, killing snapshot agent", pid, exc_info=True)
      os._exit(1)

  async def run_restore(self, pid: int) -> None:
    try:
      await asyncio.to_thread(self.restorer.restore, pid)
    except Exception:
      logger.critical("restore failed for pid %s; GPU state is unknown, killing snapshot agent", pid, exc_info=True)
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
  run_id = payload.get("run_id")

  assert run_id is not None, "run_id is required"

  match command:
    case "REGISTER":
      return await agent.register(run_id, payload["pid"], connection_id=connection_id)
    case "ACQUIRE":
      return await agent.acquire(run_id)
    case "RELEASE":
      return await agent.release(run_id)
    case "UNREGISTER":
      return await agent.unregister(run_id)
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
