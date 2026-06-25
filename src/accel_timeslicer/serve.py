import argparse
import asyncio
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

from .checkpoint import CudaCheckpointRestorer
from .llmd import LlmDCheckpointRestorer
from .single_node import SingleNodeTimeSlicer
from .workload import DEFAULT_TIME_SLICE_GROUP, WorkloadRef

try:
  from timeslice.snapshot_agent import SnapshotAgentClient as LlmDClient
except ImportError:
  LlmDClient = None

logger = logging.getLogger(__name__)


async def start_time_slicer(time_slicer: SingleNodeTimeSlicer, socket_path: str) -> asyncio.Server:
  socket = Path(socket_path)
  socket.parent.mkdir(parents=True, exist_ok=True)
  socket.unlink(missing_ok=True)

  return await asyncio.start_unix_server(partial(handle_connection, time_slicer), path=socket_path)


async def start_tcp_time_slicer(time_slicer: SingleNodeTimeSlicer, host: str, port: int) -> asyncio.Server:
  return await asyncio.start_server(partial(handle_connection, time_slicer), host=host, port=port)


async def handle_connection(time_slicer: SingleNodeTimeSlicer, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
  connection_id = id(writer)
  try:
    while line := await reader.readline():
      response = await dispatch(time_slicer, line, connection_id)
      writer.write(json.dumps(response).encode("utf-8") + b"\n")
      await writer.drain()
  finally:
    await time_slicer.connection_closed(connection_id)
    writer.close()
    await writer.wait_closed()


async def dispatch(time_slicer: SingleNodeTimeSlicer, line: bytes, connection_id: int) -> dict[str, Any]:
  payload = json.loads(line.decode("utf-8"))

  command = payload.get("command", "").upper()
  workload = workload_from_payload(payload)

  match command:
    case "REGISTER":
      return await time_slicer.register(workload, connection_id=connection_id)
    case "ACQUIRE":
      return await time_slicer.acquire(workload)
    case "RELEASE":
      return await time_slicer.release(workload)
    case "UNREGISTER":
      return await time_slicer.unregister(workload)
    case _:
      return {"ok": False, "error": f"unknown command '{command}'"}


def workload_from_payload(payload: dict[str, Any]) -> WorkloadRef:
  job_id = payload.get("job_id") or payload.get("snapshot_id")
  if job_id is None:
    raise ValueError("workload requires job_id")
  return WorkloadRef(
    job_id=job_id,
    group=payload.get("group") or DEFAULT_TIME_SLICE_GROUP,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run the OpenRL accelerator time-slicer.")
  parser.add_argument("--socket", default=os.getenv("OPEN_RL_ACCEL_TIMESLICER_SOCKET", "/tmp/open-rl/accel-timeslicer.sock"))
  parser.add_argument("--listen-host", default=None)
  parser.add_argument("--port", type=int, default=None)
  parser.add_argument("--backend", choices=["cuda", "llmd"], default=os.getenv("OPEN_RL_ACCEL_TIMESLICER_BACKEND", "cuda"))
  parser.add_argument("--cuda-checkpoint-bin", default=os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint"))
  parser.add_argument("--cuda-checkpoint-timeout-ms", type=int, default=None)
  parser.add_argument("--llmd-snapshot-endpoint", default=os.getenv("LLMD_SNAPSHOT_AGENT_ENDPOINT", "127.0.0.1:9001"))
  parser.add_argument("--llmd-backend", default=os.getenv("LLMD_SNAPSHOT_BACKEND", "CUDA"))
  parser.add_argument("--llmd-poll-interval-sec", type=float, default=float(os.getenv("LLMD_SNAPSHOT_POLL_INTERVAL_SEC", "1.0")))
  return parser.parse_args()


async def main_async() -> None:
  args = parse_args()
  if args.backend == "llmd":
    if LlmDClient is None:
      raise RuntimeError("--backend llmd requires the llm-d timeslice snapshot client package")
    restorer = LlmDCheckpointRestorer(LlmDClient(endpoint=args.llmd_snapshot_endpoint), args.llmd_backend, args.llmd_poll_interval_sec)
    time_slicer = SingleNodeTimeSlicer(restorer=restorer)
  else:
    restorer = CudaCheckpointRestorer(args.cuda_checkpoint_bin, args.cuda_checkpoint_timeout_ms)
    time_slicer = SingleNodeTimeSlicer(restorer=restorer)
  if args.port is None:
    server = await start_time_slicer(time_slicer, args.socket)
    logger.info("listening on unix://%s", args.socket)
  else:
    host = args.listen_host or "0.0.0.0"
    server = await start_tcp_time_slicer(time_slicer, host, args.port)
    logger.info("listening on tcp://%s:%s", host, args.port)
  async with server:
    await server.serve_forever()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="[ACCEL_TIMESLICER] %(levelname)s %(message)s")
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
