"""Small torch.distributed boundary for trainer workers launched by torchrun."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def world_size() -> int:
  return int(os.getenv("WORLD_SIZE", "1"))


def rank() -> int:
  return int(os.getenv("RANK", "0"))


def local_rank() -> int:
  return int(os.getenv("LOCAL_RANK", "0"))


def is_distributed() -> bool:
  return world_size() > 1


def is_primary() -> bool:
  return rank() == 0


def initialize() -> None:
  """Join the process group torchrun created and select this rank's GPU."""
  if not is_distributed() or dist.is_initialized():
    return
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", timeout=timedelta(minutes=30))
  else:
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))


def close() -> None:
  if dist.is_initialized():
    dist.destroy_process_group()


def pin_executor_threads() -> None:
  """Give the event loop's executor threads this rank's CUDA device.

  Torch's current device is thread-local and every worker call is handed to a
  thread with asyncio.to_thread. Left alone, a thread the pool spawns later
  still points at cuda:0, and a device-less allocation on it lands there:
  correct on rank 0, wrong on every other rank, surfacing as a NCCL fault.
  """
  if not torch.cuda.is_available() or not is_distributed():
    return
  device = local_rank()
  torch.cuda.set_device(device)
  asyncio.get_running_loop().set_default_executor(
    ThreadPoolExecutor(thread_name_prefix="trainer-worker", initializer=torch.cuda.set_device, initargs=(device,))
  )


def barrier() -> None:
  if is_distributed():
    dist.barrier()


def broadcast_object(value: Any = None) -> Any:
  """Rank 0's value on every rank."""
  if not is_distributed():
    return value
  values = [value if is_primary() else None]
  dist.broadcast_object_list(values, src=0)
  return values[0]


# Reductions over one process group. None stands for a process training alone.


def group_rank(group: dist.ProcessGroup | None) -> int:
  return 0 if group is None else dist.get_rank(group)


def group_size(group: dist.ProcessGroup | None) -> int:
  return 1 if group is None else dist.get_world_size(group)


def all_reduce(value: float, op: dist.ReduceOp, group: dist.ProcessGroup | None) -> float:
  if group is None:
    return value
  device = torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")
  tensor = torch.tensor([value], dtype=torch.float64, device=device)
  dist.all_reduce(tensor, op=op, group=group)
  return float(tensor.item())


def all_reduce_sum(value: float, group: dist.ProcessGroup | None) -> float:
  return all_reduce(value, dist.ReduceOp.SUM, group)


def all_reduce_max(value: float, group: dist.ProcessGroup | None) -> float:
  return all_reduce(value, dist.ReduceOp.MAX, group)


def all_gather_object(value: Any, group: dist.ProcessGroup | None) -> list[Any]:
  if group is None:
    return [value]
  values: list[Any] = [None] * dist.get_world_size(group)
  dist.all_gather_object(values, value, group=group)
  return values
