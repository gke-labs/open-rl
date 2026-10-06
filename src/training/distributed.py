"""Small torch.distributed boundary for a trainer launched under torchrun."""

import os
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
  """Initialize the process group created by torchrun and select this rank's GPU."""
  if not is_distributed() or dist.is_initialized():
    return
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank())
  dist.init_process_group(
    backend=os.getenv("OPEN_RL_CONTROL_BACKEND", "gloo"),
    timeout=timedelta(seconds=int(os.getenv("OPEN_RL_DISTRIBUTED_TIMEOUT", "1800"))),
  )


def broadcast_object(value: Any = None) -> Any:
  if not is_distributed():
    return value
  values = [value if is_primary() else None]
  dist.broadcast_object_list(values, src=0)
  return values[0]


def close() -> None:
  if dist.is_initialized():
    dist.destroy_process_group()
