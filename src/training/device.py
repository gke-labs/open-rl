# Torch device selection for trainer workers.

import os

import torch


def resolve_device() -> torch.device:
  """Return the device named by OPEN_RL_DEVICE, else the first available of cuda, mps, cpu.

  tpu is never auto-detected: torch only knows the "tpu" device type once
  torch_tpu is imported, and torch_tpu has no availability check.
  """
  name = os.environ.get("OPEN_RL_DEVICE", "").strip().lower()
  if name == "tpu":
    try:
      import torch_tpu  # noqa: F401
    except ImportError as exc:
      raise ImportError(f"OPEN_RL_DEVICE=tpu needs the torch_tpu package: {exc}") from exc
  if name:
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")
