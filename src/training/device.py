# Torch device selection for trainer workers.

import os

import torch


def resolve_device() -> torch.device:
  """Return the device named by OPEN_RL_DEVICE, else the first available of cuda, mps, cpu.

  tpu is used only when asked for. Importing torch_tpu registers the "tpu"
  device type, but only on a host with TPU chips and when
  TORCH_DEVICE_BACKEND_AUTOLOAD is not 0.
  """
  name = os.environ.get("OPEN_RL_DEVICE", "").strip().lower()
  if name.partition(":")[0] == "tpu":
    try:
      import torch_tpu  # noqa: F401
    except ImportError as exc:
      raise ImportError(f"OPEN_RL_DEVICE={name} needs the torch_tpu package: {exc}") from exc
    try:
      return torch.device(name)
    except RuntimeError as exc:
      raise RuntimeError("torch_tpu did not register the tpu device: no TPU chips found, or TORCH_DEVICE_BACKEND_AUTOLOAD=0") from exc
  if name:
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")
