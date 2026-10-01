# Process-level torch device selection, shared by the trainer workers and the
# delta weight-transfer engine.

import os

import torch


def resolve_device(import_torch_tpu: bool = True) -> torch.device:
  """The torch device this process should use.

  OPEN_RL_DEVICE picks it explicitly; unset falls back to cuda/mps/cpu
  auto-detection. "tpu" is never auto-detected: an importable torch_tpu wheel
  only means the package is installed, not that this process may claim the
  host's TPU chips.
  """
  override = os.environ.get("OPEN_RL_DEVICE", "").lower() or None
  if override == "tpu" and import_torch_tpu:
    # torch alone cannot construct a "tpu" device: torch_tpu registers the
    # PrivateUse1 backend on import — and only when TORCH_DEVICE_BACKEND_AUTOLOAD
    # is not 0 (worker_manager resets it to 1 for TPU trainers).
    try:
      import torch_tpu  # noqa: F401
    except ImportError as exc:
      raise ImportError(f"OPEN_RL_DEVICE=tpu but this interpreter has no working torch_tpu: {exc}") from exc
  if override:
    try:
      return torch.device(override)
    except RuntimeError:
      if import_torch_tpu:
        raise
      # Callers that must not import torch_tpu (the sampler process reaches
      # the TPU through JAX) treat an unregistered device type as another
      # process's device and fall through to auto-detection.
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")
