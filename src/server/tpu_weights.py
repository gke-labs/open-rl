"""Transposed linear weights on the TPU vLLM platform.

Once loaded, the TPU platform stores each linear layer's weight transposed and
marks the layer with _tpu_weight_flipped. A full reload rebuilds the weight in
checkpoint layout, and the platform transposes it again only if the flag is
clear, setting the flag once it has.
"""

import torch

FLIPPED_ATTR = "_tpu_weight_flipped"


def flipped_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
  """The layers holding transposed weights. None on GPU."""
  return [module for module in model.modules() if getattr(module, FLIPPED_ATTR, False)]


def clear_flipped(modules: list[torch.nn.Module]) -> None:
  for module in modules:
    setattr(module, FLIPPED_ATTR, False)


def check_flipped_again(modules: list[torch.nn.Module], path: str) -> None:
  """Fail if a reload left any of these layers untransposed."""
  missed = [module for module in modules if not getattr(module, FLIPPED_ATTR, False)]
  if missed:
    raise RuntimeError(f"{len(missed)} of {len(modules)} TPU linear layers were not reloaded from {path}")
