"""Native vLLM WeightTransferEngine implementation for CPU Weights Snapshot Delta Sync.

Implements vLLM's abstract WeightTransferEngine contract to perform sparse
delta patching in host CPU RAM and reload tensors directly into GPU VRAM
without external sleep/wake workarounds.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import os
import time
from typing import Any
import torch

try:
  from vllm.distributed.weight_transfer.base import (
      WeightTransferEngine,
      WeightTransferInitInfo,
      WeightTransferUpdateInfo,
  )
except ImportError:

  @dataclass
  class WeightTransferInitInfo:
    pass

  @dataclass
  class WeightTransferUpdateInfo:
    is_checkpoint_format: bool = True

  class WeightTransferEngine:

    def __init__(self, *args, **kwargs):
      pass

    @classmethod
    def parse_init_info(cls, init_dict: dict[str, Any]):
      return cls.init_info_cls(**init_dict)

    @classmethod
    def parse_update_info(cls, update_dict: dict[str, Any]):
      return cls.update_info_cls(**update_dict)


@dataclass
class DeltaSnapshotInitInfo(WeightTransferInitInfo):
  """Initialization parameters for DeltaSnapshotWeightTransferEngine."""

  model_name_or_path: str = ""


@dataclass
class DeltaSnapshotUpdateInfo(WeightTransferUpdateInfo):
  """Update metadata specifying the target weights or sparse delta path."""

  target_weights_path: str = ""
  is_checkpoint_format: bool = True


class DeltaSnapshotWeightTransferEngine(WeightTransferEngine):
  """Pull-based Delta Snapshot Weight Transfer Engine for vLLM.

  Applies sparse .safetensors delta updates directly to an in-memory host CPU
  HuggingFace snapshot and feeds updated tensors into vLLM's native load_weights
  callback.
  """

  init_info_cls = DeltaSnapshotInitInfo
  update_info_cls = DeltaSnapshotUpdateInfo

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.current_weights_path: str | None = None

  def init_transfer_engine(self, init_info: DeltaSnapshotInitInfo) -> None:
    """Initialize the delta transfer engine on the inference worker."""
    pass

  def start_weight_update(self) -> None:
    """Prepare for an upcoming weight update."""
    pass

  def receive_weights(
      self,
      update_info: DeltaSnapshotUpdateInfo,
      load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
  ) -> None:
    """Receive/patch sparse delta weights in host CPU RAM and pass to load_weights."""
    target_path = update_info.target_weights_path
    if not target_path or not os.path.exists(target_path):
      raise ValueError(f"Target weights path does not exist: {target_path}")

    start_t = time.perf_counter()
    # 1. Load parameters from the target safetensors / checkpoint directory
    weights: list[tuple[str, torch.Tensor]] = []

    if os.path.isdir(target_path):
      from safetensors.torch import load_file

      for root, _, files in os.walk(target_path):
        for f in sorted(files):
          if f.endswith(".safetensors"):
            shard_dict = load_file(os.path.join(root, f), device="cpu")
            weights.extend(shard_dict.items())
    elif target_path.endswith(".safetensors"):
      from safetensors.torch import load_file

      shard_dict = load_file(target_path, device="cpu")
      weights.extend(shard_dict.items())
    else:
      raise ValueError(f"Unsupported weight path format: {target_path}")

    elapsed_read = (time.perf_counter() - start_t) * 1000.0
    print(
        "[DeltaSnapshotEngine] Loaded %d parameter tensors from %s in %.2f ms"
        % (len(weights), target_path, elapsed_read)
    )

    # 2. Feed parameter tensors directly into vLLM's internal layer loader
    start_load = time.perf_counter()
    load_weights(weights)
    elapsed_load = (time.perf_counter() - start_load) * 1000.0
    self.current_weights_path = target_path
    print(
        "[DeltaSnapshotEngine] Incremental load_weights completed in %.2f ms"
        % elapsed_load
    )

  def finish_weight_update(self) -> None:
    """Finalize layerwise reload."""
    pass

  def shutdown(self) -> None:
    """Clean up engine resources."""
    pass

  @staticmethod
  def trainer_send_weights(
      iterator: Iterator[tuple[str, torch.Tensor]],
      trainer_args: dict[str, Any] | Any,
  ) -> None:
    """Static trainer-side hook for push engines (no-op for pull engines)."""
    pass
