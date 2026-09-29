"""vLLM 0.29 weight-transfer plugin: apply sparse patches or a full checkpoint from disk while generation is paused."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.distributed.weight_transfer.base import WeightTransferEngine, WeightTransferInitInfo, WeightTransferUpdateInfo
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.model_executor.model_loader.checkpoint_weight_patch import CheckpointWeightPatch, load_checkpoint_weight_patches
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

from server.model_metadata import SPARSE_DELTA_VERSION

logger = logging.getLogger(__name__)


def read_weight_metadata(path: Path) -> dict:
  metadata_path = path / "metadata.json"
  if path.is_dir() and metadata_path.exists():
    with metadata_path.open() as stream:
      metadata = json.load(stream)
    if not isinstance(metadata, dict):
      raise ValueError("Weight metadata must be an object")
    return metadata
  return {}


def read_sparse_patches(path: Path, metadata: dict, device: torch.device | str = "cpu") -> list[CheckpointWeightPatch]:
  """Read and validate a whole sparse delta before any model writes; reject old fused coordinates.

  Only indices and values move to the device. Staging on CPU would make the
  native loader copy full checkpoint tensors over PCIe, losing the sparse savings.
  """
  if metadata.get("format_version") != SPARSE_DELTA_VERSION:
    raise ValueError("Sparse weights require format_version=2 (native checkpoint coordinates); regenerate legacy deltas with the upgraded trainer.")
  names = metadata.get("layer_names")
  shapes = metadata.get("layer_shapes")
  if not isinstance(names, list) or not isinstance(shapes, list) or len(names) != len(shapes):
    raise ValueError("Sparse layer_names and layer_shapes must be matching lists")
  if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
    raise ValueError("Sparse layer_names must be unique non-empty strings")
  tensors = load_file(str(path / "delta.safetensors"), device="cpu")
  expected = {f"{i}.{kind}" for i in range(len(names)) for kind in ("indices", "values")}
  if set(tensors) != expected:
    raise ValueError("Sparse tensor keys do not match the declared layers")
  patches = []
  for i, (name, shape) in enumerate(zip(names, shapes, strict=True)):
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
      raise ValueError(f"Invalid checkpoint shape for {name}")
    indices, values = tensors[f"{i}.indices"], tensors[f"{i}.values"]
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
      raise ValueError(f"Sparse indices must be a 1D integer tensor: {name}")
    if indices.numel() > 1 and torch.any(indices[1:] <= indices[:-1]).item():
      raise ValueError(f"Sparse indices must be strictly increasing: {name}")
    if values.ndim != 1 or not values.is_floating_point() or indices.numel() != values.numel():
      raise ValueError(f"Sparse values must be a matching 1D floating tensor: {name}")
    if indices.numel():
      patches.append(CheckpointWeightPatch(name, tuple(shape), values.dtype, values.to(device), indices.to(device)))
  return patches


@dataclass
class DeltaSnapshotUpdateInfo(WeightTransferUpdateInfo):
  target_weights_path: str


class DeltaSnapshotWeightTransferEngine(WeightTransferEngine):
  """Pull checkpoint files while the sampler has generation paused.

  The registry name stays delta_snapshot for deployment compatibility. The
  sampler process tracks the loaded version and refuses to serve after a failure.
  """

  init_info_cls = WeightTransferInitInfo
  update_info_cls = DeltaSnapshotUpdateInfo
  supports_draft_weight_update = False

  def init_transfer_engine(self, init_info: WeightTransferInitInfo) -> None:
    pass

  def start_weight_update(self) -> None:
    pass

  def finish_weight_update(self) -> None:
    pass

  def shutdown(self) -> None:
    pass

  def receive_weights(self, update_info: DeltaSnapshotUpdateInfo) -> None:
    path = Path(update_info.target_weights_path)
    if not path.exists():
      raise ValueError(f"Target weights path does not exist: {path}")
    metadata = read_weight_metadata(path)
    if metadata.get("format") == "sparse_delta":
      patches = read_sparse_patches(path, metadata, self.device)
      # The reader verified sorted, unique indices on CPU; skip re-sorting on GPU.
      load_checkpoint_weight_patches(self.model, patches, validate_unique_indices=False)
      logger.info("Applied %d checkpoint-coordinate sparse patches from %s", len(patches), path)
    else:
      files = sorted(str(file) for file in path.glob("*.safetensors")) if path.is_dir() else [str(path)]
      if not files or any(not file.endswith(".safetensors") or Path(file).name == "delta.safetensors" for file in files):
        raise ValueError(f"No full checkpoint safetensors found at {path}")
      # Dense checkpoints need layerwise post-processing; sparse patches must bypass it.
      initialize_layerwise_reload(self.model)
      self.model.load_weights(safetensors_weights_iterator(files, use_tqdm_on_load=False))
      finalize_layerwise_reload(self.model, self.model_config)
      logger.info("Loaded full checkpoint from %s", path)


def register_delta_weight_transfer() -> None:
  """vLLM general plugin: register in the frontend and spawned worker processes."""
  WeightTransferEngineFactory.register_engine("delta_snapshot", DeltaSnapshotWeightTransferEngine)
