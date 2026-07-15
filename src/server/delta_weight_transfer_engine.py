"""Native vLLM WeightTransferEngine implementation for CPU Weights Snapshot Delta Sync.

Implements vLLM's abstract WeightTransferEngine contract to perform sparse
delta patching in host CPU RAM and reload tensors directly into GPU VRAM
without external sleep/wake workarounds.
"""

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
  base_model_path: str = ""


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
    self._cpu_snapshot: dict[str, torch.Tensor] = {}
    self._base_model: str = os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))
    if args and hasattr(args[0], "model"):
      self._base_model = args[0].model

  @staticmethod
  def _get_real_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "is_meta", False) and not getattr(tensor.data, "is_meta", False):
      return tensor
    try:
      from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

      if "." in name:
        mod_name, p_name = name.rsplit(".", 1)
      else:
        mod_name, p_name = "", name
      modules = dict(model.named_modules())
      mod = modules.get(mod_name)
      if mod is not None and mod in LAYERWISE_INFO:
        info = LAYERWISE_INFO[mod]
        if hasattr(info, "kernel_tensors") and info.kernel_tensors is not None:
          params, buffers = info.kernel_tensors
          if p_name in params:
            return params[p_name]
          if p_name in buffers:
            return buffers[p_name]
    except Exception:
      pass
    return tensor

  def _ensure_cpu_snapshot(self, base_model: str, model: torch.nn.Module | None) -> None:
    if self._cpu_snapshot:
      return
    base_model = base_model or self._base_model or os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))
    print(f"[DeltaSnapshotEngine] Initializing CPU weights snapshot for sparse delta patching (base model: '{base_model}')...")

    if base_model:
      start_t = time.perf_counter()
      from vllm.model_executor.model_loader.weight_utils import (
        download_weights_from_hf,
        safetensors_weights_iterator,
      )

      if os.path.isdir(base_model):
        hf_folder = base_model
      else:
        hf_folder = download_weights_from_hf(base_model, cache_dir=None, allow_patterns=["*.safetensors"])

      hf_weights_files = sorted([os.path.join(hf_folder, f) for f in os.listdir(hf_folder) if f.endswith(".safetensors") and "delta" not in f])
      for name, tensor in safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load=False):
        if not name.endswith(".indices") and "delta" not in name:
          self._cpu_snapshot[name] = tensor.pin_memory() if torch.cuda.is_available() else tensor.clone()
      if self._cpu_snapshot:
        elapsed = (time.perf_counter() - start_t) * 1000.0
        print(
          f"[DeltaSnapshotEngine] CPU weights snapshot initialized with {len(self._cpu_snapshot)} "
          f"HuggingFace tensors from base model '{base_model}' via vLLM weight iterator in {elapsed:.2f} ms."
        )
        return

    if model is not None:
      start_t = time.perf_counter()
      for name, param in model.named_parameters():
        real_t = self._get_real_tensor(model, name, param)
        self._cpu_snapshot[name] = real_t.data.cpu().pin_memory() if torch.cuda.is_available() else real_t.data.cpu().clone()
      for name, buf in model.named_buffers():
        real_t = self._get_real_tensor(model, name, buf)
        self._cpu_snapshot[name] = real_t.data.cpu().pin_memory() if torch.cuda.is_available() else real_t.data.cpu().clone()
      elapsed = (time.perf_counter() - start_t) * 1000.0
      print(f"[DeltaSnapshotEngine] CPU weights snapshot initialized with {len(self._cpu_snapshot)} vLLM tensors from model in {elapsed:.2f} ms.")
      return

    raise RuntimeError(f"Failed to initialize CPU weights snapshot: neither base safetensors for '{base_model}' nor model instance available.")

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
    import json

    target_path = update_info.target_weights_path
    if not target_path or not os.path.exists(target_path):
      raise ValueError(f"Target weights path does not exist: {target_path}")

    start_t = time.perf_counter()

    # Check for sparse_delta metadata format
    is_sparse_delta = False
    metadata_path = os.path.join(target_path, "metadata.json") if os.path.isdir(target_path) else ""
    if metadata_path and os.path.exists(metadata_path):
      try:
        with open(metadata_path) as f:
          meta = json.load(f)
        is_sparse_delta = meta.get("format") == "sparse_delta"
      except Exception:
        pass

    if is_sparse_delta:
      delta_file = os.path.join(target_path, "delta.safetensors")
      if not os.path.exists(delta_file):
        raise ValueError(f"Sparse delta metadata present but delta.safetensors not found at: {target_path}")

      from safetensors.torch import load_file

      sparse_delta = load_file(delta_file, device="cpu")
      elapsed_read = (time.perf_counter() - start_t) * 1000.0
      print(f"[DeltaSnapshotEngine] Loaded sparse delta from {delta_file} in {elapsed_read:.2f} ms")

      # Initialize base CPU snapshot if not yet populated
      if not self._cpu_snapshot:
        model = getattr(load_weights, "__self__", None)
        if model is None and getattr(load_weights, "__closure__", None):
          for cell in load_weights.__closure__:
            if hasattr(cell.cell_contents, "named_parameters"):
              model = cell.cell_contents
        base_model = getattr(update_info, "base_model_path", "") or self._base_model
        self._ensure_cpu_snapshot(base_model, model)

      meta_names: list[str] | None = None
      if metadata_path and os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            meta = json.load(f)
          if "layer_names" in meta:
            meta_names = json.loads(meta["layer_names"]) if isinstance(meta["layer_names"], str) else meta["layer_names"]
        except Exception:
          pass

      if meta_names is None:
        raise ValueError("Missing 'layer_names' metadata in sparse delta directory.")

      indices_flat = sparse_delta["delta.indices_flat"].to(torch.int64)
      values_flat = sparse_delta["delta.values_flat"]
      layer_lengths = sparse_delta["delta.layer_lengths"].tolist()

      if len(meta_names) != len(layer_lengths):
        raise ValueError(f"Mismatch between layer_names ({len(meta_names)}) and layer_lengths ({len(layer_lengths)}) in sparse delta.")

      if len(meta_names) == 0 or sum(layer_lengths) == 0 or indices_flat.numel() == 0:
        self.current_weights_path = target_path
        print("[DeltaSnapshotEngine] Verified patch: 0 tensors changed (NO-OP PATCH DETECTED - Skipping GPU reload)")
        return

      t0_apply = time.perf_counter()
      split_indices = torch.split(indices_flat, layer_lengths)
      split_values = torch.split(values_flat, layer_lengths)

      for i, name in enumerate(meta_names):
        if name not in self._cpu_snapshot:
          raise KeyError(f"Parameter '{name}' found in sparse delta but missing from CPU snapshot.")
        snap_flat = self._cpu_snapshot[name].view(-1)
        snap_flat[split_indices[i]] = split_values[i]

      t_apply_ms = (time.perf_counter() - t0_apply) * 1000.0
      print(
        f"[DeltaSnapshotEngine] Applied packed 1D sparse delta ({len(meta_names)} layers, "
        f"{indices_flat.numel()} total elements) to CPU snapshot in {t_apply_ms:.2f} ms"
      )

      weights_to_load = list(self._cpu_snapshot.items())
    else:
      # Full snapshot safetensors path
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
      print(f"[DeltaSnapshotEngine] Loaded {len(weights)} parameter tensors from {target_path} in {elapsed_read:.2f} ms")

      changed_weights = []
      no_op_tensors = 0
      for name, incoming_tensor in weights:
        if (
          name in self._cpu_snapshot
          and self._cpu_snapshot[name].shape == incoming_tensor.shape
          and self._cpu_snapshot[name].dtype == incoming_tensor.dtype
          and torch.equal(self._cpu_snapshot[name], incoming_tensor)
        ):
          no_op_tensors += 1
        else:
          self._cpu_snapshot[name] = incoming_tensor
          changed_weights.append((name, incoming_tensor))

      if len(changed_weights) == 0 and len(weights) > 0:
        self.current_weights_path = target_path
        print(f"[DeltaSnapshotEngine] Verified patch: 0/{len(weights)} tensors changed (NO-OP PATCH DETECTED - Skipping GPU reload)")
        return

      print(f"[DeltaSnapshotEngine] Verified patch: {len(changed_weights)}/{len(weights)} tensors changed ({no_op_tensors} no-op tensors skipped)")
      weights_to_load = changed_weights

    # Feed genuinely changed parameter tensors directly into vLLM's internal layer loader
    start_load = time.perf_counter()
    load_weights(weights_to_load)
    elapsed_load = (time.perf_counter() - start_load) * 1000.0
    self.current_weights_path = target_path
    print(f"[DeltaSnapshotEngine] Incremental load_weights completed ({len(weights_to_load)} tensors) in {elapsed_load:.2f} ms")

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


try:
  from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

  WeightTransferEngineFactory.register_engine(
    "delta_snapshot",
    DeltaSnapshotWeightTransferEngine,
  )
except (ImportError, ValueError):
  pass
