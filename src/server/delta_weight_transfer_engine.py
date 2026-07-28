"""Native vLLM WeightTransferEngine implementation for CPU Weights Snapshot Delta Sync.

Implements vLLM's abstract WeightTransferEngine contract to perform sparse
delta patching in host CPU RAM and reload tensors directly into GPU VRAM
without external sleep/wake workarounds.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load_file

try:
  from vllm.logger import init_logger

  logger = init_logger("vllm.distributed.weight_transfer.delta_snapshot")
except ImportError:
  import logging

  logger = logging.getLogger("vllm.distributed.weight_transfer.delta_snapshot")

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


@dataclass
class SparseWeightPatch:
  """A sparse in-place patch for one existing parameter."""

  name: str
  indices: torch.Tensor
  values: torch.Tensor


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

    # Explicit instance attribute initializations
    self.model: torch.nn.Module | None = kwargs.get("model", getattr(self, "model", None))
    self.vllm_config: Any = kwargs.get("vllm_config", getattr(self, "vllm_config", None))
    self.model_config: Any = kwargs.get(
      "model_config",
      getattr(self, "model_config", getattr(self.vllm_config, "model_config", None)),
    )
    self.device: torch.device | None = kwargs.get(
      "device",
      getattr(
        self,
        "device",
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
      ),
    )
    self.current_weights_path: str | None = None
    self._cpu_snapshot: dict[str, torch.Tensor] = {}
    self._base_model: str = ""

    if self.model_config is not None and getattr(self.model_config, "model", None):
      self._base_model = self.model_config.model
    else:
      self._base_model = os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))

    device_str = str(self.device) if self.device is not None else "unknown"
    model_type = type(self.model).__name__ if self.model is not None else "None"
    tp_size = getattr(getattr(self, "parallel_config", None), "tensor_parallel_size", 1)
    pp_size = getattr(getattr(self, "parallel_config", None), "pipeline_parallel_size", 1)
    logger.info(
      f"[DeltaSnapshotEngine] Initialized DeltaSnapshotWeightTransferEngine | "
      f"base_model='{self._base_model}' | device={device_str} | model={model_type} | "
      f"parallel=(TP={tp_size}, PP={pp_size})"
    )

  @staticmethod
  def _get_real_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "is_meta", False) and not getattr(tensor.data, "is_meta", False):
      return tensor
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
    return tensor

  def _validate_patch(self, patch: SparseWeightPatch, param: torch.Tensor) -> None:
    """Defensive pre-condition guards before executing VRAM mutations."""
    if not param.data.is_contiguous():
      raise NotImplementedError(f"Sparse weight updates require contiguous params: {patch.name}")
    if patch.indices.ndim != 1 or patch.values.ndim != 1:
      raise ValueError(f"Sparse weight patches must be 1D flattened updates: {patch.name}")
    if patch.indices.numel() != patch.values.numel():
      raise ValueError(f"`indices` and `values` must have matching lengths for {patch.name}")
    if patch.values.dtype != param.dtype:
      raise ValueError(f"Sparse values dtype {patch.values.dtype} does not match parameter dtype {param.dtype}")

  def _ensure_cpu_snapshot(self, base_model: str, model: torch.nn.Module | None = None) -> None:
    if self._cpu_snapshot:
      return

    base_model = base_model or self._base_model or os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))
    model = model or self.model
    logger.info(f"[DeltaSnapshotEngine] Initializing CPU weights snapshot for sparse delta patching (base model: '{base_model}')...")

    if model is not None:
      start_t = time.perf_counter()
      for name, param in model.named_parameters():
        real_t = self._get_real_tensor(model, name, param)
        self._cpu_snapshot[name] = real_t.data.cpu().pin_memory() if torch.cuda.is_available() else real_t.data.cpu().clone()
      for name, buf in model.named_buffers():
        real_t = self._get_real_tensor(model, name, buf)
        self._cpu_snapshot[name] = real_t.data.cpu().pin_memory() if torch.cuda.is_available() else real_t.data.cpu().clone()
      elapsed = (time.perf_counter() - start_t) * 1000.0
      logger.info(
        f"[DeltaSnapshotEngine] CPU weights snapshot initialized with {len(self._cpu_snapshot)} vLLM tensors from model in {elapsed:.2f} ms."
      )
      return

    if base_model:
      start_t = time.perf_counter()
      from vllm.model_executor.model_loader.weight_utils import (
        download_weights_from_hf,
        safetensors_weights_iterator,
      )

      if os.path.isdir(base_model):
        hf_folder = base_model
      else:
        try:
          hf_folder = download_weights_from_hf(
            base_model,
            cache_dir=None,
            allow_patterns=["*.safetensors"],
            trust_remote_code=True,
          )
        except TypeError:
          # Fallback if vLLM download_weights_from_hf does not accept trust_remote_code
          hf_folder = download_weights_from_hf(
            base_model,
            cache_dir=None,
            allow_patterns=["*.safetensors"],
          )

      hf_weights_files = sorted([os.path.join(hf_folder, f) for f in os.listdir(hf_folder) if f.endswith(".safetensors") and "delta" not in f])
      for name, tensor in safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load=False):
        if not name.endswith(".indices") and "delta" not in name:
          self._cpu_snapshot[name] = tensor.pin_memory() if torch.cuda.is_available() else tensor.clone()
      if self._cpu_snapshot:
        elapsed = (time.perf_counter() - start_t) * 1000.0
        logger.info(
          f"[DeltaSnapshotEngine] CPU weights snapshot initialized with {len(self._cpu_snapshot)} "
          f"HuggingFace tensors from base model '{base_model}' via vLLM weight iterator in {elapsed:.2f} ms."
        )
        return

    raise RuntimeError(f"Failed to initialize CPU weights snapshot: neither base safetensors for '{base_model}' nor model instance available.")

  def init_transfer_engine(self, init_info: DeltaSnapshotInitInfo) -> None:
    """Initialize the delta transfer engine on the inference worker."""
    pass

  def start_weight_update(self) -> None:
    """No-op: sparse patches are applied in place; no dense layerwise reload required."""
    pass

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

  def _resolve_gpu_param_and_offset(self, hf_name: str) -> tuple[torch.Tensor, int]:
    """Resolves a HuggingFace parameter name to (gpu_param, 1d_element_offset) on self.model."""
    if self.model is None:
      raise RuntimeError("[InPlaceGPU] self.model is not available for in-place GPU parameter resolution.")

    # 1. Direct match on self.model
    try:
      param = self.model.get_parameter(hf_name)
      return param, 0
    except (AttributeError, KeyError):
      pass

    # Extract model config for dimension calculation
    hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
    if hf_config is not None and hasattr(hf_config, "get_text_config"):
      hf_config = hf_config.get_text_config()

    hidden_size = getattr(hf_config, "hidden_size", None)
    num_heads = getattr(hf_config, "num_attention_heads", None)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None and hidden_size is not None and num_heads is not None:
      head_dim = hidden_size // num_heads

    intermediate_size = getattr(hf_config, "intermediate_size", None)

    # 2. Packed QKV attention mapping (q_proj, k_proj, v_proj -> qkv_proj)
    if ".q_proj." in hf_name or ".k_proj." in hf_name or ".v_proj." in hf_name:
      qkv_name = hf_name.replace(".q_proj.", ".qkv_proj.").replace(".k_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
      try:
        qkv_param = self.model.get_parameter(qkv_name)
        if hidden_size is not None and num_heads is not None and head_dim is not None:
          q_numel = num_heads * head_dim * hidden_size
          k_numel = num_kv_heads * head_dim * hidden_size
          if ".q_proj." in hf_name:
            return qkv_param, 0
          elif ".k_proj." in hf_name:
            return qkv_param, q_numel
          elif ".v_proj." in hf_name:
            return qkv_param, q_numel + k_numel
      except (AttributeError, KeyError):
        pass

    # 3. Packed MLP mapping (gate_proj, up_proj -> gate_up_proj)
    if ".gate_proj." in hf_name or ".up_proj." in hf_name:
      gate_up_name = hf_name.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
      try:
        gate_up_param = self.model.get_parameter(gate_up_name)
        if hidden_size is not None and intermediate_size is not None:
          gate_numel = intermediate_size * hidden_size
          if ".gate_proj." in hf_name:
            return gate_up_param, 0
          elif ".up_proj." in hf_name:
            return gate_up_param, gate_numel
      except (AttributeError, KeyError):
        pass

    raise KeyError(f"[InPlaceGPU] Unable to resolve HuggingFace parameter name '{hf_name}' to a GPU parameter on model {type(self.model).__name__}.")

  def _build_bulk_tensor_slices(
    self,
    resolved_ops: list[tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]],
    changed_elements: int,
    param_dtype: torch.dtype,
  ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, int, int]]]:
    """Allocates flat bulk 1D CPU index & value tensors and computes GPU parameter slice offsets (DRY helper)."""
    bulk_indices_cpu = torch.empty(changed_elements, dtype=torch.long)
    bulk_values_cpu = torch.empty(changed_elements, dtype=param_dtype)

    curr_offset = 0
    op_slices: list[tuple[torch.Tensor, int, int]] = []
    for gpu_param, offset, idx_cpu, val_cpu in resolved_ops:
      n = idx_cpu.numel()
      end_offset = curr_offset + n
      bulk_indices_cpu[curr_offset:end_offset] = idx_cpu.to(dtype=torch.long) + offset
      bulk_values_cpu[curr_offset:end_offset] = val_cpu.to(dtype=param_dtype)
      op_slices.append((gpu_param, curr_offset, end_offset))
      curr_offset = end_offset

    return bulk_indices_cpu, bulk_values_cpu, op_slices

  def _apply_gpu_in_place(
    self,
    meta_names: list[str],
    split_indices: tuple[torch.Tensor, ...] | list[torch.Tensor],
    split_values: tuple[torch.Tensor, ...] | list[torch.Tensor],
    target_path: str,
  ) -> None:
    """Applies sparse 1D patches directly to GPU parameters in-place in VRAM without CPU snapshot or load_weights."""
    t0_start = time.perf_counter()
    changed_elements = sum(idx.numel() for idx in split_indices)
    logger.info(
      f"[DeltaSnapshotEngine] [IN_PLACE_GPU] Starting direct in-place GPU weight patch across "
      f"{len(meta_names)} layers ({changed_elements} total changed elements)..."
    )

    if changed_elements == 0:
      self.current_weights_path = target_path
      return

    t0_resolve = time.perf_counter()
    resolved_ops = []
    for i, name in enumerate(meta_names):
      if split_indices[i].numel() == 0:
        continue
      gpu_param, offset = self._resolve_gpu_param_and_offset(name)
      resolved_ops.append((gpu_param, offset, split_indices[i], split_values[i]))

    t_resolve_ms = (time.perf_counter() - t0_resolve) * 1000.0

    t0_copy = time.perf_counter()
    if resolved_ops:
      target_device = self.device or resolved_ops[0][0].device
      param_dtype = resolved_ops[0][0].dtype

      bulk_indices_cpu, bulk_values_cpu, op_slices = self._build_bulk_tensor_slices(resolved_ops, changed_elements, param_dtype)

      # Pin host CPU memory for fast PCIe DMA transfer if CUDA is available
      if torch.cuda.is_available() and target_device.type == "cuda":
        bulk_indices_cpu = bulk_indices_cpu.pin_memory()
        bulk_values_cpu = bulk_values_cpu.pin_memory()

      # Single bulk Host-to-Device transfer across PCIe
      bulk_indices_gpu = bulk_indices_cpu.to(device=target_device, non_blocking=True)
      bulk_values_gpu = bulk_values_cpu.to(device=target_device, non_blocking=True)

      # Mutate VRAM in-place using slices from the bulk GPU tensors
      for gpu_param, start_idx, end_idx in op_slices:
        flat_param = gpu_param.data.view(-1)
        idx_slice = bulk_indices_gpu[start_idx:end_idx]
        val_slice = bulk_values_gpu[start_idx:end_idx]
        patch = SparseWeightPatch(name=str(gpu_param), indices=idx_slice, values=val_slice)
        self._validate_patch(patch, flat_param)
        flat_param.index_copy_(0, idx_slice, val_slice)

      if torch.cuda.is_available() and target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    t_copy_ms = (time.perf_counter() - t0_copy) * 1000.0
    t_total_ms = (time.perf_counter() - t0_start) * 1000.0

    self.current_weights_path = target_path
    logger.info(
      f"[DeltaSnapshotEngine] [IN_PLACE_GPU] Successfully applied in-place GPU patch ({len(meta_names)} layers, "
      f"{changed_elements} elements mutated directly in VRAM) in {t_total_ms:.2f} ms "
      f"(Resolve: {t_resolve_ms:.2f} ms, GPU Copy: {t_copy_ms:.2f} ms)."
    )

  def _validate_receive_args(
    self,
    update_info: DeltaSnapshotUpdateInfo,
    load_weights: Callable[[list[tuple[str, torch.Tensor]]], None] | None,
  ) -> Callable[[list[tuple[str, torch.Tensor]]], None] | None:
    """Validates update arguments and resolves load_weights callback."""
    target_path = update_info.target_weights_path
    if not target_path or not os.path.exists(target_path):
      raise ValueError(f"Target weights path does not exist: {target_path}")

    if load_weights is None and self.model is not None and hasattr(self.model, "load_weights"):
      load_weights = self.model.load_weights

    return load_weights

  def _load_metadata(self, target_path: str) -> dict[str, Any]:
    """Reads metadata.json from target weights path."""
    metadata_path = os.path.join(target_path, "metadata.json") if os.path.isdir(target_path) else ""
    if metadata_path and os.path.exists(metadata_path):
      with open(metadata_path) as f:
        return json.load(f)
    return {}

  def _determine_transfer_mode(self, meta: dict[str, Any]) -> tuple[str, bool]:
    """Determines (mode_str, use_in_place_gpu) from WeightSyncConfig and metadata."""
    from server.model_metadata import WeightSyncConfig

    is_sparse_delta = meta.get("format") == "sparse_delta"
    weight_sync_cfg = WeightSyncConfig.from_env()
    strategy = weight_sync_cfg.strategy
    delta_apply_method = weight_sync_cfg.delta_apply_method
    in_place_env = os.getenv("OPEN_RL_IN_PLACE_DELTA", "").lower() in ("1", "true", "yes")

    if strategy == "full" or not is_sparse_delta:
      return "full", False
    elif (delta_apply_method == "patch_in_place" or in_place_env) and self.model is not None:
      return "patch_in_place", True
    else:
      return "full_replace", False

  def _parse_sparse_delta_file(self, target_path: str, meta: dict[str, Any]) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], int]:
    """Parses sparse safetensors file returning (meta_names, split_indices, split_values, changed_elements)."""
    delta_file = os.path.join(target_path, "delta.safetensors")
    if not os.path.exists(delta_file):
      raise ValueError(f"Sparse delta metadata present but delta.safetensors not found at: {target_path}")

    sparse_delta = load_file(delta_file, device="cpu")

    raw_names = meta.get("layer_names")
    meta_names: list[str] = json.loads(raw_names) if isinstance(raw_names, str) else (raw_names or [])

    indices_flat = sparse_delta.get("delta.indices_flat")
    if indices_flat is None:
      indices_flat = sparse_delta.get("indices")

    values_flat = sparse_delta.get("delta.values_flat")
    if values_flat is None:
      values_flat = sparse_delta.get("values")

    layer_lengths_tensor = sparse_delta.get("delta.layer_lengths")
    if layer_lengths_tensor is None:
      layer_lengths_tensor = sparse_delta.get("layer_lengths")

    if indices_flat is not None and values_flat is not None and layer_lengths_tensor is not None:
      indices_flat = indices_flat.to(torch.int64)
      layer_lengths = layer_lengths_tensor.tolist()

      if meta_names and len(meta_names) != len(layer_lengths):
        raise ValueError(f"Mismatch between layer_names ({len(meta_names)}) and layer_lengths ({len(layer_lengths)}) in sparse delta.")

      split_indices = list(torch.split(indices_flat, layer_lengths))
      split_values = list(torch.split(values_flat, layer_lengths))
      changed_elements = indices_flat.numel()
    else:
      # Support legacy per-layer key format (e.g. layer.0.weight.indices / values)
      layer_names_set = set()
      for k in sparse_delta:
        if k.endswith(".indices"):
          layer_names_set.add(k[:-8])
      if not meta_names:
        meta_names = sorted(list(layer_names_set))

      split_indices = []
      split_values = []
      for name in meta_names:
        idx_key = f"{name}.indices"
        val_key = f"{name}.values"
        if idx_key in sparse_delta and val_key in sparse_delta:
          split_indices.append(sparse_delta[idx_key].to(torch.int64))
          split_values.append(sparse_delta[val_key])
        else:
          split_indices.append(torch.empty(0, dtype=torch.int64))
          split_values.append(torch.empty(0, dtype=torch.float32))

      changed_elements = sum(idx.numel() for idx in split_indices)

    return meta_names, split_indices, split_values, changed_elements

  def _apply_sparse_delta_to_cpu_snapshot(
    self,
    meta_names: list[str],
    split_indices: list[torch.Tensor],
    split_values: list[torch.Tensor],
    changed_elements: int,
    target_path: str,
    update_info: DeltaSnapshotUpdateInfo,
    load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
  ) -> list[tuple[str, torch.Tensor]]:
    """Applies sparse delta tensors to host CPU snapshot and returns updated layer list."""
    if not self._cpu_snapshot:
      base_model = update_info.base_model_path or self._base_model
      logger.info(f"[DeltaSnapshotEngine] CPU snapshot uninitialized. Populating base CPU snapshot for '{base_model}'...")
      model = self.model or getattr(load_weights, "__self__", None)
      self._ensure_cpu_snapshot(base_model, model)

    logger.info(f"[DeltaSnapshotEngine] Applying sparse delta patch across {len(meta_names)} layers ({changed_elements} changed elements)...")
    t0_apply = time.perf_counter()

    for i, name in enumerate(meta_names):
      if name not in self._cpu_snapshot:
        raise KeyError(f"Parameter '{name}' found in sparse delta but missing from CPU snapshot.")
      if i < len(split_indices) and split_indices[i].numel() > 0:
        snap_flat = self._cpu_snapshot[name].view(-1)
        snap_flat[split_indices[i]] = split_values[i]

    t_apply_ms = (time.perf_counter() - t0_apply) * 1000.0
    total_numel = sum(t.numel() for t in self._cpu_snapshot.values())
    pct_changed = (changed_elements / total_numel * 100.0) if total_numel > 0 else 0.0
    logger.info(
      f"[DeltaSnapshotEngine] Applied packed 1D sparse delta ({len(meta_names)} layers, "
      f"{changed_elements}/{total_numel} elements [{pct_changed:.3f}% changed]) to CPU snapshot in {t_apply_ms:.2f} ms"
    )

    return list(self._cpu_snapshot.items())

  def _read_full_weights_shards(self, target_path: str) -> list[tuple[str, torch.Tensor]]:
    """Reads full dense safetensors shards from target_path."""
    weights: list[tuple[str, torch.Tensor]] = []
    if os.path.isdir(target_path):
      for root, _, files in os.walk(target_path):
        for f in sorted(files):
          if f.endswith(".safetensors") and "delta" not in f:
            shard_dict = load_file(os.path.join(root, f), device="cpu")
            weights.extend(shard_dict.items())
    elif target_path.endswith(".safetensors"):
      shard_dict = load_file(target_path, device="cpu")
      weights.extend(shard_dict.items())
    else:
      raise ValueError(f"Unsupported weight path format: {target_path}")

    return weights

  def receive_weights(
    self,
    update_info: DeltaSnapshotUpdateInfo,
    load_weights: Callable[[list[tuple[str, torch.Tensor]]], None] | None = None,
  ) -> None:
    """Receive/patch sparse delta weights in host CPU RAM and pass to load_weights."""
    resolved_loader = self._validate_receive_args(update_info, load_weights)
    target_path = update_info.target_weights_path

    start_t = time.perf_counter()
    logger.info(f"[DeltaSnapshotEngine] Starting weight update from target path: '{target_path}'")

    meta = self._load_metadata(target_path)
    mode_str, use_in_place_gpu = self._determine_transfer_mode(meta)
    logger.info(f"[DeltaSnapshotEngine] Weight update mode: apply_method={mode_str}")

    is_sparse_delta = meta.get("format") == "sparse_delta"
    if is_sparse_delta:
      meta_names, split_indices, split_values, changed_elements = self._parse_sparse_delta_file(target_path, meta)

      if changed_elements == 0:
        self.current_weights_path = target_path
        logger.info("[DeltaSnapshotEngine] Verified patch: 0 tensors changed (NO-OP PATCH DETECTED - Skipping GPU reload)")
        return

      if use_in_place_gpu:
        self._apply_gpu_in_place(meta_names, split_indices, split_values, target_path)
        return

      weights_to_load = self._apply_sparse_delta_to_cpu_snapshot(
        meta_names, split_indices, split_values, changed_elements, target_path, update_info, resolved_loader
      )
    else:
      weights = self._read_full_weights_shards(target_path)
      elapsed_read = (time.perf_counter() - start_t) * 1000.0
      logger.info(f"[DeltaSnapshotEngine] Loaded {len(weights)} parameter tensors from {target_path} in {elapsed_read:.2f} ms")

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
        logger.info(f"[DeltaSnapshotEngine] Verified patch: 0/{len(weights)} tensors changed (NO-OP PATCH DETECTED - Skipping GPU reload)")
        return

      logger.info(
        f"[DeltaSnapshotEngine] Verified patch: {len(changed_weights)}/{len(weights)} tensors changed ({no_op_tensors} no-op tensors skipped)"
      )
      weights_to_load = changed_weights

    # Feed genuinely changed parameter tensors directly into vLLM's internal layer loader
    if resolved_loader is None:
      raise ValueError("load_weights callback was not provided and self.model does not have load_weights attribute.")
    logger.info(f"[DeltaSnapshotEngine] Feeding {len(weights_to_load)} tensors into load_weights callback...")
    start_load = time.perf_counter()
    resolved_loader(weights_to_load)
    elapsed_load = (time.perf_counter() - start_load) * 1000.0
    self.current_weights_path = target_path
    logger.info(f"[DeltaSnapshotEngine] Incremental load_weights completed ({len(weights_to_load)} tensors) in {elapsed_load:.2f} ms")


try:
  from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

  WeightTransferEngineFactory.register_engine(
    "delta_snapshot",
    DeltaSnapshotWeightTransferEngine,
  )
except (ImportError, ValueError):
  pass
