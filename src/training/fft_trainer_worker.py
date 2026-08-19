# Full fine-tuning trainer worker lifecycle.

import gc
import itertools
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: str | None = None


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


from server.model_metadata import WeightSyncConfig


class FFTTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.model: PreTrainedModel | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.cpu_offload: bool = True
    self.weight_sync_cfg: WeightSyncConfig = WeightSyncConfig.from_env()
    self._is_offloaded: bool = False
    self._latest_delta_tensors: dict[str, torch.Tensor] = {}
    self._latest_total_changed: int = 0
    self._latest_total_elements: int = 0
    self._param_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._grad_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._opt_shadow: dict[tuple[torch.nn.Parameter, str], tuple[torch.device, torch.Tensor]] = {}
    self._prev_weights_shadow: dict[str, torch.Tensor] = {}
    self.model_layer_names: list[str] = []
    self.total_model_elements: int = 0

  def set_weight_sync_strategy(self, strategy: str) -> None:
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync_cfg.strategy = strategy

  def _get_prev_cpu_weight(self, name: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    if param in self._param_shadow:
      return self._param_shadow[param][1]
    return None

  def _update_prev_cpu_weight(self, name: str, param: torch.nn.Parameter, indices: torch.Tensor, values: torch.Tensor) -> None:
    if param in self._param_shadow:
      self._param_shadow[param][1].view(-1)[indices.to(torch.int64).cpu()] = values

  def load_base_model(self, base_model_name: str) -> None:
    """Load one full model for one fine-tuning job process."""
    if self.model is not None and self.base_model_name == base_model_name:
      print(f"Full fine-tuning model {base_model_name} already loaded.")
      return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_device = "auto" if num_gpus > 1 else self.device
    print(f"Loading full fine-tuning model {base_model_name} (target device map: {target_device}, visible GPUs: {num_gpus})...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    self.model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map=target_device)
    print("Successfully loaded full fine-tuning model.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: FFTConfig | None = None) -> None:
    """Load the per-job model if needed, then prepare it for full fine-tuning."""
    if config is not None:
      self.cpu_offload = config.cpu_offload
      if hasattr(config, "weight_sync_strategy") and config.weight_sync_strategy:
        self.set_weight_sync_strategy(config.weight_sync_strategy)
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."

    for param in self.model.parameters():
      param.requires_grad_(True)
    self.trainable_params = trainable_model_parameters(self.model)
    self.model_layer_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
    self.total_model_elements = sum(p.numel() for p in self.model.parameters())
    if self.weight_sync_cfg.strategy == "delta":
      for param in self.model.parameters():
        if param.requires_grad and param not in self._param_shadow:
          cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          cpu_buf.copy_(param.data, non_blocking=True)
          self._param_shadow[param] = (param.device, cpu_buf)

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on full fine-tuning model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.model.train()

  def _prepare_for_save(self) -> bool:
    was_offloaded = self._is_offloaded
    if was_offloaded and self.model is not None:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = self._param_shadow[tensor][1]
    return was_offloaded

  def _cleanup_after_save(self, was_offloaded: bool) -> None:
    if was_offloaded and self.model is not None:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = torch.empty(0, dtype=tensor.dtype, device=self._param_shadow[tensor][0])

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save model while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    name = alias or "fft-model"
    save_path = name if os.path.isabs(name) else os.path.join(tmp_dir, "fft", name)
    os.makedirs(save_path, exist_ok=True)

    was_offloaded = self._prepare_for_save()
    try:
      self.model.save_pretrained(save_path)
      if self.tokenizer is not None:
        self.tokenizer.save_pretrained(save_path)
    finally:
      self._cleanup_after_save(was_offloaded)

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": "weights",
      "model_id": alias,
      "timestamp": time.time(),
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved full fine-tuning model to {save_path}")
    return {"path": save_path}

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save state while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    if self.weight_sync_cfg.strategy == "delta" and not include_optimizer:
      return self.save_state_delta(model_id=model_id, state_path=state_path, kind=kind)

    os.makedirs(state_path, exist_ok=True)
    was_offloaded = self._prepare_for_save()
    try:
      self.model.save_pretrained(state_path)
      if self.tokenizer is not None:
        self.tokenizer.save_pretrained(state_path)

      if include_optimizer and self.optimizer is not None:
        torch.save(self.optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))
    finally:
      self._cleanup_after_save(was_offloaded)

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved full fine-tuning state to {state_path}")
    return {"path": state_path}

  def save_state_delta(
    self,
    model_id: str,
    state_path: str,
    kind: str = "sampler",
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save state delta while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    os.makedirs(state_path, exist_ok=True)
    total_changed = 0
    total_elements = 0
    layer_names_list: list[str] = []
    indices_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []
    layer_lengths_list: list[int] = []

    t_collect_start = time.perf_counter()
    if self._latest_delta_tensors and "names" in self._latest_delta_tensors:
      layer_names_list = self._latest_delta_tensors["names"]
      indices_list = self._latest_delta_tensors["indices_list"]
      values_list = self._latest_delta_tensors["values_list"]
      layer_lengths_list = self._latest_delta_tensors["layer_lengths_list"]
      total_changed = self._latest_total_changed
      total_elements = self._latest_total_elements
    else:
      layer_names_list = self.model_layer_names
      layer_lengths_list = [0] * len(layer_names_list)
      total_changed = 0
      total_elements = self.total_model_elements
      indices_list = []
      values_list = []

    if indices_list:
      indices_flat = torch.cat(indices_list).to(torch.int32).contiguous()
      values_flat = torch.cat(values_list).contiguous()
    else:
      fallback_dtype = next(self.model.parameters()).dtype if self.model else torch.float32
      indices_flat = torch.empty(0, dtype=torch.int32, device="cpu")
      values_flat = torch.empty(0, dtype=fallback_dtype, device="cpu")

    layer_lengths_tensor = torch.tensor(layer_lengths_list, dtype=torch.int64, device="cpu")
    packed_delta = {
      "delta.indices_flat": indices_flat,
      "delta.values_flat": values_flat,
      "delta.layer_lengths": layer_lengths_tensor,
    }

    t_collect_end = time.perf_counter()
    collect_time = t_collect_end - t_collect_start

    import safetensors.torch

    delta_path = os.path.join(state_path, "delta.safetensors")
    t_save_start = time.perf_counter()
    safetensors.torch.save_file(
      packed_delta,
      delta_path,
      metadata={"layer_names": json.dumps(layer_names_list)},
    )
    t_save_end = time.perf_counter()
    save_file_time = t_save_end - t_save_start

    logger.info(
      f"[SAVE_STATE_DELTA] model_id={model_id} kind={kind} | "
      f"collect_time={collect_time:.4f}s | "
      f"safetensors_save_time={save_file_time:.4f}s | "
      f"total_delta_save_time={collect_time + save_file_time:.4f}s | "
      f"changed={total_changed}/{total_elements} ({100.0 * total_changed / max(1, total_elements):.2f}%) across {len(layer_names_list)} layers"
    )

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "format": "sparse_delta",
      "kind": kind,
      "model_id": model_id,
      "changed_elements": total_changed,
      "total_elements": total_elements,
      "layer_names": layer_names_list,
      "density_pct": round(100.0 * total_changed / max(1, total_elements), 3),
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved sparse delta ({metadata['density_pct']}% changed elements, {total_changed}/{total_elements}) to {state_path}")
    return {"path": state_path, "density_pct": metadata["density_pct"]}

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")

    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    self.base_model_name = base_model
    self.tokenizer = AutoTokenizer.from_pretrained(state_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_device = "auto" if num_gpus > 1 else self.device
    self.model = AutoModelForCausalLM.from_pretrained(state_path, dtype=dtype, device_map=target_device)
    self.prepare_model_for_training()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=1e-4)
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        print(f"Restored optimizer state from {optimizer_path}")

    print(f"Loaded full fine-tuning state from {state_path}")
    return {"model_id": model_id, "base_model": base_model}

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    res = super().forward_backward(self.model, data, loss_fn, loss_config)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def _remap_hf_to_vllm_fused(
    self,
    layer_names_list: list[str],
    indices_list: list[torch.Tensor],
  ) -> tuple[list[str], list[torch.Tensor]]:
    """Remaps HF layer names (q_proj, k_proj, v_proj, gate_proj, up_proj) and offsets indices to vLLM fused names."""
    config = getattr(self.model, "config", None)
    if config is None:
      return layer_names_list, indices_list
    # Multimodal wrappers (e.g. gemma-4 ForConditionalGeneration) nest the LM
    # dims under text_config.
    if getattr(config, "hidden_size", None) is None and getattr(config, "text_config", None) is not None:
      config = config.text_config

    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_size is not None and num_heads is not None:
      head_dim = hidden_size // num_heads

    intermediate_size = getattr(config, "intermediate_size", None)

    q_numel = (num_heads * head_dim * hidden_size) if (hidden_size and num_heads and head_dim) else None
    k_numel = (num_kv_heads * head_dim * hidden_size) if (hidden_size and num_kv_heads and head_dim) else None
    gate_numel = (intermediate_size * hidden_size) if (hidden_size and intermediate_size) else None
    # Bias rows fuse with bias-sized offsets (Qwen2.5 attention has QKV
    # biases; using weight-sized offsets sent bias indices out of bounds).
    q_bias_numel = (num_heads * head_dim) if (num_heads and head_dim) else None
    k_bias_numel = (num_kv_heads * head_dim) if (num_kv_heads and head_dim) else None

    mapped_names: list[str] = []
    mapped_indices: list[torch.Tensor] = []

    for name, idx in zip(layer_names_list, indices_list):
      is_bias = name.endswith(".bias")
      if (".q_proj." in name or ".k_proj." in name or ".v_proj." in name) and q_numel is not None and k_numel is not None:
        qkv_name = name.replace(".q_proj.", ".qkv_proj.").replace(".k_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
        qn, kn = (q_bias_numel, k_bias_numel) if is_bias else (q_numel, k_numel)
        offset = 0 if ".q_proj." in name else (qn if ".k_proj." in name else qn + kn)
        mapped_names.append(qkv_name)
        mapped_indices.append(idx + offset)
        continue

      if (".gate_proj." in name or ".up_proj." in name) and gate_numel is not None:
        gate_up_name = name.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
        # (No known FFT target has MLP biases; if one appears, intermediate_size
        # is the bias-sized gate offset.)
        offset = 0 if ".gate_proj." in name else (intermediate_size if is_bias else gate_numel)
        mapped_names.append(gate_up_name)
        mapped_indices.append(idx + offset)
        continue

      mapped_names.append(name)
      mapped_indices.append(idx)

    return mapped_names, mapped_indices

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if not self.trainable_params:
      self.trainable_params = trainable_model_parameters(self.model)

    if self.optimizer is None:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      print(f"Initializing AdamW optimizer for full fine-tuning model with lr={lr}")
      self.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
      )

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    t_clip_start = time.perf_counter()
    total_norm = torch.nn.utils.clip_grad_norm_(
      self.trainable_params,
      max_grad_norm,
    )
    t_clip_end = time.perf_counter()
    clip_time = t_clip_end - t_clip_start

    t_step_start = time.perf_counter()
    self.optimizer.step()
    self.optimizer.zero_grad()
    t_step_end = time.perf_counter()
    step_time = t_step_end - t_step_start

    delta_compute_time = 0.0
    if self.weight_sync_cfg.strategy == "delta" and self.model is not None and hasattr(self.model, "named_parameters"):
      t_delta_start = time.perf_counter()
      self._latest_delta_tensors.clear()
      self._latest_total_changed = 0
      self._latest_total_elements = self.total_model_elements

      layer_names_list: list[str] = []
      indices_list: list[torch.Tensor] = []
      values_list: list[torch.Tensor] = []
      layer_lengths_list: list[int] = []

      for name, param in self.model.named_parameters():
        if not param.requires_grad:
          continue
        prev_tensor = self._get_prev_cpu_weight(name, param)
        if prev_tensor is None:
          cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          cpu_buf.copy_(param.data, non_blocking=True)
          self._param_shadow[param] = (param.device, cpu_buf)
          prev_tensor = cpu_buf

        prev_gpu = prev_tensor.to(param.device, non_blocking=True)

        diff_mask = param.data.view(-1).ne(prev_gpu.view(-1))
        indices = diff_mask.nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
          idx_cpu = indices.to(torch.int32).contiguous().cpu()
          val_cpu = param.data.view(-1)[diff_mask].contiguous().cpu()
          layer_names_list.append(name)
          indices_list.append(idx_cpu)
          values_list.append(val_cpu)
          layer_lengths_list.append(int(idx_cpu.numel()))
          self._latest_total_changed += int(idx_cpu.numel())
          self._update_prev_cpu_weight(name, param, idx_cpu, val_cpu)
        del prev_gpu, diff_mask, indices

      if self.weight_sync_cfg.delta_format == "vllm_fused":
        layer_names_list, indices_list = self._remap_hf_to_vllm_fused(layer_names_list, indices_list)

      self._latest_delta_tensors = {
        "names": layer_names_list,
        "indices_list": indices_list,
        "values_list": values_list,
        "layer_lengths_list": layer_lengths_list,
      }

      t_delta_end = time.perf_counter()
      delta_compute_time = t_delta_end - t_delta_start
      logger.info(
        f"[OPTIM_STEP] model_id={model_id} | delta_compute_time={delta_compute_time:.4f}s | "
        f"changed={self._latest_total_changed}/{self._latest_total_elements} "
        f"({100.0 * self._latest_total_changed / max(1, self._latest_total_elements):.2f}%) across {len(layer_names_list)} layers"
      )

    logger.info(
      f"[OPTIM_STEP] model_id={model_id} | clip_grad_time={clip_time:.4f}s | "
      f"optimizer_step_time={step_time:.4f}s | delta_compute_time={delta_compute_time:.4f}s | "
      f"total_optim_time={clip_time + step_time + delta_compute_time:.4f}s"
    )

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm.item()),
        "time/compute_delta_diff": self.sanitize_float(delta_compute_time),
        "time/optimizer_step": self.sanitize_float(step_time),
        "time/clip_grad_norm": self.sanitize_float(clip_time),
      },
    }

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    return super().generate(self.model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)

  def sleep(self) -> None:
    """Offload GPU tensors to pinned host CPU memory and empty CUDA allocator cache."""
    if not self.cpu_offload or self.model is None or self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()

    # Phase 1: Launch Batched Asynchronous DMA copies WITHOUT freeing GPU tensors!
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor.device.type == "cuda":
        orig_device = tensor.device
        if tensor in self._param_shadow and self._param_shadow[tensor][1].shape == tensor.shape:
          cpu_buf = self._param_shadow[tensor][1]
        else:
          cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          self._param_shadow[tensor] = (orig_device, cpu_buf)
        cpu_buf.copy_(tensor.data, non_blocking=True)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor.grad.device.type == "cuda":
        orig_device = tensor.grad.device
        if tensor in self._grad_shadow and self._grad_shadow[tensor][1].shape == tensor.grad.shape:
          cpu_buf = self._grad_shadow[tensor][1]
        else:
          cpu_buf = torch.empty(tensor.grad.shape, dtype=tensor.grad.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          self._grad_shadow[tensor] = (orig_device, cpu_buf)
        cpu_buf.copy_(tensor.grad.data, non_blocking=True)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.device.type == "cuda":
              orig_device = v.device
              opt_key = (param, k)
              if opt_key in self._opt_shadow and self._opt_shadow[opt_key][1].shape == v.shape:
                cpu_buf = self._opt_shadow[opt_key][1]
              else:
                cpu_buf = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=torch.cuda.is_available())
                self._opt_shadow[opt_key] = (orig_device, cpu_buf)
              cpu_buf.copy_(v, non_blocking=True)

    # Phase 2: Single Barrier Synchronization point!
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    # Phase 3: Now that DMA has finished, safely deallocate GPU VRAM!
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor in self._param_shadow:
        orig_device = self._param_shadow[tensor][0]
        tensor.data = torch.empty(0, dtype=tensor.dtype, device=orig_device)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor in self._grad_shadow:
        orig_device = self._grad_shadow[tensor][0]
        tensor.grad.data = torch.empty(0, dtype=tensor.grad.dtype, device=orig_device)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          for k in list(state.keys()):
            opt_key = (param, k)
            if opt_key in self._opt_shadow:
              orig_device, cpu_buf = self._opt_shadow[opt_key]
              state[k] = cpu_buf

    if torch.cuda.is_available():
      gc.collect()
      torch.cuda.empty_cache()
      if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    self._is_offloaded = True
    print(f"[FFT Worker] Offloaded weights & states to pinned CPU memory in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def wake_up(self) -> None:
    """Reload pinned CPU shadow tensors back to CUDA VRAM without destroying host shadow buffers."""
    if not self.cpu_offload or self.model is None or not self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()

    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor in self._param_shadow:
        orig_device, cpu_data = self._param_shadow[tensor]
        tensor.data = cpu_data.to(orig_device, non_blocking=True)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor in self._grad_shadow:
        orig_device, cpu_grad = self._grad_shadow[tensor]
        tensor.grad.data = cpu_grad.to(orig_device, non_blocking=True)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          state.pop("_orig_devices", None)
          target_device = param.device
          for k, v in list(state.items()):
            opt_key = (param, k)
            if opt_key in self._opt_shadow:
              orig_device, cpu_buf = self._opt_shadow[opt_key]
              state[k] = cpu_buf.to(orig_device, non_blocking=True)
            elif isinstance(v, torch.Tensor) and v.device.type == "cpu" and k != "step":
              state[k] = v.to(target_device, non_blocking=True)

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    self._is_offloaded = False
    print(f"[FFT Worker] Reloaded weights & states to CUDA in {(time.perf_counter() - start_t) * 1000:.1f} ms.")
