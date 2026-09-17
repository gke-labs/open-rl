# LoRA trainer: one shared base model hosting many adapters.

import json
import os
import time
import traceback
from datetime import datetime
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.commands import CreateModel, CreateModelFromState, SaveWeightsForSampler
from training.trainer_worker import BaseTrainerWorker, Datum, tmp_dir
from training.types import LoraConfig, SamplerWeights

__all__ = ["LoraAdapter", "LoraConfig", "LoraTrainingWorker", "active_adapter_parameters"]


def active_adapter_parameters(model: PeftModelForCausalLM, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


class LoraAdapter:
  """The Trainer for one adapter on a LoraTrainingWorker.

  Every call activates the adapter on the shared PEFT model first, so several
  adapters can be trained from one process without the request loop knowing.
  """

  def __init__(self, host: "LoraTrainingWorker", adapter_id: str):
    self.host = host
    self.adapter_id = adapter_id

  @property
  def model_id(self) -> str:
    return self.adapter_id

  def activate(self) -> PeftModelForCausalLM:
    assert self.host.peft_model is not None, "Model must be loaded first."
    self.host.peft_model.set_adapter(self.adapter_id)
    return self.host.peft_model

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    return BaseTrainerWorker.forward_backward(self.host, self.activate(), data, loss_fn, loss_config)

  def optim_step(self, adam_params: dict[str, Any]) -> dict[str, Any]:
    self.activate()
    state = self.host.adapter_state(self.adapter_id)
    params = state["trainable_params"]
    if state.get("optimizer") is None:
      state["optimizer"] = self.host.build_optimizer(params, adam_params, label=f"'{self.adapter_id}'")
    total_norm, *_ = self.host.step_optimizer(state["optimizer"], params, adam_params)
    self.host.save_adapter(self.adapter_id)
    return {"metrics": {"grad_norm:mean": self.host.sanitize_float(total_norm), **self.host.ratio_metrics()}}

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    """Save adapter weights (and optionally optimizer state) to a specific path."""
    peft_model = self.activate()
    os.makedirs(state_path, exist_ok=True)
    peft_model.save_pretrained(state_path, selected_adapters=[self.adapter_id])

    optimizer = self.host.adapter_state(self.adapter_id).get("optimizer")
    has_optimizer = include_optimizer and optimizer is not None
    if has_optimizer:
      torch.save(optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))

    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(self.host.checkpoint_metadata(self.adapter_id, kind=kind, has_optimizer=has_optimizer), f)
    print(f"Saved state for '{self.adapter_id}' to {state_path}")
    return {"path": state_path}

  def load_from_state(self, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    return self.host.load_adapter_state(self.adapter_id, state_path, restore_optimizer)

  def publish_sampler_weights(self, command: SaveWeightsForSampler) -> SamplerWeights:
    return self.host.save_adapter(self.adapter_id, command.alias)

  def save_weights(self, alias: str | None = None) -> dict[str, Any]:
    return {"path": self.host.save_adapter(self.adapter_id, alias).path}

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    return BaseTrainerWorker.generate(self.host, self.activate(), prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)


class LoraTrainingWorker(BaseTrainerWorker):
  """Hosts N LoRA adapters on one base model. create() and trainer() hand out
  LoraAdapter trainers; the host itself keeps the base model, the PEFT
  wrapper and the per-adapter optimizer state."""

  def __init__(self):
    super().__init__()
    self.base_model: PreTrainedModel | None = None
    self.peft_model: PeftModelForCausalLM | None = None
    self.adapter_states: dict[str, dict[str, Any]] = {}
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}

  # -- worker API -----------------------------------------------------------

  def create(self, command: CreateModel) -> LoraAdapter:
    self.create_model(command.base_model, command.model_id, command.lora_config)
    return LoraAdapter(self, command.model_id)

  def restore(self, command: CreateModelFromState) -> LoraAdapter:
    self.load_adapter_state(command.model_id, command.state_path, command.restore_optimizer)
    return LoraAdapter(self, command.model_id)

  def trainer(self, model_id: str) -> LoraAdapter:
    self.adapter_state(model_id)
    return LoraAdapter(self, model_id)

  def adapter_state(self, adapter_id: str) -> dict[str, Any]:
    try:
      return self.adapter_states[adapter_id]
    except KeyError:
      raise ValueError(f"Adapter '{adapter_id}' is not loaded on this worker") from None

  # -- base model and adapters ----------------------------------------------

  def load_base_model(self, base_model_name: str) -> None:
    """Eagerly load the massive base model tensors into VRAM."""
    if self.base_model is not None and self.base_model_name == base_model_name:
      print(f"Base model {base_model_name} already loaded.")
      return

    print(f"Loading base model {base_model_name} to {self.device}...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map=self.device)
    print("Successfully loaded.")

  def target_lora_modules(self, config: LoraConfig) -> list[str]:
    assert self.base_model is not None

    cache_key = (config.train_attn, config.train_mlp, config.train_unembed)
    if cache_key in self.lora_target_modules:
      return self.lora_target_modules[cache_key]

    target_suffixes: list[str] = []
    if config.train_attn:
      target_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if config.train_mlp:
      # TODO: Revisit MLP targets for packed/MoE module names across supported backends.
      target_suffixes.extend(["gate_proj", "up_proj", "down_proj"])
    if config.train_unembed:
      # getattr because not every config defines tie_word_embeddings (transformers
      # v5 dropped it from the PretrainedConfig base class). When absent,
      # transformers performs no tying, so False matches the loaded model.
      if getattr(self.base_model.config, "tie_word_embeddings", False):
        # An adapter on a tied lm_head shares the embedding tensor: PEFT warns,
        # merging would corrupt embed_tokens, and vLLM refuses lm_head adapter
        # weights for tied models. Keep every produced adapter vLLM-loadable.
        print(
          f"[LoRA] Ignoring train_unembed=True: {self.base_model_name} ties lm_head to embed_tokens, "
          "and the resulting adapter could not be loaded by vLLM."
        )
      else:
        target_suffixes.append("lm_head")

    if not target_suffixes:
      raise ValueError("No trainable LoRA targets remain (train_unembed is ignored on tied-embeddings models; enable train_attn or train_mlp)")

    # Once an adapter exists, PEFT has wrapped the targeted layers and the
    # Linear sits under base_layer; a later config must still find it.
    target_names = set(target_suffixes)
    target_modules = [
      name
      for name, module in self.base_model.named_modules()
      if name.rsplit(".", 1)[-1] in target_names and isinstance(getattr(module, "base_layer", module), torch.nn.Linear)
    ]
    if not target_modules:
      raise ValueError(f"No supported LoRA target modules found for suffixes: {target_suffixes}")
    self.lora_target_modules[cache_key] = target_modules
    return target_modules

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    self.adapter_states.pop(adapter_id, None)
    if not any([config.train_attn, config.train_mlp, config.train_unembed]):
      raise ValueError("At least one LoRA training target must be enabled.")

    print(f"Creating LoRA adapter '{adapter_id}'...")
    peft_config = PeftLoraConfig(
      task_type="CAUSAL_LM",
      r=config.rank,
      lora_alpha=config.lora_alpha,
      lora_dropout=config.lora_dropout,
      bias="none",
      target_modules=self.target_lora_modules(config),
      modules_to_save=None,
    )

    if config.seed is not None:
      torch.manual_seed(config.seed)
    if self.peft_model is None:
      self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
    else:
      self.peft_model.add_adapter(adapter_id, peft_config)

    self.register_adapter(adapter_id)
    print(f"LoRA adapter '{adapter_id}' created and set to active.")
    self.save_adapter(adapter_id)

  def register_adapter(self, adapter_id: str) -> dict[str, Any]:
    assert self.peft_model is not None
    self.peft_model.set_adapter(adapter_id)
    state: dict[str, Any] = {"trainable_params": active_adapter_parameters(self.peft_model, adapter_id), "optimizer": None}
    self.adapter_states[adapter_id] = state
    self.enable_gradient_checkpointing(self.peft_model)
    self.peft_model.train()
    return state

  def create_model(self, base_model_name: str, model_id: str, config: LoraConfig) -> None:
    """Load the shared base model if needed, then create a trainable LoRA adapter."""
    self.load_base_model(base_model_name)
    self.create_adapter(model_id, config)

  def save_adapter(self, adapter_id: str, alias: str | None = None) -> SamplerWeights:
    """Write the adapter where the LoRA sampler workers load it: peft/<id>/<id>/."""
    adapter_root = os.path.join(tmp_dir(), "peft", adapter_id)
    adapter_dir = os.path.join(adapter_root, adapter_id)
    if self.peft_model is None:
      print(f"[LoRA] Cannot save adapter '{adapter_id}': no active PEFT model initialized.")
      return SamplerWeights(kind="adapter", path=adapter_dir)
    try:
      os.makedirs(adapter_root, exist_ok=True)
      self.peft_model.set_adapter(adapter_id)
      self.peft_model.save_pretrained(adapter_root, selected_adapters=[adapter_id])

      metadata = {
        "model_id": adapter_id,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time(),
        **({"alias": alias} if alias is not None else {}),
      }
      with open(os.path.join(adapter_root, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      print(f"Auto-saved adapter '{adapter_id}' to {adapter_root}")
    except Exception as e:
      print(f"[ERROR] Failed to auto-save weights for {adapter_id}: {e}")
      traceback.print_exc()
    return SamplerWeights(kind="adapter", path=adapter_dir)

  def load_adapter_state(self, adapter_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """Create (or replace) an adapter from a saved state directory."""
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")
    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    src_adapter_id = metadata.get("model_id")
    adapter_dir = (
      os.path.join(state_path, src_adapter_id) if src_adapter_id and os.path.exists(os.path.join(state_path, src_adapter_id)) else state_path
    )

    self.load_base_model(base_model)
    assert self.base_model is not None
    if self.peft_model is None:
      self.peft_model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_dir, adapter_name=adapter_id, is_trainable=True)
    else:
      if adapter_id in self.peft_model.peft_config:
        self.peft_model.delete_adapter(adapter_id)
        self.adapter_states.pop(adapter_id, None)
      self.peft_model.load_adapter(adapter_dir, adapter_name=adapter_id, is_trainable=True)

    state = self.register_adapter(adapter_id)
    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and metadata.get("has_optimizer") and os.path.exists(optimizer_path):
      optimizer = torch.optim.AdamW(state["trainable_params"], lr=1e-4)
      optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
      state["optimizer"] = optimizer
      print(f"Restored optimizer state for '{adapter_id}' from {optimizer_path}")

    print(f"Loaded state for '{adapter_id}' from {state_path}")
    return {"model_id": adapter_id, "is_lora": True, "base_model": base_model}
