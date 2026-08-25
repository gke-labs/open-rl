# LoRA trainer worker lifecycle and adapter management.

import json
import math
import os
import time
import traceback
from datetime import datetime
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class LoraConfig(BaseModel):
  rank: int = 16
  seed: int | None = None
  lora_alpha: int = 16
  lora_dropout: float = 0.05
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = False


def active_adapter_parameters(model: PeftModelForCausalLM, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


class LoraTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.base_model: PreTrainedModel | None = None
    self.peft_model: PeftModelForCausalLM | None = None
    self.base_model_name: str | None = None
    self.adapter_states: dict[str, dict[str, Any]] = {}
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}

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

    target_names = set(target_suffixes)
    target_modules = [
      name for name, module in self.base_model.named_modules() if name.rsplit(".", 1)[-1] in target_names and isinstance(module, torch.nn.Linear)
    ]
    if not target_modules:
      raise ValueError(f"No supported LoRA target modules found for suffixes: {target_suffixes}")
    self.lora_target_modules[cache_key] = target_modules
    return target_modules

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    if adapter_id in self.adapter_states:
      del self.adapter_states[adapter_id]

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

    self.peft_model.set_adapter(adapter_id)
    self.adapter_states[adapter_id] = {"trainable_params": active_adapter_parameters(self.peft_model, adapter_id), "optimizer": None}

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()
    print(f"LoRA adapter '{adapter_id}' created and set to active.")

    self.save_adapter(adapter_id)

  def create_model(self, base_model_name: str, model_id: str, config: LoraConfig) -> None:
    """Load the shared base model if needed, then create a trainable LoRA adapter."""
    self.load_base_model(base_model_name)
    self.create_adapter(model_id, config)

  def save_adapter(self, adapter_id: str, alias: str | None = None) -> None:
    """Save adapter weights to disk for reliability and sharing."""
    try:
      tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
      save_path = os.path.join(tmp_dir, "peft", adapter_id)
      os.makedirs(save_path, exist_ok=True)

      # Save the adapter weights
      self.peft_model.set_adapter(adapter_id)
      self.peft_model.save_pretrained(save_path, selected_adapters=[adapter_id])

      # Save minimal metadata
      metadata = {"model_id": adapter_id, "created_at": datetime.now().isoformat(), "timestamp": time.time()}
      if alias is not None:
        metadata["alias"] = alias
      with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

      print(f"Auto-saved adapter '{adapter_id}' to {save_path}")
    except Exception as e:
      print(f"[ERROR] Failed to auto-save weights for {adapter_id}: {e}")
      traceback.print_exc()

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    """Save adapter weights (and optionally optimizer state) to a specific path."""
    assert self.peft_model is not None, "Model must be loaded first."

    self.peft_model.set_adapter(model_id)
    os.makedirs(state_path, exist_ok=True)
    self.peft_model.save_pretrained(state_path, selected_adapters=[model_id])

    adapter_state = self.adapter_states.get(model_id)
    optimizer = adapter_state.get("optimizer") if adapter_state is not None else None
    if include_optimizer and optimizer is not None:
      torch.save(optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved state for '{model_id}' to {state_path}")
    return {"path": state_path}

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """Create an adapter from a saved state directory.

    Expects the directory to contain a metadata.json describing base_model
    and (optionally) an adapter subdirectory with the saved LoRA weights.
    """
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")

    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    src_adapter_id = metadata.get("model_id")
    adapter_dir = state_path
    if src_adapter_id and os.path.exists(os.path.join(state_path, src_adapter_id)):
      adapter_dir = os.path.join(state_path, src_adapter_id)

    self.load_base_model(base_model)
    assert self.base_model is not None

    if self.peft_model is None:
      self.peft_model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_dir, adapter_name=model_id, is_trainable=True)
    else:
      if model_id in self.peft_model.peft_config:
        self.peft_model.delete_adapter(model_id)
        if model_id in self.adapter_states:
          del self.adapter_states[model_id]
      self.peft_model.load_adapter(adapter_dir, adapter_name=model_id, is_trainable=True)

    self.peft_model.set_adapter(model_id)
    params = active_adapter_parameters(self.peft_model, model_id)
    adapter_state = {"trainable_params": params, "optimizer": None}
    self.adapter_states[model_id] = adapter_state

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        lr = 1e-4
        optimizer = torch.optim.AdamW(params, lr=lr)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        adapter_state["optimizer"] = optimizer
        print(f"Restored optimizer state for '{model_id}' from {optimizer_path}")

    print(f"Loaded state for '{model_id}' from {state_path}")
    return {"model_id": model_id, "is_lora": True, "base_model": base_model}

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None) -> dict[str, Any]:
    assert self.peft_model is not None, "Model must be loaded first."
    if model_id:
      self.peft_model.set_adapter(model_id)
    return super().forward_backward(self.peft_model, data, loss_fn, loss_config)

  def optim_step(self, adam_params: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Apply accumulated gradients and update model weights."""
    assert self.peft_model is not None, "Model must be loaded first."
    if not model_id:
      raise ValueError("model_id is required for optim_step")

    self.peft_model.set_adapter(model_id)
    try:
      adapter_state = self.adapter_states[model_id]
    except KeyError as e:
      raise ValueError(f"Adapter '{model_id}' has no cached trainable parameters") from e
    params = adapter_state["trainable_params"]

    if adapter_state.get("optimizer") is None:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      print(f"Initializing AdamW optimizer for '{model_id}' with lr={lr}")
      adapter_state["optimizer"] = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
      )

    optimizer = adapter_state["optimizer"]
    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    total_norm = torch.nn.utils.clip_grad_norm_(
      params,
      max_grad_norm,
    )

    optimizer.step()
    optimizer.zero_grad()

    self.save_adapter(model_id)

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm.item()),
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
    if model_id:
      self.peft_model.set_adapter(model_id)
    return super().generate(self.peft_model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)
