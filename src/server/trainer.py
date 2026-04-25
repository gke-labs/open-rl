# This file contains the core training engine logic for Open-RL, handling forward/backward passes and optimization steps.

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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class TensorData(BaseModel):
  data: list[int] | list[float]


class LoraConfig(BaseModel):
  rank: int = 16
  seed: int | None = None
  lora_alpha: int = 16
  lora_dropout: float = 0.05
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = False


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


class TrainerEngine:
  def __init__(self):
    # The raw pre-trained base model (e.g., Gemma, Qwen) loaded in VRAM
    self.base_model: PreTrainedModel | None = None

    # The model wrapped with PEFT/LoRA adapters that we actually train
    self.peft_model: PeftModelForCausalLM | None = None

    # The tokenizer associated with the base model
    self.tokenizer: PreTrainedTokenizerBase | None = None

    # String identifier of the currently loaded base model
    self.base_model_name: str | None = None

    # Store optimizers per model_id (adapter ID)
    self.optimizers: dict[str, torch.optim.Optimizer] = {}

    # Decide device
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

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

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    if adapter_id in self.optimizers:
      del self.optimizers[adapter_id]

    if not any([config.train_attn, config.train_mlp, config.train_unembed]):
      raise ValueError("At least one LoRA training target must be enabled.")

    print(f"Creating LoRA adapter '{adapter_id}'...")

    target_suffixes: list[str] = []
    if config.train_attn:
      target_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if config.train_mlp:
      target_suffixes.extend(["gate_proj", "up_proj", "down_proj"])

    # Decide target_modules based on model type and config
    explicit_target_modules = os.getenv("OPEN_RL_TARGET_MODULES")
    use_text_tower_lora = getattr(self.base_model.config, "model_type", None) == "gemma4"
    target_modules: str | list[str]
    layers_to_transform = None
    layers_pattern = None

    if explicit_target_modules == "all-linear":
      target_modules = "all-linear"
      print("[trainer] Forcing PEFT target_modules=all-linear via OPEN_RL_TARGET_MODULES")
    elif use_text_tower_lora:
      if target_suffixes:
        target_modules = target_suffixes
        model_config = getattr(self.base_model.config, "text_config", self.base_model.config)
        layers_to_transform = list(range(model_config.num_hidden_layers))
        layers_pattern = "language_model.layers"
      else:
        target_modules = []
    elif config.train_attn and config.train_mlp and config.train_unembed:
      target_modules = "all-linear"
    else:
      target_modules = target_suffixes

    peft_config = PeftLoraConfig(
      task_type="CAUSAL_LM",
      r=config.rank,
      lora_alpha=config.lora_alpha,
      lora_dropout=config.lora_dropout,
      bias="none",
      target_modules=target_modules,
      layers_to_transform=layers_to_transform,
      layers_pattern=layers_pattern,
      modules_to_save=["lm_head", "embed_tokens"] if config.train_unembed else None,
    )

    if config.seed is not None:
      torch.manual_seed(config.seed)
    if self.peft_model is None:
      self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
    else:
      self.peft_model.add_adapter(adapter_id, peft_config)

    self.peft_model.set_adapter(adapter_id)

    self.peft_model.train()
    print(f"LoRA adapter '{adapter_id}' created and set to active.")

    self.save_adapter(adapter_id)

  def save_adapter(self, adapter_id: str, alias: str | None = None) -> None:
    """Save adapter weights to disk for reliability and sharing."""
    try:
      tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
      save_path = os.path.join(tmp_dir, "peft", adapter_id)
      os.makedirs(save_path, exist_ok=True)

      # Save the adapter weights
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

    optimizer = self.optimizers.get(model_id)
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
      if model_id in getattr(self.peft_model, "peft_config", {}):
        self.peft_model.delete_adapter(model_id)
        self.optimizers.pop(model_id, None)
      self.peft_model.load_adapter(adapter_dir, adapter_name=model_id, is_trainable=True)

    self.peft_model.set_adapter(model_id)
    self.peft_model.train()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        lr = 1e-4
        params = [p for p in self.peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        self.optimizers[model_id] = optimizer
        print(f"Restored optimizer state for '{model_id}' from {optimizer_path}")

    print(f"Loaded state for '{model_id}' from {state_path}")
    return {"model_id": model_id, "is_lora": True, "base_model": base_model}

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None) -> dict[str, Any]:
    """Core training step: forward pass, loss computation, and backward pass."""
    assert self.peft_model is not None, "Model must be loaded first."

    total_loss = 0.0
    loss_fn_outputs = []

    # Ensure model is in train mode
    self.peft_model.train()

    for datum in data:
      # 1. Common Setup: Extract tokens and get logprobs
      target_logprobs, targets_tensor, weights_tensor = self._get_logprobs(datum)

      # 2. Specialized Loss Calculation
      match loss_fn:
        case "cross_entropy":
          loss = self._compute_cross_entropy_loss(target_logprobs, weights_tensor)
        case "importance_sampling":
          loss = self._compute_importance_sampling_loss(target_logprobs, weights_tensor, datum)
        case "ppo":
          loss = self._compute_ppo_loss(target_logprobs, weights_tensor, datum, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      # 3. Common Cleanup: Backward pass
      loss.backward()
      total_loss += loss.item()

      # Save logprobs for return
      logprobs_list = target_logprobs.detach().cpu().tolist()
      logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]

      loss_fn_outputs.append({"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}})

    mean_loss = total_loss / max(1, len(data))

    return {
      "metrics": {"loss:mean": self._sanitize_float(mean_loss), "loss:sum": self._sanitize_float(total_loss)},
      "loss_fn_outputs": loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def _get_logprobs(self, datum: Datum) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (target_logprobs, targets_tensor, weights_tensor)."""
    # model_input is now just a flat list of tokens!
    inputs_tensor = torch.tensor([datum.model_input], dtype=torch.long, device=self.device)

    # Extract targets
    targets_data = datum.loss_fn_inputs["target_tokens"].data
    targets_tensor = torch.tensor(targets_data, dtype=torch.long, device=self.device)

    # Extract weights with default fallback to 1.0
    if "weights" in datum.loss_fn_inputs:
      weights_data = datum.loss_fn_inputs["weights"].data
    else:
      weights_data = [1.0] * len(targets_data)

    weights_tensor = torch.tensor(weights_data, dtype=torch.float32, device=self.device)

    outputs = self.peft_model(inputs_tensor, use_cache=False)
    logits = outputs.logits[0]  # Shape: (SeqLen, VocabSize)

    seq_len = min(logits.size(0), targets_tensor.size(0))
    sliced_logits = logits[:seq_len]
    sliced_targets = targets_tensor[:seq_len]

    target_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1).gather(dim=-1, index=sliced_targets.unsqueeze(-1)).squeeze(-1)

    if weights_tensor.numel() > 0:
      weights_tensor = weights_tensor[:seq_len]

    return target_logprobs, targets_tensor, weights_tensor

  def _compute_cross_entropy_loss(self, target_logprobs: torch.Tensor, weights_tensor: torch.Tensor) -> torch.Tensor:
    """Simple cross entropy loss."""
    elementwise_loss = -target_logprobs * weights_tensor
    return elementwise_loss.sum()

  def _compute_importance_sampling_loss(self, target_logprobs: torch.Tensor, weights_tensor: torch.Tensor, datum: Datum) -> torch.Tensor:
    """Importance sampling loss for RL."""
    if "logprobs" not in datum.loss_fn_inputs or "advantages" not in datum.loss_fn_inputs:
      raise ValueError("importance_sampling requires 'logprobs' and 'advantages' in loss_fn_inputs")

    ref_logprobs = datum.loss_fn_inputs["logprobs"].data
    advantages = datum.loss_fn_inputs["advantages"].data
    ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
    advantages_tensor = torch.tensor(advantages, dtype=target_logprobs.dtype, device=self.device)

    seq_len = min(target_logprobs.size(0), ref_tensor.size(0), advantages_tensor.size(0), weights_tensor.size(0))
    target_logprobs = target_logprobs[:seq_len]
    ref_tensor = ref_tensor[:seq_len]
    advantages_tensor = advantages_tensor[:seq_len]
    weights_tensor = weights_tensor[:seq_len]

    diff = target_logprobs - ref_tensor
    diff = torch.clamp(diff, min=-20.0, max=20.0)
    ratio = torch.exp(diff)

    elementwise_loss = -(ratio * advantages_tensor) * weights_tensor
    elementwise_loss = torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)
    return elementwise_loss.sum()

  def _compute_ppo_loss(self, target_logprobs: torch.Tensor, weights_tensor: torch.Tensor, datum: Datum, loss_config: dict | None) -> torch.Tensor:
    """PPO loss for RL."""
    if "logprobs" not in datum.loss_fn_inputs or "advantages" not in datum.loss_fn_inputs:
      raise ValueError("ppo requires 'logprobs' and 'advantages' in loss_fn_inputs")

    ref_logprobs = datum.loss_fn_inputs["logprobs"].data
    advantages = datum.loss_fn_inputs["advantages"].data

    ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
    advantages_tensor = torch.tensor(advantages, dtype=target_logprobs.dtype, device=self.device)

    seq_len = min(target_logprobs.size(0), ref_tensor.size(0), advantages_tensor.size(0), weights_tensor.size(0))
    target_logprobs = target_logprobs[:seq_len]
    ref_tensor = ref_tensor[:seq_len]
    advantages_tensor = advantages_tensor[:seq_len]
    weights_tensor = weights_tensor[:seq_len]

    diff = target_logprobs - ref_tensor
    diff = torch.clamp(diff, min=-20.0, max=20.0)
    ratio = torch.exp(diff)

    epsilon = loss_config.get("clip_range", 0.2) if loss_config else 0.2

    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_tensor

    elementwise_objective = torch.min(surr1, surr2)

    # Optional KL penalty against the reference policy.
    # Uses the unbiased estimator (ratio - 1) - log(ratio), which is
    # non-negative and zero when the policy matches the reference.
    kl_coeff = loss_config.get("kl_coeff", 0.0) if loss_config else 0.0
    if kl_coeff > 0:
      kl = (ratio - 1) - diff
      elementwise_objective = elementwise_objective - kl_coeff * kl

    return -(elementwise_objective * weights_tensor).sum()

  def _sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val

  def set_active_adapter(self, adapter_id: str) -> None:
    """Switch which LoRA adapter is active."""
    if self.peft_model is not None:
      self.peft_model.set_adapter(adapter_id)

  def optim_step(self, adam_params: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Apply accumulated gradients and update model weights."""
    assert self.peft_model is not None, "Model must be loaded first."
    if not model_id:
      raise ValueError("model_id is required for optim_step")

    if model_id not in self.optimizers:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      print(f"Initializing AdamW optimizer for '{model_id}' with lr={lr}")
      params = [p for p in self.peft_model.parameters() if p.requires_grad]
      self.optimizers[model_id] = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
      )

    optimizer = self.optimizers[model_id]
    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    total_norm = 0.0
    for param in self.peft_model.parameters():
      if param.grad is not None:
        total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5

    clip_norm = adam_params.get("grad_clip_norm", 0.0)
    if clip_norm and clip_norm > 0.0:
      torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), clip_norm)

    optimizer.step()
    optimizer.zero_grad()

    self.save_adapter(model_id)

    return {
      "metrics": {
        "grad_norm:mean": self._sanitize_float(total_norm),
      },
    }

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
  ) -> dict[str, Any]:
    """Generate completions from the current model."""
    assert self.peft_model is not None, "Model must be loaded first."

    if model_id:
      self.peft_model.set_adapter(model_id)
    self.peft_model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.peft_model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=None,
        top_k=None,
        num_return_sequences=num_samples,
        output_scores=True,
        return_dict_in_generate=True,
      )

    sequences_out = []
    for seq_idx in range(num_samples):
      gen_sequences = outputs.sequences[seq_idx]
      generated_tokens = gen_sequences[len(prompt_tokens) :].cpu().tolist()

      logprobs = []
      for token_step_idx in range(len(generated_tokens)):
        score_tensor = outputs.scores[token_step_idx]
        logprob_dist = torch.nn.functional.log_softmax(score_tensor[seq_idx], dim=-1)
        token_id = generated_tokens[token_step_idx]
        logprob = logprob_dist[token_id].item()
        logprobs.append(self._sanitize_float(logprob))

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    return {"sequences": sequences_out}


def main() -> None:
  from clock_cycle import main as clock_cycle_main

  clock_cycle_main()


if __name__ == "__main__":
  main()
