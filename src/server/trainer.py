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

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


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


def active_adapter_parameters(model: Any, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


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

    # Store per-adapter training state by model_id (adapter ID).
    self.adapter_states: dict[str, dict[str, Any]] = {}
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}

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

  def _target_lora_modules(self, config: LoraConfig) -> list[str]:
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
      target_suffixes.append("lm_head")

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

    if config.train_unembed:
      print("[WARN] train_unembed=True is not supported with vLLM LoRA sampling; ignoring it and using train_unembed=False.")
      config.train_unembed = False

    if not any([config.train_attn, config.train_mlp, config.train_unembed]):
      raise ValueError("At least one LoRA training target must be enabled.")

    print(f"Creating LoRA adapter '{adapter_id}'...")

    peft_config = PeftLoraConfig(
      task_type="CAUSAL_LM",
      r=config.rank,
      lora_alpha=config.lora_alpha,
      lora_dropout=config.lora_dropout,
      bias="none",
      target_modules=self._target_lora_modules(config),
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
    """Core training step: forward pass, loss computation, and backward pass."""
    assert self.peft_model is not None, "Model must be loaded first."
    if model_id:
      self.peft_model.set_adapter(model_id)

    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    self.peft_model.train()

    for batch in self._make_training_batches(data):
      batch_indices = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]

      target_logprobs, weights, aux_inputs, lengths = self._get_logprobs_batch(batch_data)
      match loss_fn:
        case "cross_entropy":
          elementwise_loss = self._cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          elementwise_loss = self._importance_sampling_loss(target_logprobs, weights, aux_inputs)
        case "ppo":
          elementwise_loss = self._ppo_loss(target_logprobs, weights, aux_inputs, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      per_datum_loss = elementwise_loss.sum(dim=1)
      loss = per_datum_loss.sum()
      loss.backward()
      total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

    mean_loss = total_loss / max(1, len(data))
    completed_loss_fn_outputs = []
    for output in loss_fn_outputs:
      if output is None:
        raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")
      completed_loss_fn_outputs.append(output)

    return {
      "metrics": {"loss:mean": self._sanitize_float(mean_loss), "loss:sum": self._sanitize_float(total_loss)},
      "loss_fn_outputs": completed_loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def _make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    """Group examples for the single padded forward/backward path."""
    if len(data) <= 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0"))

    if token_budget <= 0:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    ordered_data = sorted(enumerate(data), key=lambda item: len(item[1].model_input))
    batches: list[list[tuple[int, Datum]]] = []
    batch: list[tuple[int, Datum]] = []
    batch_max_len = 0

    for item in ordered_data:
      length = len(item[1].model_input)
      next_max_len = max(batch_max_len, length)
      next_size = len(batch) + 1
      over_token_budget = next_max_len * next_size > token_budget

      if batch and over_token_budget:
        batches.append(batch)
        batch = []
        batch_max_len = 0

      batch.append(item)
      batch_max_len = max(batch_max_len, length)

    if batch:
      batches.append(batch)

    return batches

  def _get_logprobs_batch(self, data: list[Datum]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[int]]:
    assert self.peft_model is not None

    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_tensor = torch.full((batch_size, max_input_len), pad_token_id, dtype=torch.long, device=self.device)
    attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long, device=self.device)
    for row, datum in enumerate(data):
      input_len = input_lengths[row]
      input_tensor[row, :input_len] = torch.tensor(datum.model_input, dtype=torch.long, device=self.device)
      attention_mask[row, :input_len] = 1

    target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
    lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
    max_target_len = max(lengths)
    loss_dtype = torch.float32

    targets = torch.zeros((batch_size, max_target_len), dtype=torch.long, device=self.device)
    weights = torch.zeros((batch_size, max_target_len), dtype=loss_dtype, device=self.device)
    aux_inputs: dict[str, torch.Tensor] = {}

    aux_keys = set().union(*(datum.loss_fn_inputs.keys() for datum in data)) - {"target_tokens", "weights"}
    for key in aux_keys:
      aux_inputs[key] = torch.zeros((batch_size, max_target_len), dtype=loss_dtype, device=self.device)

    for row, datum in enumerate(data):
      seq_len = lengths[row]
      targets_data = datum.loss_fn_inputs["target_tokens"].data[:seq_len]
      targets[row, :seq_len] = torch.tensor(targets_data, dtype=torch.long, device=self.device)

      weights_data = datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row]
      weights[row, :seq_len] = torch.tensor(weights_data[:seq_len], dtype=loss_dtype, device=self.device)

      for key, aux_tensor in aux_inputs.items():
        if key not in datum.loss_fn_inputs:
          continue
        values = datum.loss_fn_inputs[key].data[:seq_len]
        aux_tensor[row, :seq_len] = torch.tensor(values, dtype=loss_dtype, device=self.device)

    outputs = self.peft_model(input_tensor, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :max_target_len, :]
    target_logprobs = torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return target_logprobs, weights, aux_inputs, lengths

  def _cross_entropy_loss(self, target_logprobs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return -target_logprobs * weights

  def _importance_sampling_loss(
    self,
    target_logprobs: torch.Tensor,
    weights: torch.Tensor,
    aux_inputs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    ref, advantages = self._reference_logprobs_and_advantages(aux_inputs, "importance_sampling")
    ratio = self._policy_ratio(target_logprobs, ref)
    elementwise_loss = -(ratio * advantages) * weights
    return torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)

  def _ppo_loss(
    self,
    target_logprobs: torch.Tensor,
    weights: torch.Tensor,
    aux_inputs: dict[str, torch.Tensor],
    loss_config: dict | None,
  ) -> torch.Tensor:
    ref, advantages = self._reference_logprobs_and_advantages(aux_inputs, "ppo")
    diff = torch.clamp(target_logprobs - ref, min=-20.0, max=20.0)
    ratio = torch.exp(diff)
    epsilon = loss_config.get("clip_range", 0.2) if loss_config else 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    elementwise_objective = torch.min(surr1, surr2)
    kl_coeff = loss_config.get("kl_coeff", 0.0) if loss_config else 0.0
    if kl_coeff > 0:
      kl = (ratio - 1) - diff
      elementwise_objective = elementwise_objective - kl_coeff * kl
    return -(elementwise_objective * weights)

  def _reference_logprobs_and_advantages(self, aux_inputs: dict[str, torch.Tensor], loss_fn: str) -> tuple[torch.Tensor, torch.Tensor]:
    ref = aux_inputs.get("logprobs")
    advantages = aux_inputs.get("advantages")
    if ref is None or advantages is None:
      raise ValueError(f"{loss_fn} requires 'logprobs' and 'advantages' in loss_fn_inputs")
    return ref, advantages

  def _policy_ratio(self, target_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(target_logprobs - ref_logprobs, min=-20.0, max=20.0))

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
        "grad_norm:mean": self._sanitize_float(total_norm.item()),
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
    """Generate completions from the current model."""
    assert self.peft_model is not None, "Model must be loaded first."

    if model_id:
      self.peft_model.set_adapter(model_id)
    self.peft_model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self._prompt_logprobs(input_tensor) if include_prompt_logprobs else None

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

    result = {"sequences": sequences_out}
    if prompt_logprobs is not None:
      result["prompt_logprobs"] = prompt_logprobs
    return result

  def _prompt_logprobs(self, input_tensor: torch.Tensor) -> list[float | None]:
    assert self.peft_model is not None, "Model must be loaded first."

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.peft_model(input_tensor, attention_mask=attention_mask)
      logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

    prompt_tokens = input_tensor[0].tolist()
    prompt_logprobs: list[float | None] = [None]
    for token_idx, token_id in enumerate(prompt_tokens[1:]):
      logprob = logprob_dist[token_idx, token_id].item()
      prompt_logprobs.append(self._sanitize_float(logprob))

    return prompt_logprobs


def main() -> None:
  from clock_cycle import main as clock_cycle_main

  clock_cycle_main()


if __name__ == "__main__":
  main()
