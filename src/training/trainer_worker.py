# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.commands import CreateModel, CreateModelFromState, SaveWeightsForSampler
from training.distributed import all_gather_object, all_reduce_max, all_reduce_sum, group_rank, group_size, local_rank
from training.types import Datum, SamplerWeights, TensorData

__all__ = ["BaseTrainerWorker", "Datum", "TensorData", "chunk_target_logprob", "tmp_dir"]

# Position marker for a filler pass, see forward_backward.
FILLER_DATUM_INDEX = -1
ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"

RATIO_STATS_ZERO = {"max_abs_log_ratio": 0.0, "tokens": 0.0, "tokens_abs_log_ratio_gt1": 0.0, "tokens_abs_log_ratio_gt5": 0.0}


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def chunk_target_logprob(
  hidden_chunk: torch.Tensor,
  weight: torch.Tensor,
  bias: torch.Tensor | None,
  target_chunk: torch.Tensor,
  softcap: float | None,
) -> torch.Tensor:
  """logit[target] - logsumexp for one chunk of hidden states.

  The [chunk, vocab] logits are local to this call, so a backend that runs it
  under activation checkpointing never stores full-sequence logits.
  """
  logits = torch.nn.functional.linear(hidden_chunk, weight, bias)
  if logits.dtype in (torch.float16, torch.bfloat16):
    logits = logits.float()
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  target_logit = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
  return target_logit - torch.logsumexp(logits, dim=-1)


class BaseTrainerWorker:
  """Loss computation shared by every backend, plus the worker API the request
  loop drives.

  A worker owns one trainable model: create() builds it, trainer() returns it,
  and the Trainer methods (forward_backward, optim_step, save_state,
  load_from_state, publish_sampler_weights, save_weights, generate) act on it.
  A host that serves several adapters from one base model overrides create()
  and trainer() to hand back a bound adapter instead of itself.
  """

  # Whether samplers receive whole checkpoints (True) or LoRA adapters (False).
  full_parameter = False

  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None
    self.base_model_name: str | None = None
    self.model_id: str | None = None
    self.ratio_stats = dict(RATIO_STATS_ZERO)

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  @property
  def is_lora(self) -> bool:
    return not self.full_parameter

  # -- worker API -----------------------------------------------------------

  def create(self, command: CreateModel) -> "BaseTrainerWorker":
    """Build the model the command describes and return its Trainer."""
    self.model_id = command.model_id
    config = command.full_config if self.full_parameter else command.lora_config
    self.create_model(command.base_model, command.model_id, config)
    return self

  def restore(self, command: CreateModelFromState) -> "BaseTrainerWorker":
    self.model_id = command.model_id
    self.load_from_state(command.state_path, command.restore_optimizer)
    return self

  def trainer(self, model_id: str) -> "BaseTrainerWorker":
    """The Trainer for model_id. A single-model worker is its own trainer."""
    return self

  def publish_sampler_weights(self, command: SaveWeightsForSampler) -> SamplerWeights:
    """Write what the samplers load. A full-parameter worker writes a whole
    checkpoint at the path the sampling ref names."""
    ref = command.path or command.sampling_session_id
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    path = os.path.join(tmp_dir(), "sampler_full", ref.removeprefix("tinker://").lstrip("/"))
    self.save_state(path, include_optimizer=False, kind="sampler")
    return SamplerWeights(kind="checkpoint", path=path)

  def save_weights(self, alias: str | None = None) -> dict[str, Any]:
    return self.save_model(alias or self.model_id)

  # Hooks for the GPU lease a dedicated worker runs under. sleep and wake_up
  # bracket the lease; save_needs_gpu says whether a save must run inside it.
  def sleep(self) -> None:
    pass

  def wake_up(self) -> None:
    pass

  def save_needs_gpu(self) -> bool:
    return False

  def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
    if not ENABLE_GRADIENT_CHECKPOINTING:
      return
    try:
      if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
      if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
      print("Gradient checkpointing and input require grads enabled.")
    except Exception as e:
      print(f"Failed to enable gradient checkpointing: {e}")

  def checkpoint_metadata(self, model_id: str | None, kind: str = "state", has_optimizer: bool = False, **extra: Any) -> dict[str, Any]:
    return {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": has_optimizer,
      "model_id": model_id,
      "timestamp": time.time(),
      **extra,
    }

  def build_optimizer(
    self,
    params: list[torch.nn.Parameter],
    adam_params: dict[str, Any],
    label: str = "model",
    **kwargs: Any,
  ) -> torch.optim.Optimizer:
    lr = adam_params.get("learning_rate", 1e-4)
    print(f"Initializing AdamW optimizer for {label} with lr={lr}")
    return torch.optim.AdamW(
      params,
      lr=lr,
      betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
      eps=adam_params.get("eps", 1e-12),
      weight_decay=adam_params.get("weight_decay", 0.0),
      **kwargs,
    )

  def clip_gradients(self, params: list[torch.nn.Parameter], max_grad_norm: float) -> tuple[float, float]:
    total_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
    return total_norm, clip_coef

  def step_optimizer(
    self,
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    adam_params: dict[str, Any],
    default_clip: float = 0.0,
  ) -> tuple[float, float, float, float]:
    if (lr := adam_params.get("learning_rate")) is not None:
      for group in optimizer.param_groups:
        group["lr"] = lr
    max_grad_norm = adam_params.get("grad_clip_norm") or default_clip or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf
    t0 = time.perf_counter()
    total_norm, clip_coef = self.clip_gradients(params, max_grad_norm)
    t1 = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    t2 = time.perf_counter()
    return total_norm, clip_coef, t1 - t0, t2 - t1

  def data_parallel_group(self) -> dist.ProcessGroup | None:
    """The process group whose ranks split the datums of one forward_backward.

    None means this process trains alone. Ranks that share one model replica
    (tensor or context parallel) must see identical datums, so a backend with
    such ranks returns its data-parallel subgroup, not the world.
    """
    return None

  def data_parallel_loss_scale(self) -> float:
    """Factor applied to every rank's loss before backward.

    forward_backward wants the sum of the gradients over all datums. A backend
    whose gradient reduction averages over the data-parallel group (FSDP)
    returns the group size to undo that mean; one that sums returns 1.
    """
    return 1.0

  def forward_backward(self, model: PreTrainedModel, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs.

    Under data parallelism each rank owns a round-robin shard of the datums and
    every rank runs the same number of passes (short ranks run zero-scaled
    fillers) so the collectives inside backward line up.
    """
    if not data:
      return {"metrics": {"loss:mean": 0.0, "loss:sum": 0.0}, "loss_fn_outputs": [], "loss_fn_output_type": "ArrayRecord"}

    group = self.data_parallel_group()
    dp_rank, dp_size = group_rank(group), group_size(group)
    loss_scale = self.data_parallel_loss_scale()
    local_indices = list(range(dp_rank, len(data), dp_size))
    local_batches = self.make_training_batches([data[idx] for idx in local_indices])
    if dp_size > 1:
      filler_passes = int(all_reduce_max(len(local_batches), group)) - len(local_batches)
      if filler_passes > 0:
        # Only the collectives matter, so the cheapest datum will do.
        filler = min(data, key=lambda datum: len(datum.model_input))
        local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    model.train()

    for batch in local_batches:
      is_filler = batch[0][0] == FILLER_DATUM_INDEX
      batch_indices = [] if is_filler else [local_indices[position] for position, _ in batch]
      batch_data = [datum for _, datum in batch]

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)

      old_logprobs = advantages = None
      if loss_fn in ("importance_sampling", "ppo"):
        old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
        advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)

      # A batch without gradient (a GRPO group whose rewards all tied) still
      # costs a full backward unless its forward runs without a graph. Under
      # data parallelism the pass must still happen for the collectives.
      skip_backward = dp_size == 1 and self.batch_has_no_gradient(loss_fn, loss_config, weights, advantages)
      with torch.no_grad() if skip_backward else nullcontext():
        target_logprobs = self.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)
      if old_logprobs is not None and not is_filler:
        self.record_ratio_stats(target_logprobs, old_logprobs, weights, advantages)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          elementwise_loss = losses.importance_sampling_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
          )
        case "ppo":
          elementwise_loss = losses.ppo_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
            loss_config,
          )
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      per_datum_loss = elementwise_loss.sum(dim=1)
      loss = per_datum_loss.sum()
      if not skip_backward:
        (loss * (0.0 if is_filler else loss_scale)).backward()
      if not is_filler:
        total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

    if dp_size > 1:
      total_loss = all_reduce_sum(total_loss, group)
      for part in all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}, group):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output

    mean_loss = total_loss / max(1, len(data))
    if any(output is None for output in loss_fn_outputs):
      raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")

    return {
      "metrics": {"loss:mean": self.sanitize_float(mean_loss), "loss:sum": self.sanitize_float(total_loss)},
      "loss_fn_outputs": loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def batch_has_no_gradient(self, loss_fn: str, loss_config: dict | None, weights: torch.Tensor, advantages: torch.Tensor | None) -> bool:
    if advantages is None or bool(((advantages != 0) & (weights != 0)).any()):
      return False
    has_kl_penalty = loss_fn == "ppo" and bool(loss_config and loss_config.get("kl_coeff", 0.0) > 0)
    return not has_kl_penalty

  def record_ratio_stats(self, target_logprobs: torch.Tensor, old_logprobs: torch.Tensor, weights: torch.Tensor, advantages: torch.Tensor) -> None:
    """Tail of the sampler/trainer log-ratio over action tokens, accumulated
    until the next optim_step reports it. The mean KL is blind to a handful of
    tokens with a huge ratio, and under an unclipped loss those few tokens can
    be most of the gradient. Prompt positions carry weight 1 but no advantage
    and a sampled logprob of 0, so only positions with an advantage count."""
    active = (weights != 0) & (advantages != 0)
    if not bool(active.any()):
      return
    log_ratio = (target_logprobs.detach().float() - old_logprobs.float())[active].abs()
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=1e4)
    stats = self.ratio_stats
    stats["max_abs_log_ratio"] = max(stats["max_abs_log_ratio"], float(log_ratio.max()))
    stats["tokens"] += float(active.sum())
    stats["tokens_abs_log_ratio_gt1"] += float((log_ratio > 1.0).sum())
    stats["tokens_abs_log_ratio_gt5"] += float((log_ratio > 5.0).sum())

  def ratio_metrics(self) -> dict[str, float]:
    """Reduce the accumulated tail stats over the data-parallel group and reset them."""
    group = self.data_parallel_group()
    stats, self.ratio_stats = self.ratio_stats, dict(RATIO_STATS_ZERO)
    max_abs = all_reduce_max(stats["max_abs_log_ratio"], group)
    tokens = all_reduce_sum(stats["tokens"], group)
    gt1 = all_reduce_sum(stats["tokens_abs_log_ratio_gt1"], group)
    gt5 = all_reduce_sum(stats["tokens_abs_log_ratio_gt5"], group)
    return {
      "ratio/max_abs_log:max": max_abs,
      "ratio/tokens_abs_log_gt1:sum": gt1,
      "ratio/tokens_abs_log_gt5:sum": gt5,
      "ratio/frac_abs_log_gt1:mean": gt1 / tokens if tokens else 0.0,
    }

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
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

  def pad_sequences(
    self,
    sequences: list[list[int] | list[float]],
    lengths: list[int],
    dtype: torch.dtype,
    pad_value: int | float = 0,
  ) -> torch.Tensor:
    """Return padded values with shape [batch, max(lengths)]."""
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.device)
    for row, sequence in enumerate(sequences):
      length = lengths[row]
      padded[row, :length] = padded.new_tensor(sequence[:length])
    return padded

  def pad_model_inputs(
    self,
    data: list[Datum],
  ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, pad_token_id)
    attention_mask = input_ids.new_zeros((batch_size, max_input_len))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1

    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(
    self,
    data: list[Datum],
    input_lengths: list[int],
  ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return target_token_ids and weights with shape [batch, max_target_len]."""
    batch_size = len(data)
    target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
    lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
    target_token_ids = self.pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long)
    weight_sequences = [
      datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
    ]
    weights = self.pad_sequences(weight_sequences, lengths, torch.float32)

    return target_token_ids, weights, lengths

  def compute_target_logprobs(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return selected target logprobs with shape [batch, max_target_len]."""
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, : target_token_ids.shape[1], :]
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)

  def generate(
    self,
    model: PreTrainedModel,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    """Generate completions from model."""
    model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self.prompt_logprobs(model, input_tensor) if include_prompt_logprobs else None

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = model.generate(
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
        logprobs.append(self.sanitize_float(logprob))

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    result = {"sequences": sequences_out}
    if prompt_logprobs is not None:
      result["prompt_logprobs"] = prompt_logprobs
    return result

  def prompt_logprobs(self, model: PreTrainedModel, input_tensor: torch.Tensor) -> list[float | None]:
    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = model(input_tensor, attention_mask=attention_mask)
      logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

    prompt_tokens = input_tensor[0].tolist()
    prompt_logprobs: list[float | None] = [None]
    for token_idx, token_id in enumerate(prompt_tokens[1:]):
      logprob = logprob_dist[token_idx, token_id].item()
      prompt_logprobs.append(self.sanitize_float(logprob))

    return prompt_logprobs

  def sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val
