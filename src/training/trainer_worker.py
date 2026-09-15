# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from typing import Any

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.device import resolve_device


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


class BaseTrainerWorker:
  # Smallest padded sequence length when shape bucketing is on. Compile cost on
  # XLA is near-flat in sequence length, so generous rungs are cheap.
  MIN_LENGTH_BUCKET = 64

  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None

    # TPU requires an explicit OPEN_RL_DEVICE=tpu (Dockerfile.tpu sets it);
    # a torch_tpu venv can still run a cpu trainer, e.g. next to a TPU sampler.
    self.device = resolve_device()

  def forward_backward(self, model: PreTrainedModel, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs."""
    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    model.train()

    for batch in self.make_training_batches(data):
      batch_indices = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]

      if self.shape_bucketing_enabled():
        bucketed_rows = self.bucket_size(len(batch_data))
        batch_data.extend(self.make_padding_datum(batch_data[0]) for _ in range(bucketed_rows - len(batch_data)))

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)
      padded_target_len = weights.shape[1]
      target_logprobs = self.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          old_logprob_data = [datum.loss_fn_inputs["logprobs"].data for datum in batch_data]
          advantage_data = [datum.loss_fn_inputs["advantages"].data for datum in batch_data]
          old_logprobs = self.pad_sequences(old_logprob_data, lengths, torch.float32, width=padded_target_len)
          advantages = self.pad_sequences(advantage_data, lengths, torch.float32, width=padded_target_len)
          elementwise_loss = losses.importance_sampling_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
          )
        case "ppo":
          old_logprob_data = [datum.loss_fn_inputs["logprobs"].data for datum in batch_data]
          advantage_data = [datum.loss_fn_inputs["advantages"].data for datum in batch_data]
          old_logprobs = self.pad_sequences(old_logprob_data, lengths, torch.float32, width=padded_target_len)
          advantages = self.pad_sequences(advantage_data, lengths, torch.float32, width=padded_target_len)
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
      "metrics": {"loss:mean": self.sanitize_float(mean_loss), "loss:sum": self.sanitize_float(total_loss)},
      "loss_fn_outputs": completed_loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
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

    bucketing = self.shape_bucketing_enabled()

    for item in ordered_data:
      length = len(item[1].model_input)
      next_max_len = max(batch_max_len, length)
      next_size = len(batch) + 1
      if bucketing:
        # Cost the padded shape the batch will actually run at.
        next_max_len = self.bucket_size(next_max_len, self.MIN_LENGTH_BUCKET)
        next_size = self.bucket_size(next_size)
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

  def shape_bucketing_enabled(self) -> bool:
    """When set, every batch is padded to a power-of-two (batch, length) pair so XLA backends see a bounded set of shapes."""
    return os.getenv("OPEN_RL_TRAIN_SHAPE_BUCKETS", "").lower() in ("1", "true", "yes")

  def bucket_size(self, value: int, minimum: int = 1) -> int:
    """Round value up to the next power-of-two rung, starting at minimum."""
    bucket = minimum
    while bucket < value:
      bucket *= 2
    return bucket

  def make_padding_datum(self, template: Datum) -> Datum:
    """Return a zero-weight dummy datum used to pad a batch up to its bucketed row count."""
    loss_fn_inputs = {key: TensorData(data=[0]) for key in template.loss_fn_inputs}
    loss_fn_inputs["weights"] = TensorData(data=[0.0])
    return Datum(loss_fn_inputs=loss_fn_inputs, model_input=[0])

  def pad_sequences(
    self,
    sequences: list[list[int] | list[float]],
    lengths: list[int],
    dtype: torch.dtype,
    pad_value: int | float = 0,
    width: int | None = None,
  ) -> torch.Tensor:
    """Return padded values with shape [batch, width or max(lengths)]."""
    padded = torch.full((len(sequences), width or max(lengths)), pad_value, dtype=dtype, device=self.device)
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
    if self.shape_bucketing_enabled():
      max_input_len = self.bucket_size(max_input_len, self.MIN_LENGTH_BUCKET)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, pad_token_id, width=max_input_len)
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
    max_target_len = max(lengths)
    if self.shape_bucketing_enabled():
      max_target_len = self.bucket_size(max_target_len, self.MIN_LENGTH_BUCKET)
    target_token_ids = self.pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long, width=max_target_len)
    weight_sequences = [
      datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
    ]
    weights = self.pad_sequences(weight_sequences, lengths, torch.float32, width=max_target_len)

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
