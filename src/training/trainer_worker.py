# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.distributed import local_rank
from training.types import DTYPES, Datum, deployment_dtype

# Position marker for a filler pass, see _run_batches.
FILLER_DATUM_INDEX = -1

TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def trainer_dtype_from_env() -> str:
  """OPEN_RL_TRAINER_DTYPE, which the worker manager always sets; a worker
  started by hand falls back to the deployment dtype (OPEN_RL_DTYPE)."""
  value = os.getenv("OPEN_RL_TRAINER_DTYPE")
  if value is None:
    return deployment_dtype()
  if value not in DTYPES:
    raise ValueError(f"OPEN_RL_TRAINER_DTYPE must be one of {', '.join(DTYPES)}, got {value!r}")
  return value


def shard_datum_indices(count: int, shard_rank: int, shard_count: int) -> list[int]:
  """Round-robin datum ownership for one data-parallel rank."""
  return list(range(shard_rank, count, shard_count))


class BaseTrainerWorker:
  # Whether backward runs collectives that every data-parallel rank must join
  # (FSDP). Such backends need equal pass counts across ranks.
  backward_runs_collectives = False

  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")
    # The dtype the API server resolved for this worker's models. Set once per
    # process: a LoRA worker loads its shared base model before any command.
    self.dtype_name = trainer_dtype_from_env()

  def model_dtype(self) -> torch.dtype:
    """The torch dtype to load and train in. Raises rather than falling back
    when this device cannot run it."""
    dtype = TORCH_DTYPES[self.dtype_name]
    if dtype == torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
      raise RuntimeError(
        f"trainer_dtype=bfloat16 was requested but {torch.cuda.get_device_name(self.device)} does not support bfloat16; "
        "use a GPU with compute capability 8.0+ or trainer_dtype=float32 where the trainer supports it"
      )
    return dtype

  def check_command_dtype(self, dtype: str | None) -> None:
    """Refuse a model whose dtype is not the one this process runs in."""
    if dtype is not None and dtype != self.dtype_name:
      raise ValueError(f"model asks for trainer_dtype={dtype} but this trainer runs in {self.dtype_name}")

  def is_ready(self) -> bool:
    """Whether a model is loaded and the worker can serve requests. The
    request processor announces this to the API server, which shows it to
    clients as the worker's state."""
    return False

  def forward_backward(
    self,
    model: PreTrainedModel,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None = None,
    forward_only: bool = False,
  ) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs.

    With ``forward_only`` (TrainingClient.forward) the pass runs without
    autograd and in eval mode, and no gradient is accumulated: the cookbook's
    NLL evaluator calls it on held-out data right before a training step, so
    a backward here would fold the test set into the next optim_step.
    """
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    if forward_only:
      model.eval()
    else:
      model.train()

    with torch.set_grad_enabled(not forward_only):
      total_loss = self._run_batches(model, data, loss_fn, loss_config, forward_only, loss_fn_outputs)
    return self._finish(data, loss_fn_outputs, total_loss)

  # Data-parallel sharding hooks. By default every rank runs every datum; a
  # backend that reduces gradients across a DP group overrides these with it.

  def shard_rank(self) -> int:
    return 0

  def shard_count(self) -> int:
    return 1

  def shard_all_reduce_max(self, value: int) -> int:
    return value

  def shard_all_reduce_sum(self, value: float) -> float:
    return value

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    return [value]

  def _run_batches(
    self,
    model: PreTrainedModel,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[dict[str, Any] | None],
  ) -> float:
    """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss.

    Under data parallelism each rank runs a round-robin shard of the datums and
    scales its loss by the shard count, which undoes FSDP's gradient mean. When
    backward runs collectives, short ranks run zero-scaled filler passes so
    every rank makes the same number of backward calls.
    """
    shard_count = self.shard_count()
    local_indices = shard_datum_indices(len(data), self.shard_rank(), shard_count)
    local_data = [data[idx] for idx in local_indices]
    local_batches = self.make_training_batches(local_data)
    if shard_count > 1 and self.backward_runs_collectives:
      filler_passes = self.shard_all_reduce_max(len(local_batches)) - len(local_batches)
      filler = local_data[0] if local_data else data[0]
      local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    off_policy_cfg = losses.parse_off_policy_config(loss_config) if loss_fn in ("importance_sampling", "ppo") else losses.OffPolicyConfig()

    total_loss = 0.0
    for batch in local_batches:
      batch_positions = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]
      is_filler = batch_positions == [FILLER_DATUM_INDEX]
      batch_indices = [] if is_filler else [local_indices[position] for position in batch_positions]

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)

      topk_token_ids: torch.Tensor | None = None
      sampler_topk_logprobs: torch.Tensor | None = None
      trainer_topk_logprobs: torch.Tensor | None = None
      if off_policy_cfg.score_centering:
        topk_token_ids, sampler_topk_logprobs = self.pad_topk_inputs(batch_data, lengths, weights)

      if off_policy_cfg.uses_logit_replay or off_policy_cfg.score_centering:
        target_logprobs, trainer_topk_logprobs = self.compute_target_and_topk_logprobs(
          model,
          input_ids,
          attention_mask,
          target_token_ids,
          top_p=off_policy_cfg.top_p,
          temperature=off_policy_cfg.temperature,
          topk_token_ids=topk_token_ids,
        )
      else:
        target_logprobs = self.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.importance_sampling_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
            loss_config,
            trainer_topk_logprobs=trainer_topk_logprobs,
            sampler_topk_logprobs=sampler_topk_logprobs,
          )
        case "ppo":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.ppo_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
            loss_config,
            trainer_topk_logprobs=trainer_topk_logprobs,
            sampler_topk_logprobs=sampler_topk_logprobs,
          )
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      per_datum_loss = elementwise_loss.sum(dim=1)
      loss = per_datum_loss.sum()
      if not forward_only:
        (loss * (0.0 if is_filler else float(shard_count))).backward()
      if not is_filler:
        total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

    if shard_count > 1:
      total_loss = self.shard_all_reduce_sum(total_loss)
      for part in self.shard_all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output
    return total_loss

  def _finish(self, data: list[Datum], loss_fn_outputs: list[dict[str, Any] | None], total_loss: float) -> dict[str, Any]:
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

  def pad_topk_inputs(
    self,
    data: list[Datum],
    lengths: list[int],
    weights: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded top-K token IDs and sampler logprobs with shape ``[batch, max_target_len, K]``.

    Accepts ``topk_token_ids`` and ``topk_logprobs`` either already aligned to
    ``target_tokens`` (length ``T``) or covering only the generated completion
    suffix (length ``C < T`` where the leading ``T - C`` prompt positions have
    ``weights == 0``).
    """
    batch_size = len(data)
    max_len = max(lengths)
    parsed_rows: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    max_k = 0

    for row, datum in enumerate(data):
      if "topk_token_ids" not in datum.loss_fn_inputs or "topk_logprobs" not in datum.loss_fn_inputs:
        raise ValueError("loss_fn_config['score_centering'] requires 'topk_token_ids' and 'topk_logprobs' in datum.loss_fn_inputs")
      ids_td = datum.loss_fn_inputs["topk_token_ids"]
      lps_td = datum.loss_fn_inputs["topk_logprobs"]
      if len(ids_td.data) != len(lps_td.data) or not ids_td.data:
        raise ValueError("topk_token_ids and topk_logprobs must be non-empty and have matching element counts")

      length = lengths[row]
      if ids_td.shape and len(ids_td.shape) == 2:
        t_rows, k = int(ids_td.shape[0]), int(ids_td.shape[1])
      elif lps_td.shape and len(lps_td.shape) == 2:
        t_rows, k = int(lps_td.shape[0]), int(lps_td.shape[1])
      elif len(ids_td.data) % length == 0:
        t_rows, k = length, len(ids_td.data) // length
      else:
        raise ValueError(
          f"Cannot infer 2D [T, K] shape for topk_token_ids of length {len(ids_td.data)} (target length {length}); specify shape=[T, K]"
        )

      if k <= 0 or t_rows * k != len(ids_td.data):
        raise ValueError(f"Invalid top-K shape [{t_rows}, {k}] for {len(ids_td.data)} elements")

      ids_tensor = torch.tensor(ids_td.data, dtype=torch.long, device=self.device).reshape(t_rows, k).clamp_min(0)
      lps_tensor = torch.tensor(lps_td.data, dtype=torch.float32, device=self.device).reshape(t_rows, k)
      parsed_rows.append((ids_tensor, lps_tensor, k))
      max_k = max(max_k, k)

    padded_ids = torch.zeros((batch_size, max_len, max_k), dtype=torch.long, device=self.device)
    padded_lps = torch.full((batch_size, max_len, max_k), -99999.0, dtype=torch.float32, device=self.device)

    for row, (ids_tensor, lps_tensor, k) in enumerate(parsed_rows):
      length = lengths[row]
      t_rows = ids_tensor.shape[0]
      if t_rows >= length:
        padded_ids[row, :length, :k] = ids_tensor[:length, :k]
        padded_lps[row, :length, :k] = lps_tensor[:length, :k]
      else:
        prefix_len = length - t_rows
        if torch.any(weights[row, :prefix_len] != 0):
          raise ValueError(
            f"topk_token_ids has {t_rows} rows for target length {length}, but the leading {prefix_len} positions have non-zero weights"
          )
        padded_ids[row, prefix_len:length, :k] = ids_tensor[:, :k]
        padded_lps[row, prefix_len:length, :k] = lps_tensor[:, :k]

    return padded_ids, padded_lps

  def compute_target_logprobs(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return selected target logprobs with shape [batch, max_target_len]."""
    target_logprobs, _ = self.compute_target_and_topk_logprobs(model, input_ids, attention_mask, target_token_ids)
    return target_logprobs

  def compute_target_and_topk_logprobs(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    top_p: float = 1.0,
    temperature: float = 1.0,
    topk_token_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return target logprobs ``[batch, seq_len]`` and optional top-K logprobs ``[batch, seq_len, K]``."""
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, : target_token_ids.shape[1], :]
    if top_p < 1.0 or temperature != 1.0:
      logits = losses.apply_top_p(logits, top_p=top_p, temperature=temperature)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_logprobs = log_probs.gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
    trainer_topk_logprobs = log_probs.gather(dim=-1, index=topk_token_ids) if topk_token_ids is not None else None
    return target_logprobs, trainer_topk_logprobs

  def generate(
    self,
    model: PreTrainedModel,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
    top_p: float = 1.0,
    top_k: int = -1,
    topk_sample_logprobs: int = 0,
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
        top_p=top_p if (do_sample and 0.0 < top_p < 1.0) else None,
        top_k=top_k if (do_sample and top_k > 0) else None,
        num_return_sequences=num_samples,
        output_scores=True,
        return_dict_in_generate=True,
      )

    sequences_out = []
    for seq_idx in range(num_samples):
      gen_sequences = outputs.sequences[seq_idx]
      generated_tokens = gen_sequences[len(prompt_tokens) :].cpu().tolist()

      logprobs = []
      topk_logprobs: list[list[tuple[int, float]]] | None = [] if topk_sample_logprobs > 0 else None
      for token_step_idx in range(len(generated_tokens)):
        score_tensor = outputs.scores[token_step_idx]
        logprob_dist = torch.nn.functional.log_softmax(score_tensor[seq_idx], dim=-1)
        token_id = generated_tokens[token_step_idx]
        logprob = logprob_dist[token_id].item()
        logprobs.append(self.sanitize_float(logprob))
        if topk_logprobs is not None:
          k = min(int(topk_sample_logprobs), int(logprob_dist.shape[-1]))
          vals, idxs = torch.topk(logprob_dist, k=k, dim=-1)
          topk_logprobs.append([(int(tid), self.sanitize_float(float(lp))) for tid, lp in zip(idxs.cpu().tolist(), vals.cpu().tolist(), strict=True)])

      seq_entry: dict[str, Any] = {"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"}
      if topk_logprobs is not None:
        seq_entry["topk_logprobs"] = topk_logprobs
        seq_entry["topk_k"] = int(topk_sample_logprobs)
      sequences_out.append(seq_entry)

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
