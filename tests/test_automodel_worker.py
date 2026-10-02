"""Automodel worker checks that run on CPU without nemo-automodel."""

import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server.training_requests_processor import build_worker
from training import automodel_worker
from training.automodel_worker import AutomodelTrainingWorker, lora_target_patterns
from training.trainer_worker import BaseTrainerWorker
from training.types import Datum, FFTConfig, LoraConfig, TensorData


class ClipGradientsTest(unittest.TestCase):
  """The per-tensor norm stack equals torch's clip_grad_norm_ on plain tensors."""

  def make_worker(self):
    torch.manual_seed(1)
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    for param in params:
      param.grad = torch.randn_like(param)
    worker = AutomodelTrainingWorker()
    worker.trainable_params = params
    return worker, params

  def test_clips_like_torch(self) -> None:
    worker, params = self.make_worker()
    reference = [torch.nn.Parameter(param.detach().clone()) for param in params]
    for ref, param in zip(reference, params, strict=True):
      ref.grad = param.grad.clone()

    total = worker.clip_gradients(0.5)
    expected_total = torch.nn.utils.clip_grad_norm_(reference, 0.5)

    self.assertAlmostEqual(total, expected_total.item(), places=5)
    for ref, param in zip(reference, params, strict=True):
      torch.testing.assert_close(param.grad, ref.grad)

  def test_no_clipping_under_the_threshold(self) -> None:
    worker, params = self.make_worker()
    before = [param.grad.clone() for param in params]
    worker.clip_gradients(float("inf"))
    for grad, param in zip(before, params, strict=True):
      torch.testing.assert_close(param.grad, grad)


class LoraTargetsTest(unittest.TestCase):
  """The adapter wraps what the client's LoraConfig asks for, the way the HF LoRA worker does."""

  def test_targets_follow_the_config(self) -> None:
    attn = lora_target_patterns(LoraConfig(train_attn=True, train_mlp=False))
    self.assertIn("model.*.layers.*.q_proj", attn)
    self.assertIn("model.*.layers.*.in_proj_qkv", attn)
    self.assertFalse(any(pattern.endswith(("gate_proj", "up_proj", "down_proj", "lm_head")) for pattern in attn))
    both = lora_target_patterns(LoraConfig())
    self.assertIn("model.*.layers.*.gate_proj", both)
    self.assertNotIn("lm_head", both)

  def test_unembed_is_never_wrapped(self) -> None:
    self.assertNotIn("lm_head", lora_target_patterns(LoraConfig(train_unembed=True)))
    with self.assertRaises(ValueError):
      lora_target_patterns(LoraConfig(train_attn=False, train_mlp=False, train_unembed=True))


class ChunkedLogprobsTest(unittest.TestCase):
  """Chunked projection from hidden states matches the full-logits path, values and gradients."""

  def test_matches_full_logits(self) -> None:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 5:] = 0
    targets = torch.randint(0, 64, (2, 7))

    def logprobs_and_grad(worker):
      model.zero_grad()
      logprobs = worker.compute_target_logprobs(model, input_ids, attention_mask, targets)
      logprobs[attention_mask.bool()].sum().backward()
      return logprobs.detach(), model.model.embed_tokens.weight.grad.clone()

    expected, expected_grad = logprobs_and_grad(BaseTrainerWorker())
    with patch.object(automodel_worker, "LOGPROB_CHUNK", 3):
      actual, actual_grad = logprobs_and_grad(AutomodelTrainingWorker())

    mask = attention_mask.bool()
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(actual_grad, expected_grad)


class LinearLoRA(torch.nn.Module):
  """The part of Automodel's LinearLoRA the worker touches: frozen base,
  trainable A and B, scale, dropout and init_lora_weights."""

  def __init__(self, base: torch.nn.Linear, rank: int):
    super().__init__()
    self.base = base
    self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
    self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
    self.scale = 1.0
    self.dropout_p = 0.0

  def init_lora_weights(self, init_method: str) -> None:
    with torch.no_grad():
      torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
      self.lora_B.weight.fill_(0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.dropout(x, self.dropout_p, self.training)
    return self.base(x) + self.lora_B(self.lora_A(x) * self.scale)


@dataclass
class PeftConfig:
  """The fields of Automodel's PeftConfig the worker reads."""

  dim: int
  alpha: float
  dropout: float = 0.0


def tiny_lora_model(rank: int) -> LlamaForCausalLM:
  torch.manual_seed(0)
  config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
  model = LlamaForCausalLM(config)
  for param in model.parameters():
    param.requires_grad_(False)
  for layer in model.model.layers:
    layer.self_attn.q_proj = LinearLoRA(layer.self_attn.q_proj, rank)
    layer.mlp.gate_proj = LinearLoRA(layer.mlp.gate_proj, rank)
  return model


def datums(rows: list[list[int]]) -> list[Datum]:
  return [
    Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * (len(row) - 1))})
    for row in rows
  ]


@patch.object(automodel_worker, "MAX_LORA_RANK", 4)
class MultiAdapterTest(unittest.TestCase):
  """Adapters of any shape sharing one trainer train exactly as they would alone."""

  DATA = {"a": datums([[3, 5, 8, 13, 21], [2, 7, 1, 8]]), "b": datums([[9, 4, 6, 1, 1, 2], [5, 5, 3]])}
  CONFIGS = {
    "a": LoraConfig(rank=2, lora_alpha=8, lora_dropout=0.0, train_mlp=False),
    "b": LoraConfig(rank=4, lora_alpha=4, lora_dropout=0.0),
  }
  ADAM = {"learning_rate": 0.05}

  def make_worker(self) -> AutomodelTrainingWorker:
    worker = AutomodelTrainingWorker()
    worker.device = torch.device("cpu")

    def load_model(base_model_name: str) -> None:
      worker.base_model_name = base_model_name
      worker.model = tiny_lora_model(automodel_worker.MAX_LORA_RANK)

    worker.load_model = load_model
    worker.build_peft_config = lambda config: PeftConfig(dim=config.rank, alpha=config.lora_alpha)
    return worker

  def create(self, worker: AutomodelTrainingWorker, model_id: str, seed: int) -> None:
    worker.create_model("tiny", model_id, self.CONFIGS[model_id].model_copy(update={"seed": seed}))

  def step(self, worker: AutomodelTrainingWorker, model_id: str) -> None:
    worker.forward_backward(self.DATA[model_id], "cross_entropy", model_id=model_id)
    worker.optim_step(self.ADAM, model_id)

  def weights(self, worker: AutomodelTrainingWorker, model_id: str) -> list[torch.Tensor]:
    worker.activate(model_id)
    return [param.detach().clone() for param in worker.trainable_params]

  def test_interleaved_adapters_match_each_trained_alone(self) -> None:
    shared = self.make_worker()
    self.create(shared, "a", seed=1)
    self.create(shared, "b", seed=2)
    # Both adapters hold grads at once before either steps.
    shared.forward_backward(self.DATA["a"], "cross_entropy", model_id="a")
    shared.forward_backward(self.DATA["b"], "cross_entropy", model_id="b")
    shared.optim_step(self.ADAM, "a")
    shared.optim_step(self.ADAM, "b")
    self.step(shared, "b")
    self.step(shared, "a")

    for model_id, seed in (("a", 1), ("b", 2)):
      alone = self.make_worker()
      self.create(alone, model_id, seed)
      self.step(alone, model_id)
      self.step(alone, model_id)
      for actual, expected in zip(self.weights(shared, model_id), self.weights(alone, model_id), strict=True):
        torch.testing.assert_close(actual, expected)

  def test_a_new_adapter_starts_fresh(self) -> None:
    worker = self.make_worker()
    self.create(worker, "a", seed=1)
    self.step(worker, "a")
    self.create(worker, "b", seed=1)
    fresh = self.make_worker()
    self.create(fresh, "b", seed=1)
    for actual, expected in zip(self.weights(worker, "b"), self.weights(fresh, "b"), strict=True):
      torch.testing.assert_close(actual, expected)

  def test_unused_rank_and_modules_stay_zero(self) -> None:
    worker = self.make_worker()
    self.create(worker, "a", seed=1)
    self.create(worker, "b", seed=2)
    for _ in range(2):
      self.step(worker, "b")
      self.step(worker, "a")
    worker.activate("a")
    for name, module in worker.lora_modules():
      A, B = module.lora_A.weight, module.lora_B.weight
      if name.endswith("gate_proj"):
        self.assertEqual(A.count_nonzero(), 0)
        self.assertEqual(B.count_nonzero(), 0)
      else:
        self.assertEqual(A[2:].count_nonzero(), 0)
        self.assertEqual(B[:, 2:].count_nonzero(), 0)
        self.assertGreater(B[:, :2].count_nonzero(), 0)
      self.assertEqual(module.scale, 4.0)

  def test_saved_alpha_keeps_the_scale(self) -> None:
    worker = self.make_worker()
    self.create(worker, "a", seed=1)
    saved = worker.saved_peft_config(self.CONFIGS["a"])
    self.assertEqual(saved.dim, 4)
    self.assertEqual(saved.alpha / saved.dim, 8 / 2)

  def test_too_large_a_rank_or_another_base_is_refused(self) -> None:
    worker = self.make_worker()
    self.create(worker, "a", seed=1)
    with self.assertRaises(ValueError):
      worker.create_model("tiny", "big", LoraConfig(rank=8))
    with self.assertRaises(RuntimeError):
      worker.create_model("other", "c", LoraConfig(rank=2))

  def test_deleting_the_active_adapter_leaves_the_others_as_they_were(self) -> None:
    shared = self.make_worker()
    self.create(shared, "a", seed=1)
    self.create(shared, "b", seed=2)
    self.step(shared, "b")
    self.step(shared, "a")
    shared.delete_model("a")
    self.assertNotIn("a", shared.adapters)
    self.step(shared, "b")

    alone = self.make_worker()
    self.create(alone, "b", seed=2)
    self.step(alone, "b")
    self.step(alone, "b")
    for actual, expected in zip(self.weights(shared, "b"), self.weights(alone, "b"), strict=True):
      torch.testing.assert_close(actual, expected)
    with self.assertRaises(ValueError):
      shared.optim_step(self.ADAM, "a")

  def test_an_unknown_adapter_is_refused(self) -> None:
    worker = self.make_worker()
    self.create(worker, "a", seed=1)
    with self.assertRaises(ValueError):
      worker.optim_step(self.ADAM, "nope")


class SelectionTest(unittest.TestCase):
  def test_the_backend_env_picks_the_automodel_worker(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": "automodel"}):
      self.assertIsInstance(build_worker(is_lora=True), AutomodelTrainingWorker)
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": ""}):
      self.assertNotIsInstance(build_worker(is_lora=True), AutomodelTrainingWorker)

  def test_full_fine_tuning_is_refused(self) -> None:
    with self.assertRaises(ValueError):
      AutomodelTrainingWorker().create_model("m", "id", FFTConfig())


if __name__ == "__main__":
  unittest.main()
