"""A forward-only pass (TrainingClient.forward) must not accumulate gradients.

The cookbook's NLL evaluator calls forward() on held-out data immediately
before a training step. The worker only zeroes gradients in optim_step, so a
backward pass here would fold the test set into the next update.
"""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.trainer_worker import BaseTrainerWorker
from training.types import Datum, TensorData


def _tiny_model() -> LlamaForCausalLM:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
  )
  return LlamaForCausalLM(config)


def _datums() -> list[Datum]:
  rows = [[3, 5, 8, 13, 21, 34], [2, 7, 1, 8]]
  return [
    Datum(
      model_input=row[:-1],
      loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * (len(row) - 1))},
    )
    for row in rows
  ]


class ForwardOnlyTest(unittest.TestCase):
  def setUp(self) -> None:
    self.worker = BaseTrainerWorker()
    self.worker.device = torch.device("cpu")
    self.model = _tiny_model()

  def _grad_norm(self) -> float:
    return sum(float(p.grad.norm()) for p in self.model.parameters() if p.grad is not None)

  def test_forward_only_leaves_no_gradients(self) -> None:
    result = self.worker.forward_backward(self.model, _datums(), "cross_entropy", forward_only=True)
    self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    self.assertEqual(len(result["loss_fn_outputs"]), 2)
    self.assertEqual(len(result["loss_fn_outputs"][0]["logprobs"]["data"]), 5)
    self.assertGreater(result["metrics"]["loss:sum"], 0.0)

  def test_training_pass_still_accumulates(self) -> None:
    self.worker.forward_backward(self.model, _datums(), "cross_entropy")
    self.assertGreater(self._grad_norm(), 0.0)

  def test_forward_only_after_training_does_not_change_gradients(self) -> None:
    self.worker.forward_backward(self.model, _datums(), "cross_entropy")
    before = [p.grad.clone() for p in self.model.parameters() if p.grad is not None]
    self.worker.forward_backward(self.model, _datums(), "cross_entropy", forward_only=True)
    after = [p.grad for p in self.model.parameters() if p.grad is not None]
    self.assertEqual(len(before), len(after))
    for a, b in zip(before, after, strict=True):
      self.assertTrue(torch.equal(a, b))

  def test_forward_only_returns_the_same_logprobs_as_a_training_pass(self) -> None:
    # No dropout in the tiny config, so eval and train mode agree.
    forward_only = self.worker.forward_backward(self.model, _datums(), "cross_entropy", forward_only=True)
    trained = self.worker.forward_backward(self.model, _datums(), "cross_entropy")
    for lhs, rhs in zip(forward_only["loss_fn_outputs"], trained["loss_fn_outputs"], strict=True):
      for a, b in zip(lhs["logprobs"]["data"], rhs["logprobs"]["data"], strict=True):
        self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
  unittest.main()
