import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from training.trainer_worker import BaseTrainerWorker
from training.types import Datum

VOCAB = 96
TINY = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
LOSSES = {"cross_entropy": None, "importance_sampling": None, "ppo": {"clip_range": 0.2, "kl_coeff": 0.03}}


def qwen3() -> torch.nn.Module:
  torch.manual_seed(0)
  return Qwen3ForCausalLM(Qwen3Config(**TINY))


def cpu_worker() -> BaseTrainerWorker:
  with patch.dict(os.environ, {"OPEN_RL_DEVICE": "cpu"}):
    return BaseTrainerWorker()


def bucketing(enabled: bool):
  return patch.object(BaseTrainerWorker, "shape_bucketing_enabled", return_value=enabled)


def token_budget(tokens: int):
  return patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": str(tokens)})


def datum(length: int, seed: int) -> Datum:
  rng = random.Random(seed)
  return Datum(
    model_input=[rng.randrange(VOCAB) for _ in range(length)],
    loss_fn_inputs={
      "target_tokens": {"data": [rng.randrange(VOCAB) for _ in range(length)]},
      "weights": {"data": [rng.choice([0.0, 0.5, 1.0]) for _ in range(length)]},
      "logprobs": {"data": [-rng.uniform(0.5, 5.0) for _ in range(length)]},
      "advantages": {"data": [rng.uniform(-1.0, 1.0) for _ in range(length)]},
    },
  )


def rollouts() -> list[Datum]:
  # Three rows of mixed length: one batch pads to 4 rows x 64 positions with bucketing on.
  return [datum(5, 1), datum(9, 2), datum(7, 3)]


def run(model, data, loss_fn, *, enabled: bool, shapes: list | None = None):
  model.zero_grad(set_to_none=True)
  hook = model.model.register_forward_pre_hook(lambda _module, args: shapes.append(tuple(args[0].shape))) if shapes is not None else None
  with bucketing(enabled), token_budget(4096):
    result = cpu_worker().forward_backward(model, data, loss_fn, LOSSES[loss_fn])
  if hook:
    hook.remove()
  grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
  return result, grads


class TestShapeBucketingEnabled(unittest.TestCase):
  def test_off_on_cpu(self) -> None:
    self.assertFalse(cpu_worker().shape_bucketing_enabled())

  def test_on_for_tpu_device(self) -> None:
    worker = cpu_worker()
    worker.device = SimpleNamespace(type="tpu")
    self.assertTrue(worker.shape_bucketing_enabled())

  def test_padded_shapes(self) -> None:
    worker = cpu_worker()
    with bucketing(True):
      input_ids, attention_mask, input_lengths = worker.pad_model_inputs(rollouts())
      target_token_ids, weights, _lengths = worker.pad_targets_and_weights(rollouts(), input_lengths)
    self.assertEqual(input_ids.shape, (3, 64))
    self.assertEqual(attention_mask.shape, (3, 64))
    self.assertEqual(target_token_ids.shape, (3, 64))
    self.assertEqual(weights.shape, (3, 64))
    self.assertEqual(attention_mask.sum(dim=1).tolist(), [5, 9, 7])


class TestBucketSize(unittest.TestCase):
  def test_rows(self) -> None:
    for value, expected in [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (16, 16), (17, 32)]:
      with self.subTest(value=value):
        self.assertEqual(BaseTrainerWorker.bucket_size(value), expected)

  def test_lengths_start_at_minimum(self) -> None:
    minimum = BaseTrainerWorker.MIN_LENGTH_BUCKET
    self.assertEqual(minimum, 64)
    for value, expected in [(1, 64), (63, 64), (64, 64), (65, 128), (128, 128), (129, 256), (1000, 1024)]:
      with self.subTest(value=value):
        self.assertEqual(BaseTrainerWorker.bucket_size(value, minimum), expected)


class TestBucketedLossMatches(unittest.TestCase):
  def test_loss_and_gradients_match_unbucketed(self) -> None:
    for loss_fn in LOSSES:
      with self.subTest(loss_fn=loss_fn):
        model = qwen3()
        expected_shapes, actual_shapes = [], []
        expected, expected_grads = run(model, rollouts(), loss_fn, enabled=False, shapes=expected_shapes)
        actual, actual_grads = run(model, rollouts(), loss_fn, enabled=True, shapes=actual_shapes)
        self.assertEqual((expected_shapes, actual_shapes), ([(3, 9)], [(4, 64)]))
        torch.testing.assert_close(actual["metrics"]["loss:sum"], expected["metrics"]["loss:sum"], rtol=1e-5, atol=1e-6)
        for actual_output, expected_output in zip(actual["loss_fn_outputs"], expected["loss_fn_outputs"], strict=True):
          self.assertEqual(actual_output["logprobs"]["shape"], expected_output["logprobs"]["shape"])
          torch.testing.assert_close(actual_output["logprobs"]["data"], expected_output["logprobs"]["data"], rtol=1e-5, atol=1e-5)
        self.assertTrue(expected_grads)
        self.assertEqual(actual_grads.keys(), expected_grads.keys())
        for param, grad in expected_grads.items():
          torch.testing.assert_close(actual_grads[param], grad, rtol=0, atol=1e-5 * grad.abs().max().item(), msg=param)

  def test_padding_rows_add_nothing(self) -> None:
    padding = BaseTrainerWorker.padding_datum()
    for loss_fn in LOSSES:
      with self.subTest(loss_fn=loss_fn):
        model = qwen3()
        alone, _grads = run(model, [padding], loss_fn, enabled=False)
        self.assertEqual(alone["metrics"]["loss:sum"], 0.0)
        self.assertTrue(all(param.grad is None or not param.grad.any() for param in model.parameters()))

        expected, expected_grads = run(model, rollouts(), loss_fn, enabled=False)
        actual, actual_grads = run(model, [*rollouts(), padding, padding], loss_fn, enabled=False)
        torch.testing.assert_close(actual["metrics"]["loss:sum"], expected["metrics"]["loss:sum"])
        for param, grad in expected_grads.items():
          torch.testing.assert_close(actual_grads[param], grad, msg=param)


class TestBucketedPacker(unittest.TestCase):
  def test_batches_fit_the_budget_at_padded_shape(self) -> None:
    rng = random.Random(0)
    data = [datum(rng.randint(1, 300), seed) for seed in range(40)]
    worker = cpu_worker()
    for budget in (256, 1024, 4096, 16384):
      with self.subTest(budget=budget), bucketing(True), token_budget(budget):
        batches = worker.make_training_batches(data)
        self.assertCountEqual([idx for batch in batches for idx, _ in batch], range(len(data)))
        for batch in batches:
          rows = worker.bucket_size(len(batch))
          length = worker.bucket_size(max(len(d.model_input) for _, d in batch), worker.MIN_LENGTH_BUCKET)
          self.assertTrue(len(batch) == 1 or rows * length <= budget, f"{len(batch)} rows padded to {rows} x {length} > {budget}")

  def test_unbucketed_packer_unchanged(self) -> None:
    # Five 10-token rows fit a 50-token budget unpadded, but pad to 8 x 64 with bucketing.
    data = [datum(10, seed) for seed in range(5)]
    worker = cpu_worker()
    with token_budget(50):
      self.assertEqual([len(batch) for batch in worker.make_training_batches(data)], [5])
      with bucketing(True):
        self.assertEqual([len(batch) for batch in worker.make_training_batches(data)], [1] * 5)


if __name__ == "__main__":
  unittest.main()
