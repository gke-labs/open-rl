import os
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from training.trainer_worker import BaseTrainerWorker
from training.types import Datum


def worker_on(device: str) -> BaseTrainerWorker:
  with patch.dict(os.environ, {"OPEN_RL_DEVICE": "cpu"}):
    worker = BaseTrainerWorker()
  worker.device = torch.device(device)
  return worker


def datum(tokens: list[int]) -> Datum:
  return Datum(
    model_input=tokens,
    loss_fn_inputs={
      "target_tokens": {"data": tokens[1:] + [0]},
      "weights": {"data": [1.0] * len(tokens)},
      "logprobs": {"data": [-0.5] * len(tokens)},
      "advantages": {"data": [0.25] * len(tokens)},
    },
  )


class DeviceOps(TorchDispatchMode):
  """Records every op that touches a tensor on `device`."""

  def __init__(self, device: str):
    super().__init__()
    self.device = device
    self.ops: list[str] = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    on_device = any(isinstance(a, torch.Tensor) and a.device.type == self.device for a in (*args, *kwargs.values()))
    out = func(*args, **kwargs)
    if on_device or (isinstance(out, torch.Tensor) and out.device.type == self.device):
      self.ops.append(str(func))
    return out


class TestPadding(unittest.TestCase):
  def test_values_and_padding(self) -> None:
    worker = worker_on("cpu")
    padded = worker.pad_sequences([[1, 2, 3], [4]], [3, 1], torch.long, pad_value=9, width=5)
    self.assertEqual(padded.tolist(), [[1, 2, 3, 9, 9], [4, 9, 9, 9, 9]])
    self.assertEqual(padded.dtype, torch.long)
    floats = worker.pad_sequences([[0.5, 0.25], [1.0, 2.0, 3.0]], [2, 2], torch.float32)
    self.assertEqual(floats.tolist(), [[0.5, 0.25], [1.0, 2.0]])

  def test_model_inputs(self) -> None:
    worker = worker_on("cpu")
    input_ids, attention_mask, input_lengths = worker.pad_model_inputs([datum([5, 6, 7]), datum([8])])
    self.assertEqual(input_ids.tolist(), [[5, 6, 7], [8, 0, 0]])
    self.assertEqual(attention_mask.tolist(), [[1, 1, 1], [1, 0, 0]])
    self.assertEqual(input_lengths, [3, 1])

  def test_each_padded_tensor_reaches_the_device_in_one_op(self) -> None:
    # A per-row write into a device tensor is a device kernel whose shape is the
    # row's length; on TPU each new shape compiles a program that is kept for the
    # life of the process. Padding is built on the host and moved whole.
    worker = worker_on("meta")
    data = [datum(list(range(1, 8))), datum([3, 4]), datum(list(range(10, 15)))]
    with DeviceOps("meta") as ops:
      _input_ids, _attention_mask, input_lengths = worker.pad_model_inputs(data)
      _targets, _weights, lengths = worker.pad_targets_and_weights(data, input_lengths)
      worker.pad_sequences([d.loss_fn_inputs["logprobs"].data for d in data], lengths, torch.float32)
    self.assertEqual(len(ops.ops), 5, ops.ops)
