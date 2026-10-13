"""The weight-transfer methods mixed into the TPU platform's worker."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from server.tpu_weights import FLIPPED_ATTR
from server.tpu_worker_extension import TPUWeightTransferExtension


class FakeTPUWorker(TPUWeightTransferExtension):
  """The TPU worker attributes the extension reads."""

  def __init__(self, weight_transfer_config, flipped: bool = True):
    self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    if flipped:
      setattr(self.model[0], FLIPPED_ATTR, True)
    self.model_runner = MagicMock()
    self.model_runner.get_model.return_value = self.model
    self.vllm_config = SimpleNamespace(weight_transfer_config=weight_transfer_config)


class TPUWeightTransferExtensionTest(unittest.TestCase):
  def setUp(self) -> None:
    self.create_engine = self.enterContext(patch("server.tpu_worker_extension.WeightTransferEngineFactory.create_engine"))
    self.enterContext(patch("server.tpu_worker_extension.set_current_vllm_config", side_effect=lambda config: contextlib.nullcontext()))

  def test_one_engine_on_the_models_device_serves_every_call(self) -> None:
    config = object()
    worker = FakeTPUWorker(config)
    worker.start_weight_update()
    worker.update_weights({"target_weights_path": "/ckpt"})
    worker.finish_weight_update()
    self.create_engine.assert_called_once_with(config, worker.vllm_config, torch.device("cpu"), worker.model)
    engine = self.create_engine.return_value
    engine.start_weight_update.assert_called_once_with()
    engine.update_weights.assert_called_once_with({"target_weights_path": "/ckpt"})
    engine.finish_weight_update.assert_called_once_with()

  def test_each_worker_has_its_own_engine(self) -> None:
    for worker in (FakeTPUWorker(object()), FakeTPUWorker(object())):
      worker.start_weight_update()
    self.assertEqual(self.create_engine.call_count, 2)

  def test_without_a_weight_transfer_config_it_fails(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "Weight transfer not configured"):
      FakeTPUWorker(None).start_weight_update()

  def test_a_model_without_the_flipped_flag_fails_before_any_update(self) -> None:
    with self.assertRaisesRegex(RuntimeError, FLIPPED_ATTR):
      FakeTPUWorker(object(), flipped=False).start_weight_update()
    self.create_engine.assert_not_called()

  def test_reset_encoder_cache_goes_to_the_model_runner(self) -> None:
    worker = FakeTPUWorker(object())
    worker.reset_encoder_cache()
    worker.model_runner.reset_encoder_cache.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
