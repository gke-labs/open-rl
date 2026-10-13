import unittest

import torch

from server.tpu_weights import FLIPPED_ATTR, check_flipped_again, clear_flipped, flipped_modules


def model_with_flags(*flags: bool) -> torch.nn.Sequential:
  model = torch.nn.Sequential(*(torch.nn.Linear(2, 2) for _ in flags))
  for layer, flag in zip(model, flags, strict=True):
    setattr(layer, FLIPPED_ATTR, flag)
  return model


class FlippedModulesTest(unittest.TestCase):
  def test_finds_only_flagged_layers(self) -> None:
    model = model_with_flags(True, False, True)
    self.assertEqual(flipped_modules(model), [model[0], model[2]])

  def test_a_gpu_model_has_none(self) -> None:
    self.assertEqual(flipped_modules(torch.nn.Linear(2, 2)), [])

  def test_check_fails_until_every_cleared_layer_is_flagged_again(self) -> None:
    model = model_with_flags(True, True)
    layers = flipped_modules(model)
    clear_flipped(layers)
    self.assertEqual(flipped_modules(model), [])
    with self.assertRaisesRegex(RuntimeError, "2 of 2 TPU linear layers were not reloaded from /ckpt"):
      check_flipped_again(layers, "/ckpt")
    setattr(model[0], FLIPPED_ATTR, True)
    with self.assertRaisesRegex(RuntimeError, "1 of 2"):
      check_flipped_again(layers, "/ckpt")
    setattr(model[1], FLIPPED_ATTR, True)
    check_flipped_again(layers, "/ckpt")

  def test_nothing_to_check_on_gpu(self) -> None:
    check_flipped_again([], "/ckpt")


if __name__ == "__main__":
  unittest.main()
