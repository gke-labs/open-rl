import os
import unittest
from unittest.mock import patch

from server.accelerators import check_supported, deployment_accelerator


class DeploymentAcceleratorTest(unittest.TestCase):
  def test_only_tpu_selects_tpu(self) -> None:
    for value, expected in (("tpu", "tpu"), (" TPU ", "tpu"), ("cuda", "gpu"), ("cpu", "gpu"), ("", "gpu")):
      with self.subTest(value=value), patch.dict(os.environ, {"OPEN_RL_DEVICE": value}):
        self.assertEqual(deployment_accelerator(), expected)

  def test_unset_is_gpu(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(deployment_accelerator(), "gpu")


class CheckSupportedTest(unittest.TestCase):
  def test_gpu_takes_lora_and_full(self) -> None:
    check_supported("gpu", "lora")
    check_supported("gpu", "full")

  def test_tpu_is_lora_only(self) -> None:
    with self.assertRaisesRegex(ValueError, "LoRA only"):
      check_supported("tpu", "full")

  def test_tpu_lora_is_refused_until_tpu_workers_exist(self) -> None:
    with self.assertRaisesRegex(ValueError, "not supported yet"):
      check_supported("tpu", "lora")


if __name__ == "__main__":
  unittest.main()
