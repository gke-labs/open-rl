"""Engine settings the vLLM sampler derives from its environment."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.vllm_sampler import engine_kwargs_from_env, on_tpu_platform

BASE_ENV = {"BASE_MODEL": "Qwen/Qwen3-0.6B"}
EXTENSION = "server.tpu_worker_extension.TPUWeightTransferExtension"


def kwargs(fft_enabled: bool, on_tpu: bool = False, **env: str) -> dict:
  with patch.dict(os.environ, {**BASE_ENV, **env}, clear=True), patch("server.vllm_sampler.on_tpu_platform", return_value=on_tpu):
    return engine_kwargs_from_env(fft_enabled)


class SleepModeTest(unittest.TestCase):
  def test_a_time_sliced_fft_sampler_can_sleep(self) -> None:
    self.assertTrue(kwargs(True)["enable_sleep_mode"])

  def test_an_exclusive_fft_sampler_cannot(self) -> None:
    self.assertFalse(kwargs(True, OPEN_RL_TIME_SLICING="off")["enable_sleep_mode"])

  def test_a_lora_sampler_cannot(self) -> None:
    for env in ({}, {"OPEN_RL_TIME_SLICING": "off"}):
      with self.subTest(env=env):
        self.assertFalse(kwargs(False, **env)["enable_sleep_mode"])


def platform(out_of_tree: bool, device_type: str) -> SimpleNamespace:
  return SimpleNamespace(is_out_of_tree=lambda: out_of_tree, device_type=device_type)


class WorkerExtensionTest(unittest.TestCase):
  def test_an_fft_sampler_on_tpu_gets_the_weight_transfer_extension(self) -> None:
    self.assertEqual(kwargs(True, on_tpu=True)["worker_extension_cls"], EXTENSION)

  def test_other_samplers_keep_vllms_worker(self) -> None:
    for fft_enabled, on_tpu in ((True, False), (False, True), (False, False)):
      with self.subTest(fft_enabled=fft_enabled, on_tpu=on_tpu):
        self.assertNotIn("worker_extension_cls", kwargs(fft_enabled, on_tpu=on_tpu))

  def test_the_tpu_platform_is_out_of_tree_with_tpu_devices(self) -> None:
    cases = ((platform(True, "tpu"), True), (platform(True, "cuda"), False), (platform(False, "tpu"), False), (platform(False, "cuda"), False))
    for current, expected in cases:
      with self.subTest(out_of_tree=current.is_out_of_tree(), device_type=current.device_type), patch("vllm.platforms.current_platform", current):
        self.assertEqual(on_tpu_platform(), expected)


if __name__ == "__main__":
  unittest.main()
