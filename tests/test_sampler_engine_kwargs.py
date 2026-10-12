"""Engine settings the vLLM sampler derives from its environment."""

import os
import unittest
from unittest.mock import patch

from server.vllm_sampler import engine_kwargs_from_env

BASE_ENV = {"BASE_MODEL": "Qwen/Qwen3-0.6B"}


def kwargs(fft_enabled: bool, **env: str) -> dict:
  with patch.dict(os.environ, {**BASE_ENV, **env}, clear=True):
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


if __name__ == "__main__":
  unittest.main()
