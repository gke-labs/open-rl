import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.vllm_options import detected_device_memory_bytes, sampler_batch_limits

GIB = 2**30


class SamplerBatchLimitsTest(unittest.TestCase):
  def test_limits_follow_the_device(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VLLM_MAX_NUM_SEQS", None)
      os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
      self.assertEqual(sampler_batch_limits(80 * GIB), {"max_num_seqs": 256, "max_num_batched_tokens": 16384})
      self.assertEqual(sampler_batch_limits(40 * GIB), {"max_num_seqs": 128, "max_num_batched_tokens": 8192})
      self.assertEqual(sampler_batch_limits(24 * GIB), {"max_num_seqs": 64, "max_num_batched_tokens": 4096})
      # No device visible: the old defaults' tier.
      self.assertEqual(sampler_batch_limits(0)["max_num_seqs"], 64)

  def test_tpu_device_module(self) -> None:
    # torch_tpu registers torch.tpu, whose get_device_properties reports the chip's HBM the way torch.cuda's does.
    import torch

    tpu = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(total_memory=192 * GIB))
    with patch.object(torch.accelerator, "current_accelerator", return_value=torch.device("cpu")), patch.object(torch, "get_device_module", return_value=tpu):
      self.assertEqual(detected_device_memory_bytes(), 192 * GIB)
      self.assertEqual(sampler_batch_limits()["max_num_seqs"], 256)
    with patch.object(torch.accelerator, "current_accelerator", return_value=None):
      self.assertEqual(detected_device_memory_bytes(), 0)

  def test_explicit_settings_win(self) -> None:
    with patch.dict(os.environ, {"VLLM_MAX_NUM_SEQS": "32", "VLLM_MAX_NUM_BATCHED_TOKENS": "1024"}):
      self.assertEqual(sampler_batch_limits(80 * GIB), {"max_num_seqs": 32, "max_num_batched_tokens": 1024})


if __name__ == "__main__":
  unittest.main()
