import os
import unittest
from unittest.mock import patch

from server.vllm_options import sampler_batch_limits

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

  def test_explicit_settings_win(self) -> None:
    with patch.dict(os.environ, {"VLLM_MAX_NUM_SEQS": "32", "VLLM_MAX_NUM_BATCHED_TOKENS": "1024"}):
      self.assertEqual(sampler_batch_limits(80 * GIB), {"max_num_seqs": 32, "max_num_batched_tokens": 1024})


if __name__ == "__main__":
  unittest.main()
