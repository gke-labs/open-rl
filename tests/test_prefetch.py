"""training.prefetch warms the page cache for a checkpoint before the loader
maps it. These tests cover its decisions; the effect on load time is measured
in runs/2026-10-15_cold_start."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import prefetch


class PrefetchTest(unittest.TestCase):
  def setUp(self):
    self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
    (self.dir / "model-00001-of-00002.safetensors").write_bytes(os.urandom(3 * 1024 * 1024))
    (self.dir / "model-00002-of-00002.safetensors").write_bytes(os.urandom(1024 * 1024 + 17))
    (self.dir / "config.json").write_text("{}")

  def test_lists_checkpoint_files_largest_first(self):
    files = prefetch.checkpoint_files(str(self.dir))
    self.assertEqual([f.name for f in files], ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])

  def test_an_uncached_model_id_is_left_to_the_loader(self):
    with patch.dict(os.environ, {"HF_HOME": str(self.dir / "empty-hub")}):
      self.assertEqual(prefetch.checkpoint_files("org/not-downloaded"), [])

  def test_auto_mode_skips_local_filesystems(self):
    with patch.dict(os.environ, {"OPEN_RL_PREFETCH": "auto"}), patch.object(prefetch, "fs_type", return_value="ext4"):
      self.assertIsNone(prefetch.prefetch(str(self.dir), log=lambda *_: None))

  def test_auto_mode_reads_everything_on_a_network_filesystem(self):
    with patch.dict(os.environ, {"OPEN_RL_PREFETCH": "auto", "OPEN_RL_PREFETCH_THREADS": "3"}), patch.object(prefetch, "fs_type", return_value="nfs"):
      result = prefetch.prefetch(str(self.dir), log=lambda *_: None)
    self.assertEqual(result["bytes"], 4 * 1024 * 1024 + 17)
    self.assertEqual((result["files"], result["threads"], result["fs"]), (2, 3, "nfs"))

  def test_always_reads_even_on_local_disk_and_off_never_does(self):
    with patch.dict(os.environ, {"OPEN_RL_PREFETCH": "always"}), patch.object(prefetch, "fs_type", return_value="ext4"):
      self.assertEqual(prefetch.prefetch(str(self.dir), log=lambda *_: None)["files"], 2)
    with patch.dict(os.environ, {"OPEN_RL_PREFETCH": "off"}), patch.object(prefetch, "fs_type", return_value="nfs"):
      self.assertIsNone(prefetch.prefetch(str(self.dir)))

  def test_skips_when_the_files_would_not_fit_in_memory(self):
    messages = []
    with (
      patch.dict(os.environ, {"OPEN_RL_PREFETCH": "auto"}),
      patch.object(prefetch, "fs_type", return_value="nfs"),
      patch.object(prefetch, "available_bytes", return_value=1024 * 1024),
    ):
      self.assertIsNone(prefetch.prefetch(str(self.dir), log=messages.append))
    self.assertIn("not prefetching", messages[0])

  def test_large_files_are_split_across_threads(self):
    with (
      patch.dict(os.environ, {"OPEN_RL_PREFETCH": "always", "OPEN_RL_PREFETCH_BLOCK_MIB": "1"}),
      patch.object(prefetch, "CHUNK_BYTES", 1024 * 1024),
      patch.object(prefetch, "fs_type", return_value="nfs"),
    ):
      result = prefetch.prefetch(str(self.dir), log=lambda *_: None)
    self.assertEqual(result["bytes"], 4 * 1024 * 1024 + 17)


if __name__ == "__main__":
  unittest.main()


class TrainTokenBudgetTest(unittest.TestCase):
  """The per-pass token budget follows the vocabulary and the free memory unless pinned."""

  def setUp(self):
    from training.trainer_worker import BaseTrainerWorker

    self.worker = BaseTrainerWorker()
    self.model = type("Model", (), {"config": type("Config", (), {"vocab_size": 262144})()})()

  def test_auto_is_the_logits_cap(self):
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "auto"}), patch.object(type(self.worker), "logits_token_cap", return_value=7000):
      self.assertEqual(self.worker.train_token_budget(self.model), 7000)
    with patch.dict(os.environ, {}, clear=False), patch.object(type(self.worker), "logits_token_cap", return_value=7000):
      os.environ.pop("OPEN_RL_TRAIN_TOKEN_BUDGET", None)
      self.assertEqual(self.worker.train_token_budget(self.model), 7000)

  def test_a_number_is_capped_by_what_the_logits_fit(self):
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "65536"}), patch.object(type(self.worker), "logits_token_cap", return_value=7000):
      self.assertEqual(self.worker.train_token_budget(self.model), 7000)
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "4096"}), patch.object(type(self.worker), "logits_token_cap", return_value=7000):
      self.assertEqual(self.worker.train_token_budget(self.model), 4096)

  def test_zero_keeps_one_sequence_per_pass(self):
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "0"}):
      self.assertEqual(self.worker.train_token_budget(self.model), 0)

  def test_cap_without_cuda_or_vocabulary_is_the_default(self):
    self.assertEqual(self.worker.logits_token_cap(object()), self.worker.DEFAULT_TOKEN_CAP)
