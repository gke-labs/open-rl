"""training.prefetch warms the page cache for a checkpoint before the loader
maps it. These tests cover its decisions."""

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
