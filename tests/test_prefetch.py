"""training.prefetch warms the page cache for a checkpoint before the loader
maps it. These tests cover its decisions."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from training import prefetch

TOTAL = 4 * 1024 * 1024 + 17


class PrefetchTest(unittest.TestCase):
  def setUp(self):
    self.enterContext(redirect_stdout(io.StringIO()))
    self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
    (self.dir / "model-00001-of-00002.safetensors").write_bytes(os.urandom(3 * 1024 * 1024))
    (self.dir / "model-00002-of-00002.safetensors").write_bytes(os.urandom(1024 * 1024 + 17))
    (self.dir / "pytorch_model.bin").write_bytes(b"unused")
    (self.dir / "config.json").write_text("{}")

  def test_lists_only_safetensors(self):
    files = prefetch.checkpoint_files(str(self.dir))
    self.assertEqual(sorted(f.name for f in files), ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])

  def test_an_uncached_model_id_is_left_to_the_loader(self):
    self.assertEqual(prefetch.checkpoint_files("org/not-downloaded"), [])

  def test_skips_local_filesystems(self):
    with patch.object(prefetch, "fs_type", return_value="ext4"):
      self.assertEqual(prefetch.prefetch(str(self.dir)), 0)

  def test_reads_every_chunk_on_a_network_filesystem(self):
    with patch.object(prefetch, "fs_type", return_value="nfs"), patch.object(prefetch, "CHUNK_BYTES", 1024 * 1024):
      self.assertEqual(prefetch.prefetch(str(self.dir)), TOTAL)

  def test_skips_when_the_files_would_not_fit_in_memory(self):
    with patch.object(prefetch, "fs_type", return_value="nfs"), patch.object(prefetch, "available_bytes", return_value=1024 * 1024):
      self.assertEqual(prefetch.prefetch(str(self.dir)), 0)

  def test_a_read_error_does_not_fail_the_load(self):
    with patch.object(prefetch, "fs_type", return_value="nfs"), patch.object(prefetch, "read_range", side_effect=OSError("Stale file handle")):
      self.assertEqual(prefetch.prefetch(str(self.dir)), 0)

  def test_available_memory_counts_the_cgroup_working_set(self):
    (self.dir / "meminfo").write_text(f"MemTotal: 100 kB\nMemAvailable: {64 * 1024 * 1024} kB\n")
    cgroup = self.dir / "cgroup"
    cgroup.mkdir()
    gib = 2**30
    (cgroup / "memory.max").write_text(f"{16 * gib}\n")
    (cgroup / "memory.current").write_text(f"{15 * gib}\n")
    (cgroup / "memory.stat").write_text(f"anon {5 * gib}\ninactive_file {10 * gib}\n")
    with patch.object(prefetch, "MEMINFO", self.dir / "meminfo"), patch.object(prefetch, "CGROUP", cgroup):
      self.assertEqual(prefetch.available_bytes(), 11 * gib)
      (cgroup / "memory.max").write_text("max\n")
      self.assertEqual(prefetch.available_bytes(), 64 * gib)


if __name__ == "__main__":
  unittest.main()
