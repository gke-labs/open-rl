import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.lora_snapshots import freeze_adapter, resolve_lora_path, snapshot_path
from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor
from training.commands import SaveWeightsForSampler
from training.lora_trainer_worker import LoraTrainingWorker


class LoraSnapshotTest(unittest.TestCase):
  def setUp(self) -> None:
    self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": str(self.root)}))
    self.source = self.root / "peft" / "job" / "job"
    self.source.mkdir(parents=True)
    (self.source / "adapter_config.json").write_text("{}")
    (self.source / "adapter_model.safetensors").write_bytes(b"version-one")
    self.first = "tinker://job/sampler_weights/sampler-1"
    self.second = "tinker://job/sampler_weights/sampler-2"

  def test_saved_versions_and_named_reference_keep_their_original_weights(self) -> None:
    alias = "tinker://job/sampler_weights/named"
    freeze_adapter("job", self.first, alias)
    (self.source / "adapter_model.safetensors").write_bytes(b"version-two")
    freeze_adapter("job", self.second)
    for ref, expected in [(self.first, b"version-one"), (alias, b"version-one"), (self.second, b"version-two")]:
      # Even an explicitly supplied mutable path cannot override a saved version.
      path = Path(resolve_lora_path(ref, str(self.source)))
      self.assertEqual((path / "adapter_model.safetensors").read_bytes(), expected)
    self.assertNotEqual(snapshot_path(self.first), snapshot_path(self.second))

  def test_missing_saved_version_never_uses_mutable_or_base_weights(self) -> None:
    with self.assertRaisesRegex(FileNotFoundError, "unavailable"):
      resolve_lora_path(self.first, str(self.source))
    self.assertEqual(resolve_lora_path("job", None), str(self.source))
    self.assertIsNone(resolve_lora_path("not-saved", None))

  def test_reusing_a_saved_reference_cannot_change_cached_weights(self) -> None:
    freeze_adapter("job", self.first)
    (self.source / "adapter_model.safetensors").write_bytes(b"new-weights")
    with self.assertRaisesRegex(FileExistsError, "new name or sequence ID"):
      freeze_adapter("job", self.first)
    self.assertEqual((snapshot_path(self.first) / "adapter_model.safetensors").read_bytes(), b"version-one")

  def test_a_failed_copy_does_not_publish_a_partial_adapter(self) -> None:
    with patch("server.lora_snapshots.shutil.copytree", side_effect=OSError("disk full")), self.assertRaises(OSError):
      freeze_adapter("job", self.first)
    self.assertFalse(snapshot_path(self.first).exists())
    self.assertEqual(list((self.root / "sampler_lora").iterdir()), [])

  def test_processor_acknowledges_both_references_only_after_freezing_weights(self) -> None:
    worker = LoraTrainingWorker()
    processor = TrainingRequestsProcessor(InMemoryStore(), worker)
    alias = "tinker://job/sampler_weights/named"
    command = SaveWeightsForSampler(request_id="save", model_id="job", path=alias, sampling_session_id=self.first)
    with patch.object(worker, "save_for_sampler", return_value=None):
      result = asyncio.run(processor.dispatch_operation(command))
    self.assertEqual(result["type"], "sampler_weights_saved")
    self.assertTrue(snapshot_path(result["path"]).is_dir())
    self.assertTrue(snapshot_path(result["sampling_session_id"]).is_dir())

  def test_saving_an_uninitialized_adapter_fails(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "no active PEFT model"):
      LoraTrainingWorker().save_for_sampler("job", None, self.first)


if __name__ == "__main__":
  unittest.main()
