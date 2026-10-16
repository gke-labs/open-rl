import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.lora_snapshots import freeze_adapter, snapshot_name, snapshot_root


class LoraSnapshotTest(unittest.TestCase):
  def test_a_frozen_snapshot_keeps_its_weights_is_never_overwritten_and_names_the_served_base(self) -> None:
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.enterContext(tempfile.TemporaryDirectory())}))
    live = Path(os.environ["OPEN_RL_TMP_DIR"], "peft", "job", "job")
    live.mkdir(parents=True)
    (live / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/models/qwen", "r": 8}))
    (live / "adapter_model.safetensors").write_bytes(b"version-one")
    ref = "tinker://job/sampler_weights/sampler-1"
    freeze_adapter("job", "Qwen/Qwen2.5-0.5B", ref)
    (live / "adapter_model.safetensors").write_bytes(b"version-two")
    with self.assertRaisesRegex(FileExistsError, "new name or sequence ID"):
      freeze_adapter("job", "Qwen/Qwen2.5-0.5B", ref)
    frozen = snapshot_root() / snapshot_name(ref)
    self.assertEqual((frozen / "adapter_model.safetensors").read_bytes(), b"version-one")
    self.assertEqual(json.loads((frozen / "adapter_config.json").read_text()), {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B", "r": 8})
