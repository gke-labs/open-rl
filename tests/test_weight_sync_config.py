import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

from server.model_metadata import TrainingModelMetadata, WeightSyncConfig, decode_model_metadata
from server.worker_manager import metadata_for


class TestWeightSyncConfig(unittest.TestCase):
  def test_legacy_fields_are_ignored(self):
    """Stored metadata may still carry delta_format and delta_apply_method."""
    legacy = {"strategy": "full", "delta_format": "vllm_fused", "delta_apply_method": "patch_in_place"}
    meta = decode_model_metadata(json.dumps({"base_model": "m", "created_at": 1.0, "weight_sync_config": legacy}))
    self.assertEqual(meta.weight_sync_config, WeightSyncConfig(strategy="full"))

  def test_metadata_persistence_and_store_retrieval(self):
    meta = TrainingModelMetadata(
      base_model="Qwen/Qwen3-8B",
      created_at=123456789.0,
      fine_tuning_type="full",
      weight_sync_config=asdict(WeightSyncConfig(strategy="delta")),
    )
    mock_store = MagicMock()
    mock_store.get_value_sync.return_value = meta.model_dump_json()
    with unittest.mock.patch("server.worker_manager.get_state_store", return_value=mock_store):
      meta_res = metadata_for("test-model-123")
      self.assertEqual(meta_res.base_model, "Qwen/Qwen3-8B")
      self.assertEqual(meta_res.weight_sync_config.strategy, "delta")

  def test_from_env_reconstruction(self):
    env_vars = {"BASE_MODEL": "Qwen/Qwen2.5-0.5B", "OPEN_RL_WEIGHT_SYNC_STRATEGY": "full", "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT": "native"}
    self.assertEqual(WeightSyncConfig.from_env(env_vars), WeightSyncConfig(strategy="full"))
    self.assertEqual(WeightSyncConfig.from_env({}).strategy, "delta")


if __name__ == "__main__":
  unittest.main()
