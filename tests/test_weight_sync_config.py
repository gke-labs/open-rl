import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

from server.model_metadata import TrainingModelMetadata, WeightSyncConfig, extract_weight_sync_config
from server.worker_manager import _fetch_metadata_from_store


class TestWeightSyncConfig(unittest.TestCase):
  def test_default_header_parsing(self):
    """Test that missing headers apply single-location defaults correctly."""
    cfg = extract_weight_sync_config({})
    self.assertEqual(cfg.strategy, "delta")
    self.assertEqual(cfg.delta_format, "vllm_fused")
    self.assertEqual(cfg.delta_apply_method, "patch_in_place")
    self.assertTrue(cfg.enable_prefetching)

    # Calling with None
    cfg_none = extract_weight_sync_config(None)
    self.assertEqual(cfg_none.strategy, "delta")
    self.assertEqual(cfg_none.delta_format, "vllm_fused")
    self.assertEqual(cfg_none.delta_apply_method, "patch_in_place")
    self.assertTrue(cfg_none.enable_prefetching)

  def test_explicit_header_overrides(self):
    """Test passing explicit x-open-rl-weight-sync-* HTTP headers."""
    headers = {
      "x-open-rl-weight-sync-strategy": "full",
      "x-open-rl-weight-sync-delta-format": "native",
      "x-open-rl-weight-sync-delta-apply-method": "full_replace",
      "x-open-rl-weight-sync-enable-prefetching": "false",
    }
    cfg = extract_weight_sync_config(headers)
    self.assertEqual(cfg.strategy, "full")
    self.assertEqual(cfg.delta_format, "native")
    self.assertEqual(cfg.delta_apply_method, "full_replace")
    self.assertFalse(cfg.enable_prefetching)

  def test_header_case_insensitivity_and_alias_backwards_compatibility(self):
    """Test that uppercase headers and legacy alias header names parse correctly."""
    headers = {
      "x-open-rl-weight-sync-strategy": "DELTA",
      "x-open-rl-weight-sync-format": "NATIVE",
      "x-open-rl-weight-sync-apply-method": "PATCH_IN_PLACE",
      "x-open-rl-weight-sync-enable-prefetching": "TRUE",
    }
    cfg = extract_weight_sync_config(headers)
    self.assertEqual(cfg.strategy, "delta")
    self.assertEqual(cfg.delta_format, "native")
    self.assertEqual(cfg.delta_apply_method, "patch_in_place")
    self.assertTrue(cfg.enable_prefetching)

  def test_invalid_enum_fallbacks(self):
    """Test that invalid enum header values safely fall back to defaults."""
    headers = {
      "x-open-rl-weight-sync-strategy": "invalid_mode",
      "x-open-rl-weight-sync-delta-format": "unknown_format",
      "x-open-rl-weight-sync-delta-apply-method": "invalid_method",
    }
    cfg = extract_weight_sync_config(headers)
    self.assertEqual(cfg.strategy, "delta")
    self.assertEqual(cfg.delta_format, "vllm_fused")
    self.assertEqual(cfg.delta_apply_method, "patch_in_place")
    self.assertTrue(cfg.enable_prefetching)

  def test_metadata_persistence_and_store_retrieval(self):
    """Test TrainingModelMetadata serialization and worker manager store retrieval."""
    cfg = extract_weight_sync_config(
      {
        "x-open-rl-weight-sync-strategy": "delta",
        "x-open-rl-weight-sync-delta-format": "vllm_fused",
        "x-open-rl-weight-sync-delta-apply-method": "patch_in_place",
        "x-open-rl-weight-sync-enable-prefetching": "true",
      }
    )

    meta = TrainingModelMetadata(
      base_model="Qwen/Qwen3-8B",
      created_at=123456789.0,
      training_kind="fft",
      weight_sync_config=asdict(cfg),
    )

    serialized = json.dumps(asdict(meta))
    mock_store = MagicMock()
    mock_store.get_value_sync.return_value = serialized

    with unittest.mock.patch("server.store.get_store", return_value=mock_store):
      meta_res = _fetch_metadata_from_store("test-model-123")
      self.assertIsNotNone(meta_res)
      self.assertEqual(meta_res.base_model, "Qwen/Qwen3-8B")
      self.assertIsNotNone(meta_res.weight_sync_config)
      self.assertEqual(meta_res.weight_sync_config.strategy, "delta")
      self.assertEqual(meta_res.weight_sync_config.delta_format, "vllm_fused")
      self.assertEqual(meta_res.weight_sync_config.delta_apply_method, "patch_in_place")
      self.assertTrue(meta_res.weight_sync_config.enable_prefetching)

  def test_from_env_reconstruction(self):
    """Test reconstructing WeightSyncConfig and TrainingModelMetadata directly from environment dictionary."""
    env_vars = {
      "BASE_MODEL": "Qwen/Qwen2.5-0.5B",
      "OPEN_RL_WEIGHT_SYNC_STRATEGY": "full",
      "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT": "native",
      "OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD": "full_replace",
      "OPEN_RL_WEIGHT_SYNC_ENABLE_PREFETCHING": "false",
    }
    cfg = WeightSyncConfig.from_env(env_vars)
    self.assertEqual(cfg.strategy, "full")
    self.assertEqual(cfg.delta_format, "native")
    self.assertEqual(cfg.delta_apply_method, "full_replace")
    self.assertFalse(cfg.enable_prefetching)

    meta = TrainingModelMetadata.from_env(env_vars)
    self.assertEqual(meta.base_model, "Qwen/Qwen2.5-0.5B")
    self.assertEqual(meta.weight_sync_config.strategy, "full")
    self.assertEqual(meta.weight_sync_config.delta_format, "native")
    self.assertEqual(meta.weight_sync_config.delta_apply_method, "full_replace")
    self.assertFalse(meta.weight_sync_config.enable_prefetching)


if __name__ == "__main__":
  unittest.main()
