"""Unit tests for DeltaSnapshotWeightTransferEngine."""

import os
import tempfile
import unittest
import torch

from src.server.delta_weight_transfer_engine import (
    DeltaSnapshotInitInfo,
    DeltaSnapshotUpdateInfo,
    DeltaSnapshotWeightTransferEngine,
)


class DeltaSnapshotWeightTransferEngineTest(unittest.TestCase):

  def test_delta_snapshot_weight_transfer_engine_contract(self):
    """Test that DeltaSnapshotWeightTransferEngine satisfies the vLLM contract."""
    engine = DeltaSnapshotWeightTransferEngine(
        config=None, parallel_config=None  # type: ignore
    )

    init_info = engine.parse_init_info(
        {"model_name_or_path": "Qwen/Qwen3-8B"}
    )
    self.assertIsInstance(init_info, DeltaSnapshotInitInfo)
    self.assertEqual(init_info.model_name_or_path, "Qwen/Qwen3-8B")

    update_info = engine.parse_update_info(
        {"target_weights_path": "/path/to/weights"}
    )
    self.assertIsInstance(update_info, DeltaSnapshotUpdateInfo)
    self.assertEqual(update_info.target_weights_path, "/path/to/weights")
    self.assertTrue(update_info.is_checkpoint_format)

  def test_receive_weights_loads_tensors_into_callback(self):
    """Test that receive_weights parses safetensors and passes to load_weights."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      dummy_weights = {
          "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
          "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
      }
      file_path = os.path.join(tmpdir, "delta.safetensors")
      save_file(dummy_weights, file_path)

      engine = DeltaSnapshotWeightTransferEngine(
          config=None, parallel_config=None  # type: ignore
      )
      update_info = DeltaSnapshotUpdateInfo(target_weights_path=file_path)

      loaded_tensors: list[tuple[str, torch.Tensor]] = []

      def mock_load_weights(weights: list[tuple[str, torch.Tensor]]) -> None:
        loaded_tensors.extend(weights)

      engine.receive_weights(update_info, mock_load_weights)

      self.assertEqual(engine.current_weights_path, file_path)
      self.assertEqual(len(loaded_tensors), 2)
      loaded_names = {k for k, _ in loaded_tensors}
      self.assertIn(
          "model.layers.0.self_attn.q_proj.weight", loaded_names
      )
      self.assertIn("model.layers.0.mlp.gate_proj.weight", loaded_names)


if __name__ == "__main__":
  unittest.main()
