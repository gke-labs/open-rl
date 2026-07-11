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
  def test_noop_patch_detection_skips_gpu_reload(self):
    """Test that applying an identical patch is identified as a no-op and skipped."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      weights_v1 = {
          "layer.0.weight": torch.ones(4, 4),
          "layer.1.weight": torch.zeros(4, 4),
      }
      file_v1 = os.path.join(tmpdir, "delta1.safetensors")
      save_file(weights_v1, file_v1)

      engine = DeltaSnapshotWeightTransferEngine(
          config=None, parallel_config=None  # type: ignore
      )

      # First update: 2 new/changed tensors
      calls_v1: list[list[tuple[str, torch.Tensor]]] = []
      engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=file_v1),
          lambda w: calls_v1.append(w),
      )
      self.assertEqual(len(calls_v1), 1)
      self.assertEqual(len(calls_v1[0]), 2)

      # Second update: identical weights (NO-OP patch)
      file_v2 = os.path.join(tmpdir, "delta2.safetensors")
      save_file(weights_v1, file_v2)

      calls_v2: list[list[tuple[str, torch.Tensor]]] = []
      engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=file_v2),
          lambda w: calls_v2.append(w),
      )
      # Must detect complete no-op and skip calling load_weights callback
      self.assertEqual(len(calls_v2), 0)
      self.assertEqual(engine.current_weights_path, file_v2)

  def test_selective_layer_filtering_skips_noop_tensors(self):
    """Test that only genuinely modified tensors are passed to load_weights."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      weights_v1 = {
          "layer.0.weight": torch.ones(4, 4),
          "layer.1.weight": torch.zeros(4, 4),
      }
      file_v1 = os.path.join(tmpdir, "delta1.safetensors")
      save_file(weights_v1, file_v1)

      engine = DeltaSnapshotWeightTransferEngine(
          config=None, parallel_config=None  # type: ignore
      )
      engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=file_v1),
          lambda w: None,
      )

      # Update 2: layer.0.weight is unchanged (no-op), layer.1.weight is modified
      weights_v2 = {
          "layer.0.weight": torch.ones(4, 4),
          "layer.1.weight": torch.full((4, 4), 2.5),
      }
      file_v2 = os.path.join(tmpdir, "delta2.safetensors")
      save_file(weights_v2, file_v2)

      loaded_calls: list[tuple[str, torch.Tensor]] = []
      engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=file_v2),
          lambda w: loaded_calls.extend(w),
      )

      # Only layer.1.weight should be passed to callback
      self.assertEqual(len(loaded_calls), 1)
      self.assertEqual(loaded_calls[0][0], "layer.1.weight")
      self.assertTrue(torch.equal(loaded_calls[0][1], torch.full((4, 4), 2.5)))


if __name__ == "__main__":
  unittest.main()
