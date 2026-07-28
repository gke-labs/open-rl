"""Unit tests for DeltaSnapshotWeightTransferEngine."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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
      config=None,
      parallel_config=None,  # type: ignore
    )

    init_info = engine.parse_init_info({"model_name_or_path": "Qwen/Qwen3-8B"})
    self.assertIsInstance(init_info, DeltaSnapshotInitInfo)
    self.assertEqual(init_info.model_name_or_path, "Qwen/Qwen3-8B")

    update_info = engine.parse_update_info({"target_weights_path": "/path/to/weights"})
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
        config=None,
        parallel_config=None,  # type: ignore
      )
      update_info = DeltaSnapshotUpdateInfo(target_weights_path=file_path)

      loaded_tensors: list[tuple[str, torch.Tensor]] = []

      def mock_load_weights(weights: list[tuple[str, torch.Tensor]]) -> None:
        loaded_tensors.extend(weights)

      engine.receive_weights(update_info, mock_load_weights)

      self.assertEqual(engine.current_weights_path, file_path)
      self.assertEqual(len(loaded_tensors), 2)
      loaded_names = {k for k, _ in loaded_tensors}
      self.assertIn("model.layers.0.self_attn.q_proj.weight", loaded_names)

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
        config=None,
        parallel_config=None,  # type: ignore
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
        config=None,
        parallel_config=None,  # type: ignore
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

  def test_receive_weights_sparse_delta_patching(self):
    """Test that receive_weights parses sparse_delta metadata, applies indices to CPU snapshot, and passes full reconstructed layer tensor."""
    import json

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      # Create metadata specifying sparse_delta format and layer_names
      with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1, "layer_names": ["layer.0.weight"]}, f)

      # Create sparse coordinate arrays in 1D flat packed format
      sparse_dict = {
        "delta.indices_flat": torch.tensor([5], dtype=torch.int32),
        "delta.values_flat": torch.tensor([99.0], dtype=torch.float32),
        "delta.layer_lengths": torch.tensor([1], dtype=torch.int64),
      }
      save_file(sparse_dict, os.path.join(tmpdir, "delta.safetensors"))

      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
      )

      # Mock active vLLM model holding initial weights (all zeros)
      class DummyModel:
        def named_parameters(self):
          return [("layer.0.weight", torch.nn.Parameter(torch.zeros(4, 4)))]

        def named_buffers(self):
          return []

      dummy_model = DummyModel()

      loaded_calls: list[tuple[str, torch.Tensor]] = []

      # Bind load_weights callback to dummy_model exactly as vLLM does
      class BoundLoader:
        def __init__(self, model):
          self.__self__ = model

        def __call__(self, weights):
          loaded_calls.extend(weights)

      mock_loader = BoundLoader(dummy_model)
      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=tmpdir), mock_loader)

      # Assert mock_loader received only the full reconstructed 2D layer tensor, NOT .indices / .values
      self.assertEqual(len(loaded_calls), 1)
      self.assertEqual(loaded_calls[0][0], "layer.0.weight")
      self.assertEqual(loaded_calls[0][1].shape, (4, 4))
      self.assertEqual(loaded_calls[0][1].view(-1)[5].item(), 99.0)

  def test_receive_weights_base_model_directory_loading(self):
    """Test that receive_weights directly populates CPU snapshot from base_model_path when provided."""
    import json
    import sys
    from unittest.mock import MagicMock

    import safetensors
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      base_dir = os.path.join(tmpdir, "base_model")
      os.makedirs(base_dir)
      base_weights = {"layer.0.weight": torch.zeros(4, 4)}
      save_file(base_weights, os.path.join(base_dir, "model.safetensors"))

      step_dir = os.path.join(tmpdir, "step_1")
      os.makedirs(step_dir)
      with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1}, f)

      sparse_dict = {
        "layer.0.weight.indices": torch.tensor([2], dtype=torch.int32),
        "layer.0.weight.values": torch.tensor([42.0], dtype=torch.float32),
      }
      save_file(sparse_dict, os.path.join(step_dir, "delta.safetensors"))

      mock_utils = MagicMock()
      mock_utils.download_weights_from_hf.side_effect = lambda bm, **kwargs: bm

      def fake_iterator(hf_weights_files, use_tqdm_on_load=False):
        for p in hf_weights_files:
          if os.path.exists(p):
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
              for key in f.keys():  # noqa: SIM118
                yield key, f.get_tensor(key)

      mock_utils.safetensors_weights_iterator.side_effect = fake_iterator

      with patch.dict(
        sys.modules,
        {
          "vllm": MagicMock(),
          "vllm.model_executor": MagicMock(),
          "vllm.model_executor.model_loader": MagicMock(),
          "vllm.model_executor.model_loader.weight_utils": mock_utils,
        },
      ):
        engine = DeltaSnapshotWeightTransferEngine(
          config=None,
          parallel_config=None,  # type: ignore
        )

        loaded_calls: list[tuple[str, torch.Tensor]] = []
        engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=step_dir, base_model_path=base_dir),
          lambda w: loaded_calls.extend(w),
        )

        self.assertEqual(len(loaded_calls), 1)
        self.assertEqual(loaded_calls[0][0], "layer.0.weight")
        self.assertEqual(loaded_calls[0][1].shape, (4, 4))
        self.assertEqual(loaded_calls[0][1].view(-1)[2].item(), 42.0)

  def test_receive_weights_hf_cache_and_env_loading(self):
    """Test that _ensure_cpu_snapshot resolves HF model IDs from OPEN_RL_BASE_MODEL (e.g. Qwen/Test-4B -> models--Qwen--Test-4B)."""
    import json
    import sys
    from unittest.mock import MagicMock

    import safetensors
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      # Mock HF hub cache structure: ~/.cache/huggingface/hub/models--Qwen--Test-4B/snapshots/commit123/
      hf_folder = os.path.join(tmpdir, "models--Qwen--Test-4B", "snapshots", "commit123")
      os.makedirs(hf_folder)
      save_file({"layer.0.weight": torch.zeros(3, 3)}, os.path.join(hf_folder, "model-00001-of-00001.safetensors"))

      step_dir = os.path.join(tmpdir, "step_1")
      os.makedirs(step_dir)
      with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1}, f)

      sparse_dict = {
        "layer.0.weight.indices": torch.tensor([4], dtype=torch.int32),
        "layer.0.weight.values": torch.tensor([88.0], dtype=torch.float32),
      }
      save_file(sparse_dict, os.path.join(step_dir, "delta.safetensors"))

      mock_utils = MagicMock()
      mock_utils.download_weights_from_hf.side_effect = lambda bm, **kwargs: hf_folder

      def fake_iterator(hf_weights_files, use_tqdm_on_load=False):
        for p in hf_weights_files:
          if os.path.exists(p):
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
              for key in f.keys():  # noqa: SIM118
                yield key, f.get_tensor(key)

      mock_utils.safetensors_weights_iterator.side_effect = fake_iterator

      # Patch os.path.expanduser and sys.modules for vllm
      with (
        patch.dict(os.environ, {"OPEN_RL_BASE_MODEL": "Qwen/Test-4B"}),
        patch("os.path.expanduser", lambda path: path.replace("~/.cache/huggingface/hub", tmpdir) if path.startswith("~") else path),
        patch.dict(
          sys.modules,
          {
            "vllm": MagicMock(),
            "vllm.model_executor": MagicMock(),
            "vllm.model_executor.model_loader": MagicMock(),
            "vllm.model_executor.model_loader.weight_utils": mock_utils,
          },
        ),
      ):
        engine = DeltaSnapshotWeightTransferEngine(
          config=None,
          parallel_config=None,  # type: ignore
        )

        loaded_calls: list[tuple[str, torch.Tensor]] = []
        engine.receive_weights(
          DeltaSnapshotUpdateInfo(target_weights_path=step_dir),
          lambda w: loaded_calls.extend(w),
        )

        self.assertEqual(len(loaded_calls), 1)
        self.assertEqual(loaded_calls[0][0], "layer.0.weight")
        self.assertEqual(loaded_calls[0][1].shape, (3, 3))
        self.assertEqual(loaded_calls[0][1].view(-1)[4].item(), 88.0)

  def test_in_place_gpu_weight_patching(self):
    """Test that direct in-place GPU patching updates model parameters in-place without load_weights callback."""
    import json

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
      # Mock vLLM model holding parameters directly
      qkv_param = torch.nn.Parameter(torch.zeros(1152, 896), requires_grad=False)

      class MockModel(torch.nn.Module):
        def get_parameter(self, name):
          if "qkv_proj" in name:
            return qkv_param
          raise KeyError(name)

      model = MockModel()

      # Mock hf_config for offset resolution
      mock_config = MagicMock()
      mock_config.hidden_size = 896
      mock_config.num_attention_heads = 14
      mock_config.num_key_value_heads = 2
      mock_config.head_dim = 64
      mock_config.intermediate_size = 4864
      mock_hf_config = MagicMock()
      mock_hf_config.get_text_config.return_value = mock_config

      vllm_config = MagicMock()
      vllm_config.model_config.hf_config = mock_hf_config
      vllm_config.model_config.model = "Qwen/Qwen2.5-0.5B-Instruct"

      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        model=model,
      )

      # Create sparse delta patch targeting k_proj (which maps to qkv_proj with offset)
      sparse_dict = {
        "delta.indices_flat": torch.tensor([10, 20], dtype=torch.int32),
        "delta.values_flat": torch.tensor([42.0, 99.0], dtype=torch.float32),
        "delta.layer_lengths": torch.tensor([2], dtype=torch.int64),
      }
      save_file(sparse_dict, os.path.join(tmpdir, "delta.safetensors"))

      meta = {
        "format": "sparse_delta",
        "layer_names": ["model.layers.0.self_attn.k_proj.weight"],
      }
      with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump(meta, f)

      # Enable in-place GPU mode explicitly
      with patch.dict(os.environ, {"OPEN_RL_IN_PLACE_DELTA": "1"}):
        update_info = DeltaSnapshotUpdateInfo(target_weights_path=tmpdir)
        engine.receive_weights(update_info)

      q_numel = 14 * 64 * 896  # 802816
      qkv_flat = qkv_param.data.view(-1)
      self.assertEqual(qkv_flat[q_numel + 10].item(), 42.0)
      self.assertEqual(qkv_flat[q_numel + 20].item(), 99.0)


if __name__ == "__main__":
  unittest.main()
