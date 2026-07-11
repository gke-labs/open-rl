import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Add src to path so we can import server.vllm_sampler and training.fft_trainer_worker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

# Mock vllm modules before importing the sampler patch
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.v1"] = MagicMock()
sys.modules["vllm.v1.worker"] = MagicMock()
gpu_worker_mock = MagicMock()
sys.modules["vllm.v1.worker.gpu_worker"] = gpu_worker_mock


class MockWorkerBase:
  def __init__(self, vllm_config=None, local_rank=0, rank=0, distributed_init_method="", is_driver_worker=False):
    self.model_config = MagicMock()
    self.model_config.model = "mock-base-model-path"
    self.load_config = MagicMock()
    self.model_runner = MagicMock()


# Inject MockWorker into the mocked module
class MockWorker(MockWorkerBase):
  def reload_weights(self, *args, **kwargs):
    pass


gpu_worker_mock.Worker = MockWorker

# Mock model loader
model_loader_mock = MagicMock()
sys.modules["vllm.model_executor.model_loader"] = model_loader_mock

# Now import the patching function
from server.vllm_sampler import patch_vllm_worker_for_delta_sync


class SamplerPatchTest(unittest.TestCase):
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    # Reset mock worker state
    if hasattr(MockWorker, "_openrl_delta_patched"):
      delattr(MockWorker, "_openrl_delta_patched")

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_custom_reload_weights_correct_setup(self):
    # Let's write the clean version here:
    # Reset Worker class to original mock state
    class CleanMockWorker(MockWorkerBase):
      def reload_weights(self, *args, **kwargs):
        self.original_reload_called = True
        self.original_reload_args = args
        self.original_reload_kwargs = kwargs

    gpu_worker_mock.Worker = CleanMockWorker

    patch_vllm_worker_for_delta_sync()

    worker = CleanMockWorker()

    # Mock base model iterator loading:
    base_weight = torch.zeros(10, 10, dtype=torch.bfloat16)
    mock_base_weights = [("fc.weight", base_weight.clone())]

    mock_loader_instance = MagicMock()
    mock_loader_instance.get_all_weights.return_value = mock_base_weights
    model_loader_mock.get_model_loader.return_value = mock_loader_instance

    mock_model = MagicMock()
    worker.model_runner.get_model.return_value = mock_model

    # Save a mock delta
    delta_tensors = {
      "fc.weight.indices": torch.tensor([2], dtype=torch.int32),  # flat index 2 = row 0, col 2
      "fc.weight.values": torch.tensor([42.0], dtype=torch.bfloat16),
    }
    weights_path = os.path.join(self.test_dir, "step_1")
    os.makedirs(weights_path, exist_ok=True)
    import safetensors.torch

    safetensors.torch.save_file(delta_tensors, os.path.join(weights_path, "delta.safetensors"))
    with open(os.path.join(weights_path, "metadata.json"), "w") as f:
      json.dump({"format": "sparse_delta", "changed_elements": 1}, f)

    # Trigger reload
    worker.reload_weights(weights_path=weights_path)

    # Verify:
    # 1. CPU snapshot initialized and updated
    self.assertTrue(hasattr(worker, "_bf16_snapshot"))
    self.assertIn("fc.weight", worker._bf16_snapshot)

    # Value at index 2 should be updated to 42.0
    self.assertEqual(float(worker._bf16_snapshot["fc.weight"].view(-1)[2]), 42.0)
    # Value at index 0 should remain 0.0
    self.assertEqual(float(worker._bf16_snapshot["fc.weight"].view(-1)[0]), 0.0)

    # 2. original_reload was called with the snapshot iterator
    self.assertTrue(getattr(worker, "original_reload_called", False))

    # Check that weights_iterator kwargs contains the snapshot items
    kwargs = worker.original_reload_kwargs
    self.assertIn("weights_iterator", kwargs)
    self.assertTrue(kwargs.get("is_checkpoint_format"))

    iterator = list(kwargs["weights_iterator"])
    self.assertEqual(len(iterator), 1)
    self.assertEqual(iterator[0][0], "fc.weight")
    # Verify the iterator yielded the updated snapshot tensor
    self.assertEqual(float(iterator[0][1].view(-1)[2]), 42.0)


if __name__ == "__main__":
  unittest.main()
