import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from training.fft_trainer_worker import FFTTrainingWorker


class SimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(10, 10, bias=False)

  def forward(self, x):
    return self.fc(x)


class DeltaWeightSyncTest(unittest.TestCase):
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_sparse_delta_encoding_and_lossless_overwrite(self):
    worker = FFTTrainingWorker()
    worker.cpu_offload = False
    worker.base_model_name = "test-simple-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    # Simulate an Adam update where 2 out of 100 elements change (2% sparsity)
    orig_w0 = worker.model.fc.weight.data.clone()
    worker.model.fc.weight.data[0, 2] = 42.0
    worker.model.fc.weight.data[5, 7] = -13.37

    worker.optim_step({})
    state_path = os.path.join(self.test_dir, "step_1")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # 1. Verify metadata
    metadata_path = os.path.join(state_path, "metadata.json")
    self.assertTrue(os.path.exists(metadata_path))
    with open(metadata_path) as f:
      meta = json.load(f)
    self.assertEqual(meta["format"], "sparse_delta")
    self.assertEqual(meta["changed_elements"], 2)
    self.assertEqual(meta["total_elements"], 100)
    self.assertEqual(meta["density_pct"], 2.0)

    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    import safetensors.torch

    sparse_delta = safetensors.torch.load_file(delta_file)

    self.assertIn("delta.indices_flat", sparse_delta)
    self.assertIn("delta.values_flat", sparse_delta)
    self.assertEqual(sparse_delta["delta.indices_flat"].numel(), 2)
    self.assertEqual(sparse_delta["delta.indices_flat"].dtype, torch.int32)

    # 3. Verify Lossless Selective Overwrite reproduces exact target W1
    simulated_sampler_weight = orig_w0.clone()
    indices = sparse_delta["delta.indices_flat"]
    values = sparse_delta["delta.values_flat"]
    simulated_sampler_weight.view(-1)[indices.to(torch.int64)] = values

    self.assertTrue(
      torch.equal(simulated_sampler_weight, worker.model.fc.weight.data),
      "Lossless selective overwrite must produce bitwise identical tensors (0 ULP drift)",
    )

    # 4. Verify worker's CPU shadow was updated to W1 so next step diffs correctly
    self.assertTrue(
      torch.equal(worker._param_shadow[worker.model.fc.weight][1], worker.model.fc.weight.data.cpu()),
      "Worker shadow must be updated after delta save",
    )

  def test_weight_sync_strategy_selection(self):
    worker = FFTTrainingWorker()
    self.assertEqual(worker.weight_sync_strategy, "delta")
    worker.set_weight_sync_strategy("full")
    self.assertEqual(worker.weight_sync_strategy, "full")
    with self.assertRaises(ValueError):
      worker.set_weight_sync_strategy("invalid_strategy")

  def test_save_state_delta_with_offloading(self):
    """Test that save_state_delta() succeeds cleanly when model is offloaded (_is_offloaded=True and param.data size 0)."""
    worker = FFTTrainingWorker()
    worker.base_model_name = "test-offload-model"
    worker.model = SimpleModel()

    # Initialize shadow with base weights W0
    worker._param_shadow = {param: (param.device, param.data.detach().cpu().clone()) for param in worker.model.parameters() if param.requires_grad}

    # Simulate what optim_step() produces on GPU before offload_to_cpu() moves weights and sets _is_offloaded=True
    w1_fc = worker.model.fc.weight.data.detach().cpu().clone()
    w1_fc[1, 1] = 77.7
    worker._param_shadow[worker.model.fc.weight] = (torch.device("cuda" if torch.cuda.is_available() else "cpu"), w1_fc)
    worker._latest_delta_tensors = {
      "names": ["fc.weight"],
      "indices_list": [torch.tensor([1 * 10 + 1], dtype=torch.int32)],
      "values_list": [torch.tensor([77.7], dtype=torch.float32)],
      "layer_lengths_list": [1],
    }
    worker._latest_total_changed = 1
    worker._latest_total_elements = 100

    # Simulate offload state where GPU param.data is set to 0-size tensor
    worker._is_offloaded = True
    worker.model.fc.weight.data = torch.empty(0, dtype=worker.model.fc.weight.dtype, device="cpu")

    state_path = os.path.join(self.test_dir, "step_offload")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # Verify that delta.safetensors was cleanly saved from offloaded CPU buffer
    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    import safetensors.torch

    sparse_delta = safetensors.torch.load_file(delta_file)
    self.assertIn("delta.indices_flat", sparse_delta)
    self.assertEqual(sparse_delta["delta.indices_flat"].numel(), 1)
    self.assertAlmostEqual(sparse_delta["delta.values_flat"][0].item(), 77.7, places=4)


if __name__ == "__main__":
  unittest.main()
