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
    worker.base_model_name = "test-simple-model"
    worker.model = SimpleModel()

    # Initialize shadow with base weights W0
    worker._prev_weights_shadow = {
      name: param.data.detach().cpu().clone()
      for name, param in worker.model.named_parameters()
      if param.requires_grad
    }

    # Simulate an Adam update where 2 out of 100 elements change (2% sparsity)
    orig_w0 = worker.model.fc.weight.data.clone()
    worker.model.fc.weight.data[0, 2] = 42.0
    worker.model.fc.weight.data[5, 7] = -13.37

    state_path = os.path.join(self.test_dir, "step_1")
    result = worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

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

    self.assertIn("fc.weight.indices", sparse_delta)
    self.assertIn("fc.weight.values", sparse_delta)
    self.assertEqual(sparse_delta["fc.weight.indices"].numel(), 2)
    self.assertEqual(sparse_delta["fc.weight.indices"].dtype, torch.int32)

    # 3. Verify Lossless Selective Overwrite reproduces exact target W1
    simulated_sampler_weight = orig_w0.clone()
    indices = sparse_delta["fc.weight.indices"]
    values = sparse_delta["fc.weight.values"]
    simulated_sampler_weight.view(-1)[indices.to(torch.int64)] = values

    self.assertTrue(
      torch.equal(simulated_sampler_weight, worker.model.fc.weight.data),
      "Lossless selective overwrite must produce bitwise identical tensors (0 ULP drift)",
    )

    # 4. Verify worker's CPU shadow was updated to W1 so next step diffs correctly
    self.assertTrue(
      torch.equal(worker._prev_weights_shadow["fc.weight"], worker.model.fc.weight.data.cpu()),
      "Worker shadow must be updated after delta save",
    )


if __name__ == "__main__":
  unittest.main()
