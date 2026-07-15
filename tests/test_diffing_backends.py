import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.fft_trainer_worker import FFTTrainingWorker


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(16, 16, bias=False)
    self.fc2 = nn.Linear(16, 16, bias=False)


class TestUniversalStreamedDiffing(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def _create_worker_and_modify(self) -> FFTTrainingWorker:
    torch.manual_seed(42)
    model = DummyModel().to(self.device)
    worker = FFTTrainingWorker()
    worker.base_model_name = "dummy"
    worker.model = model

    # Initialize shadow weights via first save step
    worker.save_state_delta("dummy", os.path.join(self.temp_dir, "init"), kind="sampler")

    # Modify specific elements
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.5
      model.fc1.weight[3, 5] -= 0.75
      model.fc2.weight[2, 2] += 2.0

    return worker

  def test_streamed_diffing_standalone_vs_optim_step_equivalence(self):
    """Verifies standalone save_state_delta and optim_step + save_state_delta produce identical 1D Flat Packed deltas."""
    worker_standalone = self._create_worker_and_modify()
    standalone_dir = os.path.join(self.temp_dir, "save_standalone")
    meta_standalone = worker_standalone.save_state_delta("dummy", standalone_dir, kind="sampler")

    worker_optim = self._create_worker_and_modify()
    worker_optim.weight_sync_strategy = "delta"
    worker_optim.optim_step({})
    optim_dir = os.path.join(self.temp_dir, "save_optim")
    meta_optim = worker_optim.save_state_delta("dummy", optim_dir, kind="sampler")

    self.assertEqual(meta_standalone["changed_elements"], meta_optim["changed_elements"])
    self.assertEqual(meta_standalone["layer_names"], meta_optim["layer_names"])

    standalone_tensors = load_file(os.path.join(standalone_dir, "delta.safetensors"))
    optim_tensors = load_file(os.path.join(optim_dir, "delta.safetensors"))

    self.assertEqual(set(standalone_tensors.keys()), set(optim_tensors.keys()))
    for key in standalone_tensors:
      self.assertTrue(
        torch.equal(standalone_tensors[key], optim_tensors[key]),
        f"Mismatch on {key} between standalone and optim_step diffing paths",
      )


if __name__ == "__main__":
  unittest.main()
