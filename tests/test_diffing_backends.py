import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

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
    # save_state_delta asserts the worker is offloaded when cpu_offload=True
    # (the default since the time-slicing campaigns); this test exercises the
    # diffing math, not the offload lifecycle.
    worker.cpu_offload = False
    worker.prepare_model_for_training()

    # Modify specific elements
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.5
      model.fc1.weight[3, 5] -= 0.75
      model.fc2.weight[2, 2] += 2.0

    return worker

  def _read_metadata(self, save_dir: str) -> dict:
    # save_state_delta returns only {path, density_pct}; the full accounting
    # (changed_elements, layer_names, ...) is persisted in metadata.json.
    with open(os.path.join(save_dir, "metadata.json")) as f:
      return json.load(f)

  def test_streamed_diffing_optim_step_and_multi_save_idempotency(self):
    """Verifies optim_step streams diff to _latest_delta_tensors and multiple saves read it non-destructively."""
    worker = self._create_worker_and_modify()
    worker.weight_sync_strategy = "delta"
    worker.optim_step({})

    save_dir_1 = os.path.join(self.temp_dir, "save_optim_1")
    worker.save_state_delta("dummy", save_dir_1, kind="sampler")
    meta_1 = self._read_metadata(save_dir_1)

    save_dir_2 = os.path.join(self.temp_dir, "save_optim_2")
    worker.save_state_delta("dummy", save_dir_2, kind="state")
    meta_2 = self._read_metadata(save_dir_2)

    self.assertEqual(meta_1["changed_elements"], 3)
    self.assertEqual(meta_1["changed_elements"], meta_2["changed_elements"])
    self.assertEqual(meta_1["layer_names"], meta_2["layer_names"])

  def test_streamed_diffing_empty_delta_fallback(self):
    """Verifies save_state_delta before optim_step emits an exact O(1) empty delta."""
    worker = self._create_worker_and_modify()
    save_dir = os.path.join(self.temp_dir, "save_empty")
    worker.save_state_delta("dummy", save_dir, kind="sampler")
    meta = self._read_metadata(save_dir)

    self.assertEqual(meta["changed_elements"], 0)
    self.assertEqual(meta["total_elements"], worker.total_model_elements)
    self.assertEqual(meta["layer_names"], [])
    self.assertEqual(meta["layer_shapes"], [])


if __name__ == "__main__":
  unittest.main()
