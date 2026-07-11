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


class TestDiffingBackends(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def _create_worker_and_modify(self) -> tuple[FFTTrainingWorker, dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    model = DummyModel().to(self.device)
    worker = FFTTrainingWorker()
    worker.base_model_name = "dummy"
    worker.model = model

    # Initialize shadow weights via first save step (0 changed elements expected on first call)
    worker.save_state_delta("dummy", os.path.join(self.temp_dir, "init"), kind="sampler")

    # Modify specific elements
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.5
      model.fc1.weight[3, 5] -= 0.75
      model.fc2.weight[2, 2] += 2.0

    return worker, {
      "fc1": model.fc1.weight.detach().cpu().clone(),
      "fc2": model.fc2.weight.detach().cpu().clone(),
    }

  def test_cpu_gpu_and_benchmark_backends_equivalence(self):
    """Verifies CPU, GPU, and Benchmark backends produce 100% identical sparse deltas."""
    worker_gpu, _ = self._create_worker_and_modify()
    gpu_dir = os.path.join(self.temp_dir, "save_gpu")
    gpu_meta = worker_gpu.save_state_delta("dummy", gpu_dir, kind="sampler", diffing_device="gpu")

    worker_cpu, _ = self._create_worker_and_modify()
    cpu_dir = os.path.join(self.temp_dir, "save_cpu")
    cpu_meta = worker_cpu.save_state_delta("dummy", cpu_dir, kind="sampler", diffing_device="cpu")

    worker_bench, _ = self._create_worker_and_modify()
    bench_dir = os.path.join(self.temp_dir, "save_bench")
    bench_meta = worker_bench.save_state_delta("dummy", bench_dir, kind="sampler", diffing_device="benchmark")

    # 1. Assert metadata match
    self.assertEqual(gpu_meta["density_pct"], cpu_meta["density_pct"])
    self.assertEqual(cpu_meta["density_pct"], bench_meta["density_pct"])

    # 2. Assert saved delta.safetensors contain identical indices and values
    gpu_tensors = load_file(os.path.join(gpu_dir, "delta.safetensors"))
    cpu_tensors = load_file(os.path.join(cpu_dir, "delta.safetensors"))
    bench_tensors = load_file(os.path.join(bench_dir, "delta.safetensors"))

    self.assertEqual(set(gpu_tensors.keys()), set(cpu_tensors.keys()))
    for key in gpu_tensors:
      self.assertTrue(
        torch.equal(gpu_tensors[key], cpu_tensors[key]),
        f"Mismatch on {key} between GPU and CPU backends",
      )
      self.assertTrue(
        torch.equal(cpu_tensors[key], bench_tensors[key]),
        f"Mismatch on {key} between CPU and Benchmark backends",
      )


if __name__ == "__main__":
  unittest.main()
