import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from training.device import model_dtype, resolve_device
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker

SRC = Path(__file__).resolve().parents[1] / "src"
# On a TPU host, importing torch_tpu renames PrivateUse1 to "tpu"; the stub does only that.
STUB_TORCH_TPU = 'import torch\ntorch.utils.rename_privateuse1_backend("tpu")\n'


def device_env(value: str | None):
  env = {k: v for k, v in os.environ.items() if k != "OPEN_RL_DEVICE"}
  if value is not None:
    env["OPEN_RL_DEVICE"] = value
  return patch.dict(os.environ, env, clear=True)


class TestResolveDevice(unittest.TestCase):
  def test_unset_auto_detects(self) -> None:
    cases = [((True, True), "cuda:0"), ((False, True), "mps"), ((False, False), "cpu")]
    for (cuda, mps), expected in cases:
      with (
        self.subTest(expected=expected),
        device_env(None),
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
      ):
        self.assertEqual(resolve_device(), torch.device(expected))

  def test_cuda_is_this_ranks_gpu(self) -> None:
    with device_env(None), patch.dict(os.environ, {"LOCAL_RANK": "2"}), patch("torch.cuda.is_available", return_value=True):
      self.assertEqual(resolve_device(), torch.device("cuda", 2))

  def test_explicit_device_wins_over_cuda(self) -> None:
    with device_env("CPU"), patch("torch.cuda.is_available", return_value=True):
      self.assertEqual(resolve_device(), torch.device("cpu"))

  def test_tpu_without_torch_tpu_names_the_package(self) -> None:
    with device_env("tpu"), patch.dict(sys.modules, {"torch_tpu": None}), self.assertRaisesRegex(ImportError, "torch_tpu"):
      resolve_device()

  def test_tpu_not_registered_says_why(self) -> None:
    # torch_tpu imports but registers nothing, as on a host with no chips or with autoload off.
    with (
      device_env("tpu"),
      patch.dict(sys.modules, {"torch_tpu": ModuleType("torch_tpu")}),
      self.assertRaisesRegex(RuntimeError, "TORCH_DEVICE_BACKEND_AUTOLOAD"),
    ):
      resolve_device()

  def test_tpu_imports_torch_tpu(self) -> None:
    # The backend rename is process-wide and permanent, so it runs in a subprocess.
    with tempfile.TemporaryDirectory() as stub_dir:
      (Path(stub_dir) / "torch_tpu.py").write_text(STUB_TORCH_TPU)
      env = dict(os.environ, OPEN_RL_DEVICE="tpu", PYTHONPATH=os.pathsep.join([stub_dir, str(SRC)]))
      result = subprocess.run(
        [sys.executable, "-c", "from training.device import resolve_device; print(resolve_device())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
      )
    self.assertEqual(result.stdout.strip(), "tpu")


class TestLoadBaseModel(unittest.TestCase):
  def load(self, device, cuda: bool, bf16: bool = False) -> tuple[MagicMock, MagicMock]:
    with device_env("cpu"):
      worker = LoraTrainingWorker()
    worker.device = device
    is_bf16_supported = MagicMock(return_value=bf16)
    with (
      patch("training.lora_trainer_worker.AutoTokenizer"),
      patch("training.lora_trainer_worker.AutoModelForCausalLM") as auto_model,
      patch("torch.cuda.is_available", return_value=cuda),
      patch("torch.cuda.is_bf16_supported", is_bf16_supported),
    ):
      worker.load_base_model("tiny/model")
    return auto_model.from_pretrained, is_bf16_supported

  def test_tpu_loads_bf16(self) -> None:
    device = SimpleNamespace(type="tpu")
    from_pretrained, is_bf16_supported = self.load(device, cuda=False)
    from_pretrained.assert_called_once_with("tiny/model", dtype=torch.bfloat16, device_map=device)
    is_bf16_supported.assert_not_called()

  def test_cpu_loads_fp32(self) -> None:
    # Also on a CUDA host, where OPEN_RL_DEVICE=cpu chose the CPU.
    device = torch.device("cpu")
    for cuda in (False, True):
      with self.subTest(cuda=cuda):
        from_pretrained, is_bf16_supported = self.load(device, cuda=cuda, bf16=True)
        from_pretrained.assert_called_once_with("tiny/model", dtype=torch.float32, device_map=device)
        is_bf16_supported.assert_not_called()

  def test_cuda_uses_bf16_when_supported(self) -> None:
    device = torch.device("cuda")
    for bf16, dtype in [(True, torch.bfloat16), (False, torch.float32)]:
      with self.subTest(bf16=bf16):
        from_pretrained, _ = self.load(device, cuda=True, bf16=bf16)
        from_pretrained.assert_called_once_with("tiny/model", dtype=dtype, device_map=device)


class TestModelDtype(unittest.TestCase):
  def test_bf16_on_tpu_and_on_cuda_that_supports_it(self) -> None:
    cases = [
      (SimpleNamespace(type="tpu"), False, torch.bfloat16),
      (torch.device("cuda"), True, torch.bfloat16),
      (torch.device("cuda"), False, torch.float32),
      (torch.device("cpu"), True, torch.float32),
      (torch.device("mps"), True, torch.float32),
    ]
    for device, bf16, expected in cases:
      with self.subTest(device=device, bf16=bf16), patch("torch.cuda.is_bf16_supported", return_value=bf16):
        self.assertEqual(model_dtype(device), expected)


class TestLoadFFTModel(unittest.TestCase):
  def load(self, device, gpus: int, bf16: bool = False) -> MagicMock:
    with device_env("cpu"):
      worker = FFTTrainingWorker()
    worker.device = device
    with (
      patch("training.fft_trainer_worker.AutoTokenizer"),
      patch("training.fft_trainer_worker.AutoModelForCausalLM") as auto_model,
      patch("torch.cuda.is_available", return_value=gpus > 0),
      patch("torch.cuda.device_count", return_value=gpus),
      patch("torch.cuda.is_bf16_supported", return_value=bf16),
    ):
      worker.load_base_model("tiny/model")
    return auto_model.from_pretrained

  def test_tpu_loads_bf16(self) -> None:
    device = SimpleNamespace(type="tpu")
    self.load(device, gpus=0).assert_called_once_with("tiny/model", dtype=torch.bfloat16, device_map=device)

  def test_cuda_uses_bf16_when_supported(self) -> None:
    device = torch.device("cuda", 0)
    for bf16, dtype in [(True, torch.bfloat16), (False, torch.float32)]:
      with self.subTest(bf16=bf16):
        self.load(device, gpus=1, bf16=bf16).assert_called_once_with("tiny/model", dtype=dtype, device_map=device)

  def test_several_gpus_split_the_model(self) -> None:
    self.load(torch.device("cuda", 0), gpus=2, bf16=True).assert_called_once_with("tiny/model", dtype=torch.bfloat16, device_map="auto")

  def test_restore_from_state_uses_the_same_dtype(self) -> None:
    with device_env("cpu"):
      worker = FFTTrainingWorker()
    worker.device = SimpleNamespace(type="tpu")
    with tempfile.TemporaryDirectory() as state_path:
      Path(state_path, "metadata.json").write_text('{"base_model": "tiny/model"}')
      with (
        patch("training.fft_trainer_worker.AutoTokenizer"),
        patch("training.fft_trainer_worker.AutoModelForCausalLM") as auto_model,
        patch.object(worker, "prepare_model_for_training"),
        patch("torch.cuda.is_available", return_value=False),
      ):
        worker.load_from_state("model-1", state_path)
    auto_model.from_pretrained.assert_called_once_with(state_path, dtype=torch.bfloat16, device_map=worker.device)


if __name__ == "__main__":
  unittest.main()
