"""Engine settings that differ on the TPU platform."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.vllm_options import engine_args, on_tpu_platform

EXTENSION = "server.tpu_worker_extension.TPUWeightTransferExtension"


def args(is_lora: bool, on_tpu: bool) -> dict:
  with patch.dict(os.environ, {}, clear=True), patch("server.vllm_options.on_tpu_platform", return_value=on_tpu):
    return engine_args(is_lora)


def platform(out_of_tree: bool, device_type: str) -> dict[str, types.ModuleType]:
  """A fake vllm.platforms, since the unit tests run without vLLM."""
  modules = {name: types.ModuleType(name) for name in ("vllm", "vllm.platforms")}
  modules["vllm.platforms"].current_platform = SimpleNamespace(is_out_of_tree=lambda: out_of_tree, device_type=device_type)
  return modules


class TpuEngineArgsTest(unittest.TestCase):
  def test_an_fft_sampler_on_tpu_gets_the_weight_transfer_extension(self) -> None:
    self.assertEqual(args(False, on_tpu=True)["worker_extension_cls"], EXTENSION)

  def test_other_samplers_keep_vllms_worker(self) -> None:
    for is_lora, on_tpu in ((False, False), (True, True), (True, False)):
      with self.subTest(is_lora=is_lora, on_tpu=on_tpu):
        self.assertNotIn("worker_extension_cls", args(is_lora, on_tpu))

  def test_the_tpu_platform_captures_no_cuda_graphs(self) -> None:
    self.assertNotIn("compilation_config", args(False, on_tpu=True))
    self.assertIn("compilation_config", args(False, on_tpu=False))

  def test_the_tpu_platform_is_out_of_tree_with_tpu_devices(self) -> None:
    cases = (((True, "tpu"), True), ((True, "cuda"), False), ((False, "tpu"), False), ((False, "cuda"), False))
    for (out_of_tree, device_type), expected in cases:
      with self.subTest(out_of_tree=out_of_tree, device_type=device_type), patch.dict(sys.modules, platform(out_of_tree, device_type)):
        self.assertEqual(on_tpu_platform(), expected)

  def test_without_vllm_the_platform_is_not_tpu(self) -> None:
    with patch.dict(sys.modules, {"vllm": None, "vllm.platforms": None}):
      self.assertFalse(on_tpu_platform())


if __name__ == "__main__":
  unittest.main()
