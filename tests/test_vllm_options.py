import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

from server import vllm_options
from server.vllm_options import gpu_memory_utilization, split_stop

GIB = 1024**3


class SplitStopTest(unittest.TestCase):
  def test_none(self):
    self.assertEqual(split_stop(None), (None, None))

  def test_bare_string(self):
    self.assertEqual(split_stop("\n\nUser:"), (["\n\nUser:"], None))

  def test_token_ids(self):
    self.assertEqual(split_stop([151645, 151643]), (None, [151645, 151643]))

  def test_strings(self):
    self.assertEqual(split_stop(["</s>", "\n\n"]), (["</s>", "\n\n"], None))

  def test_mixed(self):
    self.assertEqual(split_stop(["</s>", 151645]), (["</s>"], [151645]))

  def test_empty_sequence(self):
    self.assertEqual(split_stop([]), (None, None))

  def test_bool_is_not_a_token_id(self):
    self.assertEqual(split_stop([True]), (None, None))


if __name__ == "__main__":
  unittest.main()


class GpuMemoryUtilizationTest(unittest.TestCase):
  def test_explicit_fraction_wins(self) -> None:
    with patch.dict(os.environ, {"VLLM_GPU_MEMORY_UTILIZATION": "0.5", "OPEN_RL_ACCELERATOR_MEMORY": str(70 * 1024**3)}):
      self.assertEqual(gpu_memory_utilization(), 0.5)

  def test_placement_claim_does_not_cap_the_device(self) -> None:
    # The claim only decides where the worker lands; once placed it owns the
    # device while it holds it, so a 7 GiB claim still gets the 0.90 default.
    with patch.dict(os.environ, {"OPEN_RL_ACCELERATOR_MEMORY": str(7 * 1024**3)}, clear=True):
      self.assertEqual(gpu_memory_utilization(), 0.90)

  def test_without_a_budget_the_default_holds(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(gpu_memory_utilization(), 0.90)

  def test_headroom_comes_off_vllms_share(self) -> None:
    # gemma-4-e2b's weight update stages a 4.38 GiB tensor; on a 22 GiB L4 vLLM
    # must leave that free on top of its own 10%, and on an 80 GB H100 a little.
    with patch.dict(os.environ, {}, clear=True):
      l4 = gpu_memory_utilization(headroom_bytes=int(4.88 * GIB), device_memory_bytes=int(22.03 * GIB))
      h100 = gpu_memory_utilization(headroom_bytes=int(4.88 * GIB), device_memory_bytes=80 * GIB)
    self.assertAlmostEqual(l4, 0.90 - 4.88 / 22.03, places=3)
    self.assertAlmostEqual(h100, 0.90 - 4.88 / 80, places=3)
    self.assertLess((0.90 - l4) * 22.03, 4.88 + 0.01)

  def test_the_share_still_holds_the_resident_weights(self) -> None:
    # gemma-4-e2b on an L4: 11 GiB of headroom would leave 0.40 of 22 GiB, less
    # than the 9 GiB of weights vLLM must load. The resident weights win.
    with patch.dict(os.environ, {}, clear=True):
      share = gpu_memory_utilization(headroom_bytes=12 * GIB, device_memory_bytes=22 * GIB, resident_bytes=9 * GIB)
    self.assertAlmostEqual(share, 10 / 22, places=3)

  def test_headroom_never_pushes_the_share_below_the_floor(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(gpu_memory_utilization(headroom_bytes=20 * GIB, device_memory_bytes=22 * GIB), vllm_options.MIN_GPU_MEMORY_UTILIZATION)

  def test_explicit_fraction_ignores_headroom(self) -> None:
    with patch.dict(os.environ, {"VLLM_GPU_MEMORY_UTILIZATION": "0.5"}, clear=True):
      self.assertEqual(gpu_memory_utilization(headroom_bytes=10 * GIB, device_memory_bytes=22 * GIB), 0.5)

  def test_unknown_device_keeps_the_default(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(gpu_memory_utilization(headroom_bytes=5 * GIB, device_memory_bytes=0), 0.90)


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int]]]) -> None:
  """A safetensors file with the given header and zero-filled data."""
  header, offset = {}, 0
  for name, (dtype, shape) in tensors.items():
    numel = 1
    for dim in shape:
      numel *= dim
    size = numel * vllm_options.SAFETENSORS_DTYPE_BYTES[dtype]
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
    offset += size
  raw = json.dumps(header).encode()
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)
    f.write(b"\0" * offset)


class WeightUpdateHeadroomTest(unittest.TestCase):
  def test_largest_tensor_is_read_from_the_headers(self) -> None:
    with tempfile.TemporaryDirectory() as d:
      write_safetensors(os.path.join(d, "model-00001.safetensors"), {"embed": ("BF16", [1024, 64]), "norm": ("F32", [64])})
      write_safetensors(os.path.join(d, "model-00002.safetensors"), {"head": ("BF16", [2048, 64])})
      self.assertEqual(vllm_options.largest_checkpoint_tensor_bytes(d), 2048 * 64 * 2)
      self.assertEqual(vllm_options.checkpoint_tensor_bytes(d), (2048 * 64 * 2, 1024 * 64 * 2 + 64 * 4 + 2048 * 64 * 2))
      with patch.dict(os.environ, {}, clear=True):
        self.assertEqual(
          vllm_options.weight_update_headroom_bytes(d),
          int(2048 * 64 * 2 * vllm_options.WEIGHT_UPDATE_STAGING_FACTOR) + vllm_options.WEIGHT_UPDATE_MARGIN_BYTES,
        )

  def test_missing_checkpoint_means_no_headroom(self) -> None:
    with patch.dict(os.environ, {}, clear=True), patch.object(vllm_options, "local_snapshot", return_value=None):
      self.assertEqual(vllm_options.weight_update_headroom_bytes("nobody/no-such-model"), 0)

  def test_explicit_headroom_wins(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_WEIGHT_UPDATE_HEADROOM_GIB": "1.5"}, clear=True):
      self.assertEqual(vllm_options.weight_update_headroom_bytes("nobody/no-such-model"), int(1.5 * GIB))
    with patch.dict(os.environ, {"OPEN_RL_WEIGHT_UPDATE_HEADROOM_GIB": "0"}, clear=True):
      self.assertEqual(vllm_options.weight_update_headroom_bytes("nobody/no-such-model"), 0)


class CudagraphCaptureSizesTest(unittest.TestCase):
  def test_powers_of_two_up_to_the_sequence_limit_by_default(self):
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VLLM_CUDAGRAPH_CAPTURE_SIZES", None)
      self.assertEqual(vllm_options.cudagraph_capture_sizes(64), [1, 2, 4, 8, 16, 32, 64])
      self.assertEqual(vllm_options.cudagraph_capture_sizes(256)[-1], 256)
      self.assertEqual(vllm_options.compilation_kwargs(8), {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}})

  def test_an_explicit_list_is_sorted_and_capped(self):
    with patch.dict(os.environ, {"VLLM_CUDAGRAPH_CAPTURE_SIZES": "64, 8,16,512"}):
      self.assertEqual(vllm_options.cudagraph_capture_sizes(128), [8, 16, 64])

  def test_default_keeps_vllms_own_list(self):
    with patch.dict(os.environ, {"VLLM_CUDAGRAPH_CAPTURE_SIZES": "default"}):
      self.assertIsNone(vllm_options.cudagraph_capture_sizes(64))
      self.assertEqual(vllm_options.compilation_kwargs(64), {})
