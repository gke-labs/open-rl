import sys
import unittest
from unittest.mock import patch

from server.estimator import GIB, MODEL_TO_KV_BYTES_PER_TOKEN, SAMPLER_KV_TOKENS, footprint, normalize_model_id, parameter_count


def kv_cache_bytes(model: str) -> int:
  return MODEL_TO_KV_BYTES_PER_TOKEN[normalize_model_id(model)] * SAMPLER_KV_TOKENS


class FootprintTest(unittest.TestCase):
  """Sizing is a formula over the parameter count: bytes per parameter on the
  device and in host memory, plus a fixed reserve per role."""

  def test_small_models_fit_a_24gb_device(self) -> None:
    for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
      for kind in ("lora", "full"):
        self.assertLess(footprint(model, kind, "trainer").accelerator_bytes, 22 * GIB, f"{model} {kind}")
        self.assertLess(footprint(model, kind, "sampler").accelerator_bytes, 22 * GIB, f"{model} {kind}")

  def test_lora_needs_less_device_than_full_fine_tuning(self) -> None:
    # 4B carries 8GB of frozen weights for LoRA but weights, grads and a
    # master copy for a full fine-tune.
    self.assertLess(footprint("Qwen/Qwen3-4B", "lora", "trainer").accelerator_bytes, 22 * GIB)
    self.assertGreater(footprint("Qwen/Qwen3-4B", "full", "trainer").accelerator_bytes, 22 * GIB)

  def test_sampler_matches_the_live_samplers(self) -> None:
    # vLLM on the sweep: handed 7.11 GiB, a Qwen3-0.6B LoRA sampler loaded
    # 2.17 GiB and had 4.39 GiB of KV cache; handed 21.25 GiB, a Qwen3-8B
    # LoRA sampler loaded 18.17 GiB and had 1.49 GiB. The model must land
    # within 0.3 GiB of the KV cache each actually got.
    for model, budget, kv in (("Qwen/Qwen3-0.6B", 7.11, 4.39), ("Qwen/Qwen3-8B", 21.25, 1.49)):
      fp = footprint(model, "lora", "sampler")
      predicted_kv = kv_cache_bytes(model) - (fp.accelerator_bytes - budget * GIB)
      self.assertAlmostEqual(predicted_kv / GIB, kv, delta=0.3, msg=model)

  def test_sampler_kv_cache_grows_with_the_model(self) -> None:
    # An 8B LoRA sampler sized at 22Gi landed on an L4 with 0.23 GiB of KV
    # cache left and crash-looped; it belongs on the 80GB tier. 4B still fits,
    # and a 27B sampler still fits one 80GB device.
    self.assertGreater(footprint("Qwen/Qwen3-8B", "lora", "sampler").accelerator_bytes, 22 * GIB)
    self.assertLess(footprint("Qwen/Qwen3-4B", "lora", "sampler").accelerator_bytes, 22 * GIB)
    self.assertLess(footprint("Qwen/Qwen3.5-27B", "lora", "sampler").accelerator_bytes, 79 * GIB)

  def test_host_memory_matches_the_measured_points(self) -> None:
    # Qwen2.5-0.5B trial runs measured 28Gi (trainer) and 20Gi (sampler).
    trainer = footprint("Qwen/Qwen2.5-0.5B", "full", "trainer")
    sampler = footprint("Qwen/Qwen2.5-0.5B", "full", "sampler")
    self.assertTrue(24 * GIB <= trainer.host_request_bytes <= 32 * GIB, trainer)
    self.assertTrue(20 * GIB <= sampler.host_request_bytes <= 28 * GIB, sampler)
    # A Qwen2.5-7B FFT trainer was OOM-killed at a 110Gi limit; the request
    # alone must already say that much.
    big = footprint("Qwen/Qwen2.5-7B", "full", "trainer")
    self.assertGreater(big.host_request_bytes, 110 * GIB)
    self.assertEqual(big.host_limit_bytes, big.host_request_bytes)

  def test_tpu_workers_get_generous_host_memory(self) -> None:
    # Every TPU run so far used a 96Gi limit and worked; the overheads start
    # near it until measured peaks lower them.
    for model in ("Qwen/Qwen3-0.6B", "google/gemma-4-E2B-it"):
      for role in ("trainer", "sampler"):
        fp = footprint(model, "lora", role, accelerator="tpu")
        self.assertTrue(90 * GIB <= fp.host_request_bytes <= 110 * GIB, f"{model} {role} {fp}")
        self.assertEqual(fp.host_limit_bytes, fp.host_request_bytes)

  def test_tpu_keeps_the_gpu_device_figure(self) -> None:
    for role in ("trainer", "sampler"):
      tpu = footprint("google/gemma-4-E2B-it", "lora", role, accelerator="tpu")
      self.assertEqual(tpu.accelerator_bytes, footprint("google/gemma-4-E2B-it", "lora", role).accelerator_bytes)

  def test_gpu_is_the_default_accelerator(self) -> None:
    for role in ("trainer", "sampler"):
      self.assertEqual(footprint("Qwen/Qwen3-0.6B", "lora", role), footprint("Qwen/Qwen3-0.6B", "lora", role, accelerator="gpu"))

  def test_resources_render_as_whole_gib(self) -> None:
    fp = footprint("Qwen/Qwen2.5-0.5B", "lora", "trainer")
    self.assertRegex(fp.accelerator, r"^\d+Gi$")
    self.assertEqual(set(fp.resources), {"requests", "limits"})
    self.assertRegex(fp.resources["limits"]["memory"], r"^\d+Gi$")

  def test_effective_size_names_are_sized_by_their_raw_weights(self) -> None:
    # gemma-4-e2b names an effective 2B but holds ~5B raw weights.
    self.assertGreater(
      footprint("google/gemma-4-E2B-it", "full", "trainer").accelerator_bytes, footprint("google/gemma-3-1b-it", "full", "trainer").accelerator_bytes
    )

  def test_unknown_models_are_sized_large(self) -> None:
    with self.assertLogs("server.estimator", level="WARNING"):
      unknown = footprint("meta-llama/Llama-3-70B", "lora", "sampler")
    self.assertEqual(unknown, footprint("Qwen/Qwen3-8B", "lora", "sampler"))

  def test_restored_models_are_sized_as_full_fine_tunes(self) -> None:
    self.assertEqual(footprint("Qwen/Qwen3-4B", "restored", "trainer"), footprint("Qwen/Qwen3-4B", "full", "trainer"))

  def test_variant_and_release_tags_resolve_to_the_base_entry(self) -> None:
    self.assertEqual(parameter_count("Qwen/Qwen3-4B-Instruct-2507"), parameter_count("Qwen/Qwen3-4B"))
    self.assertEqual(parameter_count("google/gemma-4-E2B-it"), parameter_count("gemma-4-e2b"))

  def test_sizing_needs_no_network_or_hub_client(self) -> None:
    with patch.dict(sys.modules, {"huggingface_hub": None}):
      self.assertIsNotNone(footprint("Qwen/Qwen3-8B", "lora", "trainer"))


if __name__ == "__main__":
  unittest.main()
