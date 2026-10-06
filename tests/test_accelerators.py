import unittest

from server.accelerators import check_supported, parse_accel_prefs
from server.model_metadata import TrainingModelMetadata


class ParseAccelPrefsTest(unittest.TestCase):
  def test_commas_or_bars_separate_entries(self) -> None:
    for value in ("tpu,gpu", "tpu|gpu", " TPU | gpu ", ["tpu", "gpu"]):
      with self.subTest(value=value):
        self.assertEqual(parse_accel_prefs(value), ["tpu", "gpu"])

  def test_empty_and_repeated_lists_are_refused(self) -> None:
    for value, error in (([], "at least one"), ("tpu|tpu", "twice")):
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, error):
        parse_accel_prefs(value)


class AcceleratorForTest(unittest.TestCase):
  def test_each_role_takes_the_first_entry_of_its_list(self) -> None:
    meta = TrainingModelMetadata(base_model="m", trainer_accel_prefs=["tpu", "gpu"], sampler_accel_prefs=["gpu", "tpu"])
    self.assertEqual(meta.accelerator_for("trainer"), "tpu")
    self.assertEqual(meta.accelerator_for("sampler"), "gpu")

  def test_defaults_to_gpu(self) -> None:
    meta = TrainingModelMetadata(base_model="m")
    self.assertEqual((meta.accelerator_for("trainer"), meta.accelerator_for("sampler")), ("gpu", "gpu"))


class CheckSupportedTest(unittest.TestCase):
  def test_gpu_takes_lora_and_full(self) -> None:
    check_supported("lora", ["gpu"], ["gpu"])
    check_supported("full", ["gpu"], ["gpu"])

  def test_full_fine_tuning_needs_gpu_in_both_lists(self) -> None:
    for trainer, sampler in ((["tpu"], ["gpu"]), (["gpu"], ["tpu"])):
      with self.subTest(trainer=trainer, sampler=sampler), self.assertRaisesRegex(ValueError, "LoRA only"):
        check_supported("full", trainer, sampler)

  def test_tpu_is_refused_until_tpu_workers_exist(self) -> None:
    for trainer, sampler in ((["tpu"], ["gpu"]), (["gpu"], ["tpu", "gpu"])):
      with self.subTest(trainer=trainer, sampler=sampler), self.assertRaisesRegex(ValueError, "not supported yet"):
        check_supported("lora", trainer, sampler)


if __name__ == "__main__":
  unittest.main()
