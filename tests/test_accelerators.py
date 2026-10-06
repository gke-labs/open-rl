import unittest
from unittest.mock import patch

from server.accelerators import LocalLaunch, check_supported, local_launch, parse_accel_prefs, tpu_chip_env


class ParseAccelPrefsTest(unittest.TestCase):
  def test_commas_or_bars_separate_entries(self) -> None:
    for value in ("tpu,gpu", "tpu|gpu", " TPU | gpu ", ["tpu", "gpu"]):
      with self.subTest(value=value):
        self.assertEqual(parse_accel_prefs(value), ["tpu", "gpu"])

  def test_empty_and_repeated_lists_are_refused(self) -> None:
    for value, error in (([], "at least one"), ("tpu|tpu", "twice")):
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, error):
        parse_accel_prefs(value)


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


class LocalLaunchTest(unittest.TestCase):
  def test_launch_table(self) -> None:
    tpu_trainer_env = {"OPEN_RL_DEVICE": "tpu", "TORCH_DEVICE_BACKEND_AUTOLOAD": "1"}
    cases = [
      ("gpu", "trainer", LocalLaunch(["gpu"], None, {})),
      ("gpu", "sampler", LocalLaunch(["gpu", "vllm"], None, {})),
      ("tpu", "trainer", LocalLaunch(["tpu"], ".venv-tpu-trainer", tpu_trainer_env)),
    ]
    for accelerator, role, expected in cases:
      with self.subTest(accelerator=accelerator, role=role), patch.dict("os.environ", {}, clear=True):
        self.assertEqual(local_launch(accelerator, role), expected)

  def test_tpu_samplers_are_refused_until_they_exist(self) -> None:
    with self.assertRaisesRegex(NotImplementedError, "TPU samplers"):
      local_launch("tpu", "sampler")

  def test_trainer_tpu_visible_chips_pins_the_tpu_trainer(self) -> None:
    with patch.dict("os.environ", {"TRAINER_TPU_VISIBLE_CHIPS": "2"}, clear=True):
      self.assertEqual(local_launch("tpu", "trainer").env["TPU_VISIBLE_CHIPS"], "2")
      self.assertEqual(local_launch("gpu", "trainer").env, {})


class TpuChipEnvTest(unittest.TestCase):
  def test_one_chip(self) -> None:
    self.assertEqual(
      tpu_chip_env("0"),
      {"TPU_VISIBLE_CHIPS": "0", "TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1"},
    )

  def test_more_than_one_chip_is_refused(self) -> None:
    for chips in ("0,1", "0,1,2,3"):
      with self.subTest(chips=chips), self.assertRaisesRegex(ValueError, "one chip"):
        tpu_chip_env(chips)


if __name__ == "__main__":
  unittest.main()
