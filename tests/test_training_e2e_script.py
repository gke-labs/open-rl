import os
import unittest
from unittest import mock

from scripts import run_training_e2e

TPU_TAGS = {"openrl.trainer_accel_prefs=tpu", "openrl.sampler_accel_prefs=tpu"}


def make_config(**overrides) -> run_training_e2e.RunConfig:
  return run_training_e2e.RunConfig(scenario="tiny-lora", **overrides)


def tags(env: dict[str, str]) -> set[str]:
  return set(env["TINKER_TAGS"].split(","))


class TrainingE2EAcceleratorTest(unittest.TestCase):
  def test_tpu_adds_both_accel_prefs_tags(self) -> None:
    with mock.patch.dict(os.environ, clear=True):
      env = run_training_e2e.examples_env(make_config(accelerator="tpu"))
    self.assertEqual(tags(env), TPU_TAGS)

  def test_tpu_keeps_existing_tags(self) -> None:
    with mock.patch.dict(os.environ, {"TINKER_TAGS": "team=rl,openrl.debug=true"}, clear=True):
      env = run_training_e2e.examples_env(make_config(accelerator="tpu"))
    self.assertEqual(tags(env), TPU_TAGS | {"team=rl", "openrl.debug=true"})

  def test_tpu_replaces_existing_accel_prefs(self) -> None:
    with mock.patch.dict(os.environ, {"TINKER_TAGS": "openrl.trainer_accel_prefs=gpu|tpu,team=rl"}, clear=True):
      env = run_training_e2e.examples_env(make_config(accelerator="tpu"))
    self.assertEqual(tags(env), TPU_TAGS | {"team=rl"})

  def test_gpu_and_default_leave_env_unchanged(self) -> None:
    for environ in ({}, {"TINKER_TAGS": "team=rl"}):
      with mock.patch.dict(os.environ, environ, clear=True):
        default_env = run_training_e2e.examples_env(make_config())
        gpu_env = run_training_e2e.examples_env(make_config(accelerator="gpu"))
      self.assertEqual(default_env, gpu_env)
      self.assertEqual(default_env.get("TINKER_TAGS"), environ.get("TINKER_TAGS"))

  def test_tpu_works_with_existing_backend(self) -> None:
    config = make_config(accelerator="tpu", base_url="http://127.0.0.1:9003")
    processes: list[run_training_e2e.ManagedProcess] = []
    self.assertEqual(run_training_e2e.start_backend(config, processes), "http://127.0.0.1:9003")
    self.assertEqual(processes, [])
    with mock.patch.dict(os.environ, clear=True):
      env = run_training_e2e.examples_env(config)
    self.assertEqual(tags(env), TPU_TAGS)


if __name__ == "__main__":
  unittest.main()
