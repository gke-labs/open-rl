import unittest

from server.accelerators import check_supported, parse_accel_prefs, pod_spec
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
    check_supported("lora", "pytorch", ["gpu"], ["gpu"], False)
    check_supported("full", "pytorch", ["gpu"], ["gpu"], False)
    check_supported("lora", "automodel", ["gpu"], ["gpu"], False)

  def test_shared_full_fine_tuning_refuses_tpu_anywhere_in_either_list(self) -> None:
    for trainer, sampler in ((["tpu"], ["gpu"]), (["gpu"], ["tpu"]), (["gpu", "tpu"], ["gpu"])):
      with self.subTest(trainer=trainer, sampler=sampler), self.assertRaisesRegex(ValueError, "needs openrl.exclusive=true"):
        check_supported("full", "pytorch", trainer, sampler, False)

  def test_tpu_takes_exclusive_full_fine_tuning(self) -> None:
    check_supported("full", "pytorch", ["tpu"], ["tpu"], True)

  def test_tpu_takes_lora(self) -> None:
    check_supported("lora", "pytorch", ["tpu"], ["tpu"], False)
    check_supported("lora", "automodel", ["gpu"], ["tpu"], False)

  def test_a_tpu_trainer_refuses_automodel(self) -> None:
    with self.assertRaisesRegex(ValueError, "openrl.trainer_backend=automodel needs a GPU trainer"):
      check_supported("lora", "automodel", ["gpu", "tpu"], ["gpu"], False)

  def test_a_tpu_trainer_takes_a_job_image(self) -> None:
    check_supported("lora", "ghcr.io/org/trainer:1", ["tpu"], ["tpu"], False)


class PodSpecTest(unittest.TestCase):
  def test_gpu_workers_keep_todays_pod(self) -> None:
    for role, is_lora in (("trainer", True), ("sampler", True), ("trainer", False), ("sampler", False)):
      with self.subTest(role=role, is_lora=is_lora):
        spec = pod_spec("gpu", role, is_lora)
        self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest"))
        self.assertEqual(spec.toleration_key, "nvidia.com/gpu")
        # The CRD defaults the type to GPU, so GPU Workloads leave it out.
        self.assertIsNone(spec.workload_type)
        self.assertEqual(spec.env_configmap_env, "OPEN_RL_GPU_WORKER_ENV_CONFIGMAP")
        self.assertEqual((spec.env, spec.volumes, spec.volume_mounts), ({}, [], []))

  def test_tpu_trainer(self) -> None:
    spec = pod_spec("tpu", "trainer", True)
    self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_TPU_TRAINER_IMAGE", None))
    self.assertEqual((spec.toleration_key, spec.workload_type), ("google.com/tpu", "TPU"))
    self.assertEqual(spec.env_configmap_env, "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP")
    self.assertEqual(spec.env, {"OPEN_RL_DEVICE": "tpu"})
    self.assertEqual((spec.volumes, spec.volume_mounts), ([], []))

  def test_tpu_sampler(self) -> None:
    spec = pod_spec("tpu", "sampler", True)
    self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_TPU_SAMPLER_IMAGE", None))
    self.assertEqual((spec.toleration_key, spec.workload_type), ("google.com/tpu", "TPU"))
    self.assertEqual(spec.env_configmap_env, "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP")
    self.assertEqual(spec.env, {})
    self.assertEqual(spec.volumes, [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}])
    self.assertEqual(spec.volume_mounts, [{"name": "dshm", "mountPath": "/dev/shm"}])

  def test_tpu_fft_sampler_has_its_own_image(self) -> None:
    self.assertEqual(pod_spec("tpu", "sampler", False).image_env, "OPEN_RL_TPU_FFT_SAMPLER_IMAGE")
    self.assertEqual(pod_spec("tpu", "trainer", False).image_env, "OPEN_RL_TPU_TRAINER_IMAGE")


if __name__ == "__main__":
  unittest.main()
