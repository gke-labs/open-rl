import unittest

from server.accelerators import check_supported, parse_accelerator, pod_spec
from server.model_metadata import TrainingModelMetadata


class ParseAcceleratorTest(unittest.TestCase):
  def test_case_and_whitespace_do_not_matter(self) -> None:
    for value in ("tpu", " TPU ", "Tpu"):
      with self.subTest(value=value):
        self.assertEqual(parse_accelerator(value), "tpu")

  def test_other_types_pass_through(self) -> None:
    self.assertEqual(parse_accelerator(["tpu"]), ["tpu"])


class AcceleratorForTest(unittest.TestCase):
  def test_each_role_has_its_own_setting(self) -> None:
    meta = TrainingModelMetadata(base_model="m", trainer_accel_prefs="tpu", sampler_accel_prefs="gpu")
    self.assertEqual(meta.accelerator_for("trainer"), "tpu")
    self.assertEqual(meta.accelerator_for("sampler"), "gpu")

  def test_a_record_with_a_preference_list_reads_as_its_first_entry(self) -> None:
    meta = TrainingModelMetadata(base_model="m", trainer_accel_prefs=["tpu", "gpu"], sampler_accel_prefs=["gpu", "tpu"])
    self.assertEqual((meta.trainer_accel_prefs, meta.sampler_accel_prefs), ("tpu", "gpu"))
    with self.assertRaises(ValueError):
      TrainingModelMetadata(base_model="m", trainer_accel_prefs=[])

  def test_defaults_to_gpu(self) -> None:
    meta = TrainingModelMetadata(base_model="m")
    self.assertEqual((meta.accelerator_for("trainer"), meta.accelerator_for("sampler")), ("gpu", "gpu"))


class CheckSupportedTest(unittest.TestCase):
  def test_gpu_takes_lora_and_full(self) -> None:
    check_supported("lora", "pytorch", "gpu", "gpu", False)
    check_supported("full", "pytorch", "gpu", "gpu", False)
    check_supported("lora", "automodel", "gpu", "gpu", False)

  def test_shared_full_fine_tuning_refuses_tpu_for_either_role(self) -> None:
    for trainer, sampler in (("tpu", "gpu"), ("gpu", "tpu")):
      with self.subTest(trainer=trainer, sampler=sampler), self.assertRaisesRegex(ValueError, "needs openrl.exclusive=true"):
        check_supported("full", "pytorch", trainer, sampler, False)

  def test_tpu_takes_exclusive_full_fine_tuning(self) -> None:
    for trainer, sampler in (("tpu", "tpu"), ("gpu", "tpu"), ("tpu", "gpu")):
      with self.subTest(trainer=trainer, sampler=sampler):
        check_supported("full", "pytorch", trainer, sampler, True)

  def test_tpu_takes_lora(self) -> None:
    check_supported("lora", "pytorch", "tpu", "tpu", False)
    check_supported("lora", "automodel", "gpu", "tpu", False)

  def test_a_tpu_trainer_refuses_automodel(self) -> None:
    with self.assertRaisesRegex(ValueError, "openrl.trainer_backend=automodel needs a GPU trainer"):
      check_supported("lora", "automodel", "tpu", "gpu", False)

  def test_a_tpu_trainer_takes_a_job_image(self) -> None:
    check_supported("lora", "ghcr.io/org/trainer:1", "tpu", "tpu", False)


class PodSpecTest(unittest.TestCase):
  def test_gpu_workers_keep_todays_pod(self) -> None:
    for role in ("trainer", "sampler"):
      with self.subTest(role=role):
        spec = pod_spec("gpu", role)
        self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_WORKER_IMAGE", "ghcr.io/openrl-dev/openrl/server:latest"))
        self.assertEqual(spec.toleration_key, "nvidia.com/gpu")
        # The CRD defaults the type to GPU, so GPU Workloads leave it out.
        self.assertIsNone(spec.workload_type)
        self.assertEqual(spec.env_configmap_env, "OPEN_RL_GPU_WORKER_ENV_CONFIGMAP")
        self.assertEqual((spec.env, spec.volumes, spec.volume_mounts), ({}, [], []))

  def test_tpu_trainer(self) -> None:
    spec = pod_spec("tpu", "trainer")
    self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_TPU_TRAINER_IMAGE", None))
    self.assertEqual((spec.toleration_key, spec.workload_type), ("google.com/tpu", "TPU"))
    self.assertEqual(spec.env_configmap_env, "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP")
    self.assertEqual(spec.env, {"OPEN_RL_DEVICE": "tpu", "TORCH_TPU_DEFER_AND_FUSE": "1"})
    self.assertEqual(spec.volumes, [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}}])
    self.assertEqual(spec.volume_mounts, [{"name": "dshm", "mountPath": "/dev/shm"}])

  def test_tpu_sampler(self) -> None:
    spec = pod_spec("tpu", "sampler")
    self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_TPU_SAMPLER_IMAGE", None))
    self.assertEqual((spec.toleration_key, spec.workload_type), ("google.com/tpu", "TPU"))
    self.assertEqual(spec.env_configmap_env, "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP")
    self.assertEqual(spec.env, {})
    self.assertEqual(spec.volumes, [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}}])
    self.assertEqual(spec.volume_mounts, [{"name": "dshm", "mountPath": "/dev/shm"}])


if __name__ == "__main__":
  unittest.main()
