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
    check_supported("lora", "pytorch", ["gpu"], ["gpu"])
    check_supported("full", "pytorch", ["gpu"], ["gpu"])
    check_supported("lora", "automodel", ["gpu"], ["gpu"])

  def test_full_fine_tuning_refuses_tpu_anywhere_in_either_list(self) -> None:
    for trainer, sampler in ((["tpu"], ["gpu"]), (["gpu"], ["tpu"]), (["gpu", "tpu"], ["gpu"])):
      with self.subTest(trainer=trainer, sampler=sampler), self.assertRaisesRegex(ValueError, "LoRA only"):
        check_supported("full", "pytorch", trainer, sampler)

  def test_tpu_takes_lora(self) -> None:
    check_supported("lora", "pytorch", ["tpu"], ["tpu"])
    check_supported("lora", "automodel", ["gpu"], ["tpu"])

  def test_a_tpu_trainer_runs_the_pytorch_backend_only(self) -> None:
    for backend in ("automodel", "ghcr.io/org/trainer:1"):
      with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, f"openrl.trainer_backend={backend} needs a GPU trainer"):
        check_supported("lora", backend, ["gpu", "tpu"], ["gpu"])


class PodSpecTest(unittest.TestCase):
  def test_gpu_workers_keep_todays_pod(self) -> None:
    for role in ("trainer", "sampler"):
      with self.subTest(role=role):
        spec = pod_spec("gpu", role)
        self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest"))
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
    self.assertEqual(spec.env, {"OPEN_RL_DEVICE": "tpu"})
    self.assertEqual((spec.volumes, spec.volume_mounts), ([], []))

  def test_tpu_sampler(self) -> None:
    spec = pod_spec("tpu", "sampler")
    self.assertEqual((spec.image_env, spec.default_image), ("OPEN_RL_TPU_SAMPLER_IMAGE", None))
    self.assertEqual((spec.toleration_key, spec.workload_type), ("google.com/tpu", "TPU"))
    self.assertEqual(spec.env_configmap_env, "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP")
    self.assertEqual(spec.env, {})
    self.assertEqual(spec.volumes, [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}])
    self.assertEqual(spec.volume_mounts, [{"name": "dshm", "mountPath": "/dev/shm"}])


if __name__ == "__main__":
  unittest.main()
