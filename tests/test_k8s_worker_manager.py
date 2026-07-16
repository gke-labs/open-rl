import os
import tempfile
import types
import unittest
from unittest.mock import patch

from server.k8s_worker_manager import KubernetesFFTWorkerManager, sanitize_job_id
from server.worker_manager import FFTWorkerManager, create_fft_worker_manager

POD_TEMPLATE = """\
apiVersion: v1
kind: Pod
spec:
  restartPolicy: OnFailure
  containers:
  - name: trainer-worker
    image: example/server:latest
    command: ["python", "-m", "server.training_requests_processor"]
    env:
    - name: REDIS_URL
      value: "redis://redis-service:6379"
"""


class _ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class _FakeCoreApi:
  def __init__(self, pod_phases: dict[str, str] | None = None):
    self.pod_phases = pod_phases or {}
    self.created: list[tuple[str, dict]] = []
    self.deleted: list[str] = []
    self.create_error: Exception | None = None

  def read_namespaced_pod(self, name: str, namespace: str):
    if name not in self.pod_phases:
      raise _ApiError(404)
    return types.SimpleNamespace(status=types.SimpleNamespace(phase=self.pod_phases[name]))

  def create_namespaced_pod(self, namespace: str, body: dict):
    if self.create_error is not None:
      raise self.create_error
    self.created.append((namespace, body))

  def delete_namespaced_pod(self, name: str, namespace: str):
    self.deleted.append(name)
    self.pod_phases.pop(name, None)


class KubernetesFFTWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.template_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    self.template_file.write(POD_TEMPLATE)
    self.template_file.close()
    self.addCleanup(os.unlink, self.template_file.name)
    self.env = {
      "REDIS_URL": "redis://redis-service:6379",
      "OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name,
      "OPEN_RL_WORKER_NAMESPACE": "training",
    }

  def _manager(self, core_api: _FakeCoreApi) -> KubernetesFFTWorkerManager:
    with patch.dict(os.environ, self.env, clear=True):
      return KubernetesFFTWorkerManager(core_api=core_api)

  def test_launch_stamps_name_labels_args_and_job_id_env(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch("Model_A.1")

    self.assertEqual(len(api.created), 1)
    namespace, body = api.created[0]
    self.assertEqual(namespace, "training")
    self.assertEqual(body["metadata"]["name"], "open-rl-trainer-model-a-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-trainer-worker",
        "accel-timeslicer": "true",
        "timeslice.io/group": "trainers",
        "timeslice.io/job-id": "trainer-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["args"], ["--model-id", "Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "trainer-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "trainers"}, container["env"])

  def test_launch_replaces_stale_job_id_env_from_template(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "stale-job"})
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "stale-group"})

    manager.launch("Model_A.1")

    container = api.created[0][1]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    self.assertEqual(env["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-model-a-1")
    self.assertEqual(env["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  def test_launch_sampler_stamps_sampler_identity(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch_sampler("Model_A.1")

    self.assertEqual(len(api.created), 1)
    _, body = api.created[0]
    self.assertEqual(body["metadata"]["name"], "open-rl-sampler-model-a-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-sampler-worker",
        "accel-timeslicer": "true",
        "timeslice.io/group": "samplers",
        "timeslice.io/job-id": "sampler-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_sampler"])
    self.assertEqual(container["args"], ["--model-id", "Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "sampler-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "samplers"}, container["env"])

  def test_launch_is_idempotent_while_pod_is_live(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a": "Running"})
    self._manager(api).launch("model-a")

    self.assertEqual(api.created, [])
    self.assertEqual(api.deleted, [])

  def test_launch_replaces_terminal_pod(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a": "Failed"})
    self._manager(api).launch("model-a")

    self.assertEqual(api.deleted, ["open-rl-trainer-model-a"])
    self.assertEqual(len(api.created), 1)

  def test_launch_tolerates_conflict_on_create(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(409)
    self._manager(api).launch("model-a")  # must not raise

  def test_launch_raises_on_other_api_errors(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(403)
    with self.assertRaises(_ApiError):
      self._manager(api).launch("model-a")

  def test_launch_queries_model_metadata_for_pod_env(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "gemma-4-k8s",
        "weight_sync_strategy": "full",
        "training_kind": "full",
      }
    )
    api = _FakeCoreApi()

    with patch("server.store.get_store", return_value=s):
      self._manager(api).launch("Model_A.1")
      self._manager(api).launch_sampler("Model_A.1")

    trainer_container = api.created[0][1]["spec"]["containers"][0]
    trainer_env = {item["name"]: item["value"] for item in trainer_container["env"] if "value" in item}
    self.assertEqual(trainer_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(trainer_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

    sampler_container = api.created[1][1]["spec"]["containers"][0]
    sampler_env = {item["name"]: item["value"] for item in sampler_container["env"] if "value" in item}
    self.assertEqual(sampler_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(sampler_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

  def test_requires_template_and_redis(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True), self.assertRaisesRegex(RuntimeError, "POD_TEMPLATE"):
      KubernetesFFTWorkerManager(core_api=_FakeCoreApi())
    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name}, clear=True),
      self.assertRaisesRegex(RuntimeError, "REDIS_URL"),
    ):
      KubernetesFFTWorkerManager(core_api=_FakeCoreApi())

  def test_sanitize_job_id(self) -> None:
    self.assertEqual(sanitize_job_id("Model_A.1"), "model-a-1")
    self.assertEqual(sanitize_job_id("a" * 80), "a" * 63)
    with self.assertRaises(ValueError):
      sanitize_job_id("___")


class CreateFFTWorkerManagerTest(unittest.TestCase):
  def test_default_launcher_is_subprocess(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True):
      manager = create_fft_worker_manager()
    self.assertIsInstance(manager, FFTWorkerManager)

  def test_kubernetes_launcher_is_selected_by_env(self) -> None:
    env = {"REDIS_URL": "redis://r:6379", "OPEN_RL_WORKER_MANAGER": "kubernetes"}
    with (
      patch.dict(os.environ, env, clear=True),
      patch("server.k8s_worker_manager.KubernetesFFTWorkerManager") as manager_cls,
    ):
      manager = create_fft_worker_manager()
    self.assertIs(manager, manager_cls.return_value)


if __name__ == "__main__":
  unittest.main()
