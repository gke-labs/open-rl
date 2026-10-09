import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from server import api_server
from server.estimator import footprint
from server.scheduler_worker_manager import GROUP, PLURAL, VERSION, SchedulerWorkerManager
from server.store import InMemoryStateStore, InMemoryStore
from tests.api_client import asgi_client, post_json


class ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class FakeCustomObjectsApi:
  def __init__(self):
    self.created: list[dict[str, Any]] = []
    self.deleted: list[str] = []
    self.existing: dict[str, dict] = {}
    self.deleting: set[str] = set()

  def create_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict) -> dict:
    assert (group, version, plural) == (GROUP, VERSION, PLURAL)
    name = body["metadata"]["name"]
    if name in self.existing:
      raise ApiError(409)
    self.existing[name] = body
    self.created.append(body)
    return body

  def get_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict:
    if name not in self.existing:
      raise ApiError(404)
    metadata = dict(self.existing[name]["metadata"])
    if name in self.deleting:
      metadata["deletionTimestamp"] = "2026-09-08T00:00:00Z"
    return {**self.existing[name], "metadata": metadata}

  def delete_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict:
    if name not in self.existing:
      raise ApiError(404)
    del self.existing[name]
    self.deleted.append(name)
    return {}

  def list_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, label_selector: str = "") -> dict:
    key, _, value = label_selector.partition("=")
    items = [body for name, body in sorted(self.existing.items()) if not key or body["metadata"]["labels"].get(key) == value]
    return {"items": items}


class SchedulerWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}))
    self.api = FakeCustomObjectsApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api)

  def store_with(self, model_id: str, meta: dict) -> InMemoryStore:
    s = InMemoryStateStore()
    s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
    return s

  def test_lora_trainer_and_sampler_share_an_owner(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "sampler")

    trainer, sampler = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-qwen-qwen2-5-0-5b-0-trainer")
    self.assertEqual(sampler["metadata"]["name"], "lora-qwen-qwen2-5-0-5b-0-sampler")
    # Same owner: one turn for the pair, and they hold the devices together.
    self.assertEqual(trainer["spec"]["ownerID"], "qwen-qwen2-5-0-5b")
    self.assertEqual(trainer["spec"]["ownerID"], sampler["spec"]["ownerID"])
    self.assertEqual(trainer["spec"]["trainingKind"], "lora")
    self.assertTrue(trainer["spec"]["exclusive"])
    self.assertEqual(trainer["spec"]["accelerator"], {"mode": "SingleGPU", "memory": footprint("Qwen/Qwen2.5-0.5B", "lora", "trainer").accelerator})
    t_container = trainer["spec"]["template"]["spec"]["containers"][0]
    s_container = sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["command"][-1], "server.training_requests_processor")
    self.assertEqual(s_container["command"][-1], "server.vllm_sampler")
    self.assertIn("--active-tenant-set-id", t_container["args"])

  def test_an_exclusive_fft_worker_is_placed_alone_and_never_time_sliced(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full", "exclusive": True})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")
      self.manager.ensure("Model_A.1", "sampler")

    for worker in self.api.created:
      self.assertTrue(worker["spec"]["exclusive"])
      env = {e["name"]: e.get("value") for e in worker["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual(env["OPEN_RL_TIME_SLICING"], "off")

  def test_fft_worker_is_its_own_owner(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")

    (worker,) = self.api.created
    self.assertEqual(worker["spec"]["role"], "trainer")
    # An FFT job is its own owner: its trainer (and sampler, if launched)
    # share one turn, and no other job ever matches this ID.
    self.assertEqual(worker["metadata"]["name"], "fft-model-a-1-trainer")
    self.assertEqual(worker["spec"]["ownerID"], "model-a-1")
    self.assertEqual(worker["spec"]["trainingKind"], "fft")
    self.assertFalse(worker["spec"]["exclusive"])
    self.assertEqual(worker["spec"]["accelerator"]["memory"], footprint("Qwen/Qwen3-8B", "full", "trainer").accelerator)
    container = worker["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(env["OPEN_RL_ENABLE_FFT"], "true")
    self.assertNotIn("OPEN_RL_TIME_SLICING", env)
    self.assertEqual(env["OPEN_RL_FINE_TUNING_TYPE"], "full")
    self.assertEqual(env["OPEN_RL_WORKLOAD_ID"], worker["metadata"]["name"])

  def test_mutable_worker_images_use_the_requested_pull_policy(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with (
      patch("server.worker_manager.get_state_store", return_value=s),
      patch.dict(os.environ, {"OPEN_RL_WORKER_IMAGE": "localhost:5001/open-rl-server:kind-dev", "OPEN_RL_WORKER_IMAGE_PULL_POLICY": "Always"}),
    ):
      self.manager.ensure("job-lora-1", "trainer")

    container = self.api.created[0]["spec"]["template"]["spec"]["containers"][0]
    # Reusing kind-dev must fetch the rebuilt worker rather than its cached predecessor.
    self.assertEqual(container["imagePullPolicy"], "Always")

  def test_workers_get_the_deployment_settings(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    settings = {"VLLM_MAX_MODEL_LEN": "131072", "OPEN_RL_TRAIN_TOKEN_BUDGET": "131072", "MAX_JOBS": "4"}
    with patch.dict(os.environ, settings), patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")
      self.manager.ensure("job-lora-1", "trainer")

    self.assertEqual(len(self.api.created), 2)
    for pod in self.api.created:
      env = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual({k: env.get(k) for k in settings}, settings)

  def test_launch_is_idempotent(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "trainer")
    self.assertEqual(len(self.api.created), 1)

  def test_placement_knowledge_stays_out_of_the_template(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")

    (worker,) = self.api.created
    template_spec = worker["spec"]["template"]["spec"]
    env_names = {e["name"] for e in template_spec["containers"][0]["env"]}
    # The group is the claim name: placement's output, stamped by the
    # controller. Node selection and claims likewise never appear here --
    # the controller rejects a template that carries them.
    self.assertNotIn("OPEN_RL_TIME_SLICE_GROUP", env_names)
    self.assertNotIn("nodeSelector", template_spec)
    self.assertNotIn("nodeName", template_spec)
    self.assertNotIn("affinity", template_spec)
    self.assertNotIn("resourceClaims", template_spec)

  def test_release_deletes_an_fft_jobs_workloads_and_tolerates_absence(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")
      self.manager.release("Model_A.1")
      self.manager.release("Model_A.1")

    self.assertEqual(self.api.deleted, ["fft-model-a-1-trainer"])

  def test_release_leaves_a_shared_lora_runtime_alone(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.release("job-lora-1")

    self.assertEqual(self.api.deleted, [])

  def test_exclusive_lora_models_get_runtimes_of_their_own(self) -> None:
    s = InMemoryStateStore()
    for model_id in ("job-a", "job-b"):
      meta = {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora", "exclusive": True}
      s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-a", "trainer")
      self.manager.ensure("job-b", "trainer")
      self.manager.release("job-a")

    self.assertEqual([w["metadata"]["name"] for w in self.api.created], ["lora-job-a-0-trainer", "lora-job-b-0-trainer"])
    self.assertEqual(self.api.deleted, ["lora-job-a-0-trainer"])

  def test_automodel_jobs_share_a_runtime_apart_from_pytorch(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel"}
    s = self.store_with("job-am-1", meta)
    s.kv_store["open_rl:model_meta:job-am-2"] = json.dumps({**meta, "lora_config": {"rank": 32}})
    s.kv_store["open_rl:model_meta:job-pt"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-am-1", "trainer")
      self.manager.ensure("job-am-1", "sampler")
      self.manager.ensure("job-am-2", "trainer")
      self.manager.ensure("job-pt", "trainer")
      self.manager.release("job-am-1")

    # The second Automodel job's create found the first one's trainer, so only
    # three workloads exist.
    am_trainer, am_sampler, pt_trainer = self.api.created
    self.assertEqual(am_trainer["metadata"]["name"], "lora-automodel-qwen-qwen3-0-6b-0-trainer")
    self.assertEqual(pt_trainer["metadata"]["name"], "lora-qwen-qwen3-0-6b-0-trainer")
    runtime = am_trainer["spec"]["modelID"]
    self.assertEqual(am_sampler["spec"]["ownerID"], am_trainer["spec"]["ownerID"])
    t_container = am_trainer["spec"]["template"]["spec"]["containers"][0]
    s_container = am_sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["image"], "am:1")
    self.assertEqual(t_container["command"], ["python", "-u", "-m", "server.training_requests_processor"])
    self.assertEqual(t_container["args"], ["--model-id", runtime, "--active-tenant-set-id", f"{runtime}-1"])
    self.assertEqual({e["name"]: e.get("value") for e in t_container["env"]}["OPEN_RL_TRAINER_BACKEND"], "automodel")
    self.assertNotEqual(s_container["image"], "am:1")
    self.assertEqual({e["name"]: e.get("value") for e in s_container["env"]}["OPEN_RL_MODEL_ID"], runtime)
    # Like any shared LoRA runtime, deleting one job leaves it up.
    self.assertEqual(self.api.deleted, [])

  def test_an_exclusive_automodel_job_gets_its_own_automodel_trainer(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "exclusive": True}
    s = self.store_with("job-am", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-am", "trainer")
      self.manager.release("job-am")

    (trainer,) = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-job-am-0-trainer")
    self.assertEqual(trainer["spec"]["template"]["spec"]["containers"][0]["image"], "am:1")
    self.assertEqual(self.api.deleted, ["lora-job-am-0-trainer"])

  def test_a_multi_gpu_automodel_trainer_is_one_torchrun_group(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "trainer_gpus": 4}
    s = self.store_with("job-dp", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-dp", "trainer")
      self.manager.ensure("job-dp", "sampler")

    trainer, sampler = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-job-dp-0-trainer")
    self.assertTrue(trainer["spec"]["exclusive"])
    one = footprint("Qwen/Qwen3-0.6B", "lora", "trainer")
    self.assertEqual(trainer["spec"]["accelerator"], {"mode": "MultiGPU", "devices": 4, "memory": one.accelerator})
    pod = trainer["spec"]["template"]["spec"]
    container = pod["containers"][0]
    self.assertEqual(container["command"][:6], ["python", "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"])
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(env["OPEN_RL_CONTROL_BACKEND"], "cpu:gloo,cuda:nccl")
    self.assertEqual(env["OPEN_RL_TIME_SLICING"], "off")
    self.assertNotIn("OPEN_RL_AUTOMODEL_CP", env)
    self.assertIn({"name": "dshm", "mountPath": "/dev/shm"}, container["volumeMounts"])
    self.assertEqual(container["resources"]["requests"]["memory"], f"{-(-one.host_request_bytes * 4 // 2**30)}Gi")
    # The sampler is the usual single-GPU one, alone with this job.
    self.assertEqual(sampler["metadata"]["name"], "lora-job-dp-0-sampler")
    self.assertEqual(sampler["spec"]["accelerator"]["mode"], "SingleGPU")
    self.assertTrue(sampler["spec"]["exclusive"])

  def test_a_context_parallel_trainer_gets_its_cp_size(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "trainer_gpus": 4, "trainer_cp": 4}
    s = self.store_with("job-cp", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-cp", "trainer")

    (trainer,) = self.api.created
    env = {e["name"]: e.get("value") for e in trainer["spec"]["template"]["spec"]["containers"][0]["env"]}
    self.assertEqual(env["OPEN_RL_AUTOMODEL_CP"], "4")

  def test_a_job_that_names_an_image_runs_its_trainer_from_it(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "ghcr.io/org/trainer:1"}
    s = self.store_with("job-img", meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-img", "trainer")
      self.manager.ensure("job-img", "sampler")

    trainer, sampler = self.api.created
    self.assertRegex(trainer["metadata"]["name"], r"^lora-image-[0-9a-f]{10}-qwen-qwen3-0-6b-0-trainer$")
    t_container = trainer["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["image"], "ghcr.io/org/trainer:1")
    self.assertEqual(t_container["command"], ["python", "-u", "-m", "server.training_requests_processor"])
    # The image sets its own OPEN_RL_TRAINER_BACKEND.
    self.assertNotIn("OPEN_RL_TRAINER_BACKEND", {e["name"] for e in t_container["env"]})
    self.assertNotEqual(sampler["spec"]["template"]["spec"]["containers"][0]["image"], "ghcr.io/org/trainer:1")

  def test_release_owner_deletes_a_shared_lora_pair_and_nothing_else(self) -> None:
    s = self.store_with("adapter", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    s.kv_store["open_rl:model_meta:other"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("adapter", "trainer")
      self.manager.ensure("adapter", "sampler")
      self.manager.ensure("other", "trainer")
    self.assertEqual(self.manager.release_owner("qwen-qwen2-5-0-5b"), {"Qwen/Qwen2.5-0.5B"})
    self.assertEqual(self.manager.release_owner("qwen-qwen2-5-0-5b"), set())

    self.assertEqual(self.api.deleted, ["lora-qwen-qwen2-5-0-5b-0-sampler", "lora-qwen-qwen2-5-0-5b-0-trainer"])
    self.assertEqual(list(self.api.existing), ["lora-qwen-qwen3-0-6b-0-trainer"])

  def test_release_owner_finds_workloads_by_their_spec_not_their_labels(self) -> None:
    # A workload from before this API server carries only the managed-by label.
    self.api.existing["fft-old-trainer"] = {
      "metadata": {"name": "fft-old-trainer", "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
      "spec": {"ownerID": "old", "modelID": "old"},
    }
    self.assertEqual(self.manager.release_owner("old"), {"old"})
    self.assertEqual(self.api.deleted, ["fft-old-trainer"])

  def test_ensure_waits_for_a_terminating_workload_before_recreating_it(self) -> None:
    s = self.store_with("adapter", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("adapter", "trainer")
      name = self.api.created[0]["metadata"]["name"]
      self.api.deleting.add(name)

      def finalizer_finishes(_seconds: float) -> None:
        self.api.deleting.discard(name)
        self.api.existing.pop(name, None)

      with patch("server.scheduler_worker_manager.time.sleep", side_effect=finalizer_finishes) as sleep:
        self.manager.ensure("adapter", "trainer")

    self.assertEqual(sleep.call_count, 1)
    self.assertEqual(len(self.api.created), 2)

  def test_create_worker_manager_selects_scheduler_mode(self) -> None:
    from server.worker_manager import create_worker_manager

    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_MANAGER": "scheduler"}, clear=False),
      patch("server.scheduler_worker_manager.SchedulerWorkerManager") as manager_cls,
    ):
      create_worker_manager()
      manager_cls.assert_called_once()

  def test_sampler_replicas_are_single_gpu_workloads_released_together(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 3}
    s = self.store_with("job-sr", meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-sr", "trainer")
      self.manager.ensure("job-sr", "sampler")
      self.manager.release("job-sr")

    names = [w["metadata"]["name"] for w in self.api.created]
    self.assertEqual(names, ["lora-job-sr-0-trainer", "lora-job-sr-0-sampler", "lora-job-sr-1-sampler", "lora-job-sr-2-sampler"])
    for sampler in self.api.created[1:]:
      self.assertEqual(sampler["spec"]["accelerator"]["mode"], "SingleGPU")
      self.assertEqual(sampler["spec"]["ownerID"], "job-sr")
    self.assertEqual(sorted(self.api.deleted), sorted(names))


class MixedSamplingSessionTest(unittest.IsolatedAsyncioTestCase):
  async def test_lora_and_fft_sessions_launch_their_own_sampler_types(self) -> None:
    store = InMemoryStore()
    state = InMemoryStateStore()
    for model_id, kind in (("lora-a", "lora"), ("lora-b", "lora"), ("fft-a", "full")):
      await state.set_value(f"open_rl:model_meta:{model_id}", json.dumps({"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": kind}))
    api = FakeCustomObjectsApi()
    with (
      patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}),
      patch("server.worker_manager.get_state_store", return_value=state),
      patch.object(api_server, "store", store),
      patch.object(api_server, "state", state),
      patch.object(api_server, "get_store", return_value=store),
      patch.object(api_server, "worker_manager", SchedulerWorkerManager(custom_api=api)),
    ):
      async with asgi_client() as client:
        for model_id in ("lora-a", "fft-a", "lora-b"):
          await post_json(client, "create_sampling_session", {"model_path": f"tinker://{model_id}/sampler_weights/checkpoint"})

    self.assertEqual(len(api.created), 2, "LoRA sessions should reuse one sampler while FFT gets its own")
    lora, fft = api.created
    self.assertEqual(lora["spec"]["trainingKind"], "lora")
    self.assertEqual(lora["spec"]["template"]["spec"]["containers"][0]["command"][-1], "server.vllm_sampler")
    self.assertEqual(fft["spec"]["trainingKind"], "fft")
    self.assertEqual(fft["metadata"]["name"], "fft-fft-a-sampler")


if __name__ == "__main__":
  unittest.main()
