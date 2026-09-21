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
    return {"metadata": metadata}

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
    self.assertEqual(s_container["command"][-1], "server.lora_sampler")
    self.assertIn("--active-tenant-set-id", t_container["args"])

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
    self.assertEqual(lora["spec"]["template"]["spec"]["containers"][0]["command"][-1], "server.lora_sampler")
    self.assertEqual(fft["spec"]["trainingKind"], "fft")
    self.assertEqual(fft["metadata"]["name"], "fft-fft-a-sampler")


if __name__ == "__main__":
  unittest.main()
