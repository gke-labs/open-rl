"""The read-only views of design 016: capabilities and the model view.

The ASGI client runs the real routes; the worker manager is the fake from the
session lifecycle tests, so every entry here reflects what a client behind a
port-forward would see, without a cluster."""

import asyncio
import os
import unittest
from collections import defaultdict
from unittest.mock import patch

from kubernetes import client

from server import api_server
from server.scheduler_worker_manager import SchedulerWorkerManager, worker_state
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import WORKER_STATES
from tests.api_client import asgi_client, post_json
from tests.test_scheduler_worker_manager import FakeCustomObjectsApi
from tests.test_session_lifecycle import RuntimeManager


class StatusViewsTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.store = InMemoryStore()
    self.state = InMemoryStateStore()
    self.manager = RuntimeManager()
    self.registry = SessionRegistry(self.state)
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "get_store", return_value=self.store))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))
    self.enterContext(patch.object(api_server, "session_registry", self.registry))
    self.enterContext(patch.object(api_server, "worker_manager", self.manager))
    self.enterContext(patch.object(api_server, "owner_locks", defaultdict(asyncio.Lock)))
    self.enterContext(patch.object(api_server, "OPEN_RL_VERSION", "v9.9.9"))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true", "VLLM_MAX_MODEL_LEN": "8192"}))

  async def asyncSetUp(self):
    self.client = await self.enterAsyncContext(asgi_client())

  async def get(self, path: str):
    return await self.client.get(f"/api/v1/{path}")

  async def create_model(self, user_metadata: dict | None = None, trainer_ready: bool = True) -> tuple[str, str]:
    session = (await post_json(self.client, "create_session", {"user_metadata": user_metadata or {}}))["session_id"]
    model_id = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    if trainer_ready:
      # The trainer process announces itself once its model is loaded, under
      # its runtime id: the base model for a shared LoRA trainer, the model
      # id for a job's own trainer.
      from server.worker_manager import runtime_of

      await self.state.set_value(f"open_rl:trainer_ready:{runtime_of(model_id)[1]}", "1")
    return session, model_id

  async def test_capabilities_describe_the_deployment_without_breaking_the_sdk_fields(self):
    body = (await self.get("get_server_capabilities")).json()
    # The fields the SDK reads are untouched.
    self.assertIn("supported_models", body)
    self.assertIn("single_process", body)
    info = body["openrl"]
    self.assertEqual(info["version"], "v9.9.9")
    self.assertEqual(info["worker_launcher"], "process")
    self.assertTrue(info["fft_enabled"])
    self.assertTrue(info["models"]["any_huggingface_model"])
    self.assertEqual(info["limits"]["vllm_max_model_len"], 8192)
    self.assertEqual(info["limits"]["session_ttl_seconds"], 120)
    self.assertEqual(info["skill_url"], "https://github.com/gke-labs/open-rl/releases/download/v9.9.9/openrl-skill.tar.gz")
    # Deployment facts only; nothing about other clients' models or the fleet.
    self.assertNotIn("endpoints", info)
    # The settings are described once, in the OpenAPI document; capabilities point there.
    self.assertNotIn("settings", info)
    self.assertEqual(info["settings_schema_url"], "/openapi.json#/components/schemas/OpenRLSettings")

  async def test_the_openapi_document_describes_the_settings(self):
    spec = (await self.client.get("/openapi.json")).json()
    schema = spec["components"]["schemas"]["OpenRLSettings"]
    props = schema["properties"]
    self.assertEqual(set(props), {f"openrl.{name}" for name in api_server.Settings.model_fields})
    kind = props["openrl.fine_tuning_type"]
    self.assertEqual([o for o in kind["anyOf"] if o.get("type") == "string"][0]["enum"], ["lora", "full"])
    self.assertIsNone(kind["default"])
    replicas = props["openrl.sampler_replicas"]
    self.assertEqual((replicas["type"], replicas["default"], replicas["minimum"], replicas["maximum"]), ("integer", 1, 1, 8))
    self.assertEqual(props["openrl.exclusive"]["type"], "boolean")
    self.assertFalse(schema["additionalProperties"], "unknown openrl. keys are refused")
    for name, prop in props.items():
      self.assertTrue(prop.get("description"), f"{name} needs a description; agents read it from here")
    # The request models say where the settings go.
    field = spec["components"]["schemas"]["CreateSessionRequest"]["properties"]["user_metadata"]
    self.assertIn("OpenRLSettings", field["description"])

  async def test_a_dev_build_has_no_skill_url(self):
    with patch.object(api_server, "OPEN_RL_VERSION", "sha-abc1234"):
      info = (await self.get("get_server_capabilities")).json()["openrl"]
    self.assertEqual(info["version"], "sha-abc1234")
    self.assertIsNone(info["skill_url"])

  async def test_one_model_shows_settings_workers_and_a_summary(self):
    _, model_id = await self.create_model({"openrl.fine_tuning_type": "full"})
    body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertEqual(body["model_id"], model_id)
    self.assertEqual((body["base_model"], body["fine_tuning_type"], body["status"]), ("test-base", "full", "active"))
    self.assertEqual(body["settings"]["openrl.fine_tuning_type"], "full")
    self.assertEqual(body["settings"]["openrl.weight_sync"], "delta")
    self.assertFalse(body["shares_workers"])
    self.assertTrue(body["session_live"])
    (worker,) = body["workers"]
    self.assertEqual((worker["role"], worker["state"], worker["node"]), ("trainer", "ready", "node-a"))
    self.assertEqual(worker["summary"], f"trainer ready: fake-{model_id}-trainer on node-a, 5s")
    self.assertEqual(body["summary"], worker["summary"])

  async def test_the_view_exposes_only_the_models_own_facts(self):
    _, model_id = await self.create_model()
    body = (await self.get(f"openrl/models/{model_id}")).json()
    # No owner key, no session ids, no pod names or pids: nothing another tenant could use.
    fields = {"model_id", "base_model", "fine_tuning_type", "status", "created_at", "shares_workers", "session_live", "settings", "workers"}
    self.assertEqual(set(body), fields | {"workers_error", "summary"})
    self.assertIsNone(body["workers_error"])
    self.assertEqual(set(body["workers"][0]), {"role", "state", "detail", "workload", "node", "age_seconds", "summary"})

  async def test_session_live_is_the_models_own_session_not_the_shared_runtimes(self):
    # Two LoRA jobs on one base model share workers (one owner) but have their own sessions.
    session_a, model_a = await self.create_model()
    session_b, model_b = await self.create_model()
    await self.state.delete_values(f"open_rl:session:{session_a}")
    a = (await self.get(f"openrl/models/{model_a}")).json()
    b = (await self.get(f"openrl/models/{model_b}")).json()
    self.assertFalse(a["session_live"])
    self.assertTrue(b["session_live"])
    # The binding lives on the model's metadata, nowhere else.
    from server.model_metadata import get_model_metadata

    self.assertEqual((await get_model_metadata(self.state, model_a)).session_ids, [session_a])
    self.assertTrue(a["shares_workers"])
    self.assertIn("workers stay up while other jobs use them", a["summary"])
    self.assertNotIn("expired", b["summary"])
    # Once the last session on the shared workers is gone too, the reaper is what happens next.
    await self.state.delete_values(f"open_rl:session:{session_b}")
    a = (await self.get(f"openrl/models/{model_a}")).json()
    self.assertIn("the reaper releases unused workers", a["summary"])

  async def test_a_launcher_error_degrades_the_view_instead_of_failing_it(self):
    _, model_id = await self.create_model()

    def broken(owner):
      raise RuntimeError("kube-apiserver unreachable")

    with patch.object(self.manager, "inspect_owner", broken):
      response = await self.get(f"openrl/models/{model_id}")
    self.assertEqual(response.status_code, 200)
    body = response.json()
    self.assertEqual(body["workers"], [])
    self.assertIn("kube-apiserver unreachable", body["workers_error"])
    self.assertIn("worker state unavailable", body["summary"])
    self.assertTrue(body["session_live"])

  async def test_a_trainer_is_starting_until_it_announces_itself(self):
    _, model_id = await self.create_model(trainer_ready=False)
    (trainer,) = (await self.get(f"openrl/models/{model_id}")).json()["workers"]
    self.assertEqual(trainer["state"], "starting")
    self.assertTrue(trainer["detail"].endswith("loading the model"))
    await self.state.set_value("open_rl:trainer_ready:test-base", "1")
    (trainer,) = (await self.get(f"openrl/models/{model_id}")).json()["workers"]
    self.assertEqual(trainer["state"], "ready")

  async def test_a_released_model_says_so_in_its_summary(self):
    session, model_id = await self.create_model({"openrl.fine_tuning_type": "full"})
    await self.state.delete_values(f"open_rl:session:{session}")
    self.manager.ensured.clear()  # the reaper removed the workers
    body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertFalse(body["session_live"])
    self.assertEqual(body["workers"], [])
    self.assertIn("released", body["summary"])

  async def test_an_unknown_model_is_a_404_with_the_shared_error_shape(self):
    response = await self.get("openrl/models/nope")
    self.assertEqual(response.status_code, 404)
    self.assertIn("Unknown model", response.json()["error"])

  async def test_the_view_without_a_worker_manager_says_why_there_are_no_workers(self):
    _, model_id = await self.create_model()
    with patch.object(api_server, "worker_manager", None):
      body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertEqual(body["workers"], [])
    self.assertIn("standing deployments", body["summary"])
    with patch.object(api_server, "worker_manager", None), patch.object(api_server, "is_single_process_mode", return_value=True):
      body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertIn("inside the API server", body["summary"])


class WorkerStateTest(unittest.TestCase):
  """The mapping from Workload status onto the five client states."""

  def test_every_outcome_is_one_of_the_documented_states(self):
    cases = [
      (dict(phase="Pending", reason="NoCapacity: needs 40Gi", node=None), "waiting_for_placement"),
      (dict(phase="Placing", reason="PodCreated", node=None), "waiting_for_placement"),  # claim cut, allocation not yet observed
      (dict(phase="Placing", reason="PodCreated", node="node-a"), "starting"),
      (dict(phase="Running", reason="", node="node-a"), "ready"),
      (dict(phase="Failed", reason="PodFailed", node="node-a"), "failed"),
      (dict(phase="Failed", reason="NoCapacity: no pool accepts role trainer", node=None), "failed"),
      (dict(phase="Succeeded", reason="", node="node-a"), "stopped"),
      (dict(phase="Running", reason="", node="node-a", deleting=True), "stopped"),
    ]
    for facts, expected in cases:
      with self.subTest(**facts):
        state, _ = worker_state(facts["phase"], facts["reason"], facts["node"], facts.get("deleting", False))
        self.assertEqual(state, expected)
        self.assertIn(state, WORKER_STATES)

  def test_detail_carries_the_raw_facts_for_a_human(self):
    _, detail = worker_state("Placing", "PodCreated", "gke-l4-lvw8", False)
    self.assertEqual(detail, "Placing (PodCreated) on gke-l4-lvw8")
    _, detail = worker_state("Pending", "WaitingForCapacity", None, False)
    self.assertEqual(detail, "Pending (WaitingForCapacity)")
    _, detail = worker_state("Running", "", "node-a", True)
    self.assertEqual(detail, "being released; Running on node-a")


def workload(name: str, role: str, owner: str, model_id: str, status: dict, created="2026-10-12T00:00:00Z"):
  metadata = {"name": name, "creationTimestamp": created, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}}
  return {"metadata": metadata, "spec": {"role": role, "ownerID": owner, "modelID": model_id}, "status": status}


class FakeCoreApi:
  def __init__(self, *pods):
    self.pods, self.selectors = list(pods), []

  def list_namespaced_pod(self, namespace: str, label_selector: str = "", _request_timeout=None):
    self.selectors.append(label_selector)
    return client.V1PodList(items=self.pods)


def sampler_pod(name: str, ready: bool, waiting: str | None = None, restarts: int = 0):
  if waiting:
    state = client.V1ContainerState(waiting=client.V1ContainerStateWaiting(reason=waiting))
  else:
    state = client.V1ContainerState(running=client.V1ContainerStateRunning())
  container = client.V1ContainerStatus(name="worker", image="server", image_id="", ready=ready, restart_count=restarts, state=state)
  condition = client.V1PodCondition(type="Ready", status="True" if ready else "False")
  status = client.V1PodStatus(phase="Running", conditions=[condition], container_statuses=[container])
  return client.V1Pod(metadata=client.V1ObjectMeta(name=name), status=status)


class SchedulerViewTest(unittest.TestCase):
  """The cluster side of the model view, from Workloads and a running sampler's pod readiness."""

  def setUp(self):
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://test"}))
    self.custom = FakeCustomObjectsApi()
    self.custom.existing = {
      "lora-base-0-trainer": workload(
        "lora-base-0-trainer", "trainer", "base", "base", {"phase": "Running", "podName": "orw-t", "nodeName": "node-a", "reason": "PodCreated"}
      ),
      "lora-base-0-sampler": workload(
        "lora-base-0-sampler", "sampler", "base", "base", {"phase": "Placing", "podName": "orw-s", "nodeName": "node-b"}
      ),
      "fft-job-trainer": workload("fft-job-trainer", "trainer", "job", "job", {"phase": "Pending", "reason": "NoCapacity: needs 40Gi"}),
    }
    self.manager = SchedulerWorkerManager(custom_api=self.custom)

  def test_workers_come_from_workload_status_alone(self):
    by_name = {w["workload"]: w for w in self.manager.inspect_owner("base")}
    trainer = by_name["lora-base-0-trainer"]
    self.assertEqual((trainer["role"], trainer["state"], trainer["node"]), ("trainer", "ready", "node-a"))
    self.assertEqual(trainer["detail"], "Running (PodCreated) on node-a")
    self.assertGreater(trainer["age_seconds"], 0)
    sampler = by_name["lora-base-0-sampler"]
    self.assertEqual((sampler["state"], sampler["detail"]), ("starting", "Placing on node-b"))
    (pending,) = self.manager.inspect_owner("job")
    self.assertEqual((pending["state"], pending["node"]), ("waiting_for_placement", None))
    self.assertEqual(pending["detail"], "Pending (NoCapacity: needs 40Gi)")
    self.assertEqual(self.manager.inspect_owner("nobody"), [])
    # No pod names or other pod facts leak into the entry.
    self.assertNotIn("pod", trainer)

  def test_the_view_is_cached_briefly(self):
    first = self.manager.inspect_owner("base")
    self.custom.existing.clear()
    self.assertEqual(self.manager.inspect_owner("base"), first)
    self.manager.view_cache_seconds = 0
    self.assertEqual(self.manager.inspect_owner("base"), [])

  def test_a_running_sampler_is_ready_only_once_its_pod_is_ready(self):
    # The Workload reads Running while vLLM loads and while the container crash-loops.
    running = {"phase": "Running", "podName": "orw-s", "nodeName": "node-b"}
    self.custom.existing["lora-base-0-sampler"] = workload("lora-base-0-sampler", "sampler", "base", "base", running)
    crash_looping = sampler_pod("orw-s", ready=False, waiting="CrashLoopBackOff", restarts=4)
    cases = [
      ([sampler_pod("orw-s", ready=False)], "starting", "Running on node-b, loading the model"),
      ([crash_looping], "starting", "Running on node-b, CrashLoopBackOff, restart count 4"),
      ([sampler_pod("orw-s", ready=True)], "ready", "Running on node-b"),
      ([], "starting", "Running on node-b, loading the model"),
    ]
    for pods, state, detail in cases:
      with self.subTest(detail=detail):
        core = FakeCoreApi(*pods)
        manager = SchedulerWorkerManager(custom_api=self.custom, core_api=core)
        by_role = {w["role"]: w for w in manager.inspect_owner("base")}
        self.assertEqual((by_role["sampler"]["state"], by_role["sampler"]["detail"]), (state, detail))
        # One pod list per view, and the trainer is still read from its Workload alone.
        self.assertEqual(core.selectors, ["app.kubernetes.io/component=sampler"])
        self.assertEqual(by_role["trainer"]["state"], "ready")


if __name__ == "__main__":
  unittest.main()
