import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from server import api_server, sampler_http
from server.sampler_router import RouterBusy, SamplerRouter
from server.scheduler_worker_manager import ROUTER_LABEL, ROUTER_LOOKUP_TIMEOUT, ROUTER_PORT, SAMPLER_SET_LABEL, SchedulerWorkerManager, sampler_set
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.test_scheduler_worker_manager import FakeCustomObjectsApi

ROUTED = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 2, "sampler_router": "llmd"}


def state_with(model_id: str, meta: dict) -> InMemoryStateStore:
  state = InMemoryStateStore()
  state.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
  return state


class RoutedSamplerTemplateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}))
    self.api = FakeCustomObjectsApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api)

  def workloads(self, meta: dict) -> dict[str, dict]:
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", meta)):
      self.manager.ensure("job", "trainer")
      self.manager.ensure("job", "sampler")
    return {body["metadata"]["name"]: body["spec"]["template"] for body in self.api.created}

  def test_the_first_sampler_carries_the_router_for_its_set(self) -> None:
    templates = self.workloads(ROUTED)
    first, second = templates["lora-job-0-sampler"], templates["lora-job-1-sampler"]
    name = sampler_set("job")

    self.assertEqual([c["name"] for c in first["spec"]["containers"]], ["worker", "router-proxy", "router-picker"])
    self.assertEqual(first["metadata"]["labels"], {SAMPLER_SET_LABEL: name, ROUTER_LABEL: name})
    self.assertEqual(first["spec"]["serviceAccountName"], "openrl-llmd-router")
    picker = first["spec"]["containers"][2]
    self.assertIn(f"{SAMPLER_SET_LABEL}={name}", picker["args"])

    self.assertEqual([c["name"] for c in second["spec"]["containers"]], ["worker"])
    self.assertEqual(second["metadata"]["labels"], {SAMPLER_SET_LABEL: name})
    for template in (first, second):
      env = {e["name"]: e.get("value") for e in template["spec"]["containers"][0]["env"]}
      self.assertEqual(env["OPEN_RL_SAMPLER_HTTP_PORT"], "8000")

  def test_trainers_and_unrouted_samplers_are_unchanged(self) -> None:
    templates = self.workloads({**ROUTED, "sampler_router": None})
    for template in templates.values():
      self.assertNotIn("metadata", template)
      self.assertEqual([c["name"] for c in template["spec"]["containers"]], ["worker"])
    self.api.created.clear()
    self.api.existing.clear()
    trainer = self.workloads(ROUTED)["lora-job-0-trainer"]
    self.assertNotIn("metadata", trainer)

  def test_shared_pools_reject_mixed_router_settings_in_either_order(self) -> None:
    for first, second in [(None, "llmd"), ("llmd", None)]:
      with self.subTest(first=first):
        self.api.existing.clear()
        state = state_with("a", {**ROUTED, "exclusive": False, "sampler_router": first})
        state.kv_store["open_rl:model_meta:b"] = json.dumps({**ROUTED, "exclusive": False, "sampler_router": second})
        with patch("server.worker_manager.get_state_store", return_value=state):
          self.manager.ensure("a", "sampler")
          self.manager.ensure("a", "sampler")  # Matching settings still reuse the pool.
          with self.assertRaisesRegex(ValueError, "different sampler_router setting"):
            self.manager.ensure("b", "sampler")


class RouterLookupTest(unittest.TestCase):
  def test_the_running_router_pod_is_found_by_its_set(self) -> None:
    def pod(phase: str, ip: str, deleting: bool = False, ready: bool = True) -> SimpleNamespace:
      return SimpleNamespace(
        status=SimpleNamespace(phase=phase, pod_ip=ip, conditions=[SimpleNamespace(type="Ready", status="True" if ready else "False")]),
        metadata=SimpleNamespace(deletion_timestamp="now" if deleting else None),
      )

    selectors = []

    class Pods:
      def list_namespaced_pod(self, namespace: str, label_selector: str, _request_timeout: float | None = None) -> SimpleNamespace:
        selectors.append((label_selector, _request_timeout))
        return SimpleNamespace(
          items=[pod("Running", "10.0.0.1", deleting=True), pod("Pending", ""), pod("Running", "10.0.0.3", ready=False), pod("Running", "10.0.0.2")]
        )

    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
      manager = SchedulerWorkerManager(custom_api=FakeCustomObjectsApi(), core_api=Pods())
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", ROUTED)):
      self.assertEqual(manager.router_url("job"), f"http://10.0.0.2:{ROUTER_PORT}")
    self.assertEqual(selectors, [(f"{ROUTER_LABEL}={sampler_set('job')}", ROUTER_LOOKUP_TIMEOUT)])


class EchoSampler:
  """Stands in for the engine call only; everything around it is the real app."""

  engine = SimpleNamespace(errored=False)

  def __init__(self) -> None:
    self.requests: list[dict] = []

  async def generate(self, request: dict) -> dict:
    self.requests.append(request)
    return {"sequences": [{"tokens": request["prompt_token_ids"][-2:], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}


class Routers:
  def __init__(self, url: str | None) -> None:
    self.url = url

  def router_url(self, model_id: str) -> str | None:
    return self.url


class FailingRouters:
  def router_url(self, model_id: str) -> str | None:
    raise TimeoutError("read timed out")


class GatewayRoutingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.sampler = EchoSampler()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.state = state_with("job", ROUTED)
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    transport = httpx.ASGITransport(app=sampler_http.http_app(self.sampler))
    self.router = SamplerRouter(self.state, client=httpx.AsyncClient(transport=transport))
    self.enterContext(patch.object(api_server.sampler_router, "SamplerRouter", return_value=self.router))
    self.client = self.enterContext(TestClient(api_server.app))

  def sample(self) -> dict:
    body = {"model_id": "tinker://job/sampler_weights/000003", "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}}
    promise = self.client.post("/api/v1/asample", json=body).json()
    return self.client.post("/api/v1/retrieve_future", json={"request_id": promise["request_id"]}).json()

  def test_a_routed_model_samples_through_its_router(self) -> None:
    with patch.object(api_server, "worker_manager", Routers("http://router")):
      result = self.sample()
    self.assertEqual(result["sequences"][0]["tokens"], [2, 3])
    self.assertEqual(self.sampler.requests[0]["lora_id"], "tinker://job/sampler_weights/000003")
    self.assertEqual(asyncio.run(self.store.get_sampling_requests_for_model("job")), [])

  def test_a_refused_request_fails_instead_of_returning_the_error_as_a_sample(self) -> None:
    refusing = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429, text="too many requests")))
    with patch.object(api_server, "worker_manager", Routers("http://router")), patch.object(self.router, "client", refusing):
      body = {"model_id": "job", "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}}
      promise = self.client.post("/api/v1/asample", json=body).json()
      response = self.client.post("/api/v1/retrieve_future", json={"request_id": promise["request_id"]})
    self.assertEqual(response.status_code, 400)
    self.assertIn("llm-d router returned 429", response.json()["error_message"])

  def test_an_unreachable_router_fails_without_queuing(self) -> None:
    self.assert_failed_through(Routers(None))

  def test_a_failed_router_lookup_fails_without_queuing(self) -> None:
    self.assert_failed_through(FailingRouters())

  def assert_failed_through(self, routers) -> None:
    with patch.object(api_server, "worker_manager", routers):
      result = self.sample()
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertEqual(self.store.sampling_queues, {})

  def test_missing_worker_manager_refuses_routing_instead_of_queuing(self) -> None:
    with patch.object(api_server, "worker_manager", None):
      response = self.client.post("/api/v1/asample", json={"model_id": "job"})
    self.assertEqual(response.status_code, 503)
    self.assertEqual(self.store.sampling_queues, {})

  def test_launch_errors_reach_the_client(self) -> None:
    class IncompatiblePool:
      def ensure(self, model_id, role):
        raise ValueError("different sampler_router setting")

    with patch.object(api_server, "worker_manager", IncompatiblePool()):
      response = self.client.post("/api/v1/save_weights_for_sampler", json={"model_id": "job"})
    self.assertEqual(response.status_code, 503)
    self.assertIn("different sampler_router setting", response.json()["error"])


class RoutedRequestLifecycleTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.state = InMemoryStateStore()
    self.manager = Routers("http://router")
    self.request = {"request_id": "routed-test", "model_id": "job", "lora_id": None, "prompt_token_ids": [1], "max_tokens": 2, "num_samples": 1}
    self.success = {"type": "sample", "sequences": [{"tokens": [2], "logprobs": [-0.5], "stop_reason": "length"}]}

  def make_router(self, handler, **kwargs) -> SamplerRouter:
    router = SamplerRouter(self.state, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)
    self.addAsyncCleanup(router.close)
    return router

  async def test_bad_responses_are_terminal_failures(self) -> None:
    responses = [
      httpx.Response(200, text="not-json"),
      httpx.Response(200, json=[]),
      httpx.Response(200, json={"type": "sample", "sequences": [{}]}),
      httpx.Response(200, json={"type": "sample", "sequences": []}),
      httpx.Response(200, json={"type": "RequestFailedResponse"}),
      httpx.Response(503, text="unavailable"),
    ]
    for response in responses:
      with self.subTest(response=response):
        router = self.make_router(lambda request, response=response: response)
        await router.submit(self.manager.router_url, "job", self.request)
        result = await router.result("routed-test")
        self.assertEqual(result["type"], "RequestFailedResponse")

  async def test_lost_response_is_never_replayed(self) -> None:
    calls = []

    def lose_response(request):
      calls.append(request)
      raise httpx.ReadError("connection closed after generation")

    router = self.make_router(lose_response)
    await router.submit(self.manager.router_url, "job", self.request)
    result = await router.result("routed-test")
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertEqual(len(calls), 1)
    self.assertEqual(await router.result("routed-test"), result)

  async def test_poll_timeout_does_not_cancel_work_and_success_survives_restart(self) -> None:
    gate = asyncio.Event()

    async def generate(request):
      await gate.wait()
      return httpx.Response(200, json=self.success)

    router = self.make_router(generate)
    await router.submit(self.manager.router_url, "job", self.request)
    self.assertEqual(await router.result("routed-test", timeout=0), {"type": "try_again"})
    gate.set()
    self.assertEqual(await router.result("routed-test"), self.success)
    restarted = self.make_router(generate)
    self.assertEqual(await restarted.result("routed-test"), self.success)

  async def test_restart_and_expired_receipts_do_not_poll_forever(self) -> None:
    router = self.make_router(lambda request: self.fail("must not dispatch"))
    await router.write("routed-interrupted", {"type": "try_again"})
    self.assertIn("interrupted", (await router.result("routed-interrupted"))["error_message"])
    self.assertIn("expired", (await router.result("routed-missing"))["error_message"])

  async def test_deadline_and_shutdown_cancel_pending_requests(self) -> None:
    async def stall(request):
      await asyncio.Event().wait()

    for shutdown in [False, True]:
      with self.subTest(shutdown=shutdown):
        router = self.make_router(stall, timeout=0.02 if not shutdown else 30)
        await router.submit(self.manager.router_url, "job", self.request)
        if shutdown:
          await asyncio.sleep(0)  # Enter dispatch before cancelling.
          await router.close()
        result = await asyncio.wait_for(router.result("routed-test"), timeout=1)
        self.assertEqual(result["type"], "RequestFailedResponse")
        self.assertIn("shutdown" if shutdown else "TimeoutError", result["error_message"])
        self.assertFalse(router.tasks)

  async def test_capacity_is_checked_before_accepting_work(self) -> None:
    async def stall(request):
      await asyncio.Event().wait()

    router = self.make_router(stall, capacity=1)
    await router.submit(self.manager.router_url, "job", self.request)
    with self.assertRaises(RouterBusy):
      await router.submit(self.manager.router_url, "job", {**self.request, "request_id": "routed-rejected"})
    self.assertIsNone(await self.state.get_value("open_rl:routed_sample:routed-rejected"))

  async def test_result_write_failure_does_not_leave_a_pending_future_forever(self) -> None:
    router = self.make_router(lambda request: httpx.Response(200, json=self.success))
    await router.submit(self.manager.router_url, "job", self.request)
    with patch.object(self.state, "set_value", side_effect=OSError("Redis unavailable")), self.assertLogs("server.sampler_router", level="ERROR"):
      await asyncio.wait(list(router.tasks.values()))
    result = await router.result("routed-test")
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertIn("before its result was saved", result["error_message"])

  async def test_http_disconnect_cancels_generation(self) -> None:
    cancelled = asyncio.Event()

    class WaitingSampler(EchoSampler):
      async def generate(self, request):
        try:
          await asyncio.Event().wait()
        finally:
          cancelled.set()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sampler_http.http_app(WaitingSampler()))) as client:
      with patch.object(sampler_http, "ENGINE_POLL_SECONDS", 0.01), patch.object(sampler_http.Request, "is_disconnected", return_value=True):
        response = await asyncio.wait_for(client.post("http://sampler/v1/completions", json={"openrl": self.request}), timeout=1)
    self.assertIn("disconnected", response.json()["error_message"])
    self.assertTrue(cancelled.is_set())

  async def test_engine_death_during_http_generation_resolves_and_cancels_work(self) -> None:
    cancelled = asyncio.Event()

    class DyingSampler:
      def __init__(self):
        self.engine = SimpleNamespace(errored=False)

      async def generate(self, request):
        try:
          self.engine.errored = True
          await asyncio.Event().wait()
        finally:
          cancelled.set()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sampler_http.http_app(DyingSampler()))) as client:
      with patch.object(sampler_http, "ENGINE_POLL_SECONDS", 0.01):
        response = await asyncio.wait_for(client.post("http://sampler/v1/completions", json={"openrl": self.request}), timeout=1)
    self.assertIn("engine is dead", response.json()["error_message"])
    self.assertTrue(cancelled.is_set())


class SamplerRouterSettingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}))
    # No lifespan: with FFT enabled it would build a worker manager of its own.
    self.client = TestClient(api_server.app)

  def create(self, metadata: dict) -> httpx.Response:
    return self.client.post("/api/v1/create_model", json={"base_model": "m", "user_metadata": metadata})

  def test_full_fine_tuning_is_refused(self) -> None:
    with patch.object(api_server, "worker_manager", Routers("http://router")):
      response = self.create({"openrl.sampler_router": "llmd", "openrl.fine_tuning_type": "full"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("supports LoRA only", response.json()["error"])

  def test_local_workers_are_refused(self) -> None:
    with patch.object(api_server, "worker_manager", LocalWorkerManager.__new__(LocalWorkerManager)):
      response = self.create({"openrl.sampler_router": "llmd"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("launches workers as pods", response.json()["error"])


if __name__ == "__main__":
  unittest.main()
