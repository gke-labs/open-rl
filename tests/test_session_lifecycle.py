import asyncio
import os
import unittest
from collections import defaultdict
from unittest.mock import patch

from server import api_server
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore
from tests.api_client import asgi_client, post_json


class RuntimeManager:
  def __init__(self):
    self.ensured = []
    self.released = []
    self.released_models = []

  def ensure(self, model_id, role):
    self.ensured.append((model_id, role))

  def release(self, model_id):
    self.released_models.append(model_id)

  def release_owner(self, owner):
    self.released.append(owner)
    return {"test-base"} if owner == "test-base" else set()


class SessionLifecycleTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.store = InMemoryStore()
    self.state = InMemoryStateStore()
    self.enterContext(patch.object(api_server, "state", self.state))
    self.manager = RuntimeManager()
    self.registry = SessionRegistry(self.state)
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "get_store", return_value=self.store))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))
    self.enterContext(patch.object(api_server, "session_registry", self.registry))
    self.enterContext(patch.object(api_server, "worker_manager", self.manager))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true"}))
    self.enterContext(patch.object(api_server, "owner_locks", defaultdict(asyncio.Lock)))

  async def asyncSetUp(self):
    self.client = await self.enterAsyncContext(asgi_client())

  async def post(self, path, body, **kwargs):
    return await post_json(self.client, path, body, **kwargs)

  async def expire(self, session_id):
    # What the store does on its own once the heartbeats stop.
    await self.state.delete_values(f"open_rl:session:{session_id}")

  async def reap(self):
    for owner in await self.registry.owners():
      await api_server.reap_owner(owner)

  async def test_shared_lora_owner_outlives_the_session_that_created_it(self):
    training = (await self.post("create_session", {}))["session_id"]
    adapter = (await self.post("create_model", {"base_model": "test-base", "session_id": training}))["request_id"]
    fft_headers = {"x-open-rl-fine-tuning-type": "full"}
    fft_model = (await self.post("create_model", {"base_model": "fft-base", "session_id": training}, headers=fft_headers))["request_id"]
    await self.state.set_value("open_rl:sampler_ready:test-base", "1")
    sampling = (await self.post("create_session", {}))["session_id"]
    await self.post("create_sampling_session", {"model_path": f"tinker://{adapter}/sampler_weights/test", "session_id": sampling})
    self.assertEqual(self.manager.ensured, [(adapter, "trainer"), (fft_model, "trainer"), (adapter, "sampler")])

    await self.expire(training)
    await self.reap()
    self.assertEqual(self.manager.released, [fft_model.lower()])

    await self.expire(sampling)
    await self.reap()
    self.assertEqual(self.manager.released, [fft_model.lower(), "test-base"])
    self.assertIsNone(await self.state.get_value("open_rl:sampler_ready:test-base"))
    self.assertEqual(await self.registry.owners(), [])

  async def test_an_exclusive_lora_model_has_its_own_runtime(self):
    session = (await self.post("create_session", {}))["session_id"]
    shared = (await self.post("create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    exclusive_metadata = {"openrl.exclusive": "true"}
    body = {"base_model": "test-base", "session_id": session, "user_metadata": exclusive_metadata}
    exclusive = (await self.post("create_model", body))["request_id"]

    self.assertEqual(await self.registry.owners(), sorted(["test-base", exclusive.lower()]))
    self.assertEqual([r["model_id"] for r in await self.store.get_requests(active_set_id="test-base-1")], [shared])
    self.assertEqual([r["model_id"] for r in await self.store.get_requests(active_set_id=f"{exclusive}-1")], [exclusive])

    await self.post("delete_model", {"model_id": shared})
    self.assertEqual(self.manager.released_models, [])
    await self.post("delete_model", {"model_id": exclusive})
    self.assertEqual(self.manager.released_models, [exclusive])

  async def test_deleting_a_job_frees_its_adapter_or_its_workers(self):
    session = (await self.post("create_session", {}))["session_id"]
    shared = (await self.post("create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    body = {"base_model": "test-base", "session_id": session, "user_metadata": {"openrl.exclusive": "true"}}
    exclusive = (await self.post("create_model", body))["request_id"]
    await self.store.get_requests(active_set_id="test-base-1")
    await self.store.get_requests(active_set_id=f"{exclusive}-1")

    await self.post("delete_model", {"model_id": shared})
    await self.post("delete_model", {"model_id": exclusive})

    # The shared trainer drops one adapter; the exclusive job's workers shut down.
    self.assertEqual([(r["op"], r["model_id"]) for r in await self.store.get_requests(active_set_id="test-base-1")], [("delete_model", shared)])
    self.assertEqual([r["op"] for r in await self.store.get_requests(active_set_id=f"{exclusive}-1")], ["shutdown_workers"])
    self.assertEqual(self.manager.released_models, [exclusive])

  async def test_an_owner_stays_listed_until_forgotten(self):
    await self.registry.attach("a", "base")
    self.assertTrue(await self.registry.in_use("base"))
    await self.expire("a")
    self.assertFalse(await self.registry.in_use("base"))
    self.assertEqual(await self.registry.owners(), ["base"])
    await self.registry.forget("base")
    self.assertEqual(await self.registry.owners(), [])

  async def test_a_session_attaching_during_teardown_waits_for_it(self):
    await self.registry.attach("a", "test-base")
    await self.expire("a")
    loop = asyncio.get_running_loop()
    slow = asyncio.Event()

    def release_owner(owner):
      self.manager.released.append(owner)
      loop.call_soon_threadsafe(slow.set)
      return {"test-base"}

    self.manager.release_owner = release_owner
    reap = asyncio.create_task(api_server.reap_owner("test-base"))
    await slow.wait()  # the reaper has decided and is mid-delete
    await api_server.bind_session("b", "test-base")
    await reap
    self.assertEqual(self.manager.released, ["test-base"])
    self.assertTrue(await self.registry.in_use("test-base"))
    self.assertEqual(await self.registry.owners(), ["test-base"])

  async def test_a_heartbeat_for_an_unknown_session_opens_it(self):
    await self.registry.heartbeat("after-a-wiped-store")
    self.assertTrue(await self.registry.live("after-a-wiped-store"))
