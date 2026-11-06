import asyncio
import os
import unittest
from collections import defaultdict
from unittest.mock import patch

from server import api_server
from server.model_metadata import get_model_metadata
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import owner_id, owner_of
from tests.api_client import asgi_client, post_json
from tests.test_session_lifecycle import RuntimeManager


class YieldingStateStore(InMemoryStateStore):
  """Every write gives the event loop a turn before it lands, as a write to Redis does."""

  async def set_value(self, key, value, ttl_seconds=None):
    await asyncio.sleep(0)
    await super().set_value(key, value, ttl_seconds)


class LoraRuntimeSlotsTest(unittest.IsolatedAsyncioTestCase):
  """OPEN_RL_LORA_JOBS_PER_RUNTIME caps the LoRA jobs on one runtime; the next job starts another."""

  def setUp(self):
    self.store = InMemoryStore()
    self.state = YieldingStateStore()
    self.manager = RuntimeManager()
    self.registry = SessionRegistry(self.state)
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "get_store", return_value=self.store))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))
    self.enterContext(patch.object(api_server, "session_registry", self.registry))
    self.enterContext(patch.object(api_server, "worker_manager", self.manager))
    # Pools keep requests in Redis, which this test has none of.
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "torch", "OPEN_RL_ENABLE_FFT": "true", "OPEN_RL_LORA_JOBS_PER_RUNTIME": "2"}))
    self.enterContext(patch.object(api_server, "owner_locks", defaultdict(asyncio.Lock)))
    self.enterContext(patch.object(api_server, "slot_locks", defaultdict(asyncio.Lock)))

  async def asyncSetUp(self):
    self.client = await self.enterAsyncContext(asgi_client())
    self.session = (await self.post("create_session", {}))["session_id"]

  async def post(self, path, body):
    return await post_json(self.client, path, body)

  async def create(self, session_id=None) -> str:
    return (await self.post("create_model", {"base_model": "test-base", "session_id": session_id or self.session}))["request_id"]

  async def expire(self, session_id):
    # What the store does on its own once the heartbeats stop.
    await self.state.delete_values(f"open_rl:session:{session_id}")

  async def runtime(self, model_id) -> str:
    return (await get_model_metadata(self.state, model_id)).runtime(model_id)

  async def test_a_full_runtime_sends_the_next_job_to_a_new_one(self):
    jobs = [await self.create() for _ in range(3)]

    self.assertEqual([await self.runtime(job) for job in jobs], ["test-base", "test-base", "test-base#slot1"])
    self.assertEqual([owner_of(job) for job in jobs], ["test-base", "test-base", owner_id("test-base#slot1")])
    trainers = [model_id for model_id, role in self.manager.ensured if role == "trainer"]
    self.assertEqual(trainers, jobs)
    # Each runtime's trainer reads its own tenant set.
    self.assertEqual([r["model_id"] for r in await self.store.get_requests(active_set_id="test-base#slot1-1")], [jobs[2]])

  async def test_a_deleted_job_frees_its_place(self):
    first, second = await self.create(), await self.create()
    await self.post("delete_model", {"model_id": first})

    self.assertEqual(await self.runtime(await self.create()), "test-base")
    self.assertEqual(await self.runtime(second), "test-base")

  async def test_a_job_whose_sessions_expired_frees_its_place(self):
    other = (await self.post("create_session", {}))["session_id"]
    await self.create(other)
    await self.create()
    await self.expire(other)

    self.assertEqual(await self.runtime(await self.create()), "test-base")

  async def test_a_job_bound_to_no_session_holds_its_place_until_deleted(self):
    first = (await self.post("create_model", {"base_model": "test-base"}))["request_id"]
    await self.post("create_model", {"base_model": "test-base"})
    self.assertEqual(await self.runtime(await self.create()), "test-base#slot1")

    await self.post("delete_model", {"model_id": first})
    self.assertEqual(await self.runtime(await self.create()), "test-base")

  async def test_without_a_cap_every_job_shares_one_runtime(self):
    with patch.dict(os.environ, {"OPEN_RL_LORA_JOBS_PER_RUNTIME": ""}):
      jobs = [await self.create() for _ in range(3)]

    self.assertEqual({await self.runtime(job) for job in jobs}, {"test-base"})

  async def test_jobs_created_at_once_take_distinct_places(self):
    with patch.dict(os.environ, {"OPEN_RL_LORA_JOBS_PER_RUNTIME": "1"}):
      jobs = await asyncio.gather(self.create(), self.create())

    self.assertEqual(sorted([await self.runtime(job) for job in jobs]), ["test-base", "test-base#slot1"])
    for job in jobs:
      self.assertIn(job, await self.state.set_members(api_server.runtime_jobs_key(await self.runtime(job))))

  async def test_binding_a_session_the_server_has_not_seen_keeps_the_place(self):
    job = (await self.post("create_model", {"base_model": "test-base"}))["request_id"]

    async def prune_once_the_session_is_bound():
      while "sess-cold" not in (await get_model_metadata(self.state, job)).session_ids:
        await asyncio.sleep(0)
      await api_server.prune_runtime_jobs("test-base")

    await asyncio.gather(api_server.bind_session("sess-cold", job), prune_once_the_session_is_bound())

    self.assertTrue(await self.registry.live("sess-cold"))
    self.assertEqual(await self.state.set_members(api_server.runtime_jobs_key("test-base")), {job})

  def test_the_server_refuses_to_start_with_a_cap_that_is_not_a_count(self):
    for value in ("  ", "two", "-1"):
      with patch.dict(os.environ, {"OPEN_RL_LORA_JOBS_PER_RUNTIME": value}), self.assertRaisesRegex(RuntimeError, "OPEN_RL_LORA_JOBS_PER_RUNTIME"):
        api_server.check_lora_jobs_per_runtime()
    for value in ("", "0", "4"):
      with patch.dict(os.environ, {"OPEN_RL_LORA_JOBS_PER_RUNTIME": value}):
        api_server.check_lora_jobs_per_runtime()

  async def test_fft_and_exclusive_jobs_take_no_slot(self):
    for flag in ({"openrl.exclusive": "true"}, {"openrl.fine_tuning_type": "full"}):
      body = {"base_model": "test-base", "session_id": self.session, "user_metadata": flag}
      job = (await self.post("create_model", body))["request_id"]
      self.assertEqual(await self.runtime(job), job)
    self.assertEqual(await self.state.set_members(api_server.runtime_jobs_key("test-base")), set())


if __name__ == "__main__":
  unittest.main()
