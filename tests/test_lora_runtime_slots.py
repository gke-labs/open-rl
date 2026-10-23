import asyncio
import os
import unittest
from collections import defaultdict
from unittest.mock import patch

from server import api_server
from server.model_metadata import get_model_metadata
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import owner_of
from tests.api_client import asgi_client, post_json
from tests.test_session_lifecycle import RuntimeManager


class LoraRuntimeSlotsTest(unittest.IsolatedAsyncioTestCase):
  """OPEN_RL_LORA_JOBS_PER_RUNTIME caps the LoRA jobs on one runtime; the next job starts another."""

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
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_LORA_JOBS_PER_RUNTIME": "2"}))
    self.enterContext(patch.object(api_server, "owner_locks", defaultdict(asyncio.Lock)))
    self.enterContext(patch.object(api_server, "runtime_slot_lock", asyncio.Lock()))

  async def asyncSetUp(self):
    self.client = await self.enterAsyncContext(asgi_client())
    self.session = (await self.post("create_session", {}))["session_id"]

  async def post(self, path, body):
    return await post_json(self.client, path, body)

  async def create(self, session_id=None) -> str:
    return (await self.post("create_model", {"base_model": "test-base", "session_id": session_id or self.session}))["request_id"]

  async def runtime(self, model_id) -> str:
    return (await get_model_metadata(self.state, model_id)).runtime(model_id)

  async def test_a_full_runtime_sends_the_next_job_to_a_new_one(self):
    jobs = [await self.create() for _ in range(3)]

    self.assertEqual([await self.runtime(job) for job in jobs], ["test-base", "test-base", "test-base#slot1"])
    self.assertEqual([owner_of(job) for job in jobs], ["test-base", "test-base", "test-base-slot1"])
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
    await self.state.delete_values(f"open_rl:session:{other}")

    self.assertEqual(await self.runtime(await self.create()), "test-base")

  async def test_without_a_cap_every_job_shares_one_runtime(self):
    with patch.dict(os.environ, {"OPEN_RL_LORA_JOBS_PER_RUNTIME": ""}):
      jobs = [await self.create() for _ in range(3)]

    self.assertEqual({await self.runtime(job) for job in jobs}, {"test-base"})

  async def test_fft_and_exclusive_jobs_take_no_slot(self):
    body = {"base_model": "test-base", "session_id": self.session, "user_metadata": {"openrl.exclusive": "true"}}
    exclusive = (await self.post("create_model", body))["request_id"]

    self.assertEqual(await self.runtime(exclusive), exclusive)
    self.assertEqual(await self.state.set_members(api_server.runtime_jobs_key("test-base")), set())


if __name__ == "__main__":
  unittest.main()
