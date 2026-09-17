import asyncio
import json
import unittest

from server import gateway
from server.session_registry import SessionRegistry
from server.store import InMemoryStore


def seed(store, model_id, **fields):
  asyncio.run(store.set_value(f"open_rl:model_meta:{model_id}", json.dumps({"base_model": "b", "created_at": 1.0, "status": "active", **fields})))


class RunSettlementTest(unittest.TestCase):
  def test_a_run_remembers_its_first_session_only(self) -> None:
    store = InMemoryStore()
    seed(store, "r1")
    asyncio.run(gateway.remember_session(store, "r1", "sess-a"))
    asyncio.run(gateway.remember_session(store, "r1", "sess-b"))
    self.assertEqual(json.loads(asyncio.run(store.get_value("open_rl:model_meta:r1")))["session_id"], "sess-a")

  def test_runs_of_dead_sessions_end_and_live_or_terminal_runs_do_not(self) -> None:
    store = InMemoryStore()
    registry = SessionRegistry(store)
    asyncio.run(registry.heartbeat("sess-live"))
    seed(store, "live", session_id="sess-live")
    seed(store, "dead", session_id="sess-dead")
    seed(store, "done", session_id="sess-dead", status="completed")
    seed(store, "orphan")
    settled = asyncio.run(gateway.settle_runs(store, registry))
    self.assertEqual(settled, ["dead"])
    rows = {row["model_id"]: row for row in asyncio.run(store.list_jobs_metadata())}
    self.assertEqual(rows["dead"]["status"], "ended")
    self.assertIsNotNone(rows["dead"]["completed_at"])
    self.assertEqual(rows["live"]["status"], "active")
    self.assertEqual(rows["done"]["status"], "completed")
    self.assertEqual(rows["orphan"]["status"], "active")
