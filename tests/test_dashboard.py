"""The inspection routes as an agent calls them, over a snapshot with one run and its recorded optim_step."""

import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import fleet, router
from server.dashboard.__main__ import app
from server.store import InMemoryStateStore
from server.telemetry import key

RUN = {"run_id": "r1", "name": "Run one", "model": "Qwen/base", "runtime_id": "r1", "created_at": 0, "completed_at": None}
RUN |= {"lifecycle": "placed", "status": "active", "workloads": [], "pods": [], "settings": {}, "session_ids": []}


class ApiTest(unittest.TestCase):
  def setUp(self):
    self.store = InMemoryStateStore()
    op = {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "operation": "optim_step", "status": "succeeded", "started_at": 100, "at": 150}
    asyncio.run(self.store.append_sample(key("operations", "r1"), op, 10, None))
    self.snapshot = {"observed_at": 200, "placements": [], "workers": [], "runs": [RUN], "nodes": [], "pods": [], "errors": []}
    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=self.snapshot)))
    self.enterContext(mock.patch("server.dashboard.facts.get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(router, "query", mock.AsyncMock(return_value={"series": []})))  # Prometheus has no series
    self.client = TestClient(app)

  def test_routes_are_read_only_and_a_run_summary_reads_the_store(self):
    summary = self.client.get("/api/v1/inspect/runs/r1/summary").json()
    self.assertEqual((summary["steps"], summary["last_op"]["operation"]), (1, "optim_step"))
    self.assertEqual(self.client.get("/api/v1/inspect/runs/missing/summary").status_code, 404)
    self.assertTrue(all(set(methods) == {"get"} for methods in self.client.get("/openapi.json").json()["paths"].values()))

  def test_a_session_id_names_its_one_run_and_a_session_with_several_runs_is_refused(self):
    self.snapshot["runs"] = [{**RUN, "session_ids": ["s1", "s2"]}, {**RUN, "run_id": "r2", "runtime_id": "r2", "session_ids": ["s2"]}]
    by_session = self.client.get("/api/v1/inspect/runs/s1/summary").json()
    self.assertEqual((by_session["run_id"], by_session["steps"]), ("r1", 1))
    refused = self.client.get("/api/v1/inspect/runs/s2/summary")
    self.assertEqual(refused.status_code, 409)
    self.assertIn("r1, r2", refused.text)
    self.assertEqual(self.client.get("/api/v1/inspect/runs/s3/summary").status_code, 404)


if __name__ == "__main__":
  unittest.main()
