# Unit tests for Phase 1 Admin Analytics & Accelerator Usage telemetry endpoints.

import time
import unittest

from fastapi.testclient import TestClient

from server.gateway import app
from server.store import InMemoryStore


class TestAdminAccelUsage(unittest.IsolatedAsyncioTestCase):
  async def test_in_memory_store_telemetry_recording(self):
    in_mem_store = InMemoryStore()
    claim_id = "test-gpu-claim-01"

    event1 = {
      "event_id": "evt-1",
      "resource_claim_id": claim_id,
      "node_name": "node-1",
      "gpu_index": 0,
      "job_id": "math-rl",
      "tenant_id": "tenant-math",
      "model_name": "Qwen/Qwen2.5-0.5B",
      "num_params_billions": 0.5,
      "worker_role": "trainer",
      "acquire_time": time.time() - 2.0,
      "release_time": time.time(),
      "duration_ms": 2000,
      "tokens_processed": 1000,
    }

    await in_mem_store.record_accel_usage_event(claim_id, event1)
    history = await in_mem_store.get_accel_usage_history(claim_id)

    self.assertIn(claim_id, history)
    self.assertEqual(len(history[claim_id]), 1)
    self.assertEqual(history[claim_id][0]["event_id"], "evt-1")

  async def test_admin_api_endpoint(self):
    client = TestClient(app)
    response = client.get("/api/v1/admin/accel_usage")
    self.assertEqual(response.status_code, 200)

    data = response.json()
    self.assertIn("timestamp", data)
    self.assertIn("claims", data)

  async def test_admin_dashboard_html_route(self):
    client = TestClient(app)
    response = client.get("/admin/dashboard/")
    self.assertEqual(response.status_code, 200)
    self.assertIn("Open-RL Admin Dashboard", response.text)
    self.assertIn("Accelerator Usage", response.text)

  async def test_delta_weight_mutation_tracking(self):
    in_mem_store = InMemoryStore()
    model_id = "delta-job-01"
    await in_mem_store.update_job_metadata(
      model_id,
      {
        "model_id": model_id,
        "base_model": "Qwen/Qwen2.5-0.5B",
        "weight_sync_config": {"strategy": "delta"},
        "latest_mutation_pct": 1.25,
        "latest_changed_elements": 1250,
        "latest_total_elements": 100000,
      },
    )
    meta = await in_mem_store.get_model_metadata(model_id)
    self.assertIsNotNone(meta)
    self.assertEqual(meta.get("latest_mutation_pct"), 1.25)


if __name__ == "__main__":
  unittest.main()
