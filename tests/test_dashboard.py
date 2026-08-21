import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from server import gateway
from server.dashboard import data


class DashboardEndpointsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(gateway.app)

  def tearDown(self) -> None:
    os.environ.pop("OPEN_RL_DASHBOARD_DEMO", None)

  def test_index_serves_html(self) -> None:
    resp = self.client.get("/dashboard")
    self.assertEqual(resp.status_code, 200)
    self.assertIn("open-rl operations", resp.text)

  def test_static_assets_served(self) -> None:
    for asset in ("style.css", "app.js"):
      resp = self.client.get(f"/dashboard/static/{asset}")
      self.assertEqual(resp.status_code, 200, asset)

  def test_health_reports_all_groups(self) -> None:
    resp = self.client.get("/api/v1/dashboard/health")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    groups = {check["group"] for check in body["checks"]}
    self.assertLessEqual({"Gateway", "Storage", "Kubernetes"}, groups)
    statuses = {check["status"] for check in body["checks"]}
    self.assertLessEqual(statuses, {"ok", "warn", "error", "off"})
    stat_ids = {stat["id"] for stat in body["stats"]}
    self.assertLessEqual({"runs.active", "queue.requests", "queue.launch"}, stat_ids)
    self.assertIsInstance(body["queues"], list)

  def test_cluster_degrades_without_kubernetes(self) -> None:
    resp = self.client.get("/api/v1/dashboard/cluster")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    self.assertIn("kubernetes", body)
    self.assertIn("gateway", body)
    service_ids = {s["id"] for s in body["services"]}
    self.assertLessEqual({"redis", "storage"}, service_ids)
    # Edges only exist where connectivity is configured, never invented.
    for edge in body["edges"]:
      self.assertEqual(edge["from"], "gateway")

  def test_runs_lists_filesystem_checkpoints(self) -> None:
    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      adapter_dir = os.path.join(tmp_dir, "peft", "model-abc-123")
      os.makedirs(adapter_dir)
      with open(os.path.join(adapter_dir, "metadata.json"), "w") as f:
        json.dump({"alias": "my-training-run", "base_model": "Qwen/Qwen3-8B"}, f)
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "model-def-456"))

      body = self.client.get("/api/v1/dashboard/runs").json()
      runs = {run["run_id"]: run for run in body["runs"]}
      self.assertIn("model-abc-123", runs)
      self.assertEqual(runs["model-abc-123"]["name"], "my-training-run")
      self.assertEqual(runs["model-abc-123"]["base_model"], "Qwen/Qwen3-8B")
      self.assertIn("model-def-456", runs)
      self.assertIn("checkpoint", runs["model-def-456"]["sources"])

  def test_run_detail_bundles_run_state(self) -> None:
    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "model-xyz-789"))

      resp = self.client.get("/api/v1/dashboard/runs/model-xyz-789")
      self.assertEqual(resp.status_code, 200)
      detail = resp.json()
      self.assertEqual(detail["run_id"], "model-xyz-789")
      self.assertEqual(detail["queue_depth"], 0)
      self.assertEqual(detail["pods"], [])
      self.assertEqual(detail["gpu_claims"], {})
      self.assertNotIn("logs", detail, "logs are only included when requested")

    self.assertEqual(self.client.get("/api/v1/dashboard/runs/no-such-run").status_code, 404)

  def test_stop_unknown_run_conflicts(self) -> None:
    resp = self.client.post("/api/v1/dashboard/runs/does-not-exist/stop")
    self.assertEqual(resp.status_code, 409)

  def test_demo_mode_flags_every_payload(self) -> None:
    os.environ["OPEN_RL_DASHBOARD_DEMO"] = "1"
    for path in ("cluster", "runs", "health", "problems", "pods/any-pod/logs", "runs/demo-run-1?logs=5"):
      body = self.client.get(f"/api/v1/dashboard/{path}").json()
      self.assertTrue(body["demo"], path)
      self.assertIn("fictional", body["notice"], path)
    stop = self.client.post("/api/v1/dashboard/runs/demo-run-1/stop").json()
    self.assertTrue(stop["demo"])

  def test_duty_tracker_records_per_job_allocation(self) -> None:
    tracker = data.DutyTracker(max_samples=3, min_interval_s=5.0)
    pools = [{"id": "h100", "nodes": [{"name": "n1", "gpu_capacity": 8}]}]
    pods = [
      {"node": "n1", "phase": "Running", "gpus": 3, "labels": {"timeslice.io/job-id": "trainer-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "sampler-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 2, "labels": {}, "app": "dcgm-exporter"},
      {"node": "n1", "phase": "Succeeded", "gpus": 2, "labels": {"timeslice.io/job-id": "trainer-run-b"}},
      {"node": "other-node", "phase": "Running", "gpus": 8, "labels": {"timeslice.io/job-id": "trainer-run-c"}},
    ]
    tracker.record(pools, pods, now=100.0)
    tracker.record(pools, pods, now=102.0)

    duty = tracker.duty(pools[0])
    self.assertEqual(duty["capacity"], 8)
    self.assertEqual(duty["current"], 0.75)
    self.assertEqual(len(duty["series"]), 1, "second sample should be throttled")
    self.assertEqual(duty["series"][0][1], {"run-a": 4, "dcgm-exporter": 2}, "trainer+sampler merge per run; unlabeled pods use app")
    self.assertEqual(duty["jobs"], ["run-a", "dcgm-exporter"])

    for i in range(5):
      tracker.record(pools, pods, now=110.0 + i * 10)
    self.assertEqual(len(tracker.duty(pools[0])["series"]), 3, "ring buffer should cap history")

    cpu_pool = {"id": "cpu", "nodes": [{"name": "c1", "gpu_capacity": 0}]}
    self.assertIsNone(tracker.duty(cpu_pool), "pools without GPUs have no duty cycle")

  def test_operational_stats_count_in_memory_queues(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    store = InMemoryStore()
    empty_k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}
    asyncio.run(store.put_request({"model_id": "run-a", "op": "forward_backward"}))
    asyncio.run(store.put_request({"model_id": "run-a", "op": "optim_step"}))

    stats, queues = asyncio.run(data.operational_stats(store, empty_k8s, worker_manager=None))
    by_id = {stat["id"]: stat for stat in stats}
    self.assertEqual(by_id["runs.active"]["value"], "1")
    self.assertEqual(by_id["queue.requests"]["value"], "2")
    self.assertEqual(queues, [{"model_id": "run-a", "depth": 2}])

  def test_model_pods_match_timeslice_labels(self) -> None:
    pods = [
      {"name": "open-rl-trainer-x", "labels": {"timeslice.io/job-id": "trainer-abc-123"}},
      {"name": "open-rl-sampler-x", "labels": {"timeslice.io/job-id": "sampler-abc-123"}},
      {"name": "other", "labels": {"timeslice.io/job-id": "trainer-zzz"}},
      {"name": "unlabeled", "labels": {}},
    ]
    matched = {p["name"] for p in data.model_pods("abc_123", pods)}
    self.assertEqual(matched, {"open-rl-trainer-x", "open-rl-sampler-x"})
    self.assertEqual(data.model_pods("abc", pods), [], "a run must never match another run's prefix-sharing pods")

  def test_duty_reports_overcommit_honestly(self) -> None:
    tracker = data.DutyTracker()
    pools = [{"id": "shared", "nodes": [{"name": "n1", "gpu_capacity": 1}]}]
    pods = [
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "trainer-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "sampler-run-b"}},
    ]
    tracker.record(pools, pods, now=100.0)
    duty = tracker.duty(pools[0])
    self.assertEqual(duty["current"], 2.0, "time-sliced overcommit must not be clamped to 100%")

  def test_runs_with_unknown_created_at_sort_last(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "run-with-date"))

      store = InMemoryStore()
      asyncio.run(store.put_request({"model_id": "run-no-date", "op": "forward_backward"}))
      snapshot = asyncio.run(data.runs_snapshot(store, None, pods=[]))
      order = [run["run_id"] for run in snapshot["runs"]]
      self.assertEqual(order, ["run-with-date", "run-no-date"], "runs without created_at belong at the end")


if __name__ == "__main__":
  unittest.main()
