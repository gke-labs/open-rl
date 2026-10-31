import asyncio
import copy
import json
import os
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from server import api_server
from server.model_metadata import TrainingModelMetadata
from server.observability import cloud_logging, fleet, metrics
from server.store import InMemoryStateStore
from server.worker_manager import FAILED, READY, STARTING, STOPPED, WAITING_FOR_PLACEMENT

NO_CLUSTER = {"available": True, "pods": [], "nodes": [], "workloads": None, "errors": []}
T = datetime(2026, 10, 28, 12, tzinfo=UTC).timestamp()


def meta(**fields) -> str:
  return TrainingModelMetadata(base_model="Qwen/Qwen3-8B", **fields).model_dump_json()


def workload(name: str, model_id: str, kind: str, deleting: bool = False, **status: str) -> dict:
  metadata = {"name": name} | ({"deletionTimestamp": "2026-10-29T12:05:00Z"} if deleting else {})
  status = {"phase": "Running", "nodeName": "n1", "podName": f"orw-{name}", "gpuUUIDs": [f"GPU-{name}"]} | status
  return {"metadata": metadata, "spec": {"role": "trainer", "modelID": model_id, "trainingKind": kind}, "status": status}


def cluster(*states: str) -> dict:
  return {"pods": [], "workloads": [{"model_id": "r1", "training_kind": "fft", "state": s} for s in states]}


class ApiTest(unittest.TestCase):
  def setUp(self):
    self.store = InMemoryStateStore()
    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch("server.observability.router.get_state_store", return_value=self.store))

  def get(self, path: str, cluster: dict | None = NO_CLUSTER, **records: str) -> dict:
    for key, value in records.items():
      asyncio.run(self.store.set_value(f"open_rl:{key}", value))
    if cluster is not None:
      self.enterContext(mock.patch.object(fleet, "kubernetes", mock.AsyncMock(return_value=copy.deepcopy(cluster))))
    return TestClient(api_server.app).get(path).json()

  def serve_workloads(self, *workloads: dict) -> None:
    def listing(items) -> object:
      return lambda *args, **kwargs: SimpleNamespace(data=json.dumps({"items": list(items)}))

    api = SimpleNamespace(list_namespaced_pod=listing([]), list_node=listing([]), list_cluster_custom_object=listing([]))
    api.list_namespaced_custom_object = listing(workloads)
    self.enterContext(mock.patch("server.observability.kubernetes.kubernetes_api", return_value=(api, api)))

  def released(self, runtime: str, name: str, gpus: list[str], placed_at: float, released_at: float) -> None:
    record = {"name": name, "role": "trainer", "node": "n1", "pod": f"orw-{name}", "gpu_uuids": gpus, "training_kind": "fft"}
    record.update(placed_at=placed_at, released_at=released_at)
    self.store.push_recent_sync(f"open_rl:placements:{runtime}", [json.dumps(record)], 200, 3600)


class InventoryTest(ApiTest):
  def test_a_run_is_active_only_while_the_client_has_a_live_session(self):
    body = self.get(
      "/api/v1/inspect/inventory",
      **{
        "model_meta:live": meta(session_ids=["s1"], created_at=3),
        "model_meta:idle": meta(session_ids=["s2"], created_at=2),
        "model_meta:done": meta(status="completed", session_ids=["s1"], created_at=1),
        "session:s1": "{}",
      },
    )

    runs = [(r["run_id"], r["lifecycle"], r["client_status"]) for r in body["runs"]]
    self.assertEqual(runs, [("live", "active", "active"), ("idle", "ended", "active"), ("done", "ended", "completed")])
    self.assertEqual(body["errors"], [])

  def test_runs_come_back_from_redis_when_kubernetes_is_unavailable(self):
    down = NO_CLUSTER | {"available": False, "errors": ["Kubernetes unavailable: no kubeconfig"]}
    crashing = {"name": "orw-x", "node": "n1", "owner": "x", "problem": "CrashLoopBackOff", "restarts": 4}
    healthy = crashing | {"name": "orw-y", "problem": None, "restarts": 0}
    records = {"model_meta:r1": meta(session_ids=["s1"]), "session:s1": "{}"}
    body = self.get("/api/v1/inspect/inventory", down | {"pods": [crashing, healthy]}, **records)

    self.assertEqual([(r["run_id"], r["lifecycle"]) for r in body["runs"]], [("r1", "active")])
    self.assertEqual(body["cluster"], {"available": False, "nodes": [], "pods": [{"name": "orw-x", "problem": "CrashLoopBackOff", "restarts": 4}]})
    self.assertEqual(body["errors"], ["Kubernetes unavailable: no kubeconfig"])

  def test_a_malformed_model_record_is_skipped_with_an_error(self):
    body = self.get("/api/v1/inspect/inventory", **{"model_meta:good": meta(), "model_meta:bad": "not json"})

    self.assertEqual([r["run_id"] for r in body["runs"]], ["good"])
    self.assertTrue(any("bad" in error for error in body["errors"]), body["errors"])

  def test_placements_list_the_workloads_holding_gpus_with_the_runs_they_serve(self):
    placed, released = "2026-10-29T12:00:00Z", "2026-10-29T12:10:00Z"
    self.serve_workloads(
      workload("fft-a", "fft-1", "fft", placedAt=placed),
      workload("lora-a", "Qwen/Qwen3-8B", "lora", placedAt=placed),
      workload("fft-gone", "fft-2", "fft", placedAt=placed, releasedAt=released, nodeName=""),
      workload("fft-deleting", "fft-3", "fft", deleting=True, placedAt=placed),
      workload("fft-waiting", "fft-4", "fft", nodeName=""),
    )
    records = {
      "model_meta:fft-1": meta(fine_tuning_type="full", status="completed"),
      "model_meta:lora-live": meta(),
      "model_meta:lora-done": meta(status="completed"),
    }
    body = self.get("/api/v1/inspect/inventory", None, **records)

    at = datetime(2026, 10, 29, 12, tzinfo=UTC).timestamp()
    self.assertEqual(
      body["placements"],
      [
        {"name": "fft-a", "role": "trainer", "node": "n1", "pod": "orw-fft-a", "gpu_uuids": ["GPU-fft-a"], "placed_at": at, "run_ids": ["fft-1"]},
        {
          "name": "lora-a",
          "role": "trainer",
          "node": "n1",
          "pod": "orw-lora-a",
          "gpu_uuids": ["GPU-lora-a"],
          "placed_at": at,
          "run_ids": ["lora-live"],
        },
      ],
    )


class ActivityTest(ApiTest):
  def test_a_workloads_lane_joins_its_live_and_released_placements_clipped_to_the_window(self):
    self.serve_workloads(workload("fft-a", "fft-1", "fft", placedAt="2026-10-28T11:30:00Z"))
    self.released("fft-1", "fft-a", ["GPU-fft-a"], T - 3 * 3600, T - 2 * 3600)
    self.released("fft-1", "fft-a", ["GPU-fft-a"], T - 6 * 3600, T - 5 * 3600)
    self.released("fft-1", "fft-a", ["GPU-other"], T - 3 * 3600, T - 2 * 3600)
    window = "since=2026-10-28T09:30:00Z&until=2026-10-28T11:50:00Z"
    body = self.get(f"/api/v1/inspect/activity?{window}", None, **{"model_meta:fft-1": meta(fine_tuning_type="full", created_at=T - 4 * 3600)})

    lane = {"name": "fft-a", "role": "trainer", "node": "n1", "pod": "orw-fft-a", "run_ids": ["fft-1"], "shared_gpu": False}
    idle = {"turns": [], "waits": []}  # no worker record, so no slicer turns
    self.assertEqual(
      body["lanes"],
      [
        lane | {"gpu_uuids": ["GPU-fft-a"], "intervals": [[T - 150 * 60, T - 2 * 3600], [T - 30 * 60, T - 10 * 60]], "ended": False} | idle,
        lane | {"gpu_uuids": ["GPU-other"], "intervals": [[T - 150 * 60, T - 2 * 3600]], "ended": True} | idle,
      ],
    )
    self.assertEqual(body["errors"], [])

  def test_a_lane_carries_its_workers_turns_and_waits_clipped_to_the_placement_and_the_window(self):
    self.released("fft-1", "fft-a", ["GPU-fft-a"], T - 3600, T - 1800)
    # The same pod name appears in an earlier placement; only the process alive during this one counts.
    earlier = {"id": "w0", "pod": "orw-fft-a", "started_at": T - 9000, "heartbeat_at": T - 7200, "turns": [[T - 8000, T - 7900]], "waits": []}
    worker = {"id": "w1", "pod": "orw-fft-a", "started_at": T - 3600, "heartbeat_at": T - 1800, "phase": "exited"}
    worker |= {"turns": [[T - 3500, T - 3400], [T - 2500, T - 1700]], "waits": [[T - 3600, T - 3500]]}
    for record in (earlier, worker):
      asyncio.run(self.store.set_value(f"open_rl:worker:{record['id']}", json.dumps(record)))
    window = "since=2026-10-28T11:10:00Z&until=2026-10-28T12:00:00Z"  # T - 3000 onwards
    body = self.get(f"/api/v1/inspect/activity?{window}", None, **{"model_meta:fft-1": meta(fine_tuning_type="full", created_at=T - 4000)})

    (lane,) = body["lanes"]
    self.assertEqual((lane["intervals"], lane["turns"], lane["waits"]), ([[T - 3000, T - 1800]], [[T - 2500, T - 1800]], []))

  def test_live_lanes_come_back_with_an_error_when_redis_fails(self):
    self.serve_workloads(workload("fft-a", "fft-1", "fft", placedAt="2026-10-28T11:30:00Z"))
    self.enterContext(mock.patch.object(self.store, "scan_values", mock.AsyncMock(side_effect=ConnectionError("redis down"))))
    response = TestClient(api_server.app).get("/api/v1/inspect/activity?since=2026-10-28T11:00:00Z&until=2026-10-28T12:00:00Z")

    self.assertEqual(response.status_code, 200)
    self.assertEqual([(lane["name"], lane["gpu_uuids"]) for lane in response.json()["lanes"]], [("fft-a", ["GPU-fft-a"])])
    self.assertEqual(response.json()["errors"], ["Redis read failed: redis down"])

  def test_lanes_that_held_one_gpu_at_overlapping_times_are_marked_shared(self):
    self.serve_workloads()
    self.released("fft-1", "fft-a", ["GPU-0", "GPU-1"], T - 600, T - 300)
    self.released("fft-2", "fft-b", ["GPU-1"], T - 400, T - 100)
    self.released("fft-3", "fft-c", ["GPU-0"], T - 300, T)
    runs = {f"model_meta:fft-{i}": meta(fine_tuning_type="full") for i in (1, 2, 3)}
    body = self.get("/api/v1/inspect/activity?since=2026-10-28T11:00:00Z&until=2026-10-28T12:00:00Z", None, **runs)

    self.assertEqual({lane["name"]: lane["shared_gpu"] for lane in body["lanes"]}, {"fft-a": True, "fft-b": True, "fft-c": False})


class QueryTest(ApiTest):
  def setUp(self):
    super().setUp()
    self.sent, self.answers = [], {}

    async def get(url: str, params: dict) -> dict:
      self.sent.append(params["query"])
      metric = next(m for m in ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", "DCGM_FI_DEV_GPU_UTIL") if m in params["query"])
      return {"status": "success", "data": {"result": self.answers.get(metric, [])}}

    self.enterContext(mock.patch.dict(os.environ, {"OPEN_RL_PROMETHEUS_URL": "http://prometheus:9090"}))
    self.enterContext(mock.patch.object(metrics, "get", get))
    placed = "2026-10-28T11:00:00Z"
    self.serve_workloads(workload("fft-a", "fft-1", "fft", placedAt=placed), workload("fft-b", "fft-2", "fft", placedAt=placed))
    self.runs = {"model_meta:fft-1": meta(fine_tuning_type="full"), "model_meta:fft-2": meta(fine_tuning_type="full")}

  def test_without_a_metrics_source_queries_say_how_to_configure_one(self):
    os.environ.pop("OPEN_RL_PROMETHEUS_URL")
    body = self.get("/api/v1/inspect/query?name=gpu_util&run_id=fft-1", None, **self.runs)

    self.assertEqual((body["available"], body["error"]), (False, "No metrics source configured. Set OPEN_RL_PROMETHEUS_URL."))
    self.assertEqual(self.sent, [])

  def test_a_query_reads_only_the_gpus_the_run_held(self):
    self.answers["DCGM_FI_DEV_GPU_UTIL"] = [{"metric": {"UUID": "GPU-fft-a"}, "values": [[T, "42"]]}]
    body = self.get("/api/v1/inspect/query?name=gpu_util&run_id=fft-1", None, **self.runs)

    self.assertEqual((body["metric"], body["series"]), ("DCGM_FI_DEV_GPU_UTIL", [{"labels": {"UUID": "GPU-fft-a"}, "points": [[T, 42.0]]}]))
    unescaped = [q.replace("\\", "") for q in self.sent]
    self.assertTrue(unescaped and all("GPU-fft-a" in q and "GPU-fft-b" not in q for q in unescaped), self.sent)

  def test_gpu_busy_falls_back_to_gpu_util_and_says_which_it_read(self):
    self.answers["DCGM_FI_DEV_GPU_UTIL"] = [{"metric": {"UUID": "GPU-fft-a"}, "values": [[T, "30"]]}]
    fallback = self.get("/api/v1/inspect/query?name=gpu_busy&run_id=fft-1", None, **self.runs)
    self.answers["DCGM_FI_PROF_GR_ENGINE_ACTIVE"] = [{"metric": {"UUID": "GPU-fft-a"}, "values": [[T, "25"]]}]
    profiled = self.get("/api/v1/inspect/query?name=gpu_busy&run_id=fft-1", None)

    self.assertEqual((fallback["metric"], fallback["series"][0]["points"]), ("DCGM_FI_DEV_GPU_UTIL", [[T, 30.0]]))
    self.assertEqual((profiled["metric"], profiled["series"][0]["points"]), ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", [[T, 25.0]]))


class JoinTest(unittest.TestCase):
  def test_lifecycle_follows_the_runs_workloads(self):
    live, raw = {"s1"}, meta(fine_tuning_type="full", session_ids=["s1"])
    self.assertEqual(fleet.join("r1", raw, live, cluster(STOPPED, STOPPED))["lifecycle"], "ended")
    self.assertEqual(fleet.join("r1", raw, live, cluster(WAITING_FOR_PLACEMENT, READY))["lifecycle"], "waiting")
    self.assertEqual(fleet.join("r1", raw, live, cluster(STARTING, READY))["lifecycle"], "placed")
    self.assertEqual(fleet.join("r1", raw, live, cluster(FAILED, READY))["lifecycle"], "failed")

  def test_a_finished_run_on_a_shared_runtime_does_not_claim_its_pods(self):
    pod = {"name": "orw-qwen-qwen3-8b-trainer", "node": "n1", "owner": "qwen-qwen3-8b-a82fd547", "problem": None, "restarts": 0}
    shared = {"pods": [pod], "workloads": [{"model_id": "Qwen/Qwen3-8B", "training_kind": "lora", "state": READY}]}
    finished = fleet.join("r2", meta(status="completed"), set(), shared)
    active = fleet.join("r3", meta(), set(), shared)
    self.assertEqual((finished["lifecycle"], finished["pods"], finished["nodes"]), ("ended", [], []))
    self.assertEqual((active["lifecycle"], active["pods"], active["nodes"]), ("placed", [{"name": pod["name"], "problem": None}], ["n1"]))
    self.assertTrue(finished["shared_runtime"] and active["shared_runtime"])


class LogsTest(ApiTest):
  def setUp(self):
    super().setUp()
    self.sent = []

    async def send(method: str, url: str, json: dict, headers: dict) -> httpx.Response:
      self.sent.append(json)
      entry = {"textPayload": "INFO loading", "resource": {"labels": {"pod_name": "orw-fft-r1-trainer"}}}
      page = {"entries": [entry], "nextPageToken": "t2"} if "pageToken" not in json else {"entries": []}
      return httpx.Response(200, json=page, request=httpx.Request(method, url))

    self.enterContext(mock.patch.dict(os.environ, {"OPEN_RL_WORKER_NAMESPACE": "ns", "OPEN_RL_LOG_STORE": "cloud-logging"}))
    self.enterContext(mock.patch.object(cloud_logging, "send", send))
    self.enterContext(mock.patch.object(cloud_logging, "credentials", None))
    self.enterContext(mock.patch.object(cloud_logging, "project", None))
    self.google_auth = self.enterContext(mock.patch("google.auth.default", return_value=(mock.Mock(valid=True, token="t"), "p")))

  def test_logs_read_only_the_runs_pods_and_page_with_the_stores_cursor(self):
    window = "since=2026-10-01T00:00:00Z&until=2026-10-01T01:00:00Z"
    first = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=orw-fft-r1-trainer&{window}", **{"model_meta:r1": meta(fine_tuning_type="full")})
    second = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=orw-fft-r1-trainer&{window}&cursor=t2")

    self.assertEqual([(r["pod"], r["message"]) for r in first["records"]], [("orw-fft-r1-trainer", "INFO loading")])
    self.assertEqual((first["next_cursor"], second["records"], second["next_cursor"]), ("t2", [], None))
    self.assertEqual([s.get("pageToken") for s in self.sent], [None, "t2"])
    self.assertEqual(self.sent[0]["resourceNames"], ["projects/p"])
    clauses = self.sent[0]["filter"].split(" AND ")
    self.assertIn('resource.labels.namespace_name="ns"', clauses)
    self.assertIn('resource.labels.pod_name=~"^orw-fft-r1-"', clauses)
    self.assertIn('resource.labels.pod_name="orw-fft-r1-trainer"', clauses)
    self.assertNotIn("owner", self.sent[0]["filter"])

  def test_a_named_resource_is_read_instead_of_the_credentials_project(self):
    self.google_auth.return_value = (mock.Mock(valid=True, token="t"), None)
    meta_r1 = {"model_meta:r1": meta()}
    default = self.get("/api/v1/inspect/logs?run_id=r1", **meta_r1)
    view = "projects/q/locations/global/buckets/b/views/v"
    os.environ["OPEN_RL_LOG_STORE"] = f"cloud-logging:{view}"
    named = self.get("/api/v1/inspect/logs?run_id=r1")

    self.assertFalse(default["available"])
    self.assertIn("The Google credentials name no project", default["error"])
    self.assertTrue(named["available"])
    self.assertEqual([s["resourceNames"] for s in self.sent], [[view]])

  def test_logs_stay_off_without_a_known_log_store(self):
    os.environ.pop("OPEN_RL_LOG_STORE")
    unset = self.get("/api/v1/inspect/logs?run_id=r1", **{"model_meta:r1": meta()})
    os.environ["OPEN_RL_LOG_STORE"] = "splunk"
    unknown = self.get("/api/v1/inspect/logs?run_id=r1")

    self.assertEqual((unset["available"], unset["error"]), (False, "No log store configured. Set OPEN_RL_LOG_STORE; see docs/configuration.md."))
    self.assertEqual((unknown["available"], unknown["error"]), (False, "Unknown OPEN_RL_LOG_STORE 'splunk'; supported: cloud-logging"))
    self.assertEqual(self.sent, [])

  def test_client_logs_are_the_lines_labelled_with_its_id(self):
    window = "since=2026-10-01T00:00:00Z&until=2026-10-01T01:00:00Z"
    body = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=client&{window}", **{"model_meta:r1": meta(client_log_id="c1")})

    self.assertTrue(body["available"])
    self.assertEqual(
      self.sent[0]["filter"].split(" AND "),
      [
        '(labels."k8s-pod/openrl_client"="c1" OR labels."openrl_client"="c1")',
        'timestamp>="2026-10-01T00:00:00.000000+00:00"',
        'timestamp<="2026-10-01T01:00:00.000000+00:00"',
      ],
    )

  def test_a_run_without_a_client_log_id_has_no_client_logs(self):
    body = self.get("/api/v1/inspect/logs?run_id=r1&pod=client", **{"model_meta:r1": meta()})

    self.assertEqual((body["available"], body["error"]), (False, "This run set no openrl.client_log_id; see docs/configuration.md."))
    self.assertEqual(self.sent, [])


if __name__ == "__main__":
  unittest.main()
