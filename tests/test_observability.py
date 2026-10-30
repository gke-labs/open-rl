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
from server.observability import cloud_logging, fleet
from server.store import InMemoryStateStore
from server.worker_manager import FAILED, READY, STARTING, STOPPED, WAITING_FOR_PLACEMENT

NO_CLUSTER = {"available": True, "pods": [], "nodes": [], "workloads": None, "errors": []}


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

  def get(self, path: str, cluster: dict = NO_CLUSTER, **records: str) -> dict:
    for key, value in records.items():
      asyncio.run(self.store.set_value(f"open_rl:{key}", value))
    self.enterContext(mock.patch.object(fleet, "kubernetes", mock.AsyncMock(return_value=copy.deepcopy(cluster))))
    return TestClient(api_server.app).get(path).json()


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
    def listing(items) -> object:
      return lambda *args, **kwargs: SimpleNamespace(data=json.dumps({"items": list(items)}))

    placed, released = "2026-10-29T12:00:00Z", "2026-10-29T12:10:00Z"
    workloads = (
      workload("fft-a", "fft-1", "fft", placedAt=placed),
      workload("lora-a", "Qwen/Qwen3-8B", "lora", placedAt=placed),
      workload("fft-gone", "fft-2", "fft", placedAt=placed, releasedAt=released, nodeName=""),
      workload("fft-deleting", "fft-3", "fft", deleting=True, placedAt=placed),
      workload("fft-waiting", "fft-4", "fft", nodeName=""),
    )
    api = SimpleNamespace(list_namespaced_pod=listing([]), list_node=listing([]), list_cluster_custom_object=listing([]))
    api.list_namespaced_custom_object = listing(workloads)
    self.enterContext(mock.patch("server.observability.kubernetes.kubernetes_api", return_value=(api, api)))
    records = {
      "model_meta:fft-1": meta(fine_tuning_type="full", status="completed"),
      "model_meta:lora-live": meta(),
      "model_meta:lora-done": meta(status="completed"),
    }
    for key, value in records.items():
      asyncio.run(self.store.set_value(f"open_rl:{key}", value))
    body = TestClient(api_server.app).get("/api/v1/inspect/inventory").json()

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
