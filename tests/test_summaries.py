"""The run and session summaries, read through their routes from records in the shapes the recorder writes."""

import asyncio
import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server import api_server, telemetry
from server.observability import facts, fleet, router
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore, Writes
from tests.test_observability import NO_CLUSTER, meta

POD = "orw-fft-run-a-trainer"


class SummaryTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store, self.queues, self.now = InMemoryStateStore(), InMemoryStore(), time.time()
    fleet.cache.clear()
    facts.gaps.clear()
    telemetry.buffer.clear()
    for module in (fleet, facts, router, telemetry):
      self.enterContext(mock.patch.object(module, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(facts, "get_store", return_value=self.queues))
    self.enterContext(mock.patch.object(fleet, "kubernetes", mock.AsyncMock(side_effect=lambda: json.loads(json.dumps(NO_CLUSTER)))))
    self.enterContext(mock.patch.object(router, "time", mock.Mock(time=lambda: self.now)))
    self.recorder = self.enterContext(mock.patch.object(telemetry, "recorder", telemetry.Recorder("api_server", None)))

  def run_a(self, live: bool = True, sessions: tuple[str, ...] = ("sess-a",), **fields) -> None:
    asyncio.run(self.store.set_value("open_rl:model_meta:run-a", meta(session_ids=list(sessions), created_at=self.now - 3600, **fields)))
    if live:
      asyncio.run(self.store.set_value(f"open_rl:session:{sessions[0]}", "{}"))

  def record(self, *entries: tuple) -> None:
    self.store.write_sync(telemetry.plan(entries, {"run-a": 0}, {"pod": POD}, "trainer"), telemetry.TTL_SECONDS)

  def op(self, operation: str, ago: float, seconds: float = 1.0, error_type: str | None = None, **fields) -> tuple:
    at = self.now - ago
    record = telemetry.op_record(operation, "run-a", None, "run-a", "trainer", at - seconds, error_type)
    return ("op", "run-a", record | {"at": at, "elapsed_seconds": seconds, "step": 0} | fields)

  def worker(self, **fields) -> None:
    w = json.loads(telemetry.Worker("trainer", "run-a").snapshot()) | {"pod": POD} | fields
    self.store.write_sync(Writes(values={telemetry.key("worker", w["id"]): json.dumps(w)}), telemetry.TTL_SECONDS)

  def get(self, path: str, status: int = 200, **params) -> dict:
    response = TestClient(api_server.app).get(f"/api/v1/inspect/{path}", params=params)
    self.assertEqual(response.status_code, status, response.text)
    return response.json()

  def summary(self) -> dict:
    return self.get("runs/run-a/summary")

  def line(self) -> dict:
    return next(r for r in self.get("runs", ended="true")["runs"] if r["run_id"] == "run-a")

  def test_a_request_is_lost_only_once_the_gap_outlasts_a_flush(self) -> None:
    self.run_a()
    self.record(*[("count", "run-a", "enqueued")] * 98, *[("count", "run-a", "popped")] * 94)
    asyncio.run(self.queues.put_request({"model_id": "run-a"}))
    self.assertNotIn("lost", self.summary()["requests"]["ledger"])
    self.now += telemetry.FLUSH_SECONDS + 1
    self.assertEqual(self.summary()["requests"]["ledger"], {"enqueued": 98, "popped": 94, "lost": 3})

  def test_requests_held_by_a_dead_worker_name_its_pod(self) -> None:
    self.run_a()
    self.worker(heartbeat_at=self.now - 120, outstanding=[{"op": "forward_backward", "run_id": "run-a", "since": self.now - 300, "count": 2}])
    held = {"op": "forward_backward", "count": 2, "age_s": 300, "pod": POD, "holder_alive": False}
    self.assertEqual(self.summary()["requests"]["unanswered"], [held])
    self.assertEqual((self.line()["trainer_requests_unanswered"], self.line()["held_by_dead_worker"]), (2, 2))

  def test_step_pace_and_ages_count_from_the_request_and_never_go_negative(self) -> None:
    self.run_a()
    for i in range(30):
      self.record(self.op("optim_step", 600 - 20 * i), ("step", "run-a", self.now - 600 + 20 * i))
    self.record(self.op("forward", -3))  # recorded after the request's clock read
    found, line = self.summary(), self.line()
    pace = [found[k] for k in ("steps", "step_seconds", "since_last_step_s", "since_last_op_s")]
    self.assertEqual(pace, [30, {"median": 20, "last": 20}, 20, 0])
    self.assertEqual((line["median_step_s"], line["since_last_step_s"], line["since_last_op_s"]), (20, 20, 0))
    self.now += 100  # the inventory snapshot stays cached; ages still move
    self.assertEqual([self.line()[k] for k in ("since_last_step_s", "since_last_op_s")], [120, 97])

  def test_an_ended_run_shows_no_ages(self) -> None:
    self.run_a(live=False)
    self.record(self.op("optim_step", 60), ("step", "run-a", self.now - 60), self.op("optim_step", 30), ("step", "run-a", self.now - 30))
    self.assertEqual(self.get("runs")["runs"], [])
    for found in (self.summary(), self.line()):
      self.assertEqual((found["lifecycle"], found["steps"]), ("ended", 2))
      self.assertFalse({"since_last_op_s", "since_last_step_s"} & found.keys())

  def test_a_worker_running_other_settings_than_asked_is_named(self) -> None:
    self.run_a(fine_tuning_type="full")
    self.worker(config={"env": {"OPEN_RL_FINE_TUNING_TYPE": "lora"}})
    mismatch = {"pod": POD, "setting": "openrl.fine_tuning_type", "asked": "full", "running": "lora"}
    self.assertEqual(self.summary()["config"], [mismatch])

  def test_a_failure_points_at_a_logs_read_the_logs_route_takes(self) -> None:
    self.run_a()
    self.record(self.op("forward_backward", 30, seconds=2, error_type="RuntimeError"), ("count", "run-a", "failed"))
    failures = self.summary()["failures"]
    self.assertEqual((failures["count"], failures["ops"][0]["error_type"]), (1, "RuntimeError"))
    window = {"since": facts.iso(self.now - 92), "until": facts.iso(self.now + 30)}
    self.assertEqual(failures["logs"], {"read": "logs", "args": {"run_id": "run-a", "pod": POD, **window}})
    self.assertFalse(self.get(failures["logs"]["read"], **failures["logs"]["args"])["available"])  # no log store here
    self.assertEqual(self.line()["failures"], 1)

  def test_a_nonfinite_metric_is_named(self) -> None:
    self.run_a()
    self.record(self.op("optim_step", 30, step=30, nonfinite=["loss:sum"]))
    first = {"operation": "optim_step", "step": 30, "metrics": ["loss:sum"]}
    self.assertEqual(self.summary()["optimizer"]["nonfinite"], {"count": 1, "first": first})

  def test_sampler_sends_failures_and_median_group_time(self) -> None:
    self.run_a()
    for i in range(10):
      self.record(*[("sampled", "run-a", "sampler-run-a", self.now - 620 + 60 * i, self.now - 600 + 61 * i)] * 64)
    failed = telemetry.op_record("sample", "run-a", "r1", "sampler-run-a", "dispatcher", self.now - 5, telemetry.FAILED)
    self.record(("sample", "run-a", failed))
    self.assertEqual(self.summary()["samplers"], {"sends": 641, "failed": 1, "median_s": 24.5})
    self.assertEqual(self.line()["sample_failures"], 1)

  def test_a_session_keeps_its_record_sdk_errors_and_models(self) -> None:
    self.enterContext(mock.patch.object(api_server, "session_registry", SessionRegistry(self.store)))
    client = TestClient(api_server.app)
    session = client.post("/api/v1/create_session", json={"user_metadata": {"recipe_name": "s8", "api_key": "k"}}).json()["session_id"]
    self.run_a(live=False, sessions=(session,))
    client.post("/api/v1/session_heartbeat", json={"session_id": session})
    events = [
      {"severity": "ERROR", "event_name": "APIStatusError", "event_data": {"status_code": 400, "detail": "bad"}, "timestamp": "2020-01-01T00:00Z"},
      {"severity": "CRITICAL", "error_type": "tinker.Fatal", "error_message": "x" * 900},
      {"severity": "INFO", "event_name": "session_start"},
    ]
    client.post("/api/v1/telemetry", json={"session_id": session, "events": events})
    self.recorder.flush()
    found = self.get(f"sessions/{session}/summary")
    self.assertEqual((found["metadata"], found["live"], found["recorded"]), ({"recipe_name": "s8", "api_key": "[redacted]"}, True, True))
    self.assertEqual([(m["model_id"], m["sdk_errors"]) for m in found["models"]], [("run-a", 2)])
    self.assertEqual((found["sdk_errors"]["count"], found["sdk_errors"]["last"]["reason"]), (2, "SDK Fatal"))
    self.assertEqual(len(found["sdk_errors"]["last"]["message"]), telemetry.MESSAGE_CHARS)
    self.assertEqual(self.get(f"runs/{session}/summary")["run_id"], "run-a")

  def test_a_session_without_its_record_is_summarized_from_its_runs(self) -> None:
    self.run_a(live=False, sessions=("sess-old",))
    found = self.get("sessions/sess-old/summary")
    self.assertEqual((found["recorded"], found["live"], [m["model_id"] for m in found["models"]]), (False, False, ["run-a"]))
    self.get("sessions/sess-none/summary", status=404)

  def test_idle_trainer_time_splits_into_gpu_wait_queued_rollout_and_client(self) -> None:
    self.run_a(live=False)
    t0 = self.now - 1000
    self.record(self.op("forward_backward", 1000 - 10, seconds=10), self.op("optim_step", 1000 - 110, seconds=10, queue_seconds=30))
    self.worker(waits=[[t0 + 60, t0 + 80]])
    self.record(("sampled", "run-a", "sampler-run-a", t0 + 20, t0 + 50))
    seconds = {"wall": 110, "train": 20, "checkpoint": 0, "forward": 0, "gpu_wait": 20, "queued": 20, "rollout": 30, "client": 20}
    self.assertEqual(self.summary()["goodput"], {"goodput": 0.182, "seconds": seconds, "first_step_after_s": 110})
