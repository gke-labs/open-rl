"""Run facts replay recorded incidents: each incident must show up as a fact an agent can read without a verdict."""

import asyncio
import unittest
from unittest import mock

from server.dashboard import facts, fleet
from server.dashboard.facts import request_queue, sampler_queue
from server.store import InMemoryStateStore
from server.telemetry import key

NOW = 100_000.0


def op(at, operation, status="succeeded", role="trainer", **fields):
  return {"at": at, "started_at": at - 10, "operation": operation, "status": status, "role": role, "elapsed_seconds": 10.0, "metrics": {}, **fields}


def healthy_steps(count=5, every=600.0):
  return [op(NOW - (count - i) * every, "optim_step", metrics={"grad_norm:mean": 1.0 + i}) for i in range(count)]


def worker(started_at, heartbeat_at, **fields):
  return {"id": str(started_at), "pod": "t-0", "role": "trainer", "runtime_id": "r", "started_at": started_at, "heartbeat_at": heartbeat_at, **fields}


class RunFactsTest(unittest.TestCase):
  """Records go into a real in-memory store, and the facts are read back through the same context the routes use."""

  def setUp(self):
    self.enterContext(mock.patch.object(facts, "get_state_store", side_effect=lambda: self.store))
    self.enterContext(mock.patch.object(fleet, "get_state_store", side_effect=lambda: self.store))

  def facts_of(self, ops=(), workers=(), ledger=None, queued=0, lifecycle="placed", pods=(), samples_queued=0, completed_at=None):
    self.store = InMemoryStateStore()
    fleet.cache.clear()

    async def read():
      for record in ops:
        await self.store.append_sample(key("operations", "r", *(["samples"] if record["operation"] == "sample" else [])), record, 10_000, None)
      for event, value in (ledger or {}).items():
        for _ in range(value):
          await self.store.increment_field(key("ledger", "r"), event, None)
      queues = {request_queue("r"): queued, sampler_queue("r"): samples_queued}  # the in-memory store keeps no queues
      self.store.list_length = mock.AsyncMock(side_effect=queues.get)
      run = {"run_id": "r", "runtime_id": "r", "lifecycle": lifecycle, "created_at": 0.0, "completed_at": completed_at, "workloads": []}
      run.update(pods=list(pods), settings={"openrl.fine_tuning_type": "lora", "openrl.trainer_gpus": 1})
      ctx = await facts.context({"observed_at": NOW, "workers": list(workers), "placements": [], "runs": [run]}, run)
      return ctx, facts.summary(ctx), facts.line(ctx)

    return asyncio.run(read())

  def test_requests_held_by_a_trainer_that_stopped_reporting_name_the_dead_holder(self):  # I1
    held = {"op": "forward_backward", "run_id": "r", "count": 3, "since": NOW - 207360}
    _, summary, line = self.facts_of(healthy_steps(), workers=[worker(0, NOW - 600, pod="harvey-trainer-0", outstanding=[held])])
    (request,) = summary["requests"]["unanswered"]
    self.assertEqual((request["pod"], request["holder_alive"], request["age_s"]), ("harvey-trainer-0", False, 207360))
    self.assertEqual(line["held_by_dead_worker"], 3)

  def test_a_request_lost_between_push_and_pop_shows_in_the_ledger_and_queued_samples_show_too(self):  # I4
    counts = {"enqueued": 11, "popped": 10, "answered": 10}
    _, lost, _ = self.facts_of(healthy_steps(), ledger=counts)
    self.assertEqual(lost["requests"]["ledger"]["lost"], 1)
    _, queued, line = self.facts_of(healthy_steps(), ledger=counts, queued=1, samples_queued=64)
    self.assertEqual((queued["requests"]["ledger"]["lost"], queued["requests"]["samples_queued"], line["samples_queued"]), (0, 64, 64))

  def test_a_nan_gradient_names_the_op_step_and_metric(self):  # I30
    _, summary, _ = self.facts_of([*healthy_steps(), op(NOW - 5, "optim_step", nonfinite=["grad_norm:mean"], step=5)])
    first = summary["optimizer"]["nonfinite"]["first"]
    self.assertEqual((first["operation"], first["step"], first["metrics"]), ("optim_step", 5, ["grad_norm:mean"]))

  def test_a_run_that_ended_on_a_failed_create_model_names_the_error_and_where_to_read_it(self):
    _, summary, _ = self.facts_of([op(NOW - 600, "create_model", status="failed", error_type="ValueError", pod="t-0")], lifecycle="ended")
    self.assertEqual(
      (summary["steps"], summary["last_op"]["operation"], summary["failures"]["ops"][0]["error_type"]), (0, "create_model", "ValueError")
    )
    logs = summary["failures"]["logs"]
    self.assertEqual((logs["read"], logs["args"]["pod"], logs["args"]["q"]), ("logs", "t-0", "ValueError"))

  def test_relaunch_rules(self):
    first, after = worker(NOW - 1000, NOW - 905, pod_uid="u1"), {"started_at": NOW - 880, "heartbeat_at": NOW}
    holding = [{"op": "create_model", "run_id": "r", "count": 1, "since": NOW - 950}]
    cases = {
      "container restarted in place": ([first, {**first, **after}], None, 1),
      "new pod after one died holding work": ([{**first, "outstanding": holding}, {**first, **after, "pod_uid": "u2"}], None, 1),
      "clean exit": ([{**first, "phase": "exited"}, {**first, **after}], None, 0),
      "idle pod recreated, as a shared runtime's teardown does": ([first, {**first, **after, "pod_uid": "u2"}], None, 0),
      "next run on the shared runtime, same pod name": ([first, {**first, "started_at": NOW - 300, "heartbeat_at": NOW}], NOW - 500, 0),
    }
    for name, (workers, completed_at, expected) in cases.items():
      with self.subTest(name):
        _, summary, line = self.facts_of(workers=workers, completed_at=completed_at)
        self.assertEqual((line["relaunches"], "relaunches" in summary), (expected, bool(expected)))

  def test_an_oom_killed_container_is_an_incident_and_a_healthy_run_has_no_failure_blocks(self):
    oom = {"at": NOW - 60, "pod": "t-0", "reason": "OOMKilled", "message": "worker exited with code 137"}
    _, summary, _ = self.facts_of(healthy_steps(), pods=[{"name": "t-0", "incidents": [oom]}])
    self.assertEqual(summary["incidents"][0]["reason"], "OOMKilled")
    _, healthy, _ = self.facts_of(healthy_steps(), ledger={"enqueued": 5, "popped": 5, "answered": 5})
    self.assertFalse({"failures", "relaunches", "incidents", "samplers"} & set(healthy))


if __name__ == "__main__":
  unittest.main()
