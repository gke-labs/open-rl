import asyncio
import json
import time
import unittest
from unittest import mock

from server import telemetry
from server.store import InMemoryStateStore


def call(result=None, error=None):
  return mock.AsyncMock(return_value=result or {}, side_effect=error)


class WorkerRecordTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStateStore()
    self.enterContext(mock.patch.object(telemetry, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.dict("os.environ", {"POD_NAME": "pod", "POD_UID": "uid", "NODE_NAME": "node"}))
    self.enterContext(mock.patch.dict(telemetry.run_steps, clear=True))

  async def saved(self, worker_id: str) -> dict:
    await asyncio.gather(*list(telemetry.writes))
    return json.loads(await self.store.get_value(telemetry.key("worker", worker_id)))

  async def recorded(self, *parts: str) -> list[dict]:
    await asyncio.gather(*list(telemetry.writes))  # writes are fire-and-forget
    return await self.store.read_samples(telemetry.key("operations", *parts))

  async def test_the_record_shows_settings_and_a_clean_exit(self) -> None:
    env = {"OPEN_RL_FINE_TUNING_TYPE": "lora", "OPEN_RL_API_KEY": "sk-private", "HOME": "/root"}
    with mock.patch.dict("os.environ", env):
      async with telemetry.serving("trainer", "run-a") as worker:
        record = await self.saved(worker.record["id"])
        self.assertTrue(telemetry.alive(record, record["heartbeat_at"]))
    self.assertEqual(record["config"]["env"], {"OPEN_RL_FINE_TUNING_TYPE": "lora"})
    record = await self.saved(worker.record["id"])
    self.assertEqual(record["phase"], "exited")
    self.assertFalse(telemetry.alive(record, record["heartbeat_at"]))
    self.assertNotIn("sk-private", json.dumps(record))

  async def test_the_record_stays_fresh_while_model_loading_blocks_the_event_loop(self) -> None:
    with mock.patch.object(telemetry, "HEARTBEAT_SECONDS", 0.02):
      async with telemetry.serving("trainer", "run-a") as worker:
        blocked_at = time.time()
        time.sleep(0.3)  # loading weights holds the loop, and the heartbeat must not need it
        record = json.loads(self.store.get_value_sync(telemetry.key("worker", worker.record["id"])))
    self.assertGreater(record["heartbeat_at"] - blocked_at, 0.2)

  async def test_popped_requests_stay_outstanding_and_are_counted_until_answered(self) -> None:
    async with telemetry.serving("trainer", "base") as worker:
      telemetry.popped([{"request_id": f"f{i}", "op": "forward_backward", "model_id": "run-a"} for i in range(3)] + [{"op": "no id"}])
      telemetry.popped([{"request_id": "s1", "model_id": "base", "logical_run_id": "run-b"}])
      # A shared trainer's delete_model was never counted as enqueued, so it is not outstanding either.
      telemetry.popped([{"request_id": "d1", "op": "delete_model", "model_id": "run-a"}])
      telemetry.answered("f0")
      groups = {(g["op"], g["run_id"]): g["count"] for g in json.loads(worker.snapshot())["outstanding"]}
    self.assertEqual(groups, {("forward_backward", "run-a"): 2, ("unknown", "run-b"): 1})
    self.assertEqual(await self.store.read_counts(telemetry.key("ledger", "run-a")), {"popped": 3, "answered": 1})

  async def test_an_op_raises_as_it_would_and_only_rank_zero_records_its_failure(self) -> None:
    request = {"model_id": "run-a", "op": "train", "payload": "private input"}
    with self.assertRaisesRegex(ValueError, "private error"):
      await telemetry.observe_operation(request, "trainer", "run-a", call(error=ValueError("private error")))
    with self.assertRaises(asyncio.CancelledError):
      await telemetry.observe_operation(request, "trainer", "run-a", call(error=asyncio.CancelledError()))
    with mock.patch.dict("os.environ", {"RANK": "1"}):
      await telemetry.observe_operation(request, "trainer", "run-a", call())
    (sample,) = await self.recorded("run-a")
    self.assertEqual((sample["status"], sample["error_type"]), ("failed", "ValueError"))
    self.assertNotIn("private", repr(sample))

  async def test_trainer_ops_carry_their_step_and_nonfinite_metrics_and_sampling_never_evicts_them(self) -> None:
    await self.store.set_value(telemetry.key("steps", "r"), "5")  # a restarted trainer resumes the run's count
    failed, nan = {"type": "RequestFailedResponse"}, {"metrics": {"loss": 1.5, "grad_norm": float("nan"), "kl": float("inf")}}
    for op, result in (("forward_backward", nan), ("optim_step", None), ("optim_step", failed), ("forward_backward", None)):
      await telemetry.observe_operation({"model_id": "r", "op": op}, "trainer", "r", call(result))
    await telemetry.observe_operation({"model_id": "r", "op": "forward_backward", "forward_only": True}, "trainer", "r", call())
    with mock.patch.object(telemetry, "SAMPLE_LIMIT", 3):
      for i in range(6):  # a shared sampler's requests name the base model's queue and the run they sample for
        request = {"op": "sample", "model_id": "base", "logical_run_id": "r", "request_id": f"s{i}"}
        telemetry.op_finished(request, "sampler", "base", 0.0, "EngineDied" if i % 2 else None)
    trainer = await self.recorded("r")
    self.assertEqual(
      [(s["operation"], s["status"], s["step"]) for s in trainer],
      [("forward_backward", "succeeded", 5), ("optim_step", "succeeded", 5), ("optim_step", "failed", 6)]
      + [("forward_backward", "succeeded", 6), ("forward", "succeeded", 6)],
    )
    self.assertEqual((trainer[0]["metrics"], trainer[0]["nonfinite"]), ({"loss": 1.5}, ["grad_norm", "kl"]))  # JSON has no NaN
    self.assertEqual(await self.store.get_value(telemetry.key("steps", "r")), "6")


if __name__ == "__main__":
  unittest.main()


class PlacementRecordTest(unittest.TestCase):
  def test_a_placement_keeps_its_pods_termination_after_the_pod_status_drops_it(self) -> None:
    workload = {
      "metadata": {"name": "w"},
      "spec": {"role": "trainer", "modelID": "m"},
      "status": {"placedAt": "2026-10-21T06:00:00Z", "podName": "p"},
    }
    oom = {"at": telemetry.epoch("2026-10-21T06:30:00Z"), "container": "worker", "reason": "OOMKilled", "exit_code": 137, "signal": None}
    first = telemetry.merge_placements([], {"w": telemetry.placement(workload, {"p": [oom]})}, time.time())
    later = telemetry.merge_placements(first, {"w": telemetry.placement(workload, {})}, time.time())
    self.assertEqual(later[0]["terminations"], [oom])
    self.assertIsNone(later[0]["released_at"])
