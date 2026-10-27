import asyncio
import contextlib
import json
import os
import unittest
from unittest.mock import patch

from server import telemetry
from server.store import InMemoryStateStore, InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor


class SlicerStub:
  faulted = None

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    yield

  async def register(self, workload):
    return {"ok": True}

  async def unregister(self, workload):
    return {"ok": True}

  async def close(self):
    pass


class BrokenWorker:
  cpu_offload = True

  def wake_up(self):
    raise RuntimeError("CUDA out of memory")

  def sleep(self):
    pass


class BatchStore(InMemoryStore):
  def __init__(self, batch):
    super().__init__()
    self.batch = batch
    self.futures = {}

  async def get_requests_for_model(self, model_id):
    batch, self.batch = self.batch, []
    return batch

  async def set_result(self, request_id, result):
    self.futures[request_id] = result


def processor(store, slicer):
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    return TrainingRequestsProcessor(store, BrokenWorker(), "run-a", time_slicer=slicer)


class FFTBatchFailureTest(unittest.TestCase):
  def test_a_batch_that_fails_before_answering_fails_every_request(self) -> None:
    store = BatchStore([{"request_id": "fb-1", "op": "forward_backward"}, {"request_id": "os-1", "op": "optim_step"}])
    proc = processor(store, SlicerStub())
    with self.assertRaises(RuntimeError):
      asyncio.run(proc.run_once())
    self.assertEqual(sorted(store.futures), ["fb-1", "os-1"])
    for result in store.futures.values():
      self.assertEqual(result["type"], "RequestFailedResponse")
      self.assertIn("CUDA out of memory", result["error_message"])

  def test_a_worker_the_slicer_could_not_park_exits_without_unregistering(self) -> None:
    store = BatchStore([{"request_id": "sv-1", "op": "save_state"}])
    slicer = SlicerStub()
    proc = processor(store, slicer)
    exits = []

    async def record_exit(unregister=True):
      exits.append(unregister)

    async def handled(request):
      slicer.faulted = "checkpoint failed for workload run-a; it still holds the accelerator and must exit"
      return request["request_id"], {"type": "SaveWeightsResponse"}

    proc.handle_request = handled
    proc.exit_gracefully = record_exit
    asyncio.run(proc.run_once())
    self.assertEqual(store.futures["sv-1"]["type"], "SaveWeightsResponse")
    self.assertEqual(exits, [False])


class FailedAnswerRecordTest(unittest.TestCase):
  """Every failed answer a client gets has one failed record, including failures before the op runs."""

  def setUp(self):
    self.state = InMemoryStateStore()
    self.enterContext(patch.object(telemetry, "get_state_store", return_value=self.state))

  def test_a_wake_or_parse_failure_records_every_request_it_failed(self) -> None:
    wake = processor(BatchStore([{"request_id": "fb-1", "op": "forward_backward", "model_id": "run-a"}]), SlicerStub())
    parse = processor(BatchStore([{"request_id": "bad-1", "op": "optim_step", "model_id": "run-a", "adam_params": "not a dict"}]), None)

    async def serve():
      async with telemetry.serving("trainer", "run-a"):
        for proc in (wake, parse):
          with contextlib.suppress(RuntimeError):
            await proc.run_once()
      return await self.state.read_samples(telemetry.key("operations", "run-a"))

    records = [(r["request_id"], r["operation"], r["status"]) for r in asyncio.run(serve())]
    self.assertEqual(records, [("fb-1", "forward_backward", "failed"), ("bad-1", "optim_step", "failed")])

  def test_an_exit_after_shutdown_says_the_process_exited_and_lands_its_writes(self) -> None:
    proc = processor(BatchStore([{"request_id": "s-1", "op": "shutdown_workers", "model_id": "run-a"}]), SlicerStub())
    increment = self.state.increment_field

    async def slow_increment(*args):
      await asyncio.sleep(0.05)
      return await increment(*args)

    async def serve():
      async with telemetry.serving("trainer", "run-a") as worker:
        telemetry.record("run-a", "popped")  # still in flight when the process exits
        with patch("os._exit", side_effect=SystemExit), self.assertRaises(SystemExit):
          await proc.run_once()
        record = json.loads(await self.state.get_value(telemetry.key("worker", worker.record["id"])) or "{}")
        return record.get("phase"), (await self.state.read_counts(telemetry.key("ledger", "run-a"))).get("popped")

    with patch.object(self.state, "increment_field", slow_increment):
      self.assertEqual(asyncio.run(serve()), ("exited", 1))


if __name__ == "__main__":
  unittest.main()
