import asyncio
import contextlib
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStateStore, InMemoryStore
from server.training_requests_processor import FFTTrainingRequestsProcessor


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

  async def set_future(self, request_id, result):
    self.futures[request_id] = result


def processor(store, slicer):
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    return FFTTrainingRequestsProcessor(store, InMemoryStateStore(), BrokenWorker(), "run-a", slicer)


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
    store = BatchStore([{"request_id": "sv-1", "op": "save_weights"}])
    slicer = SlicerStub()
    proc = processor(store, slicer)
    exits = []

    async def record_exit(unregister=True):
      exits.append(unregister)

    async def handled(request, model_id):
      slicer.faulted = "checkpoint failed for workload run-a; it still holds the accelerator and must exit"
      return request["request_id"], {"type": "SaveWeightsResponse"}

    proc.handle_request = handled
    proc.exit_gracefully = record_exit
    asyncio.run(proc.run_once())
    self.assertEqual(store.futures["sv-1"]["type"], "SaveWeightsResponse")
    self.assertEqual(exits, [False])


if __name__ == "__main__":
  unittest.main()
