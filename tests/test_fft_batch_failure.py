import asyncio
import contextlib
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor
from training import commands
from training.types import FFTConfig


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


class RecordingSlicer(SlicerStub):
  def __init__(self):
    self.acquired = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.acquired += 1
    yield


class LoadingWorker(BrokenWorker):
  """create_model works; wake_up still fails, so a turn taken for the load would show as an error."""

  def __init__(self):
    self.loaded = []

  def create_model(self, base_model, model_id=None, config=None):
    self.loaded.append((base_model, config.cpu_offload if config else None))


class CreateModelWithoutTurnTest(unittest.TestCase):
  def run_create_model(self, **fields) -> tuple[RecordingSlicer, LoadingWorker, dict]:
    # the wire format the API server queues: envelope keys plus the command's fields under payload
    command = commands.CreateModel(request_id="cm-1", model_id="run-a", base_model="Qwen/base", fine_tuning_type="full", **fields)
    store = BatchStore([commands.wire(command)])
    slicer, worker = RecordingSlicer(), LoadingWorker()
    with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
      proc = TrainingRequestsProcessor(store, worker, "run-a", time_slicer=slicer)
    try:
      asyncio.run(proc.run_once())
    except RuntimeError:
      pass  # a batch that fails under its turn re-raises after answering every request
    return slicer, worker, store.futures["cm-1"]

  def test_an_offloaded_full_trainer_loads_its_model_without_a_gpu_turn(self) -> None:
    slicer, worker, result = self.run_create_model()
    self.assertEqual((slicer.acquired, worker.loaded, result["type"]), (0, [("Qwen/base", True)], "model_created"))

  def test_a_trainer_that_keeps_its_model_on_the_gpu_loads_it_under_a_turn(self) -> None:
    slicer, worker, result = self.run_create_model(full_config=FFTConfig(cpu_offload=False))
    self.assertEqual((slicer.acquired, worker.loaded), (1, []))  # wake_up fails before the load, as the stub says
    self.assertEqual(result["type"], "RequestFailedResponse")


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


if __name__ == "__main__":
  unittest.main()
