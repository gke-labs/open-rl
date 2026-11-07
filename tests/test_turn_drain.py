"""A time-sliced trainer's turn takes in the requests that queue for its model while it waits or runs, so an optim_step
sent right after its forward_backward runs in the same turn instead of waiting for a turn of its own."""

import asyncio
import contextlib
import os
import sys
import time
import unittest
from unittest.mock import patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor
from tests.test_fft_batch_failure import SlicerStub
from tests.test_telemetry import ScriptedTrainer, request


class QueueStore(InMemoryStore):
  """The model's queue as a list of batches: get_requests_for_model takes the next one, queued counts what waits."""

  def __init__(self, *batches):
    super().__init__()
    self.batches = [list(b) for b in batches]
    self.futures = {}
    self.fetches = 0

  async def get_requests_for_model(self, model_id):
    self.fetches += 1
    return self.batches.pop(0) if self.batches else []

  async def queued(self, model_id):
    return len(self.batches[0]) if self.batches else 0

  async def set_result(self, request_id, result):
    self.futures[request_id] = result


class CountingSlicer(SlicerStub):
  def __init__(self):
    self.turns = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.turns += 1
    yield


def serve(store, slicer):
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    processor = TrainingRequestsProcessor(store, ScriptedTrainer(), "run-a", time_slicer=slicer)
  asyncio.run(processor.run_once())
  return processor


class TurnDrainTest(unittest.TestCase):
  def test_a_request_queued_during_the_turn_runs_in_it(self) -> None:
    store, slicer = QueueStore([request("fb-1", "forward_backward", data=[])], [request("os-1", "optim_step")]), CountingSlicer()
    serve(store, slicer)
    self.assertEqual(slicer.turns, 1)
    self.assertEqual(sorted(store.futures), ["fb-1", "os-1"])
    self.assertNotEqual(store.futures["os-1"].get("type"), "RequestFailedResponse")

  def test_the_drain_keeps_going_while_requests_keep_queueing(self) -> None:
    batches = [[request(f"fb-{i}", "forward_backward", data=[])] for i in range(3)] + [[request("os-1", "optim_step")]]
    store, slicer = QueueStore(*batches), CountingSlicer()
    serve(store, slicer)
    self.assertEqual((slicer.turns, sorted(store.futures)), (1, ["fb-0", "fb-1", "fb-2", "os-1"]))

  def test_no_budget_means_no_drain(self) -> None:
    store, slicer = QueueStore([request("fb-1", "forward_backward", data=[])], [request("os-1", "optim_step")]), CountingSlicer()
    # the class's own module: another test re-imports this module, so a reference taken at import may be stale
    with patch.object(sys.modules[TrainingRequestsProcessor.__module__], "TURN_DRAIN_SECONDS", 0):
      serve(store, slicer)
    self.assertEqual(sorted(store.futures), ["fb-1"], "the optim_step waits for the next batch")
    self.assertEqual(len(store.batches), 1)

  def test_a_save_ends_the_drain(self) -> None:
    # The save in the first batch runs after the turn's GPU ops; nothing queued behind it may run before it.
    first = [request("os-1", "optim_step"), request("sv-1", "save_weights_for_sampler", path="x")]
    store, slicer = QueueStore(first, [request("fb-2", "forward_backward", data=[])]), CountingSlicer()
    with patch.object(TrainingRequestsProcessor, "handle_request", new=_answer):
      serve(store, slicer)
    self.assertEqual(store.fetches, 1)
    self.assertEqual(len(store.batches), 1, "fb-2 stays queued for the next batch")

  def test_an_empty_queue_is_not_waited_on(self) -> None:
    store, slicer = QueueStore([request("fb-1", "forward_backward", data=[])]), CountingSlicer()
    serve(store, slicer)
    self.assertEqual(store.fetches, 1, "an empty queue is checked with queued(), never with the blocking pop")

  def test_a_shutdown_taken_while_draining_is_honoured(self) -> None:
    store = QueueStore([request("fb-1", "forward_backward", data=[])], [{"request_id": "sd", "op": "shutdown_workers", "model_id": "run-a"}])
    with patch.object(TrainingRequestsProcessor, "exit_gracefully") as exit_gracefully:
      serve(store, CountingSlicer())
    exit_gracefully.assert_called_once()


async def _answer(self, r):
  """Answers any request, so a save needs no real checkpoint."""
  return r["request_id"], {"type": "ok"}


if __name__ == "__main__":
  unittest.main()


class LongBatchTest(unittest.TestCase):
  def test_the_budget_starts_after_the_turns_own_batch(self) -> None:
    # A forward_backward that runs longer than the whole budget must still let its queued optim_step join the turn.
    class SlowTrainer(ScriptedTrainer):
      def forward_backward(self, *args, **kwargs):
        time.sleep(0.2)
        return super().forward_backward(*args, **kwargs)

    store, slicer = QueueStore([request("fb-1", "forward_backward", data=[])], [request("os-1", "optim_step")]), CountingSlicer()
    with (
      patch.object(sys.modules[TrainingRequestsProcessor.__module__], "TURN_DRAIN_SECONDS", 0.05),
      patch.dict(os.environ, {"REDIS_URL": "redis://test"}),
    ):
      processor = TrainingRequestsProcessor(store, SlowTrainer(), "run-a", time_slicer=slicer)
      asyncio.run(processor.run_once())
    self.assertEqual((slicer.turns, sorted(store.futures)), (1, ["fb-1", "os-1"]))
