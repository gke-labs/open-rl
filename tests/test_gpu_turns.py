import asyncio
import contextlib
import types
import unittest

from server import observability
from server.store import InMemoryStore


class SlicerStub:
  def __init__(self):
    self.acquired = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.acquired += 1
    yield


class GpuTurnTest(unittest.TestCase):
  def test_each_turn_is_recorded_under_the_workload(self) -> None:
    store = InMemoryStore()
    slicer = SlicerStub()
    workload = types.SimpleNamespace(name="fft-run-a-trainer")

    async def run():
      async with observability.gpu_turn(slicer, workload, store, "trainer", "run-a"):
        await asyncio.sleep(0.01)
      async with observability.gpu_turn(slicer, workload, store, "trainer", "run-a"):
        pass
      return await observability.read_turns(store, "fft-run-a-trainer")

    turns = asyncio.run(run())
    self.assertEqual(slicer.acquired, 2)
    self.assertEqual(len(turns), 2)
    self.assertEqual(turns[0]["operation"], "gpu_turn")
    self.assertEqual((turns[0]["role"], turns[0]["runtime_id"], turns[0]["workload"]), ("trainer", "run-a", "fft-run-a-trainer"))
    self.assertGreaterEqual(turns[0]["at"], turns[0]["started_at"] + 0.01)
    self.assertLessEqual(turns[0]["at"], turns[1]["started_at"])
