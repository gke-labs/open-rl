"""A lane's occupancy on a time-sliced GPU is its turns; an op recorded outside every turn ran off the GPU (an offloaded save)."""

import asyncio
import unittest
from unittest import mock

from server.dashboard import fleet
from server.store import InMemoryStateStore
from server.telemetry import key

PLACEMENT = {"workload": "fft-r-trainer", "placed_at": 50.0, "released_at": None, "model_id": "r", "pod": "t-0", "role": "trainer"}
PLACEMENT |= {"node": "n1", "gpu_uuids": ["GPU-1"], "terminations": []}
RUN = {"run_id": "r", "runtime_id": "r", "name": "Run", "model": "Qwen/base", "created_at": 50.0, "completed_at": None}


def worker(turns):
  return {"id": "w", "pod": "t-0", "role": "trainer", "started_at": 50.0, "heartbeat_at": 190.0, "phase": "ready", "turns": turns, "waits": []}


class LaneOccupancyTest(unittest.TestCase):
  def activity(self, turns):
    store = InMemoryStateStore()
    save = {"at": 125.0, "started_at": 124.0, "operation": "save_weights_for_sampler", "status": "succeeded", "role": "trainer", "metrics": {}}
    asyncio.run(store.append_sample(key("operations", "r"), save, 100, None))
    fleet.cache.clear()
    with mock.patch.object(fleet, "get_state_store", return_value=store):
      state = {"observed_at": 200.0, "placements": [PLACEMENT], "runs": [RUN], "workers": [worker(turns)]}
      return asyncio.run(fleet.activity(state, 0.0, 200.0))[0]

  def test_an_op_outside_every_turn_is_shown_but_does_not_occupy_the_gpu(self):
    lane = self.activity(turns=[[100.0, 110.0]])
    self.assertEqual((lane["exact"], lane["intervals"]), (True, [[100.0, 110.0]]))
    self.assertEqual(lane["ops"], {"save_weights_for_sampler": [[124.0, 125.0]]})

  def test_without_turns_the_ops_stand_in_for_occupancy(self):
    lane = self.activity(turns=[])
    self.assertEqual((lane["exact"], lane["intervals"]), (False, [[124.0, 125.0]]))


if __name__ == "__main__":
  unittest.main()
