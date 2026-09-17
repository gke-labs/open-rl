import asyncio
import unittest

from server.dashboard import history
from server.store import InMemoryStore

PLACEMENT = {
  "id": "w-1",
  "name": "fft-w-1",
  "label": "Qwen3-8B · abc",
  "run_ids": ["abc"],
  "runtime_id": "abc",
  "node": "n1",
  "pod": "orw-w-1",
  "role": "trainer",
  "owner_id": "abc",
  "device_count": 1,
  "devices": ["gpu-0"],
}


class PlacementHistoryTest(unittest.TestCase):
  def test_first_seen_sticks_and_last_seen_advances(self) -> None:
    store = InMemoryStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    asyncio.run(history.record(store, [PLACEMENT, {**PLACEMENT, "id": "w-2", "node": None}], now=130.0))
    entries = asyncio.run(history.read(store, since=0))
    self.assertEqual([e["id"] for e in entries], ["w-1"])
    self.assertEqual((entries[0]["first_seen"], entries[0]["last_seen"], entries[0]["devices"]), (100.0, 130.0, ["gpu-0"]))

  def test_read_filters_by_last_seen(self) -> None:
    store = InMemoryStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    self.assertEqual(asyncio.run(history.read(store, since=200.0)), [])
    self.assertEqual(len(asyncio.run(history.read(store, since=100.0))), 1)
