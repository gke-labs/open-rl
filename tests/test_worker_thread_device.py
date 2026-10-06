"""Every executor thread must hold this rank's CUDA device (run33)."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from server import training_requests_processor as processor

CONCURRENCY = 4
RANK = 2


class WorkerThreadDeviceTest(unittest.TestCase):
  def setUp(self) -> None:
    # Thread name -> the device set_device was called with on that thread.
    self.pinned: dict[str, int] = {}
    self.lock = threading.Lock()

    def record_set_device(device) -> None:
      with self.lock:
        self.pinned[threading.current_thread().name] = int(device)

    self.enterContext(patch.object(processor.torch.cuda, "is_available", return_value=True))
    self.enterContext(patch.object(processor.torch.cuda, "set_device", record_set_device))
    self.enterContext(patch.object(processor, "local_rank", return_value=RANK))

  def test_every_thread_that_runs_work_holds_this_ranks_device(self) -> None:
    # The barrier forces the pool to spawn CONCURRENCY threads instead of
    # reusing one warm thread.
    barrier = threading.Barrier(CONCURRENCY)
    used: set[str] = set()

    def task() -> None:
      barrier.wait(timeout=30)
      with self.lock:
        used.add(threading.current_thread().name)

    async def main() -> None:
      processor.pin_worker_threads_to_this_rank()
      await asyncio.gather(*(asyncio.to_thread(task) for _ in range(CONCURRENCY)))

    asyncio.run(main())
    self.assertEqual(len(used), CONCURRENCY)
    self.assertEqual({name: self.pinned.get(name) for name in used}, dict.fromkeys(used, RANK))
    self.assertEqual(self.pinned.get("MainThread"), RANK)

  def test_a_cpu_only_process_is_left_alone(self) -> None:
    async def main() -> None:
      processor.pin_worker_threads_to_this_rank()
      await asyncio.to_thread(lambda: None)

    with patch.object(processor.torch.cuda, "is_available", return_value=False):
      asyncio.run(main())
    self.assertEqual(self.pinned, {})


if __name__ == "__main__":
  unittest.main()
