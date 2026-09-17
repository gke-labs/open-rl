"""Every executor thread must hold this rank's CUDA device.

Torch's current device is thread-local and every worker call is handed off with
asyncio.to_thread, so the device has to be established per thread. Left to the
default executor, only whichever pool thread happened to run create_model ever
saw set_device; threads the pool spawned later still pointed at cuda:0. That is
invisible on rank 0, where cuda:0 is the correct answer, and wrong on every
other rank, so it survives any single-GPU test run and surfaced in production
as a NCCL fault. These cases need no GPU: they patch set_device and assert on
which threads it reached.
"""

import asyncio
import threading
import unittest
import unittest.mock

from training import distributed

# How many tasks are forced to run at once. A threading.Barrier makes the pool
# genuinely spawn this many threads rather than reusing one warm thread.
CONCURRENCY = 4
BARRIER_TIMEOUT = 30.0
RANK = 2


class WorkerThreadDeviceTest(unittest.TestCase):
  def setUp(self) -> None:
    # thread name -> device it was told to use. Recording the thread, not just
    # the argument, is the whole point: the old code called set_device with the
    # right device on the wrong thread.
    self.pinned: dict[str, int] = {}
    self.lock = threading.Lock()

    def record_set_device(device) -> None:
      index = device.index if hasattr(device, "index") else int(device)
      with self.lock:
        self.pinned[threading.current_thread().name] = index

    self.enterContext(unittest.mock.patch.object(distributed.torch.cuda, "is_available", return_value=True))
    self.enterContext(unittest.mock.patch.object(distributed.torch.cuda, "set_device", record_set_device))
    self.enterContext(unittest.mock.patch.object(distributed, "is_distributed", return_value=True))
    self.enterContext(unittest.mock.patch.object(distributed, "local_rank", return_value=RANK))

  def burst(self) -> set[str]:
    """Run CONCURRENCY tasks that must overlap, and return the threads used."""
    barrier = threading.Barrier(CONCURRENCY)
    used: set[str] = set()

    def note_thread() -> None:
      with self.lock:
        used.add(threading.current_thread().name)

    def task() -> None:
      # Nobody leaves until all CONCURRENCY tasks have arrived, so the pool
      # cannot satisfy this with a single reused thread.
      barrier.wait(timeout=BARRIER_TIMEOUT)
      note_thread()

    async def main() -> None:
      distributed.pin_executor_threads()
      # A sequential call first: the steady state that looks healthy.
      await asyncio.to_thread(note_thread)
      await asyncio.gather(*(asyncio.to_thread(task) for _ in range(CONCURRENCY)))

    asyncio.run(main())
    return used

  def test_every_thread_that_runs_work_holds_this_ranks_device(self) -> None:
    used = self.burst()

    self.assertGreaterEqual(len(used), CONCURRENCY, "the burst did not actually spread across threads")
    unpinned = sorted(name for name in used if name not in self.pinned)
    self.assertEqual(unpinned, [], f"threads ran worker code without a device: {unpinned}")
    wrong = sorted((name, self.pinned[name]) for name in used if self.pinned[name] != RANK)
    self.assertEqual(wrong, [], f"threads pinned to the wrong device: {wrong}")

  def test_the_calling_thread_is_pinned_too(self) -> None:
    # Not everything goes through to_thread; the loop thread itself touches
    # torch, so it cannot be left on whatever device it started with.
    async def main() -> None:
      distributed.pin_executor_threads()

    asyncio.run(main())
    self.assertEqual(self.pinned.get("MainThread"), RANK)

  def test_the_unpinned_default_executor_is_the_failure_mode(self) -> None:
    # The control. Same burst without the fix: the pool's threads never hear
    # about the device, so anything allocated on them lands on cuda:0.
    barrier = threading.Barrier(CONCURRENCY)
    used: set[str] = set()

    def task() -> None:
      barrier.wait(timeout=BARRIER_TIMEOUT)
      with self.lock:
        used.add(threading.current_thread().name)

    async def main() -> None:
      await asyncio.gather(*(asyncio.to_thread(task) for _ in range(CONCURRENCY)))

    asyncio.run(main())

    self.assertGreaterEqual(len(used), CONCURRENCY)
    self.assertEqual(sorted(name for name in used if name in self.pinned), [])

  def test_a_single_process_is_left_alone(self) -> None:
    # One process, one GPU: no device to pin and no reason to replace the
    # executor. The CPU test suite itself runs in this state.
    with unittest.mock.patch.object(distributed, "is_distributed", return_value=False):

      async def main() -> None:
        distributed.pin_executor_threads()
        return await asyncio.to_thread(lambda: threading.current_thread().name)

      name = asyncio.run(main())

    self.assertEqual(self.pinned, {})
    self.assertNotIn(name, self.pinned)


if __name__ == "__main__":
  unittest.main()
