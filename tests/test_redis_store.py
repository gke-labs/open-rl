# Integration tests for the Redis-backed stores. They need a real Redis: either
# set OPEN_RL_TEST_REDIS_URL, or have redis-server on PATH (a throwaway instance
# is started on a free port). Skipped entirely otherwise, so `make test` stays
# green on machines without Redis.

import asyncio
import os
import shutil
import socket
import subprocess
import time
import unittest

from server.store import RedisStore

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")
REDIS_SERVER = shutil.which("redis-server")


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


@unittest.skipUnless(TEST_REDIS_URL or REDIS_SERVER, "needs OPEN_RL_TEST_REDIS_URL or redis-server on PATH")
class RedisFutureTest(unittest.IsolatedAsyncioTestCase):
  server: subprocess.Popen | None = None
  redis_url: str

  @classmethod
  def setUpClass(cls) -> None:
    if TEST_REDIS_URL:
      cls.redis_url = TEST_REDIS_URL
      return
    port = free_port()
    cls.redis_url = f"redis://127.0.0.1:{port}"
    cls.server = subprocess.Popen(
      ["redis-server", "--port", str(port), "--save", ""],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while True:
      try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
          return
      except OSError:
        if time.monotonic() > deadline:
          raise RuntimeError("redis-server did not come up") from None
        time.sleep(0.05)

  @classmethod
  def tearDownClass(cls) -> None:
    if cls.server is not None:
      cls.server.terminate()
      cls.server.wait(timeout=10)

  def setUp(self) -> None:
    self.store = RedisStore(self.redis_url)

  async def asyncSetUp(self) -> None:
    await self.store.redis.flushdb()

  async def asyncTearDown(self) -> None:
    await self.store.redis.aclose()

  async def test_get_future_returns_already_resolved_result(self) -> None:
    await self.store.set_future("req-1", {"type": "sample", "ok": True})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample", "ok": True})

  async def test_get_future_wakes_on_resolution(self) -> None:
    async def resolve_later() -> None:
      await asyncio.sleep(0.2)
      await self.store.set_future("req-1", {"type": "sample"})

    resolver = asyncio.create_task(resolve_later())
    started = time.monotonic()
    result = await self.store.get_future("req-1", timeout=10.0)
    await resolver

    self.assertEqual(result, {"type": "sample"})
    # Woken by the publish, not by grinding through the whole long-poll window.
    self.assertLess(time.monotonic() - started, 5.0)

  async def test_result_survives_repeated_and_concurrent_reads(self) -> None:
    waiters = [asyncio.create_task(self.store.get_future("req-1", timeout=10.0)) for _ in range(5)]
    await asyncio.sleep(0.2)
    await self.store.set_future("req-1", {"type": "sample"})

    for result in await asyncio.gather(*waiters):
      self.assertEqual(result, {"type": "sample"})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})

  async def test_unresolved_future_times_out_with_try_again(self) -> None:
    result = await self.store.get_future("req-never", timeout=0.3)
    self.assertEqual(result["type"], "try_again")

  async def test_pending_markers_are_not_stored(self) -> None:
    await self.store.set_future("req-1", {"status": "pending"})
    self.assertEqual((await self.store.get_future("req-1", timeout=0.3))["type"], "try_again")


class InMemoryStoreTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    from server.store import InMemoryStore

    self.store = InMemoryStore()

  async def test_sampling_queue_put_and_get(self) -> None:
    req1 = {"model_id": "base-m1", "request_id": "r1"}
    req2 = {"model_id": "base-m1", "request_id": "r2"}
    await self.store.put_sampling_request(req1)
    await self.store.put_sampling_request(req2)

    batch = await self.store.get_sampling_requests_for_model("base-m1")
    self.assertEqual(len(batch), 2)
    self.assertEqual(batch[0]["request_id"], "r1")
    self.assertEqual(batch[1]["request_id"], "r2")

    empty_batch = await self.store.get_sampling_requests_for_model("base-m1")
    self.assertEqual(empty_batch, [])


if __name__ == "__main__":
  unittest.main()
