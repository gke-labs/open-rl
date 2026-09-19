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
from unittest.mock import patch

import redis as sync_redis
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from server import store as stores
from server.model_metadata import (
  TrainingModelMetadata,
  get_model_metadata,
  get_model_metadata_sync,
  persist_model_metadata,
  update_model_metadata,
)
from server.store import InMemoryStateStore, InMemoryStore, RedisStateStore, RedisStore

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
    client = redis.from_url(self.redis_url, decode_responses=True)
    self.store = RedisStore(client)
    self.state = RedisStateStore(client, sync_redis.Redis.from_url(self.redis_url, decode_responses=True))

  async def asyncSetUp(self) -> None:
    await self.store.redis.flushdb()

  async def asyncTearDown(self) -> None:
    await self.store.redis.aclose()
    self.state.sync_redis.close()

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
    # Returns as soon as a polling read finds the result.
    self.assertLess(time.monotonic() - started, 5.0)

  async def test_result_survives_repeated_and_concurrent_reads(self) -> None:
    waiters = [asyncio.create_task(self.store.get_future("req-1", timeout=10.0)) for _ in range(5)]
    await asyncio.sleep(0.2)
    await self.store.set_future("req-1", {"type": "sample"})

    for result in await asyncio.gather(*waiters):
      self.assertEqual(result, {"type": "sample"})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})

  async def test_read_preserves_legacy_list_and_renews_expiry(self) -> None:
    key = "open_rl:future:req-1"
    previous = '{"type": "sample", "ok": false}'
    raw = '{"type": "sample", "ok": true}'
    await self.store.redis.rpush(key, previous, raw)
    await self.store.redis.expire(key, 30)

    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample", "ok": True})
    self.assertEqual(await self.store.redis.lrange(key, 0, -1), [previous, raw])
    self.assertGreater(await self.store.redis.ttl(key), 290)

  async def test_repeated_resolution_replaces_the_result(self) -> None:
    await self.store.set_future("req-1", {"type": "first"})
    await self.store.set_future("req-1", {"type": "replacement"})
    for _ in range(2):
      self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "replacement"})
    self.assertEqual(await self.store.redis.llen("open_rl:future:req-1"), 1)

  async def test_cancelled_reader_does_not_remove_result(self) -> None:
    await self.store.set_future("req-1", {"type": "sample"})
    read = asyncio.Event()
    execute_command = self.store.redis.execute_command

    async def pause_after_read(*args, **kwargs):
      result = await execute_command(*args, **kwargs)
      if args[0] in {"LINDEX", "LPOP"}:
        read.set()
        await asyncio.Event().wait()
      return result

    # Cancel after Redis has executed the read but before get_future can
    # continue. A destructive read loses the result at this boundary.
    with patch.object(self.store.redis, "execute_command", side_effect=pause_after_read):
      reader = asyncio.create_task(self.store.get_future("req-1", timeout=1.0))
      try:
        await asyncio.wait_for(read.wait(), timeout=1.0)
      finally:
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await reader

    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})

  async def test_redis_failure_is_not_reported_as_pending(self) -> None:
    with (
      patch.object(self.store.redis, "execute_command", side_effect=RedisConnectionError("unavailable")),
      self.assertRaises(RedisConnectionError),
    ):
      await self.store.get_future("req-1", timeout=0.2)

  async def test_unresolved_future_times_out_with_try_again(self) -> None:
    result = await self.store.get_future("req-never", timeout=0.3)
    self.assertEqual(result["type"], "try_again")

  async def test_pending_markers_are_not_stored(self) -> None:
    await self.store.set_future("req-1", {"status": "pending"})
    self.assertEqual((await self.store.get_future("req-1", timeout=0.3))["type"], "try_again")

  async def test_get_requests_rotates_between_tenants(self) -> None:
    for i in range(3):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")
      await self.store.put_request({"model_id": "tenant-b", "request_id": f"b{i}"}, active_set_id="base-1")

    served = []
    for _ in range(2):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])

    self.assertEqual(served, ["tenant-a", "tenant-b"])

  async def test_busy_tenant_does_not_starve_its_peer(self) -> None:
    await self.store.put_request({"model_id": "busy", "request_id": "b0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "quiet", "request_id": "q0"}, active_set_id="base-1")

    # The busy tenant keeps enqueueing after each turn. Without rotation it holds
    # the head of the active list forever and 'quiet' is never served.
    served = []
    for i in range(4):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])
      await self.store.put_request({"model_id": "busy", "request_id": f"b{i + 1}"}, active_set_id="base-1")

    self.assertIn("quiet", served)

  async def test_get_requests_drains_only_the_depth_present_on_entry(self) -> None:
    for i in range(2):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")

    batch = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual([r["request_id"] for r in batch], ["a0", "a1"])

  async def test_values_expire_and_sets_hold_members(self) -> None:
    await self.state.set_value("k", "v", ttl_seconds=60)
    self.assertEqual(await self.state.get_value("k"), "v")
    await self.state.set_value("k", "v", ttl_seconds=0.2)
    await asyncio.sleep(0.5)
    self.assertIsNone(await self.state.get_value("k"))

    await self.state.add_to_set("s", "a")
    await self.state.add_to_set("s", "b")
    await self.state.remove_from_set("s", "a")
    self.assertEqual(await self.state.set_members("s"), {"b"})
    self.assertEqual(await self.state.set_members("missing"), set())


class InMemoryStoreTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()

  async def test_future_wakes_all_waiters_without_registration(self) -> None:
    waiters = [asyncio.create_task(self.store.get_future("req-1", timeout=1.0)) for _ in range(3)]
    await asyncio.sleep(0)
    await self.store.set_future("req-1", {"type": "sample"})

    self.assertEqual(await asyncio.gather(*waiters), [{"type": "sample"}] * 3)
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})
    self.assertEqual(self.store.futures_events, {})

  async def test_timed_out_waiter_does_not_disconnect_other_waiters(self) -> None:
    short = asyncio.create_task(self.store.get_future("req-1", timeout=0.01))
    long = asyncio.create_task(self.store.get_future("req-1", timeout=1.0))
    self.assertEqual((await short)["type"], "try_again")
    await self.store.set_future("req-1", {"type": "sample"})

    self.assertEqual(await long, {"type": "sample"})
    self.assertEqual(self.store.futures_events, {})

  async def test_cancelled_waiter_does_not_disconnect_other_waiters(self) -> None:
    cancelled = asyncio.create_task(self.store.get_future("req-1", timeout=1.0))
    waiting = asyncio.create_task(self.store.get_future("req-1", timeout=1.0))
    await asyncio.sleep(0)
    cancelled.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await cancelled
    await self.store.set_future("req-1", {"type": "sample"})

    self.assertEqual(await waiting, {"type": "sample"})
    self.assertEqual(self.store.futures_events, {})

  async def test_pending_markers_do_not_store_or_overwrite_results(self) -> None:
    await self.store.set_future("req-1", {"status": "pending"})
    self.assertNotIn("req-1", self.store.futures_store)
    self.assertEqual((await self.store.get_future("req-1", timeout=0.01))["type"], "try_again")
    self.assertEqual(self.store.futures_events, {})

    await self.store.set_future("req-1", {"type": "sample"})
    await self.store.set_future("req-1", {"status": "pending"})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})

  async def test_repeated_resolution_replaces_the_result(self) -> None:
    await self.store.set_future("req-1", {"type": "first"})
    await self.store.set_future("req-1", {"type": "replacement"})
    for _ in range(2):
      self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "replacement"})

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

  async def test_get_requests_rotates_between_tenants(self) -> None:
    for i in range(3):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")
      await self.store.put_request({"model_id": "tenant-b", "request_id": f"b{i}"}, active_set_id="base-1")

    served = []
    for _ in range(2):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])

    self.assertEqual(served, ["tenant-a", "tenant-b"])

  async def test_busy_tenant_does_not_starve_its_peer(self) -> None:
    await self.store.put_request({"model_id": "busy", "request_id": "b0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "quiet", "request_id": "q0"}, active_set_id="base-1")

    served = []
    for i in range(4):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])
      await self.store.put_request({"model_id": "busy", "request_id": f"b{i + 1}"}, active_set_id="base-1")

    self.assertIn("quiet", served)

  async def test_get_requests_drains_only_the_depth_present_on_entry(self) -> None:
    for i in range(2):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")

    batch = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual([r["request_id"] for r in batch], ["a0", "a1"])


class InMemoryStateStoreTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.state = InMemoryStateStore()

  async def test_metadata_updates_preserve_existing_fields(self) -> None:
    await self.state.set_value("open_rl:model_meta:model-1", '{"base_model": "base", "total_steps_completed": 1}')
    with patch("server.model_metadata.time.time", return_value=123.0):
      await update_model_metadata(self.state, "model-1", {"total_steps_completed": 2})
    self.assertEqual(
      await get_model_metadata(self.state, "model-1"),
      {"model_id": "model-1", "base_model": "base", "total_steps_completed": 2, "updated_at": 123.0},
    )

  async def test_metadata_updates_refuse_missing_or_corrupt_records(self) -> None:
    key = "open_rl:model_meta:model-1"
    self.assertIsNone(await get_model_metadata(self.state, "model-1"))
    with self.assertRaises(KeyError):
      await update_model_metadata(self.state, "model-1", {"status": "completed"})
    self.assertIsNone(await self.state.get_value(key))
    for raw in ("invalid JSON", "[]", "null", "{}", '{"base_model": null}'):
      with self.subTest(raw=raw):
        await self.state.set_value(key, raw)
        with self.assertRaises(ValueError):
          await get_model_metadata(self.state, "model-1")
        with self.assertRaises(ValueError):
          await update_model_metadata(self.state, "model-1", {"status": "completed"})
        self.assertEqual(await self.state.get_value(key), raw)

  async def test_persisted_model_is_readable_by_sync_worker_lookup(self) -> None:
    metadata = TrainingModelMetadata(base_model="base", created_at=123.0)
    model_id = await persist_model_metadata(self.state, metadata)
    self.assertEqual(await get_model_metadata(self.state, model_id), {**metadata.to_dict(), "model_id": model_id})
    self.assertEqual(get_model_metadata_sync(self.state, model_id), await get_model_metadata(self.state, model_id))

  async def test_values_expire_and_sets_hold_members(self) -> None:
    await self.state.set_value("k", "v", ttl_seconds=60)
    self.assertEqual(await self.state.get_value("k"), "v")
    await self.state.set_value("k", "v", ttl_seconds=0.2)
    await asyncio.sleep(0.5)
    self.assertIsNone(await self.state.get_value("k"))

    await self.state.add_to_set("s", "a")
    await self.state.add_to_set("s", "b")
    await self.state.remove_from_set("s", "a")
    self.assertEqual(await self.state.set_members("s"), {"b"})
    self.assertEqual(await self.state.set_members("missing"), set())


class StoreFactoryTest(unittest.TestCase):
  def setUp(self) -> None:
    self.clear_factories()
    self.addCleanup(self.clear_factories)

  def clear_factories(self) -> None:
    stores.get_store.cache_clear()
    stores.get_state_store.cache_clear()
    stores._redis_client.cache_clear()

  def test_redis_backends_share_one_async_client(self) -> None:
    with (
      patch.dict(os.environ, {"REDIS_URL": "redis://shared"}),
      patch("server.store.redis.from_url") as async_client,
      patch("server.store.sync_redis.Redis.from_url") as sync_client,
    ):
      transport = stores.get_store()
      state = stores.get_state_store()
      self.assertIs(transport.redis, state.redis)
      self.assertIs(state.sync_redis, sync_client.return_value)
      self.assertIs(stores.get_store(), transport)
      self.assertIs(stores.get_state_store(), state)
      async_client.assert_called_once()
      sync_client.assert_called_once()

  def test_memory_backends_are_independent_singletons(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": ""}):
      transport = stores.get_store()
      state = stores.get_state_store()
      self.assertIsInstance(transport, InMemoryStore)
      self.assertIsInstance(state, InMemoryStateStore)
      self.assertIs(stores.get_store(), transport)
      self.assertIs(stores.get_state_store(), state)


if __name__ == "__main__":
  unittest.main()
