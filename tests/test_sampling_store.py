import asyncio
import json
import uuid

from server.sampling_store import PENDING_TTL_SECONDS, RESULT_TTL_SECONDS, SamplingStore, future_key, idempotency_key
from tests.redis_server import RedisTestCase


class SamplingStoreTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}"

  def record(self, revision: str = "", key: str | None = None) -> dict:
    r = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": key}
    return {**r, "prompt_token_ids": [1]}

  async def pending(self) -> dict[str, dict]:
    return {r["request_id"]: r for r in await self.store.pending(self.pool)}

  async def test_the_same_key_returns_the_first_id_before_and_after_its_result(self) -> None:
    first = self.record(key="session:7")
    ids = [await self.store.accept(first), await self.store.accept(self.record(key="session:7"))]
    pending = await self.pending()
    self.assertEqual((ids, list(pending)), ([first["request_id"]] * 2, [first["request_id"]]))
    await self.store.set_result(pending[first["request_id"]], {"type": "sample"})
    self.assertEqual(await self.store.accept(self.record(key="session:7")), first["request_id"])

  async def test_a_key_expires_days_after_accept_and_minutes_after_its_result(self) -> None:
    await self.store.accept(self.record(key="session:7"))
    key = idempotency_key(self.pool, "session:7")
    self.assertAlmostEqual(await self.redis.ttl(key), PENDING_TTL_SECONDS, delta=60)
    [pending] = (await self.pending()).values()
    await self.store.set_result(pending, {"type": "sample"})
    self.assertAlmostEqual(await self.redis.ttl(key), RESULT_TTL_SECONDS, delta=60)

  async def test_concurrent_commits_save_exactly_one_result_and_clear_pending(self) -> None:
    r = self.record()
    await self.store.accept(r)
    [pending] = (await self.pending()).values()
    # Open the connections first, so the commits overlap instead of queueing behind connection setup.
    await asyncio.gather(*(self.redis.ping() for _ in range(5)))
    won = await asyncio.gather(*(self.store.set_result(pending, {"type": "sample", "attempt": i}) for i in range(5)))
    saved = [json.loads(raw)["attempt"] for raw in await self.redis.lrange(future_key(r["request_id"]), 0, -1)]
    self.assertEqual((won.count(True), saved, await self.pending()), (1, [won.index(True)], {}))

  async def test_pending_keeps_accept_order_without_removal_and_a_revision_stays_pinned_until_its_last_request_commits(self) -> None:
    records = self.record("/ckpt/r1"), self.record("/ckpt/r1"), self.record("/ckpt/r2")
    for r in records:
      await self.store.accept(r)
    self.assertEqual([list(await self.pending()), list(await self.pending())], [[r["request_id"] for r in records]] * 2)
    r1, r1_again, _ = (await self.pending()).values()
    await self.store.set_result(r1, {"type": "sample"})
    self.assertEqual(await self.store.pinned_revisions(self.pool), {"/ckpt/r1", "/ckpt/r2"})
    await self.store.set_result(r1_again, {"type": "sample"})
    self.assertEqual(await self.store.pinned_revisions(self.pool), {"/ckpt/r2"})
