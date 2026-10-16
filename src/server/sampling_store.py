"""Durable sampling records for sampling pools. Each pool's pending requests are one
Redis stream, read without removal. Results land in the same `open_rl:future:<id>`
list that `retrieve_future` already reads."""

import json
from typing import Any

import redis.asyncio as redis
from redis.exceptions import WatchError

RESULT_TTL_SECONDS = 300


def stream_key(pool: str) -> str:
  return f"open_rl:sampling:{pool}"


def idempotency_key(pool: str, key: str) -> str:
  return f"open_rl:sampling:{pool}:idem:{key}"


def future_key(request_id: str) -> str:
  return f"open_rl:future:{request_id}"


class SamplingStore:
  def __init__(self, client: redis.Redis):
    self.redis = client

  async def accept(self, record: dict[str, Any]) -> str:
    """The new request's id, or the id already stored under its idempotency key."""
    stream, fields = stream_key(record["pool"]), {"record": json.dumps(record)}
    if not record.get("idempotency_key"):
      await self.redis.xadd(stream, fields)
      return record["request_id"]
    key = idempotency_key(record["pool"], record["idempotency_key"])
    async with self.redis.pipeline() as pipe:
      while True:
        try:
          await pipe.watch(key)
          if existing := await pipe.get(key):
            return existing
          pipe.multi()
          pipe.set(key, record["request_id"])
          pipe.xadd(stream, fields)
          await pipe.execute()
          return record["request_id"]
        except WatchError:
          continue

  async def pending(self, pool: str, limit: int | None = 1000, after: str | None = None) -> list[dict[str, Any]]:
    """Pending records in stream order, only those newer than stream id `after` when given."""
    entries = await self.redis.xrange(stream_key(pool), min=f"({after}" if after else "-", count=limit)
    return [{**json.loads(fields["record"]), "entry_id": entry_id} for entry_id, fields in entries]

  async def wait_for_new(self, pool: str, after: str, timeout_ms: int) -> None:
    """Returns once a request newer than stream id `after` arrives, or after timeout_ms."""
    await self.redis.xread({stream_key(pool): after}, count=1, block=timeout_ms)

  async def commit(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    """Save the first final result and drop the pending entry. False if it was already committed."""
    stream, future, entry_id = stream_key(record["pool"]), future_key(record["request_id"]), record["entry_id"]
    async with self.redis.pipeline() as pipe:
      while True:
        try:
          # Every commit writes the result key, so watching it lets only one through.
          await pipe.watch(future)
          if await pipe.exists(future) or not await pipe.xrange(stream, entry_id, entry_id):
            return False
          pipe.multi()
          pipe.rpush(future, json.dumps(result))
          pipe.expire(future, RESULT_TTL_SECONDS)
          pipe.xdel(stream, entry_id)
          if key := record.get("idempotency_key"):
            pipe.expire(idempotency_key(record["pool"], key), RESULT_TTL_SECONDS)
          await pipe.execute()
          return True
        except WatchError:
          continue

  async def pinned_revisions(self, pool: str) -> set[str]:
    """Revisions that pending requests still name, so their weights must stay."""
    return {record["revision"] for record in await self.pending(pool, limit=None)} - {""}
