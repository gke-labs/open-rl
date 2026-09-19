"""Request transport and generic state backends, sharing Redis connections."""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from functools import cache
from typing import Any

import redis as sync_redis
import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError


class RequestStore(ABC):
  """Request queues and results, keyed by request ID."""

  @abstractmethod
  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None:
    """Push a request into the tenant queue and assign to an active set."""
    pass

  @abstractmethod
  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]:
    """Block until at least 1 request is available in the active set, then return all currently queued requests."""
    pass

  @abstractmethod
  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    """Block until this model has at least 1 request, then return all queued requests for it."""
    pass

  @abstractmethod
  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    """Push a sampling request into the queue for its model."""
    pass

  @abstractmethod
  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    """Block until this model has at least 1 sampling request, then return all queued requests for it."""
    pass

  @abstractmethod
  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    """Resolve a future by its request ID."""
    pass

  @abstractmethod
  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    """Block until the future resolves or the timeout is reached."""
    pass


class InMemoryStore(RequestStore):
  def __init__(self):
    # tenant_id -> queue of requests
    self.queues: dict[str, asyncio.Queue] = {}
    # active_set_id -> list of tenant model_ids for round-robin
    self.active_tenants: dict[str, list[str]] = {}
    self.active_tenants_cv = asyncio.Condition()
    self.futures_store: dict[str, dict[str, Any]] = {}
    self.futures_events: dict[str, set[asyncio.Event]] = {}
    self.sampling_queues: dict[str, asyncio.Queue] = {}

  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None:
    model_id = req_data.get("model_id", "default")
    set_key = active_set_id or "default"

    async with self.active_tenants_cv:
      if model_id not in self.queues:
        self.queues[model_id] = asyncio.Queue()

      await self.queues[model_id].put(req_data)

      tenants_list = self.active_tenants.setdefault(set_key, [])
      if model_id not in tenants_list:
        tenants_list.append(model_id)
        self.active_tenants_cv.notify_all()

  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      while True:
        tenants_list = None
        if active_set_id:
          tenants_list = self.active_tenants.get(active_set_id)
        else:
          for t_list in self.active_tenants.values():
            if t_list:
              tenants_list = t_list
              break

        if tenants_list:
          model_id = tenants_list[0]
          queue = self.queues[model_id]

          # Drain only what was pending on entry. Requests this tenant enqueues
          # while we drain wait for its next turn, so a tenant whose producer
          # outruns the consumer cannot hold the queue open indefinitely.
          pending = queue.qsize()
          if pending:
            batch = [queue.get_nowait() for _ in range(pending)]
            # Round-robin: rotate the tenant we just served to the tail.
            tenants_list.append(tenants_list.pop(0))
            return batch
          else:
            tenants_list.pop(0)
            continue

        await self.active_tenants_cv.wait()

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Per-model full fine-tuning workers require REDIS_URL; in-memory queues cannot be shared across processes")

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    if model_id not in self.sampling_queues:
      self.sampling_queues[model_id] = asyncio.Queue()
    await self.sampling_queues[model_id].put(req_data)

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    if model_id not in self.sampling_queues:
      return []
    queue = self.sampling_queues[model_id]
    if queue.empty():
      return []
    batch = [queue.get_nowait()]
    while not queue.empty():
      batch.append(queue.get_nowait())
    return batch

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return
    self.futures_store[req_id] = result
    for event in self.futures_events.get(req_id, ()):
      event.set()

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    if req_id in self.futures_store:
      return self.futures_store[req_id]

    event = asyncio.Event()
    waiters = self.futures_events.setdefault(req_id, set())
    waiters.add(event)
    try:
      await asyncio.wait_for(event.wait(), timeout=timeout)
      return self.futures_store[req_id]
    except TimeoutError:
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    finally:
      waiters.remove(event)
      if not waiters:
        del self.futures_events[req_id]


class RedisStore(RequestStore):
  def __init__(self, client: redis.Redis):
    self.redis = client
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"

  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None:
    model_id = req_data.get("model_id", "default")
    queue_key = f"open_rl:queue:{model_id}"

    active_set = f"open_rl:active_tenants_set:{active_set_id}" if active_set_id else self.active_set
    active_list = f"open_rl:active_tenants:{active_set_id}" if active_set_id else self.active_list

    # 1. Add request to tenant-specific list
    await self.redis.rpush(queue_key, json.dumps(req_data))

    # 2. Add tenant to active set and list if not already there
    # SADD returns 1 if it was newly added, 0 if it already existed
    is_new = await self.redis.sadd(active_set, model_id)
    if is_new == 1:
      await self.redis.rpush(active_list, model_id)

  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]:
    active_set = f"open_rl:active_tenants_set:{active_set_id}" if active_set_id else self.active_set
    active_list = f"open_rl:active_tenants:{active_set_id}" if active_set_id else self.active_list

    while True:
      model_id_bytes = await self.redis.lindex(active_list, 0)
      if not model_id_bytes:
        try:
          result = await self.redis.brpoplpush(active_list, active_list, timeout=5)
        except RedisTimeoutError:
          return []
        if not result:
          return []
        model_id = result.decode() if isinstance(result, bytes) else str(result)
      else:
        model_id = model_id_bytes.decode() if isinstance(model_id_bytes, bytes) else str(model_id_bytes)

      queue_key = f"open_rl:queue:{model_id}"
      # Snapshot the depth before draining: requests this tenant enqueues while
      # we drain belong to its next turn, otherwise a producer that outruns the
      # consumer keeps the drain loop alive and never yields the head slot.
      q_len = await self.redis.llen(queue_key)
      if q_len == 0:
        await self.redis.lrem(active_list, 0, model_id)
        await self.redis.srem(active_set, model_id)
        continue

      batch = []
      for _ in range(q_len):
        item = await self.redis.lpop(queue_key)
        if not item:
          break
        batch.append(json.loads(item))

      if batch:
        # Round-robin: move the tenant we just served from the head to the tail.
        # Draining a tenant's whole batch keeps training batches intact; rotating
        # afterwards keeps a continuously-enqueueing tenant from starving peers.
        await self.redis.lmove(active_list, active_list, "LEFT", "RIGHT")
        return batch

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    queue_key = f"open_rl:queue:{model_id}"
    try:
      result = await self.redis.blpop(queue_key, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]

    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    q_len = await self.redis.llen(queue_key)
    if q_len == 0:
      await self.redis.lrem(self.active_list, 0, model_id)
      await self.redis.srem(self.active_set, model_id)

    return batch

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    queue_key = f"open_rl:sampler_queue:{model_id}"
    await self.redis.rpush(queue_key, json.dumps(req_data))

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    queue_key = f"open_rl:sampler_queue:{model_id}"
    try:
      result = await self.redis.blpop(queue_key, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]

    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    return batch

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return

    key = f"open_rl:future:{req_id}"
    # Keep the list format for existing workers, with one result per request.
    async with self.redis.pipeline(transaction=True) as pipeline:
      pipeline.rpush(key, json.dumps(result))
      pipeline.ltrim(key, -1, -1)
      pipeline.expire(key, 300)
      await pipeline.execute()

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    key = f"open_rl:future:{req_id}"
    deadline = time.monotonic() + timeout
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
      raw_result = await self.redis.lindex(key, -1)

      if raw_result:
        payload = json.loads(raw_result)
        await self.redis.expire(key, 300)
        return payload

      await asyncio.sleep(0.1)


class StateStore(ABC):
  """Application state stored as values with expiry and sets of members."""

  @abstractmethod
  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    """Store a simple string value by key, gone after ttl_seconds if given."""
    pass

  @abstractmethod
  async def get_value(self, key: str) -> str | None:
    """Fetch a string value by key."""
    pass

  @abstractmethod
  def get_value_sync(self, key: str) -> str | None:
    """Synchronously fetch a string value by key."""
    pass

  @abstractmethod
  async def delete_values(self, *keys: str) -> None:
    """Delete one or more keys."""
    pass

  @abstractmethod
  async def add_to_set(self, key: str, member: str) -> None:
    pass

  @abstractmethod
  async def remove_from_set(self, key: str, member: str) -> None:
    pass

  @abstractmethod
  async def set_members(self, key: str) -> set[str]:
    pass


class InMemoryStateStore(StateStore):
  def __init__(self):
    self.kv_store: dict[str, str] = {}
    self.expiries: dict[str, float] = {}
    self.sets: dict[str, set[str]] = {}

  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    self.kv_store[key] = value
    self.expiries.pop(key, None)
    if ttl_seconds is not None:
      self.expiries[key] = time.monotonic() + ttl_seconds

  async def get_value(self, key: str) -> str | None:
    return self.get_value_sync(key)

  def get_value_sync(self, key: str) -> str | None:
    if key in self.expiries and time.monotonic() >= self.expiries[key]:
      self.kv_store.pop(key, None)
      self.expiries.pop(key, None)
    return self.kv_store.get(key)

  async def delete_values(self, *keys: str) -> None:
    for k in keys:
      self.kv_store.pop(k, None)
      self.expiries.pop(k, None)
      self.sets.pop(k, None)

  async def add_to_set(self, key: str, member: str) -> None:
    self.sets.setdefault(key, set()).add(member)

  async def remove_from_set(self, key: str, member: str) -> None:
    self.sets.get(key, set()).discard(member)

  async def set_members(self, key: str) -> set[str]:
    return set(self.sets.get(key, set()))


class RedisStateStore(StateStore):
  def __init__(self, client: redis.Redis, sync_client: sync_redis.Redis):
    self.redis = client
    self.sync_redis = sync_client

  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    await self.redis.set(key, value, px=None if ttl_seconds is None else int(ttl_seconds * 1000))

  async def get_value(self, key: str) -> str | None:
    return await self.redis.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.sync_redis.get(key)

  async def delete_values(self, *keys: str) -> None:
    if keys:
      await self.redis.delete(*keys)

  async def add_to_set(self, key: str, member: str) -> None:
    await self.redis.sadd(key, member)

  async def remove_from_set(self, key: str, member: str) -> None:
    await self.redis.srem(key, member)

  async def set_members(self, key: str) -> set[str]:
    return set(await self.redis.smembers(key))


@cache
def _redis_client(redis_url: str) -> redis.Redis:
  return redis.from_url(redis_url, decode_responses=True, health_check_interval=2, max_connections=10000)


@cache
def get_store() -> RequestStore:
  redis_url = os.environ.get("REDIS_URL")
  return RedisStore(_redis_client(redis_url)) if redis_url else InMemoryStore()


@cache
def get_state_store() -> StateStore:
  redis_url = os.environ.get("REDIS_URL")
  if redis_url:
    return RedisStateStore(_redis_client(redis_url), sync_redis.Redis.from_url(redis_url, decode_responses=True))
  return InMemoryStateStore()
