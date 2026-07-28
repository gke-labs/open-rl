# This file contains the state management and request queue implementation for the Open-RL server, supporting both in-memory and Redis backends.

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError


class RequestStore(ABC):
  @abstractmethod
  async def put_request(self, req_data: dict[str, Any]) -> None:
    """Push a request into the global queue."""
    pass

  @abstractmethod
  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    """Push a create-model request onto the queue that starts dedicated FFT workers."""
    pass

  @abstractmethod
  async def get_requests(self) -> list[dict[str, Any]]:
    """Block until at least 1 request is available, then return all currently queued requests."""
    pass

  @abstractmethod
  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
    """Block until at least 1 worker-launch request is available, then drain that queue."""
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

  @abstractmethod
  async def set_value(self, key: str, value: str) -> None:
    """Store a simple string value by key."""
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
  async def list_jobs_metadata(self) -> list[dict[str, Any]]:
    """Retrieve metadata for all registered models/jobs."""
    pass

  @abstractmethod
  async def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
    """Retrieve metadata for a specific model_id."""
    pass


class InMemoryStore(RequestStore):
  def __init__(self):
    # tenant_id -> queue of requests
    self.queues: dict[str, asyncio.Queue] = {}
    # Simple list for round-robin
    self.active_tenants: list[str] = []
    self.active_tenants_cv = asyncio.Condition()
    self.futures_store: dict[str, dict[str, Any]] = {}
    self.futures_events: dict[str, asyncio.Event] = {}
    self.kv_store: dict[str, str] = {}

  async def list_jobs_metadata(self) -> list[dict[str, Any]]:
    jobs = []
    for key, val in self.kv_store.items():
      if key.startswith("open_rl:model_meta:"):
        try:
          data = json.loads(val)
          m_id = key.replace("open_rl:model_meta:", "")
          data["model_id"] = m_id
          jobs.append(data)
        except Exception:
          pass
    return jobs

  async def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
    raw_val = await self.get_value(f"open_rl:model_meta:{model_id}")
    if raw_val:
      try:
        data = json.loads(raw_val)
        data["model_id"] = model_id
        return data
      except Exception:
        pass
    return None

  async def update_job_metadata(self, model_id: str, updates: dict[str, Any]) -> None:
    key = f"open_rl:model_meta:{model_id}"
    raw_val = await self.get_value(key)
    data = {}
    if raw_val:
      try:
        data = json.loads(raw_val)
      except Exception:
        data = {}
    data.update(updates)
    data["updated_at"] = time.time()
    await self.set_value(key, json.dumps(data))

  async def put_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")

    async with self.active_tenants_cv:
      if model_id not in self.queues:
        self.queues[model_id] = asyncio.Queue()

      await self.queues[model_id].put(req_data)

      if model_id not in self.active_tenants:
        self.active_tenants.append(model_id)
        self.active_tenants_cv.notify()

  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def get_requests(self) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      # Block until at least one tenant is active
      while not self.active_tenants:
        await self.active_tenants_cv.wait()

      # Pop left, push right (Round Robin)
      model_id = self.active_tenants.pop(0)
      self.active_tenants.append(model_id)

      queue = self.queues[model_id]
      batch = [queue.get_nowait()]

      # Drain the rest of this tenant's queue
      while not queue.empty():
        batch.append(queue.get_nowait())

      # If completely empty, remove from rotation
      if queue.empty():
        self.active_tenants.remove(model_id)

      return batch

  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Per-model full fine-tuning workers require REDIS_URL; in-memory queues cannot be shared across processes")

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    raise RuntimeError("Sampling queues require REDIS_URL")

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Sampling queues require REDIS_URL")

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    self.futures_store[req_id] = result
    if req_id in self.futures_events:
      self.futures_events[req_id].set()

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    self.futures_store.setdefault(req_id, {"status": "pending"})

    if self.futures_store[req_id].get("status") != "pending":
      return self.futures_store[req_id]

    event = asyncio.Event()
    self.futures_events[req_id] = event

    try:
      await asyncio.wait_for(event.wait(), timeout=timeout)
      return self.futures_store.get(req_id)
    except TimeoutError:
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    finally:
      self.futures_events.pop(req_id, None)

  async def set_value(self, key: str, value: str) -> None:
    self.kv_store[key] = value

  async def get_value(self, key: str) -> str | None:
    return self.kv_store.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.kv_store.get(key)

  async def delete_values(self, *keys: str) -> None:
    for k in keys:
      self.kv_store.pop(k, None)


class RedisStore(RequestStore):
  def __init__(self, redis_url: str):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2, max_connections=10000)
    import redis as sync_redis_mod

    self.sync_redis = sync_redis_mod.Redis.from_url(redis_url, decode_responses=True)
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"
    self.worker_launch_queue = "open_rl:worker_launch_queue"

  async def put_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    queue_key = f"open_rl:queue:{model_id}"

    # 1. Add request to tenant-specific list
    await self.redis.rpush(queue_key, json.dumps(req_data))

    # 2. Add tenant to active set and list if not already there
    # SADD returns 1 if it was newly added, 0 if it already existed
    is_new = await self.redis.sadd(self.active_set, model_id)
    if is_new == 1:
      await self.redis.rpush(self.active_list, model_id)

  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    await self.redis.rpush(self.worker_launch_queue, json.dumps(req_data))

  async def get_requests(self) -> list[dict[str, Any]]:
    # BRPOPLPUSH blocks until an item is available.
    # It atomically pops the rightmost element of src, pushes it to the left of dst, and returns it.
    # Wait max 5 seconds so we can check for connection death.
    try:
      result = await self.redis.brpoplpush(self.active_list, self.active_list, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    model_id = result
    queue_key = f"open_rl:queue:{model_id}"
    batch = []

    # Drain the entire queue for this tenant non-blockingly
    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    # If the queue was empty (or we just drained it all but nothing new arrived),
    # we check the length. If it's truly empty, we scrub it from the rotation.
    # This requires a tiny Lua script or a quick transaction to ensure we don't
    # delete a tenant just as a new request is pushed.

    # Quick check:
    q_len = await self.redis.llen(queue_key)
    if q_len == 0:
      # We remove it from the list AND set
      await self.redis.lrem(self.active_list, 0, model_id)
      await self.redis.srem(self.active_set, model_id)

    return batch

  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
    try:
      result = await self.redis.blpop(self.worker_launch_queue, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]

    while True:
      item = await self.redis.lpop(self.worker_launch_queue)
      if not item:
        break
      batch.append(json.loads(item))

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
    await self.redis.rpush(key, json.dumps(result))
    await self.redis.expire(key, 300)

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    key = f"open_rl:future:{req_id}"
    deadline = time.monotonic() + timeout
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
      try:
        raw_result = await self.redis.lpop(key)
      except Exception:
        raw_result = None

      if raw_result:
        payload = json.loads(raw_result)
        await self.redis.rpush(key, raw_result)
        await self.redis.expire(key, 300)
        return payload

      await asyncio.sleep(0.1)

  async def set_value(self, key: str, value: str) -> None:
    await self.redis.set(key, value)

  async def get_value(self, key: str) -> str | None:
    return await self.redis.get(key)

  def get_value_sync(self, key: str) -> str | None:
    try:
      return self.sync_redis.get(key)
    except Exception:
      return None

  async def delete_values(self, *keys: str) -> None:
    if keys:
      await self.redis.delete(*keys)

  async def list_jobs_metadata(self) -> list[dict[str, Any]]:
    keys = await self.redis.keys("open_rl:model_meta:*")
    jobs = []
    for k in keys:
      k_str = k.decode() if isinstance(k, bytes) else str(k)
      m_id = k_str.replace("open_rl:model_meta:", "")
      raw_val = await self.redis.get(k_str)
      if raw_val:
        val_str = raw_val.decode() if isinstance(raw_val, bytes) else str(raw_val)
        try:
          data = json.loads(val_str)
          data["model_id"] = m_id
          jobs.append(data)
        except Exception:
          pass
    return jobs

  async def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
    raw_val = await self.redis.get(f"open_rl:model_meta:{model_id}")
    if raw_val:
      val_str = raw_val.decode() if isinstance(raw_val, bytes) else str(raw_val)
      try:
        data = json.loads(val_str)
        data["model_id"] = model_id
        return data
      except Exception:
        pass
    return None

  async def update_job_metadata(self, model_id: str, updates: dict[str, Any]) -> None:
    key = f"open_rl:model_meta:{model_id}"
    raw_val = await self.redis.get(key)
    data = {}
    if raw_val:
      val_str = raw_val.decode() if isinstance(raw_val, bytes) else str(raw_val)
      try:
        data = json.loads(val_str)
      except Exception:
        data = {}
    data.update(updates)
    data["updated_at"] = time.time()
    await self.redis.set(key, json.dumps(data))


# Global singleton factory
_store_instance = None


def get_store() -> RequestStore:
  global _store_instance
  if _store_instance is None:
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
      print(f"[RequestStore] Initializing Redis backend at {redis_url} with RR Tenant Queues")
      _store_instance = RedisStore(redis_url)
    else:
      print("[RequestStore] Initializing In-Memory backend with RR Tenant Queues")
      _store_instance = InMemoryStore()
  return _store_instance
