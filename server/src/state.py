import os
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import redis.asyncio as redis

class StateStore(ABC):
    @abstractmethod
    async def put_request(self, req_data: Dict[str, Any]) -> None:
        """Push a request into the global queue."""
        pass

    @abstractmethod
    async def get_requests(self) -> List[Dict[str, Any]]:
        """Block until at least 1 request is available, then return all currently queued requests."""
        pass

    @abstractmethod
    async def set_future(self, req_id: str, result: Dict[str, Any]) -> None:
        """Resolve a future by its request ID."""
        pass

    @abstractmethod
    async def get_future(self, req_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        """Block until the future resolves or the timeout is reached."""
        pass

class InMemoryStore(StateStore):
    def __init__(self):
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.futures_store: Dict[str, Dict[str, Any]] = {}
        self.futures_events: Dict[str, asyncio.Event] = {}

    async def put_request(self, req_data: Dict[str, Any]) -> None:
        await self.request_queue.put(req_data)

    async def get_requests(self) -> List[Dict[str, Any]]:
        # Block until at least one item is available
        req = await self.request_queue.get()
        batch = [req]
        
        # Drain the rest natively
        while not self.request_queue.empty():
            batch.append(self.request_queue.get_nowait())
            
        return batch

    async def set_future(self, req_id: str, result: Dict[str, Any]) -> None:
        self.futures_store[req_id] = result
        if req_id in self.futures_events:
            self.futures_events[req_id].set()
            
    async def get_future(self, req_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        self.futures_store.setdefault(req_id, {"status": "pending"})
        
        if self.futures_store[req_id].get("status") != "pending":
            return self.futures_store[req_id]
            
        event = asyncio.Event()
        self.futures_events[req_id] = event
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self.futures_store.get(req_id)
        except asyncio.TimeoutError:
            return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
        finally:
            self.futures_events.pop(req_id, None)

class RedisStore(StateStore):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
        self.queue_key = "open_rl:request_queue"
        
    async def put_request(self, req_data: Dict[str, Any]) -> None:
        # Pushing to the end of the list
        await self.redis.rpush(self.queue_key, json.dumps(req_data))

    async def get_requests(self) -> List[Dict[str, Any]]:
        # BRPOP blocks until an item is available at the head of the list
        # We use a 5-second timeout instead of 0 (infinity), so that if the connection
        # silently dies (e.g. pod preempted), the event loop wakes up to realize it's dead.
        result = await self.redis.blpop(self.queue_key, timeout=5)
        if not result:
            return []
            
        batch = [json.loads(result[1])]
        
        # Drain the rest of the queue non-blockingly
        while True:
            # LPOP returns None if empty
            item = await self.redis.lpop(self.queue_key)
            if not item:
                break
            batch.append(json.loads(item))
            
        return batch

    async def set_future(self, req_id: str, result: Dict[str, Any]) -> None:
        if result.get("status") == "pending":
            # Do not push dummy pending states to the Redis List.
            # blpop in get_future will simply block until a real result is pushed.
            return
            
        key = f"open_rl:future:{req_id}"
        await self.redis.rpush(key, json.dumps(result))
        # Expire the future key after 5 minutes so we don't leak memory
        await self.redis.expire(key, 300)

    async def get_future(self, req_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        key = f"open_rl:future:{req_id}"
        
        # If we poll and the key entirely does not exist, and we never recorded it,
        # we face the "Future not found" dilemma. 
        # In this abstraction, the Gateway creates the UUID, so ideally it should log it.
        # But for MVP, we just block on it. If it doesn't resolve in timeout, we return try_again.
        
        # blpop returns (key, value) or None on timeout
        # Using int(timeout) since blpop expects integer seconds
        result = await self.redis.blpop(key, timeout=max(1, int(timeout)))
        
        if result:
            payload = json.loads(result[1])
            # If the future is resolved, we want it to remain available for potential retries
            # So we push it back. A memory store keeps it forever. Redis should expire it.
            await self.redis.rpush(key, result[1])
            await self.redis.expire(key, 300)
            return payload
            
        # Timeout occurred
        return {"type": "try_again", "request_id": req_id, "queue_state": "active"}

# Global singleton factory
_store_instance = None

def get_store() -> StateStore:
    global _store_instance
    if _store_instance is None:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            print(f"[StateStore] Initializing Redis backend at {redis_url}")
            _store_instance = RedisStore(redis_url)
        else:
            print("[StateStore] Initializing In-Memory backend")
            _store_instance = InMemoryStore()
    return _store_instance
