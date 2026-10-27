"""Which client sessions are using which owners.

An owner is the scheduler's ownerID, the trainer and sampler pair that
serves one FFT job or one LoRA base model. The tinker client heartbeats its
session every ten seconds for as long as it runs. A session that stops is
dead, and an owner whose sessions are all dead is abandoned, so a job that
exits without calling delete_model still gives its GPUs back.

Everything lives in the store, so an API server restart keeps it:

  open_rl:session:<id>         present while the session is live. Each heartbeat
                               resets its expiry, so a silent session vanishes
                               on its own. Its value is the user_metadata the
                               client opened the session with, plus its openrl.
                               tags, the defaults for every model it creates.
  open_rl:session_record:<id>  when the session started and was last heard
                               from, its redacted metadata and the models it
                               attached. It outlives the live key, for the
                               inspection API. It is telemetry, so it is
                               written in the background and a failed write
                               is dropped.
  open_rl:owner:<owner>        the sessions using this owner's workers.
  open_rl:owners               every owner that has workers.

Nothing here is atomic. The API server holds a lock per owner around attach and
around the in_use check and the teardown that follows it, so a session cannot
attach to an owner between the check and the delete.
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

from server.store import StateStore
from server.telemetry import TTL_SECONDS, best_effort, key, log_event, redact, write


class SessionRegistry:
  def __init__(self, state: StateStore, ttl_seconds: float = 120.0):
    self.state = state
    self.ttl_seconds = ttl_seconds
    self.record_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # the API server runs as one replica

  async def heartbeat(self, session_id: str) -> None:
    await self.keep_alive(session_id)
    self.note(session_id)

  async def keep_alive(self, session_id: str) -> None:
    live_key = f"open_rl:session:{session_id}"
    await self.state.set_value(live_key, await self.state.get_value(live_key) or "{}", ttl_seconds=self.ttl_seconds)

  async def update_metadata(self, session_id: str, user_metadata: dict[str, Any]) -> None:
    await self.state.set_value(f"open_rl:session:{session_id}", json.dumps(user_metadata), ttl_seconds=self.ttl_seconds)
    self.note(session_id, metadata=user_metadata)

  @best_effort
  def note(self, session_id: str, *, model_id: str | None = None, metadata: dict | None = None) -> None:
    """A session's creation, heartbeat or model attach, kept on its record. Creation and attach also log a line; a
    heartbeat, every ten seconds, only moves last_seen_at."""
    write(self.update_record(session_id, model_id, metadata))
    if metadata is not None:
      log_event("session_created", {"session_id": session_id, "metadata": metadata}, session_id=session_id)
    if model_id:
      log_event("model_attached", {"session_id": session_id, "model_id": model_id}, run_id=model_id, session_id=session_id)

  async def update_record(self, session_id: str, model_id: str | None, metadata: dict | None) -> None:
    record_key, now = key("session_record", session_id), time.time()
    # Heartbeats and attaches rewrite the same record. One at a time, or a heartbeat drops a model attached meanwhile.
    async with self.record_locks[session_id]:
      saved = await self.state.get_value(record_key)
      record = json.loads(saved) if saved else {"session_id": session_id, "created_at": now, "model_ids": []}
      record["last_seen_at"] = now
      if metadata is not None:
        record["metadata"] = redact(metadata)
      if model_id and model_id not in record["model_ids"]:
        record["model_ids"].append(model_id)
      await self.state.set_value(record_key, json.dumps(record), ttl_seconds=TTL_SECONDS)

  async def user_metadata(self, session_id: str | None) -> dict[str, Any]:
    raw = await self.state.get_value(f"open_rl:session:{session_id}") if session_id else None
    data = json.loads(raw) if raw else {}
    # Sessions opened before metadata was stored hold "1".
    return data if isinstance(data, dict) else {}

  async def live(self, session_id: str) -> bool:
    return await self.state.get_value(f"open_rl:session:{session_id}") is not None

  async def attach(self, session_id: str, owner: str) -> None:
    """The session is using this owner's workers from now on."""
    await self.keep_alive(session_id)
    await self.state.add_to_set("open_rl:owners", owner)
    await self.state.add_to_set(f"open_rl:owner:{owner}", session_id)

  async def owners(self) -> list[str]:
    return sorted(await self.state.set_members("open_rl:owners"))

  async def in_use(self, owner: str) -> bool:
    """Whether any of the owner's sessions is still live. Drops the dead ones."""
    for session_id in await self.state.set_members(f"open_rl:owner:{owner}"):
      if not await self.live(session_id):
        await self.state.remove_from_set(f"open_rl:owner:{owner}", session_id)
    return bool(await self.state.set_members(f"open_rl:owner:{owner}"))

  async def forget(self, owner: str) -> None:
    """The owner's workers are gone."""
    await self.state.delete_values(f"open_rl:owner:{owner}")
    await self.state.remove_from_set("open_rl:owners", owner)
