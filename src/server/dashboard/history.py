"""Durable placement history: which worker held which GPUs on which node, and when.

Recorded in the run store every few seconds while a placement is live, so the
Nodes page can draw allocation bars for the whole selected window and the
record outlives both the worker and the gateway pod.
"""

import asyncio
import json
import time
from typing import Any

SET_KEY = "open_rl:placements"
KEY_PREFIX = "open_rl:placement:"
RETENTION_SECONDS = 7 * 86400
RECORD_INTERVAL_SECONDS = 10

FIELDS = ("id", "name", "label", "run_ids", "runtime_id", "node", "pod", "role", "owner_id", "device_count", "devices")


async def record(store, placements: list[dict], now: float | None = None) -> None:
  """Upsert every live placement: first_seen on first sight, last_seen on every sight."""
  now = now or time.time()
  for placement in placements:
    if not placement.get("node"):
      continue
    key = KEY_PREFIX + placement["id"]
    previous = await store.get_value(key)
    first_seen = json.loads(previous)["first_seen"] if previous else now
    entry = {**{field: placement.get(field) for field in FIELDS}, "first_seen": first_seen, "last_seen": now}
    await store.set_value(key, json.dumps(entry), ttl_seconds=RETENTION_SECONDS)
    await store.add_to_set(SET_KEY, placement["id"])


async def read(store, since: float) -> list[dict[str, Any]]:
  """Placements seen at or after `since`, newest first. Expired keys drop out of the set as they are met."""
  found = []
  for placement_id in await store.set_members(SET_KEY):
    raw = await store.get_value(KEY_PREFIX + placement_id)
    if raw is None:
      await store.remove_from_set(SET_KEY, placement_id)
      continue
    entry = json.loads(raw)
    if entry["last_seen"] >= since:
      found.append(entry)
  found.sort(key=lambda entry: entry["first_seen"], reverse=True)
  return found


async def recorder(store, placements_now) -> None:
  """Background loop: ask for the current placements and record them."""
  while True:
    try:
      await record(store, await placements_now())
    except Exception:
      pass  # the next pass records again; a missed tick costs one interval of history
    await asyncio.sleep(RECORD_INTERVAL_SECONDS)
