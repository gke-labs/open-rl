"""Idle tracking for LoRA worker pods.

A LoRA worker outlives any single adapter. It is keyed by base model and serves
every adapter derived from that base, so ``/api/v1/delete_model`` deliberately
does not stop one -- tearing it down when a single tenant finishes would take
out the others. Nothing else stopped it either, so the pod and the GPU claim it
holds stayed alive indefinitely. The Gateway now reaps workers that have gone
idle, and that needs a signal only the worker itself can produce: a request
popped off the queue an hour ago may still be running, so "nothing new was
enqueued" does not mean "nothing is happening".

Each worker publishes ``{last_activity, busy}`` under its own pod name.
``busy`` is what makes a long batch safe -- a worker mid-request is never reaped
however stale its timestamp -- and ``last_activity`` is what ages out a worker
that is genuinely doing nothing. Note that ``last_activity`` tracks the last
time the worker had *work*, not the last time it wrote: a heartbeat refreshed
every 30s while idle would never look idle.
"""

import json
import os
import time
from typing import Any

HEARTBEAT_KEY_PREFIX = "open_rl:worker_heartbeat:"

# The pod name is stamped into the container by render_lora_pod(), which already
# knows it. Outside Kubernetes there is no pod to reap, so heartbeats are off.
POD_NAME_ENV = "OPEN_RL_POD_NAME"


def heartbeat_key(pod_name: str) -> str:
  return f"{HEARTBEAT_KEY_PREFIX}{pod_name}"


class WorkerHeartbeat:
  """Publishes a worker's idle state, cheaply enough to call from a polling loop.

  Writes are throttled to ``write_interval`` while the state is unchanged, but a
  ``busy`` transition always publishes immediately: the reaper's decision hinges
  on that flag, and a stale ``True`` would keep a finished worker alive for a
  whole interval while a stale ``False`` is worse still.
  """

  def __init__(self, store: Any, pod_name: str | None, write_interval: float = 30.0):
    self._store = store
    self._pod_name = pod_name
    self._key = heartbeat_key(pod_name) if pod_name else None
    self._write_interval = write_interval
    # A worker that has never served a request is not instantly idle: it counts
    # as active from startup, so the reaper's timeout is measured from boot.
    self._last_activity = time.time()
    self._busy = False
    self._last_write = 0.0

  @property
  def enabled(self) -> bool:
    return self._key is not None

  async def touch(self, busy: bool) -> None:
    if self._key is None:
      return
    now = time.time()
    if busy:
      self._last_activity = now
    if busy == self._busy and (now - self._last_write) < self._write_interval:
      return
    self._busy = busy
    self._last_write = now
    payload = json.dumps({"last_activity": self._last_activity, "busy": busy})
    try:
      await self._store.set_value(self._key, payload)
    except Exception:
      # A worker must not die because its bookkeeping did. Losing a heartbeat
      # only risks a premature reap, and the pod is relaunched on demand.
      self._last_write = 0.0


def heartbeat_from_env(store: Any, write_interval: float = 30.0) -> WorkerHeartbeat:
  return WorkerHeartbeat(store, os.getenv(POD_NAME_ENV), write_interval=write_interval)


async def read_heartbeat(store: Any, pod_name: str) -> dict[str, Any] | None:
  """Return a worker's published heartbeat, or None if absent or unreadable."""
  try:
    raw = await store.get_value(heartbeat_key(pod_name))
  except Exception:
    return None
  if not raw:
    return None
  try:
    parsed = json.loads(raw)
  except (TypeError, ValueError):
    return None
  return parsed if isinstance(parsed, dict) else None


def idle_seconds(heartbeat: dict[str, Any] | None, fallback_start: float | None, now: float | None = None) -> float | None:
  """Seconds this worker has been idle, or None when it is busy or unknowable.

  ``fallback_start`` is the pod's start time. It covers a worker that has
  published nothing -- one running an image without heartbeat support, or one
  still loading its model -- which then ages out from boot rather than staying
  immortal. Returning None means "do not reap".
  """
  now = time.time() if now is None else now

  def from_start() -> float | None:
    return None if fallback_start is None else max(0.0, now - fallback_start)

  if heartbeat is None:
    return from_start()
  last_activity = heartbeat.get("last_activity")
  if not isinstance(last_activity, (int, float)):
    return from_start()
  # Worker pod names are deterministic, so a relaunched worker inherits its
  # predecessor's key. A heartbeat older than the pod itself is that
  # predecessor's: trusting it would either reap a worker that is still loading
  # its model or, on a stale ``busy``, keep one alive forever.
  if fallback_start is not None and float(last_activity) < fallback_start:
    return from_start()
  if heartbeat.get("busy"):
    return None
  return max(0.0, now - float(last_activity))
