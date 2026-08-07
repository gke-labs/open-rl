"""Teardown of idle LoRA worker pods.

FFT workers are stopped explicitly: ``/api/v1/delete_model`` enqueues a shutdown
sentinel and the pods exit. LoRA workers deliberately do not get that sentinel,
because one worker serves every adapter derived from its base model and stopping
it when a single tenant finishes would take out the others. The consequence was
that nothing stopped them at all -- the pods ran forever, and because the claim
reconciler correctly refuses to reclaim a claim a live pod holds, their GPU
claims leaked permanently.

This closes that loop from the Gateway side. A worker publishes its own idle
state (see ``server.worker_heartbeat``); the Gateway groups the pods by base
model and deletes a group once *every* pod in it has been idle past the timeout.
The grouping is the important part: a trainer and its sampler take turns, so a
sampler that has been quiet for an hour may still be halfway through a job whose
trainer is busy. Reaping per-pod would tear that job in half.

Deleting the pods is all this does. The existing claim reconciler notices the
freed claims on its next pass and deletes them.
"""

import asyncio
import os
import traceback
from typing import Any

from server.worker_heartbeat import heartbeat_key, idle_seconds, read_heartbeat

IDLE_TIMEOUT_ENV = "OPEN_RL_LORA_WORKER_IDLE_TIMEOUT_SECONDS"
CHECK_INTERVAL_ENV = "OPEN_RL_LORA_WORKER_IDLE_CHECK_INTERVAL_SECONDS"
STARTUP_GRACE_ENV = "OPEN_RL_LORA_WORKER_STARTUP_GRACE_SECONDS"
DEFAULT_IDLE_TIMEOUT_SECONDS = 1800.0
DEFAULT_CHECK_INTERVAL_SECONDS = 300.0
# How long a worker may stay silent after its pod starts before that silence is
# read as "idle" rather than "still coming up". It has to cover a multi-GB image
# pull plus the process reaching its first heartbeat write.
DEFAULT_STARTUP_GRACE_SECONDS = 900.0


async def select_idle_pods(
  store: Any,
  pods: list[dict[str, Any]],
  timeout_seconds: float,
  now: float | None = None,
  startup_grace: float = DEFAULT_STARTUP_GRACE_SECONDS,
) -> list[dict[str, Any]]:
  """Pods belonging to a base-model group in which every member is idle past ``timeout_seconds``.

  A pod whose idle time is unknown -- busy, still within its startup grace, or
  lacking both a heartbeat and a start time -- disqualifies its whole group.
  Unknown means "do not reap".
  """
  groups: dict[str, list[dict[str, Any]]] = {}
  for pod in pods:
    base_model = pod.get("base_model")
    if not base_model:
      # Without a group key we cannot tell which pods share this one's job, so
      # there is no safe way to reap it.
      continue
    groups.setdefault(base_model, []).append(pod)

  reapable: list[dict[str, Any]] = []
  for members in groups.values():
    idle_group = True
    for pod in members:
      heartbeat = await read_heartbeat(store, pod["name"])
      idle = idle_seconds(heartbeat, pod.get("start_time"), now=now, startup_grace=startup_grace)
      if idle is None or idle < timeout_seconds:
        idle_group = False
        break
    if idle_group:
      reapable.extend(members)
  return reapable


async def reap_idle_lora_workers(
  manager: Any,
  store: Any,
  timeout_seconds: float,
  startup_grace: float = DEFAULT_STARTUP_GRACE_SECONDS,
) -> list[str]:
  """Delete every LoRA worker whose base-model group has gone idle. Returns the pod names deleted."""
  pods = await asyncio.to_thread(manager.list_lora_worker_pods)
  if not pods:
    return []

  reaped: list[str] = []
  for pod in await select_idle_pods(store, pods, timeout_seconds, startup_grace=startup_grace):
    name = pod["name"]
    try:
      await asyncio.to_thread(manager.delete_pod, name)
    except Exception:
      traceback.print_exc()
      continue
    reaped.append(name)
    # The replacement pod reuses this name and so this key. It would overwrite
    # the entry on startup anyway, but not until its model is loaded, and
    # idle_seconds() would have to fall back to the pod start time until then.
    try:
      await store.delete_values(heartbeat_key(name))
    except Exception:
      pass
  return reaped


async def run_lora_idle_reaper(
  manager: Any,
  store: Any,
  interval: float,
  timeout_seconds: float,
  startup_grace: float = DEFAULT_STARTUP_GRACE_SECONDS,
) -> None:
  while True:
    await asyncio.sleep(interval)
    try:
      reaped = await reap_idle_lora_workers(manager, store, timeout_seconds, startup_grace=startup_grace)
      if reaped:
        print(f"[GATEWAY] Reaped {len(reaped)} idle LoRA worker pod(s): {', '.join(reaped)}")
    except asyncio.CancelledError:
      raise
    except Exception:
      traceback.print_exc()


def start_lora_idle_reaper(manager: Any | None, store: Any) -> asyncio.Task | None:
  """Start the reaper when the worker manager runs pods we can list (Kubernetes mode only)."""
  if manager is None or not hasattr(manager, "list_lora_worker_pods"):
    return None
  timeout_seconds = float(os.getenv(IDLE_TIMEOUT_ENV, str(DEFAULT_IDLE_TIMEOUT_SECONDS)))
  if timeout_seconds <= 0:
    print(f"[GATEWAY] Idle LoRA worker teardown disabled ({IDLE_TIMEOUT_ENV} <= 0)")
    return None
  interval = float(os.getenv(CHECK_INTERVAL_ENV, str(DEFAULT_CHECK_INTERVAL_SECONDS)))
  if interval <= 0:
    print(f"[GATEWAY] Idle LoRA worker teardown disabled ({CHECK_INTERVAL_ENV} <= 0)")
    return None
  startup_grace = max(0.0, float(os.getenv(STARTUP_GRACE_ENV, str(DEFAULT_STARTUP_GRACE_SECONDS))))
  print(f"[GATEWAY] Idle LoRA worker teardown after {timeout_seconds:.0f}s, checked every {interval:.0f}s, {startup_grace:.0f}s startup grace")
  return asyncio.create_task(run_lora_idle_reaper(manager, store, interval, timeout_seconds, startup_grace=startup_grace))
