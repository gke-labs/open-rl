import asyncio
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from server import lora_reaper
from server.store import InMemoryStore
from server.worker_heartbeat import WorkerHeartbeat, heartbeat_from_env, heartbeat_key, idle_seconds, read_heartbeat


class WorkerHeartbeatTest(unittest.IsolatedAsyncioTestCase):
  async def test_disabled_without_a_pod_name(self) -> None:
    store = InMemoryStore()
    heartbeat = WorkerHeartbeat(store, pod_name=None)

    self.assertFalse(heartbeat.enabled)
    await heartbeat.touch(busy=True)
    self.assertEqual(store.kv_store, {})

  async def test_reads_the_pod_name_from_the_environment(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_POD_NAME": "open-rl-trainer-qwen-1"}):
      self.assertTrue(heartbeat_from_env(InMemoryStore()).enabled)
    with patch.dict(os.environ, {}, clear=True):
      self.assertFalse(heartbeat_from_env(InMemoryStore()).enabled)

  async def test_busy_transitions_publish_immediately(self) -> None:
    store = InMemoryStore()
    # A write interval long enough that throttling would swallow every write
    # after the first if transitions were not exempt.
    heartbeat = WorkerHeartbeat(store, "pod-a", write_interval=3600.0)

    await heartbeat.touch(busy=False)
    await heartbeat.touch(busy=True)
    published = await read_heartbeat(store, "pod-a")
    self.assertTrue(published["busy"])

    await heartbeat.touch(busy=False)
    published = await read_heartbeat(store, "pod-a")
    self.assertFalse(published["busy"])

  async def test_unchanged_state_is_throttled(self) -> None:
    store = InMemoryStore()
    heartbeat = WorkerHeartbeat(store, "pod-a", write_interval=3600.0)

    await heartbeat.touch(busy=False)
    first = await read_heartbeat(store, "pod-a")
    # An idle poll loop calls touch() many times a second; only the first should
    # reach the store.
    with patch.object(store, "set_value", side_effect=AssertionError("throttled write escaped")):
      for _ in range(10):
        await heartbeat.touch(busy=False)
    self.assertEqual(await read_heartbeat(store, "pod-a"), first)

  async def test_a_failed_write_does_not_propagate(self) -> None:
    store = InMemoryStore()
    heartbeat = WorkerHeartbeat(store, "pod-a", write_interval=3600.0)

    with patch.object(store, "set_value", side_effect=RuntimeError("redis down")):
      await heartbeat.touch(busy=True)
    # The next attempt is not throttled: the state it wanted to publish never landed.
    await heartbeat.touch(busy=True)
    self.assertTrue((await read_heartbeat(store, "pod-a"))["busy"])

  async def test_unreadable_heartbeats_read_as_absent(self) -> None:
    store = InMemoryStore()
    self.assertIsNone(await read_heartbeat(store, "missing"))

    await store.set_value(heartbeat_key("pod-a"), "not json")
    self.assertIsNone(await read_heartbeat(store, "pod-a"))

    await store.set_value(heartbeat_key("pod-a"), json.dumps([1, 2]))
    self.assertIsNone(await read_heartbeat(store, "pod-a"))


class IdleSecondsTest(unittest.TestCase):
  def test_busy_workers_are_never_idle(self) -> None:
    self.assertIsNone(idle_seconds({"last_activity": 0.0, "busy": True}, fallback_start=0.0, now=10_000.0))

  def test_idle_is_measured_from_last_activity(self) -> None:
    self.assertEqual(idle_seconds({"last_activity": 900.0, "busy": False}, fallback_start=0.0, now=1000.0), 100.0)

  def test_a_worker_without_a_heartbeat_ages_from_its_pod_start(self) -> None:
    # An image predating heartbeat support, or a worker still loading its model.
    self.assertEqual(idle_seconds(None, fallback_start=400.0, now=1000.0), 600.0)
    self.assertEqual(idle_seconds({"busy": False}, fallback_start=400.0, now=1000.0), 600.0)

  def test_unknowable_idleness_is_not_reapable(self) -> None:
    self.assertIsNone(idle_seconds(None, fallback_start=None))

  def test_a_heartbeat_older_than_its_pod_belongs_to_the_previous_pod(self) -> None:
    # Pod names are deterministic, so a relaunched worker inherits the key. A
    # stale busy=True would make the new pod immortal...
    stale = {"last_activity": 10.0, "busy": True}
    self.assertEqual(idle_seconds(stale, fallback_start=500.0, now=1000.0), 500.0)
    # ...and a stale idle timestamp would reap it before it finished booting.
    stale_idle = {"last_activity": 10.0, "busy": False}
    self.assertEqual(idle_seconds(stale_idle, fallback_start=990.0, now=1000.0), 10.0)


class _FakeManager:
  """Stands in for KubernetesWorkerManager over the two calls the reaper makes."""

  def __init__(self, pods: list[dict[str, Any]]):
    self.pods = pods
    self.deleted: list[str] = []

  def list_lora_worker_pods(self) -> list[dict[str, Any]]:
    return list(self.pods)

  def delete_pod(self, pod_name: str) -> None:
    self.deleted.append(pod_name)


class _LocalManager:
  """Stands in for LocalWorkerManager, which runs no pods to reap."""


def _pod(name: str, base_model: str, start_time: float = 0.0) -> dict[str, Any]:
  return {"name": name, "base_model": base_model, "role": "trainer", "claim": "claim-a", "phase": "Running", "start_time": start_time}


class ReapIdleLoraWorkersTest(unittest.IsolatedAsyncioTestCase):
  async def _publish(self, store: InMemoryStore, pod_name: str, last_activity: float, busy: bool = False) -> None:
    await store.set_value(heartbeat_key(pod_name), json.dumps({"last_activity": last_activity, "busy": busy}))

  async def test_reaps_a_group_only_when_every_member_is_idle(self) -> None:
    store = InMemoryStore()
    now = 10_000.0
    manager = _FakeManager(
      [
        _pod("trainer-a", "qwen3-0-6b"),
        _pod("sampler-a", "qwen3-0-6b"),
        _pod("trainer-b", "qwen3-8b"),
        _pod("sampler-b", "qwen3-8b"),
      ]
    )
    await self._publish(store, "trainer-a", last_activity=now - 3600)
    await self._publish(store, "sampler-a", last_activity=now - 3600)
    # The 8B sampler has been quiet for an hour, but its trainer is mid-step:
    # reaping the pair would tear a live job in half.
    await self._publish(store, "trainer-b", last_activity=now - 5, busy=True)
    await self._publish(store, "sampler-b", last_activity=now - 3600)

    selected = await lora_reaper.select_idle_pods(store, manager.pods, timeout_seconds=1800.0, now=now)

    self.assertEqual(sorted(p["name"] for p in selected), ["sampler-a", "trainer-a"])

  async def test_a_group_inside_the_timeout_is_left_alone(self) -> None:
    store = InMemoryStore()
    now = 10_000.0
    pods = [_pod("trainer-a", "qwen3-0-6b"), _pod("sampler-a", "qwen3-0-6b")]
    await self._publish(store, "trainer-a", last_activity=now - 60)
    await self._publish(store, "sampler-a", last_activity=now - 3600)

    self.assertEqual(await lora_reaper.select_idle_pods(store, pods, timeout_seconds=1800.0, now=now), [])

  async def test_ungrouped_pods_are_never_reaped(self) -> None:
    store = InMemoryStore()
    # Without a base-model label there is no way to know which pods share this
    # one's job, so idleness alone is not enough to act on.
    pods = [{"name": "trainer-a", "base_model": None, "start_time": 0.0}]

    self.assertEqual(await lora_reaper.select_idle_pods(store, pods, timeout_seconds=1.0, now=10_000.0), [])

  async def test_reap_deletes_the_pods_and_clears_their_heartbeats(self) -> None:
    store = InMemoryStore()
    manager = _FakeManager([_pod("trainer-a", "qwen3-0-6b"), _pod("sampler-a", "qwen3-0-6b")])
    await self._publish(store, "trainer-a", last_activity=0.0)
    await self._publish(store, "sampler-a", last_activity=0.0)

    reaped = await lora_reaper.reap_idle_lora_workers(manager, store, timeout_seconds=1.0)

    self.assertEqual(sorted(reaped), ["sampler-a", "trainer-a"])
    self.assertEqual(sorted(manager.deleted), ["sampler-a", "trainer-a"])
    # A replacement pod reuses the name; leaving the key behind would hand it its
    # predecessor's idle timestamp.
    self.assertIsNone(await read_heartbeat(store, "trainer-a"))

  async def test_a_failed_delete_is_not_reported_as_reaped(self) -> None:
    store = InMemoryStore()
    manager = _FakeManager([_pod("trainer-a", "qwen3-0-6b")])
    await self._publish(store, "trainer-a", last_activity=0.0)

    def boom(pod_name: str) -> None:
      raise RuntimeError("API server unavailable")

    manager.delete_pod = boom
    self.assertEqual(await lora_reaper.reap_idle_lora_workers(manager, store, timeout_seconds=1.0), [])

  async def test_no_pods_means_no_api_calls(self) -> None:
    manager = _FakeManager([])
    self.assertEqual(await lora_reaper.reap_idle_lora_workers(manager, InMemoryStore(), timeout_seconds=1.0), [])


class LoraIdleReaperLoopTest(unittest.IsolatedAsyncioTestCase):
  async def test_loop_runs_on_its_interval_and_survives_failures(self) -> None:
    store = InMemoryStore()
    manager = _FakeManager([])
    calls = 0

    def list_pods() -> list[dict[str, Any]]:
      nonlocal calls
      calls += 1
      raise RuntimeError("API server unavailable")

    manager.list_lora_worker_pods = list_pods
    task = asyncio.create_task(lora_reaper.run_lora_idle_reaper(manager, store, interval=0.01, timeout_seconds=1.0))
    await asyncio.sleep(0.1)
    task.cancel()

    self.assertGreater(calls, 1, "a transient API error must not kill the only thing reaping workers")

  async def test_starts_only_for_managers_that_run_pods(self) -> None:
    store = InMemoryStore()
    self.assertIsNone(lora_reaper.start_lora_idle_reaper(None, store))
    self.assertIsNone(lora_reaper.start_lora_idle_reaper(_LocalManager(), store))

    task = lora_reaper.start_lora_idle_reaper(_FakeManager([]), store)
    self.assertIsNotNone(task)
    task.cancel()

  async def test_can_be_disabled(self) -> None:
    store = InMemoryStore()
    for env in ({lora_reaper.IDLE_TIMEOUT_ENV: "0"}, {lora_reaper.CHECK_INTERVAL_ENV: "0"}):
      with patch.dict(os.environ, env):
        self.assertIsNone(lora_reaper.start_lora_idle_reaper(_FakeManager([]), store))


if __name__ == "__main__":
  unittest.main()
