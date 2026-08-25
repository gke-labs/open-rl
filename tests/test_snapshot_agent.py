import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

from snapshot_agent.client import SnapshotAgentClient
from snapshot_agent.serve import SnapshotAgent, start_snapshot_agent


class RecordingRestorer:
  def __init__(self):
    self.calls: list[tuple[str, int]] = []

  def checkpoint(self, pid: int) -> None:
    self.calls.append(("checkpoint", pid))

  def restore(self, pid: int) -> None:
    self.calls.append(("restore", pid))


class BlockingRestorer(RecordingRestorer):
  def __init__(self):
    super().__init__()
    self.checkpoint_started = threading.Event()
    self.finish_checkpoint = threading.Event()
    self.restore_started = threading.Event()
    self.finish_restore = threading.Event()
    self.block_checkpoint = False
    self.block_restore = False

  def checkpoint(self, pid: int) -> None:
    super().checkpoint(pid)
    if self.block_checkpoint:
      self.checkpoint_started.set()
      self.finish_checkpoint.wait(timeout=5.0)

  def restore(self, pid: int) -> None:
    super().restore(pid)
    if self.block_restore:
      self.restore_started.set()
      self.finish_restore.wait(timeout=5.0)


class SnapshotAgentTest(unittest.IsolatedAsyncioTestCase):
  async def test_agent_grants_only_one_active_process_at_a_time(self) -> None:
    restorer = RecordingRestorer()
    agent = SnapshotAgent(restorer)
    await agent.register(101)
    await agent.register(202)

    self.assertTrue((await agent.acquire(101))["ok"])
    blocked = asyncio.create_task(agent.acquire(202))
    await asyncio.sleep(0.05)
    self.assertFalse(blocked.done())

    release = await agent.release(101)
    self.assertTrue(release["ok"])
    granted_b = await asyncio.wait_for(blocked, timeout=1.0)
    self.assertTrue(granted_b["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])
    self.assertEqual(agent.active_pid, 202)

  async def test_first_acquire_is_cold_and_later_acquire_restores_after_checkpoint(self) -> None:
    restorer = RecordingRestorer()
    agent = SnapshotAgent(restorer)
    await agent.register(101)
    await agent.register(202)

    self.assertTrue((await agent.acquire(101))["ok"])
    self.assertTrue((await agent.release(101))["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

    self.assertTrue((await agent.acquire(202))["ok"])
    self.assertTrue((await agent.release(202))["ok"])
    self.assertTrue((await agent.acquire(101))["ok"])

    self.assertEqual(restorer.calls, [("checkpoint", 101), ("checkpoint", 202), ("restore", 101)])

  async def test_release_with_no_waiters_checkpoints_process(self) -> None:
    restorer = RecordingRestorer()
    agent = SnapshotAgent(restorer)
    await agent.register(101)

    self.assertTrue((await agent.acquire(101))["ok"])
    release = await agent.release(101)

    self.assertTrue(release["ok"])
    self.assertIsNone(agent.active_pid)
    self.assertTrue(agent.processes[101].checkpointed)
    self.assertFalse(agent.processes[101].failed)
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

  async def test_waiting_acquire_is_not_granted_until_release_checkpoint_finishes(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    agent = SnapshotAgent(restorer)
    await agent.register(101)
    await agent.register(202)

    self.assertTrue((await agent.acquire(101))["ok"])
    release_a = asyncio.create_task(agent.release(101))

    checkpoint_started = await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0)
    self.assertTrue(checkpoint_started)

    acquire_b = asyncio.create_task(agent.acquire(202))
    await asyncio.sleep(0.05)
    self.assertFalse(release_a.done())
    self.assertFalse(acquire_b.done())

    restorer.finish_checkpoint.set()

    self.assertTrue((await asyncio.wait_for(release_a, timeout=1.0))["ok"])
    self.assertTrue((await asyncio.wait_for(acquire_b, timeout=1.0))["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

  async def test_checkpointed_process_is_not_granted_until_restore_finishes(self) -> None:
    restorer = BlockingRestorer()
    agent = SnapshotAgent(restorer)
    await agent.register(101)

    self.assertTrue((await agent.acquire(101))["ok"])
    self.assertTrue((await agent.release(101))["ok"])

    restorer.block_restore = True
    acquire_a = asyncio.create_task(agent.acquire(101))

    restore_started = await asyncio.to_thread(restorer.restore_started.wait, 1.0)
    self.assertTrue(restore_started)
    self.assertFalse(acquire_a.done())

    restorer.finish_restore.set()

    self.assertTrue((await asyncio.wait_for(acquire_a, timeout=1.0))["ok"])
    self.assertFalse(agent.processes[101].checkpointed)
    self.assertEqual(restorer.calls, [("checkpoint", 101), ("restore", 101)])

  async def test_unregister_waiting_process_prevents_later_grant(self) -> None:
    agent = SnapshotAgent(RecordingRestorer())
    await agent.register(101)
    await agent.register(202)

    self.assertTrue((await agent.acquire(101))["ok"])
    acquire_b = asyncio.create_task(agent.acquire(202))
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_b.done())

    self.assertTrue((await agent.unregister(202))["ok"])
    self.assertTrue((await agent.release(101))["ok"])

    result = await asyncio.wait_for(acquire_b, timeout=1.0)
    self.assertFalse(result["ok"])
    self.assertIsNone(agent.active_pid)

  async def test_duplicate_commands_return_explicit_errors(self) -> None:
    agent = SnapshotAgent(RecordingRestorer())
    await agent.register(101)

    self.assertFalse((await agent.register(101))["ok"])
    self.assertTrue((await agent.acquire(101))["ok"])
    self.assertFalse((await agent.acquire(101))["ok"])
    self.assertTrue((await agent.release(101))["ok"])
    self.assertFalse((await agent.release(101))["ok"])
    self.assertTrue((await agent.unregister(101))["ok"])
    self.assertFalse((await agent.unregister(101))["ok"])

  async def test_waiters_are_granted_in_fifo_order(self) -> None:
    agent = SnapshotAgent(RecordingRestorer())
    for pid in [101, 202, 303, 404]:
      await agent.register(pid)

    self.assertTrue((await agent.acquire(101))["ok"])

    grant_order: list[int] = []

    async def acquire_then_release(pid: int) -> None:
      await agent.acquire(pid)
      grant_order.append(pid)
      await agent.release(pid)

    waiters = []
    for pid in [303, 202, 404]:
      waiters.append(asyncio.create_task(acquire_then_release(pid)))
      await asyncio.sleep(0.01)

    self.assertTrue((await agent.release(101))["ok"])
    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)

    self.assertEqual(grant_order, [303, 202, 404])


class SnapshotAgentSocketTest(unittest.IsolatedAsyncioTestCase):
  async def test_persistent_socket_clients_alternate(self) -> None:
    restorer = RecordingRestorer()
    agent = SnapshotAgent(restorer)
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "snapshot-agent.sock")
      server = await start_snapshot_agent(agent, socket_path)
      client_a = SnapshotAgentClient(socket_path)
      client_b = SnapshotAgentClient(socket_path)
      try:
        await client_a.register(101)
        await client_b.register(202)

        async with client_a.acquire(101):
          blocked = asyncio.create_task(acquire_once(client_b, 202))
          await asyncio.sleep(0.05)
          self.assertFalse(blocked.done())

        self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), 202)
        self.assertEqual(restorer.calls, [("checkpoint", 101), ("checkpoint", 202)])
      finally:
        await client_a.close()
        await client_b.close()
        server.close()
        await server.wait_closed()

  async def test_closing_active_socket_marks_run_failed(self) -> None:
    agent = SnapshotAgent(RecordingRestorer())
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "snapshot-agent.sock")
      server = await start_snapshot_agent(agent, socket_path)
      client = SnapshotAgentClient(socket_path)
      try:
        await client.register(101)
        await client.request({"command": "ACQUIRE", "pid": 101})
        await client.close()
        await asyncio.sleep(0.05)

        self.assertIsNone(agent.active_pid)
        self.assertTrue(agent.processes[101].failed)
      finally:
        server.close()
        await server.wait_closed()


async def acquire_once(client: SnapshotAgentClient, pid: int) -> int:
  async with client.acquire(pid):
    return pid


if __name__ == "__main__":
  unittest.main()
