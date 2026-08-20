import asyncio
import inspect
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from accel_timeslicer.checkpoint import CudaCheckpointRestorer
from accel_timeslicer.llmd import LlmDCheckpointRestorer
from accel_timeslicer.serve import start_tcp_time_slicer, start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import WorkloadRef


class RecordingRestorer:
  def __init__(self):
    self.calls: list[tuple[str, WorkloadRef]] = []

  def checkpoint(self, target: WorkloadRef) -> None:
    self.calls.append(("checkpoint", target))

  def restore(self, target: WorkloadRef) -> None:
    self.calls.append(("restore", target))

  def labels(self) -> list[tuple[str, str, str]]:
    return [(op, target.job_id, target.group) for op, target in self.calls]

  def simple_labels(self) -> list[tuple[str, str]]:
    return [(op, target.job_id) for op, target in self.calls]


class BlockingRestorer(RecordingRestorer):
  def __init__(self):
    super().__init__()
    self.checkpoint_started = threading.Event()
    self.finish_checkpoint = threading.Event()
    self.restore_started = threading.Event()
    self.finish_restore = threading.Event()
    self.block_checkpoint = False
    self.block_restore = False

  def checkpoint(self, target: WorkloadRef) -> None:
    super().checkpoint(target)
    if self.block_checkpoint:
      self.checkpoint_started.set()
      self.finish_checkpoint.wait(timeout=5.0)

  def restore(self, target: WorkloadRef) -> None:
    super().restore(target)
    if self.block_restore:
      self.restore_started.set()
      self.finish_restore.wait(timeout=5.0)


class NoSnapshotRestorer(RecordingRestorer):
  def checkpoint(self, target: WorkloadRef) -> bool:
    super().checkpoint(target)
    return False


class SingleNodeTimeSlicerTest(unittest.IsolatedAsyncioTestCase):
  async def test_agent_grants_only_one_active_process_at_a_time(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    blocked = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    await asyncio.sleep(0.05)
    self.assertFalse(blocked.done())

    release = await agent.release(WorkloadRef(job_id="101"))
    self.assertTrue(release["ok"])
    granted_b = await asyncio.wait_for(blocked, timeout=1.0)
    self.assertTrue(granted_b["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
    self.assertEqual(agent.active_workload, "shared-accelerator:202")

  async def test_first_acquire_is_cold_and_later_acquire_restores_after_checkpoint(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202"), ("restore", "101")])

  async def test_release_with_no_waiters_checkpoints_process(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    release = await agent.release(WorkloadRef(job_id="101"))

    self.assertTrue(release["ok"])
    self.assertIsNone(agent.active_workload)
    self.assertTrue(agent.workloads["shared-accelerator:101"].checkpointed)
    self.assertFalse(agent.workloads["shared-accelerator:101"].failed)
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

  async def test_release_without_snapshot_does_not_restore_later(self) -> None:
    restorer = NoSnapshotRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    workload = WorkloadRef(job_id="101")

    await agent.register(workload)
    self.assertTrue((await agent.acquire(workload))["ok"])
    self.assertTrue((await agent.release(workload))["ok"])
    self.assertFalse(agent.workloads[workload.key].checkpointed)

    self.assertTrue((await agent.acquire(workload))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

  async def test_waiting_acquire_is_not_granted_until_release_checkpoint_finishes(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    release_a = asyncio.create_task(agent.release(WorkloadRef(job_id="101")))

    checkpoint_started = await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0)
    self.assertTrue(checkpoint_started)

    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    await asyncio.sleep(0.05)
    self.assertFalse(release_a.done())
    self.assertFalse(acquire_b.done())

    restorer.finish_checkpoint.set()

    self.assertTrue((await asyncio.wait_for(release_a, timeout=1.0))["ok"])
    self.assertTrue((await asyncio.wait_for(acquire_b, timeout=1.0))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

  async def test_checkpointed_process_is_not_granted_until_restore_finishes(self) -> None:
    restorer = BlockingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])

    restorer.block_restore = True
    acquire_a = asyncio.create_task(agent.acquire(WorkloadRef(job_id="101")))

    restore_started = await asyncio.to_thread(restorer.restore_started.wait, 1.0)
    self.assertTrue(restore_started)
    self.assertFalse(acquire_a.done())

    restorer.finish_restore.set()

    self.assertTrue((await asyncio.wait_for(acquire_a, timeout=1.0))["ok"])
    self.assertFalse(agent.workloads["shared-accelerator:101"].checkpointed)
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("restore", "101")])

  async def test_unregister_waiting_process_prevents_later_grant(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_b.done())

    self.assertTrue((await agent.unregister(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])

    result = await asyncio.wait_for(acquire_b, timeout=1.0)
    self.assertFalse(result["ok"])
    self.assertIsNone(agent.active_workload)

  async def test_duplicate_commands_return_explicit_errors(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    await agent.register(WorkloadRef(job_id="101"))

    self.assertTrue((await agent.register(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.unregister(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.unregister(WorkloadRef(job_id="101")))["ok"])

  async def test_waiters_are_granted_in_fifo_order(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    for pid in [101, 202, 303, 404]:
      await agent.register(WorkloadRef(job_id=str(pid)))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

    grant_order: list[int] = []

    async def acquire_then_release(pid: int) -> None:
      workload = WorkloadRef(job_id=str(pid))
      await agent.acquire(workload)
      grant_order.append(pid)
      await agent.release(workload)

    waiters = []
    for pid in [303, 202, 404]:
      waiters.append(asyncio.create_task(acquire_then_release(pid)))
      await asyncio.sleep(0.01)

    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)

    self.assertEqual(grant_order, [303, 202, 404])

  async def test_register_can_use_stable_snapshot_id_for_backend_calls(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    workload = WorkloadRef(job_id="job-a")
    await agent.register(workload)

    self.assertTrue((await agent.acquire(workload))["ok"])
    self.assertTrue((await agent.release(workload))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "job-a")])


class SingleNodeTimeSlicerSocketTest(unittest.IsolatedAsyncioTestCase):
  async def test_persistent_socket_clients_alternate(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "accel-timeslicer.sock")
      server = await start_time_slicer(agent, socket_path)
      client_a = SocketTimeSlicerClient(socket_path)
      client_b = SocketTimeSlicerClient(socket_path)
      try:
        await client_a.register(WorkloadRef(job_id="101"))
        await client_b.register(WorkloadRef(job_id="202"))

        async with client_a.acquire(WorkloadRef(job_id="101")):
          blocked = asyncio.create_task(acquire_once(client_b, WorkloadRef(job_id="202")))
          await asyncio.sleep(0.05)
          self.assertFalse(blocked.done())

        self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), "202")
        self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202")])
      finally:
        await client_a.close()
        await client_b.close()
        server.close()
        await server.wait_closed()

  async def test_closing_active_socket_marks_run_failed(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "accel-timeslicer.sock")
      server = await start_time_slicer(agent, socket_path)
      client = SocketTimeSlicerClient(socket_path)
      try:
        await client.register(WorkloadRef(job_id="101"))
        await client.request({"command": "ACQUIRE", "job_id": "101"})
        await client.close()
        await asyncio.sleep(0.05)

        self.assertIsNone(agent.active_workload)
        self.assertTrue(agent.workloads["shared-accelerator:101"].failed)
      finally:
        server.close()
        await server.wait_closed()

  async def test_tcp_clients_share_single_node_agent(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    server = await start_tcp_time_slicer(agent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client_a = SocketTimeSlicerClient(host="127.0.0.1", port=port)
    client_b = SocketTimeSlicerClient(host="127.0.0.1", port=port)
    try:
      await client_a.register(WorkloadRef(job_id="101"))
      await client_b.register(WorkloadRef(job_id="202"))

      async with client_a.acquire(WorkloadRef(job_id="101")):
        blocked = asyncio.create_task(acquire_once(client_b, WorkloadRef(job_id="202")))
        await asyncio.sleep(0.05)
        self.assertFalse(blocked.done())

      self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), "202")
      self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202")])
    finally:
      await client_a.close()
      await client_b.close()
      server.close()
      await server.wait_closed()

  async def test_env_client_registers_time_slice_job_id(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    server = await start_tcp_time_slicer(agent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    env = {
      "OPEN_RL_ACCEL_TIMESLICER_HOST": "127.0.0.1",
      "OPEN_RL_ACCEL_TIMESLICER_PORT": str(port),
      "OPEN_RL_TIME_SLICE_JOB_ID": "job-a",
      "OPEN_RL_TIME_SLICE_GROUP": "ignored-group",
    }
    with patch.dict("os.environ", env, clear=True):
      client = time_slicer_client_from_env()
      workload = workload_from_env(101)
    try:
      await client.register(workload)
      async with client.acquire(workload):
        pass

      self.assertEqual(restorer.labels(), [("checkpoint", "job-a", "shared-accelerator")])
    finally:
      await client.close()
      server.close()
      await server.wait_closed()

  async def test_workload_from_env_uses_fixed_group_and_model_id_when_job_id_env_is_absent(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_TIME_SLICE_GROUP": "ignored-group"}, clear=True):
      workload = workload_from_env(101, job_id="model-a")

    self.assertEqual(workload.job_id, "model-a")
    self.assertEqual(workload.group, "shared-accelerator")


class CudaCheckpointRestorerTest(unittest.TestCase):
  def test_checkpoint_discovers_pids_from_workload_identity(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[101, 202]),
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      restorer.checkpoint(workload)

    self.assertEqual(
      [call.args[0] for call in run_cuda_checkpoint.call_args_list],
      [
        ["--action", "lock", "--pid", "101"],
        ["--action", "lock", "--pid", "202"],
        ["--action", "checkpoint", "--pid", "101"],
        ["--action", "checkpoint", "--pid", "202"],
      ],
    )

  def test_restore_uses_checkpointed_pids_without_rediscovery(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[101, 202]) as discover_pids,
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      restorer.checkpoint(workload)
      discover_pids.side_effect = AssertionError("restore must not query nvidia-smi after checkpoint")
      restorer.restore(workload)

    self.assertEqual(discover_pids.call_count, 1)
    self.assertEqual(
      [call.args[0] for call in run_cuda_checkpoint.call_args_list],
      [
        ["--action", "lock", "--pid", "101"],
        ["--action", "lock", "--pid", "202"],
        ["--action", "checkpoint", "--pid", "101"],
        ["--action", "checkpoint", "--pid", "202"],
        ["--action", "restore", "--pid", "101"],
        ["--action", "restore", "--pid", "202"],
        ["--action", "unlock", "--pid", "101"],
        ["--action", "unlock", "--pid", "202"],
      ],
    )
    self.assertEqual(restorer.checkpointed_pids, {})

  def test_checkpoint_with_no_gpu_pids_skips_snapshot(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[]),
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      self.assertFalse(restorer.checkpoint(workload))

    run_cuda_checkpoint.assert_not_called()
    self.assertEqual(restorer.checkpointed_pids, {})

  def test_restore_without_prior_checkpoint_fails(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with self.assertRaisesRegex(RuntimeError, "no checkpointed PIDs"):
      restorer.restore(workload)

  def test_process_discovery_checks_gpu_pids_and_process_group_leaders(self) -> None:
    from accel_timeslicer.process_discovery import discover_workload_gpu_pids, workload_root_pids

    workload = WorkloadRef(job_id="trainer-model-a")

    def environ(pid: int) -> dict[str, str]:
      if pid == 11:
        return {"OPEN_RL_TIME_SLICE_JOB_ID": "trainer-model-a", "OPEN_RL_TIME_SLICE_GROUP": "shared-accelerator"}
      if pid == 99:
        return {"OPEN_RL_TIME_SLICE_JOB_ID": "other"}
      return {}

    def pgid(pid: int) -> int | None:
      return {12: 11, 99: 98}.get(pid)

    with (
      patch("accel_timeslicer.process_discovery.process_environ", side_effect=environ),
      patch("accel_timeslicer.process_discovery.process_group_id", side_effect=pgid),
      patch("accel_timeslicer.process_discovery.nvidia_smi_compute_pids", return_value=[12, 99]),
    ):
      self.assertEqual(discover_workload_gpu_pids(workload), [12])
      self.assertEqual(workload_root_pids(workload), [11])


class LlmDCheckpointRestorerTest(unittest.TestCase):
  def test_installed_llmd_client_matches_checkpoint_restorer_contract(self) -> None:
    try:
      from timeslice.snapshot_agent import SnapshotAgentClient
      from timeslice.snapshot_agent.types import GetOperationResponse
    except ModuleNotFoundError as exc:
      if exc.name and exc.name.split(".")[0] == "timeslice":
        self.skipTest("timeslice cluster extra is not installed")
      raise

    for name in ["snapshot_and_wait", "restore_and_wait"]:
      parameters = inspect.signature(getattr(SnapshotAgentClient, name)).parameters
      self.assertEqual(
        list(parameters)[:5],
        ["self", "job_id", "group", "poll_interval_sec", "backend_config"],
      )
      self.assertEqual(parameters["poll_interval_sec"].default, 1.0)

    self.assertTrue(callable(SnapshotAgentClient.close))
    self.assertEqual(GetOperationResponse.__annotations__["status"], str)
    self.assertIn("error", GetOperationResponse.__annotations__)

  def test_checkpoint_and_restore_wait_for_llmd_operations_by_job_id(self) -> None:
    class Client:
      def __init__(self):
        self.calls = []

      def snapshot_and_wait(self, job_id, group="", poll_interval_sec=1.0, backend_config=None):
        self.calls.append(("snapshot", job_id, group, poll_interval_sec, backend_config))
        return SimpleNamespace(status="OPERATION_STATUS_COMPLETE")

      def restore_and_wait(self, job_id, group="", poll_interval_sec=1.0, backend_config=None):
        self.calls.append(("restore", job_id, group, poll_interval_sec, backend_config))
        return SimpleNamespace(status="OPERATION_STATUS_COMPLETE")

      def close(self):
        pass

    client = Client()
    cuda_config = object()
    restorer = LlmDCheckpointRestorer(client, cuda_config, 0.25)

    target = WorkloadRef(job_id="job-a")
    restorer.checkpoint(target)
    restorer.restore(target)

    self.assertEqual(
      client.calls,
      [
        ("snapshot", "job-a", "shared-accelerator", 0.25, cuda_config),
        ("restore", "job-a", "shared-accelerator", 0.25, cuda_config),
      ],
    )

  def test_workload_requires_job_id(self) -> None:
    class Client:
      def snapshot_and_wait(self, *_args, **_kwargs):
        raise AssertionError("must not call llm-d without job id")

      def restore_and_wait(self, *_args, **_kwargs):
        raise AssertionError("must not call llm-d without job id")

      def close(self):
        pass

    with self.assertRaisesRegex(ValueError, "job_id"):
      WorkloadRef(job_id="")


async def acquire_once(client: SocketTimeSlicerClient, workload: WorkloadRef) -> str:
  async with client.acquire(workload):
    return workload.job_id


if __name__ == "__main__":
  unittest.main()
