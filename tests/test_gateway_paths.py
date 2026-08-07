import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from server import gateway
from server.store import InMemoryStore


class GetInfoTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-a"}))

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_info({"model_id": "model-a"}))
    self.assertEqual(response.status_code, 404)

  def test_create_model_requires_base_model_payload(self) -> None:
    response = asyncio.run(gateway.create_model({}))
    self.assertEqual(response.status_code, 400)

  def test_create_model_accepts_base_model_payload(self) -> None:
    created = asyncio.run(gateway.create_model({"base_model": "my-model"}))
    model_id = created["request_id"]
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["model_id"], model_id)
    self.assertEqual(queued[0]["payload"], {})
    meta = json.loads(gateway.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "my-model")


class GatewayPathTest(unittest.TestCase):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    old_tmp_dir = gateway.TMP_DIR
    with tempfile.TemporaryDirectory() as tmp_dir:
      gateway.TMP_DIR = tmp_dir
      self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)

      self.assertEqual(
        gateway.checkpoint_state_path("job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        gateway.checkpoint_state_path("job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(gateway.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


class ClaimReconcilerTest(unittest.IsolatedAsyncioTestCase):
  class _K8sManager:
    def __init__(self) -> None:
      self.calls = 0

    def reconcile_managed_claims(self) -> list[str]:
      self.calls += 1
      return ["claim-idle"]

  class _LocalManager:
    """Stands in for LocalWorkerManager, which provisions no DRA claims."""

  async def test_reconciler_runs_on_its_interval(self) -> None:
    manager = self._K8sManager()
    task = asyncio.create_task(gateway.run_claim_reconciler(manager, interval=0.01))
    await asyncio.sleep(0.1)
    task.cancel()

    self.assertGreater(manager.calls, 1, "reconcile loop should fire repeatedly, not once")

  async def test_reconciler_survives_a_failing_pass(self) -> None:
    manager = self._K8sManager()

    def boom() -> list[str]:
      manager.calls += 1
      raise RuntimeError("API server unavailable")

    manager.reconcile_managed_claims = boom
    task = asyncio.create_task(gateway.run_claim_reconciler(manager, interval=0.01))
    await asyncio.sleep(0.1)
    task.cancel()

    # A transient API error must not silently kill the only thing reclaiming GPUs.
    self.assertGreater(manager.calls, 1)

  async def test_reconciler_starts_only_for_claim_provisioning_managers(self) -> None:
    self.assertIsNone(gateway.start_claim_reconciler(None))
    self.assertIsNone(gateway.start_claim_reconciler(self._LocalManager()))

    task = gateway.start_claim_reconciler(self._K8sManager())
    self.assertIsNotNone(task)
    task.cancel()

  async def test_reconciler_can_be_disabled(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_CLAIM_RECONCILE_INTERVAL_SECONDS": "0"}):
      self.assertIsNone(gateway.start_claim_reconciler(self._K8sManager()))


if __name__ == "__main__":
  unittest.main()
