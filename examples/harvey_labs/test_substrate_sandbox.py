import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from harvey_labs.agent_sandbox import AgentSandboxFactory
from harvey_labs.config import RunConfig
from harvey_labs.sandbox import SandboxRequest, configured_sandbox_factory, podman_sandbox_factory
from harvey_labs.substrate_sandbox import SubstrateLabSandbox, SubstrateSandboxFactory


class SubstrateSandboxTest(unittest.IsolatedAsyncioTestCase):
  def make_sandbox(self, idle: float | None = 0.04):
    events: list[tuple[str, str, str]] = []

    async def record(op: str, env_id: str, *, atespace: str):
      events.append((op, env_id, atespace))

    client = SimpleNamespace(
      suspend=lambda env_id, *, atespace: record("suspend", env_id, atespace=atespace),
      delete=lambda env_id, *, atespace: record("delete", env_id, atespace=atespace),
    )
    req = SandboxRequest(lab_root=Path("."), run_id="r", documents_dir=Path("."), workspace_dir=Path("."), output_dir=Path("."), command_timeout=60)
    sb = SubstrateLabSandbox(
      client,
      SimpleNamespace(id="env-1", atespace="ate-env"),
      req,
      asyncio.get_running_loop(),
      set(),
      idle,
      tool_definitions=[],
      executor=SimpleNamespace(get_metrics=dict),
    )
    return sb, events

  async def test_idle_suspend_and_cleanup(self):
    sb, _ = self.make_sandbox()
    for _ in range(3):
      await asyncio.sleep(0.015)
      async with sb.active():
        pass
    self.assertEqual(sb.suspends, 0)
    async with sb.active():
      await asyncio.sleep(0.07)
      self.assertEqual(sb.suspends, 0)
    await asyncio.sleep(0.07)
    self.assertEqual(sb.tool_metrics()["substrate_suspends"], 1)

    sb2, events2 = self.make_sandbox()
    async with sb2.active():
      pass
    await sb2.cleanup()
    await asyncio.sleep(0.07)
    self.assertEqual(events2, [("delete", "env-1", "ate-env")])

  def test_factory_selection(self):
    self.assertIs(configured_sandbox_factory(RunConfig()), podman_sandbox_factory)
    self.assertIsInstance(configured_sandbox_factory(RunConfig(sandbox_warmpool="pool")), AgentSandboxFactory)
    cfg = RunConfig(sandbox_substrate_endpoint="127.0.0.1:7777", sandbox_substrate_image="img@sha256:1")
    self.assertIsInstance(configured_sandbox_factory(cfg), SubstrateSandboxFactory)
    with self.assertRaises(ValueError):
      configured_sandbox_factory(RunConfig(sandbox_substrate_endpoint="127.0.0.1:7777", sandbox_warmpool="pool"))
    with self.assertRaises(ValueError):
      configured_sandbox_factory(RunConfig(sandbox_substrate_endpoint="127.0.0.1:7777"))


if __name__ == "__main__":
  unittest.main()
