"""Unit tests for examples/common/agent_sandbox.py.

Run with the examples environment (tinker-cookbook is needed for the
interface types):

    PYTHONPATH=examples:examples/text-to-sql uv --project examples run python -m unittest tests.test_agent_sandbox

The fake sandbox below executes commands locally with the same contract as the
runtime image (shlex-split, no shell, cwd = base dir, uploads confined to the
base dir), and maps /opt/run_sql.py to the real script in the repo, so the
SQL round trip is exercised end to end against utils.rewards.run_sql.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common.agent_sandbox import AgentSandboxBackend, AgentSandboxPool, run_sql_in_sandbox
from tinker_cookbook.sandbox import SandboxInterface, SandboxTerminatedError
from utils.rewards import run_sql

REPO = Path(__file__).resolve().parents[1]
RUN_SQL_SCRIPT = REPO / "examples" / "text-to-sql" / "sandbox" / "run_sql.py"


class FakeCommands:
  def __init__(self, base: Path):
    self.base = base
    self.calls: list[str] = []

  async def run(self, command: str, timeout: int = 60) -> SimpleNamespace:
    self.calls.append(command)
    args = [sys.executable if a == "python" else str(RUN_SQL_SCRIPT) if a == "/opt/run_sql.py" else a for a in shlex.split(command)]
    proc = await asyncio.to_thread(subprocess.run, args, cwd=self.base, capture_output=True, text=True, timeout=timeout)
    return SimpleNamespace(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)


class FakeFiles:
  def __init__(self, base: Path):
    self.base = base

  def _safe(self, path: str) -> Path:
    full = (self.base / path.lstrip("/")).resolve()
    if os.path.commonpath([self.base.resolve(), full]) != str(self.base.resolve()):
      raise RuntimeError("Access denied")
    return full

  async def write(self, path: str, content: bytes | str, timeout: int = 60) -> None:
    full = self._safe(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content.encode() if isinstance(content, str) else content)

  async def read(self, path: str, timeout: int = 60) -> bytes:
    full = self._safe(path)
    if not full.is_file():
      raise RuntimeError("404 File not found")
    return full.read_bytes()


class FakeSandbox:
  created = 0

  def __init__(self, base: Path, ready: bool = True):
    FakeSandbox.created += 1
    self.claim_name = f"sandbox-claim-{FakeSandbox.created:04d}"
    self.namespace = "openrl-system"
    self._commands = FakeCommands(base)
    self._files = FakeFiles(base)
    self.ready = ready
    self.terminated = False

  @property
  def commands(self):
    return None if self.terminated else self._commands

  @property
  def files(self):
    return None if self.terminated else self._files

  async def status(self) -> tuple[str, str]:
    return ("SandboxReady", "ok") if self.ready else ("SandboxNotReady", "pod evicted")

  async def terminate(self) -> None:
    self.terminated = True


def run(coro):
  return asyncio.run(coro)


SCHEMA = "CREATE TABLE t(a INT, b REAL, c TEXT); INSERT INTO t VALUES (1, 0.1234567891234, 'x'), (2, 2.5, NULL), (3, 1.0, 'y');"


class BackendTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = tempfile.TemporaryDirectory()
    self.base = Path(self.tmp.name)
    self.sandbox = FakeSandbox(self.base)
    self.backend = AgentSandboxBackend(self.sandbox, base_dir="/app/work")

  def tearDown(self) -> None:
    self.tmp.cleanup()

  def test_satisfies_cookbook_interface(self) -> None:
    self.assertIsInstance(self.backend, SandboxInterface)
    self.assertEqual(self.backend.sandbox_id, self.sandbox.claim_name)

  def test_write_run_read_round_trip(self) -> None:
    async def go():
      await self.backend.write_file("dir/hello.txt", "hi there")
      listed = await self.backend.run_command("cat dir/hello.txt")
      absolute = await self.backend.read_file("/app/work/dir/hello.txt")
      missing = await self.backend.read_file("nope.txt")
      return listed, absolute, missing

    listed, absolute, missing = run(go())
    self.assertEqual((listed.exit_code, listed.stdout), (0, "hi there"))
    self.assertGreater(listed.metrics["duration_ms"], 0)
    self.assertEqual(absolute.stdout, "hi there")
    self.assertEqual(missing.exit_code, 1)

  def test_workdir_and_executable(self) -> None:
    async def go():
      await self.backend.write_file("bin/say.sh", "#!/bin/sh\necho from $(pwd)\n", executable=True)
      return await self.backend.run_command("./say.sh", workdir="bin")

    result = run(go())
    self.assertEqual(result.exit_code, 0, result.stderr)
    self.assertTrue(result.stdout.strip().endswith("/bin"))

  def test_paths_outside_base_dir_are_rejected(self) -> None:
    with self.assertRaises(ValueError):
      run(self.backend.write_file("/etc/passwd", "x"))

  def test_output_is_truncated(self) -> None:
    result = run(self.backend.run_command("python -c \"print('x' * 5000)\"", max_output_bytes=100))
    self.assertLess(len(result.stdout), 200)
    self.assertIn("truncated", result.stdout)

  def test_heartbeat_reports_dead_sandbox(self) -> None:
    run(self.backend.send_heartbeat())
    self.sandbox.ready = False
    with self.assertRaises(SandboxTerminatedError):
      run(self.backend.send_heartbeat())

  def test_cleanup_terminates_once_and_closes(self) -> None:
    closes = []

    async def on_cleanup():
      closes.append(1)

    backend = AgentSandboxBackend(self.sandbox, on_cleanup=on_cleanup)
    run(backend.cleanup())
    run(backend.cleanup())
    self.assertTrue(self.sandbox.terminated)
    self.assertEqual(closes, [1])
    with self.assertRaises(SandboxTerminatedError):
      run(backend.run_command("true"))


class RunSqlTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = tempfile.TemporaryDirectory()
    self.backend = AgentSandboxBackend(FakeSandbox(Path(self.tmp.name)))

  def tearDown(self) -> None:
    self.tmp.cleanup()

  def test_matches_local_run_sql(self) -> None:
    for query in (
      "select a, b, c from t order by a",
      "select count(*), avg(b) from t",
      "select * from nope",
      "select a from t where",
      "select group_concat(c) from t",
    ):
      with self.subTest(query=query):
        local = run_sql(SCHEMA, query)
        remote = run(run_sql_in_sandbox(self.backend, SCHEMA, query))
        self.assertEqual(remote, local)

  def test_job_file_is_removed(self) -> None:
    run(run_sql_in_sandbox(self.backend, SCHEMA, "select 1"))
    self.assertEqual(list((Path(self.tmp.name) / "jobs").iterdir()), [])

  def test_runaway_query_times_out_like_local(self) -> None:
    query = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT count(*) FROM r"
    self.assertEqual(run(run_sql_in_sandbox(self.backend, SCHEMA, query)), (None, "interrupted"))

  def test_transport_failure_is_an_error_not_an_exception(self) -> None:
    class Broken(AgentSandboxBackend):
      async def run_command(self, *a, **k):
        raise ConnectionError("pod gone")

    rows, error = run(run_sql_in_sandbox(Broken(FakeSandbox(Path(self.tmp.name))), SCHEMA, "select 1"))
    self.assertIsNone(rows)
    self.assertTrue(error.startswith("sandbox: ConnectionError"))


class PoolTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = tempfile.TemporaryDirectory()
    self.sandboxes: list[FakeSandbox] = []

  def tearDown(self) -> None:
    self.tmp.cleanup()

  async def factory(self) -> AgentSandboxBackend:
    sandbox = FakeSandbox(Path(self.tmp.name))
    self.sandboxes.append(sandbox)
    return AgentSandboxBackend(sandbox)

  def test_leases_are_bounded_and_claims_are_released(self) -> None:
    async def go():
      async with AgentSandboxPool(size=2, warm_pool="wp", namespace="ns", factory=self.factory) as pool:
        in_flight = 0
        peak = 0

        async def use():
          nonlocal in_flight, peak
          async with pool.lease() as sandbox:
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            self.assertIsInstance(sandbox, AgentSandboxBackend)
            in_flight -= 1

        await asyncio.gather(*(use() for _ in range(6)))
        return peak

    peak = run(go())
    self.assertEqual(peak, 2)
    self.assertEqual(len(self.sandboxes), 2)
    self.assertTrue(all(s.terminated for s in self.sandboxes))

  def test_dead_sandbox_is_replaced(self) -> None:
    async def go():
      async with AgentSandboxPool(size=1, warm_pool="wp", namespace="ns", factory=self.factory) as pool:
        self.sandboxes[0].ready = False
        with self.assertRaises(SandboxTerminatedError):
          async with pool.lease() as sandbox:
            await sandbox.send_heartbeat()
        async with pool.lease() as sandbox:
          await sandbox.send_heartbeat()
          return pool.replacements, sandbox.sandbox_id

    replacements, sandbox_id = run(go())
    self.assertEqual(replacements, 1)
    self.assertEqual(sandbox_id, self.sandboxes[1].claim_name)
    self.assertTrue(self.sandboxes[0].terminated)


if __name__ == "__main__":
  unittest.main()
