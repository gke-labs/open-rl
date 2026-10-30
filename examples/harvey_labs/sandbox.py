"""Cookbook sandbox contract plus LAB tool dispatch and artifact collection."""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import chz
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


@chz.chz
class SandboxRequest:
  """Prepared episode inputs. Factories must create an isolated workspace.

  Mount or upload documents at /workspace/documents (read-only), skill scripts
  from workspace_dir at /workspace, and write deliverables to /workspace/output.
  The image must provide LAB's parse-doc command and skill dependencies.
  """

  lab_root: Path
  run_id: str
  documents_dir: Path
  workspace_dir: Path
  output_dir: Path
  command_timeout: int


class LabSandbox(SandboxInterface, Protocol):
  """Keep LAB semantics outside the generic cookbook execution interface.

  Remote implementations must implement LAB's canonical tool schemas and path
  semantics, including glob/grep, without assuming shared host mounts.
  """

  @property
  def tool_definitions(self) -> list[dict[str, Any]]: ...

  async def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str: ...

  def tool_metrics(self) -> dict[str, Any]: ...

  async def collect_outputs(self, destination: Path) -> None:
    """Materialize binary deliverables for the local judge; preserve directories.

    Reject paths and symlinks escaping destination. Raise on transfer failure
    so incomplete output is never silently graded as a complete submission.
    """
    ...


# A factory owns cleanup if creation fails or is cancelled before it returns.
# Once returned, the env group owns cleanup, including partial group failures.
SandboxFactory = Callable[[SandboxRequest], Awaitable[LabSandbox]]


def add_lab_to_path(lab_root: Path) -> None:
  resolved = str(lab_root.resolve())
  if resolved not in sys.path:
    sys.path.insert(0, resolved)


class PodmanLabSandbox:
  """Adapt the existing LAB harness without duplicating its tools or mounts."""

  def __init__(self, sandbox: Any, executor: Any, tool_definitions: list[dict[str, Any]]):
    self._sandbox = sandbox
    self._executor = executor
    self.tool_definitions = tool_definitions

  @property
  def sandbox_id(self) -> str:
    return self._sandbox.container_name

  async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult:
    result = await asyncio.to_thread(self._sandbox.exec, command, cwd=workdir or "/workspace", timeout=timeout)
    cap = max_output_bytes if max_output_bytes is not None else 128 * 1024
    return SandboxResult(
      stdout=result.stdout.encode()[:cap].decode(errors="replace"),
      stderr=result.stderr.encode()[:cap].decode(errors="replace"),
      exit_code=124 if result.timed_out else result.returncode,
      metrics={"timed_out": result.timed_out},
    )

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    data = await asyncio.to_thread(self._sandbox.read_file, path)
    return SandboxResult(stdout=data[:max_bytes].decode(errors="replace"), stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    await asyncio.to_thread(self._sandbox.write_file, path, content)
    if executable:
      return await self.run_command(f"chmod +x -- {shlex.quote(path)}", timeout=timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def send_heartbeat(self, timeout: int = 30) -> None:
    pass  # Local Podman containers have no idle expiry.

  async def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
    return await asyncio.to_thread(self._executor.execute, name, arguments)

  def tool_metrics(self) -> dict[str, Any]:
    return self._executor.get_metrics()

  async def collect_outputs(self, destination: Path) -> None:
    if destination.resolve() != self._sandbox.output_dir.resolve():
      raise ValueError("Podman output destination must match its episode bind mount")
    # Files are already present through the bind mount, including binary outputs.

  async def cleanup(self) -> None:
    await asyncio.to_thread(self._sandbox.stop)


def configured_sandbox_factory(sandbox_warmpool: str | None, sandbox_namespace: str) -> SandboxFactory:
  if not sandbox_warmpool:
    return podman_sandbox_factory
  from .agent_sandbox import AgentSandboxFactory

  return AgentSandboxFactory(sandbox_warmpool, sandbox_namespace)


async def podman_sandbox_factory(request: SandboxRequest) -> LabSandbox:
  add_lab_to_path(request.lab_root)
  from harness.tools import ToolExecutor, get_all_tool_definitions
  from sandbox.sandbox import DEFAULT_IMAGE, Sandbox

  sandbox = Sandbox(
    documents_dir=request.documents_dir,
    output_dir=request.output_dir,
    workspace_dir=request.workspace_dir,
    image=DEFAULT_IMAGE,
    default_timeout=request.command_timeout,
  )
  # A cancelled to_thread keeps running. Wait for startup to settle before
  # stopping the container, otherwise it can start after cleanup has finished.
  startup = asyncio.create_task(asyncio.to_thread(sandbox.start))
  try:
    await asyncio.shield(startup)
    return PodmanLabSandbox(sandbox, ToolExecutor(sandbox=sandbox, shell_timeout=request.command_timeout), get_all_tool_definitions())
  except BaseException:
    await asyncio.gather(startup, return_exceptions=True)
    await asyncio.to_thread(sandbox.stop)
    raise
