"""LAB sandboxes as Agent Substrate environments, driven through ate-env-api.

Each episode runs LAB's sandbox image unmodified as an ate-env rootfs image.
Documents and the workspace are uploaded at start; deliverables are pulled
back as a tar.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import signal
import uuid
from pathlib import Path
from typing import Any

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from .remote_sandbox import OUTPUT, WORKSPACE, GuestSandbox, RemoteFiles, remote_tool_executor, tar_bytes
from .sandbox import LabSandbox, SandboxRequest, add_lab_to_path

logger = logging.getLogger(__name__)

READY_TIMEOUT = 300
# Slack over the in-guest coreutils timeout to cover snapshot resume and gRPC round trips.
RPC_SLACK = 180


async def wait_ready(env: Any) -> None:
  """A fresh environment fails guest calls until its actor has booted."""
  loop = asyncio.get_running_loop()
  deadline = loop.time() + READY_TIMEOUT
  delay = 0.5
  while True:
    try:
      async with asyncio.timeout(30):
        if (await env.shell("true")).exit_code == 0:
          return
    except Exception:
      if loop.time() > deadline:
        raise
    if loop.time() > deadline:
      raise TimeoutError(f"Substrate environment {env.id} not ready after {READY_TIMEOUT}s")
    await asyncio.sleep(delay)
    delay = min(delay * 2, 5)


class SubstrateLabSandbox(GuestSandbox):
  """Async LabSandbox over one ate-env environment.

  With idle_suspend_seconds set, the environment is suspended once no tool
  call or output collection has run for that long, which frees its worker
  while the model generates. The next guest call wakes it; ate-env-api waits
  for the resume. A call never starts while a suspend is in flight, and a
  suspend never starts while a call is in flight.
  """

  def __init__(
    self,
    client: Any,
    env: Any,
    request: SandboxRequest,
    loop: asyncio.AbstractEventLoop,
    live: set,
    idle_suspend_seconds: float | None = None,
    *,
    tool_definitions: list[dict[str, Any]] | None = None,
    executor: Any = None,
  ):
    self.client = client
    self.env = env
    self.request = request
    self.loop = loop
    self.live = live
    if tool_definitions is None or executor is None:
      add_lab_to_path(request.lab_root)
      from harness.tools import get_all_tool_definitions

      tool_definitions = get_all_tool_definitions()
      executor = remote_tool_executor(RemoteFiles(self), request.command_timeout)
    self.tool_definitions = tool_definitions
    self.executor = executor
    self.idle_suspend_seconds = idle_suspend_seconds
    self.busy = 0
    self.activity = 0
    self.suspend_lock = asyncio.Lock()
    self.idle_task: asyncio.Task | None = None
    self.suspends = 0

  @contextlib.asynccontextmanager
  async def active(self):
    """Mark a span of guest work so the idle suspend stays out of its way."""
    self.activity += 1
    async with self.suspend_lock:
      self.busy += 1
    try:
      yield
    finally:
      self.busy -= 1
      if self.busy == 0 and self.idle_suspend_seconds is not None:
        self.idle_task = self.loop.create_task(self.suspend_when_idle(self.activity))

  async def suspend_when_idle(self, activity: int) -> None:
    await asyncio.sleep(self.idle_suspend_seconds)
    async with self.suspend_lock:
      if self.busy or self.activity != activity:
        return
      try:
        async with asyncio.timeout(120):
          await self.client.suspend(self.sandbox_id, atespace=self.env.atespace)
        self.suspends += 1
      except Exception as exc:
        # The environment stays running; the next call proceeds as usual.
        logger.warning("could not suspend %s: %s", self.sandbox_id, exc)

  @property
  def sandbox_id(self) -> str:
    return self.env.id

  async def shell(self, command: str, timeout: int) -> SandboxResult:
    async with asyncio.timeout(timeout + RPC_SLACK):
      result = await self.env.shell(command)
    timed_out = result.exit_code in (124, 137)
    return SandboxResult(stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code, metrics={"timed_out": timed_out})

  async def read_bytes(self, path: str) -> bytes:
    from ate_env.errors import NotFoundError

    try:
      return await self.env.read_file_bytes(path)
    except NotFoundError as exc:
      raise FileNotFoundError(path) from exc
    except Exception as exc:
      raise OSError(f"{path}: {exc}") from exc

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.env.write_file(path, data)
    if executable:
      return await self.shell(f"chmod +x -- {shlex.quote(path)}", timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
    async with self.active():
      return await asyncio.to_thread(self.executor.execute, name, arguments)

  def tool_metrics(self) -> dict[str, Any]:
    return {**self.executor.get_metrics(), "substrate_suspends": self.suspends}

  async def collect_outputs(self, destination: Path) -> None:
    async with self.active():
      await super().collect_outputs(destination)

  async def cleanup(self) -> None:
    self.live.discard(self)
    self.idle_suspend_seconds = None
    if self.idle_task is not None:
      self.idle_task.cancel()
    async with asyncio.timeout(60):
      await self.client.delete(self.sandbox_id, atespace=self.env.atespace)


class SubstrateSandboxFactory:
  """One ate-env client shared by every sandbox of a run."""

  def __init__(self, endpoint: str, image: str, atespace: str, idle_suspend_seconds: float | None = None):
    self.endpoint = endpoint
    self.image = image
    self.atespace = atespace
    self.idle_suspend_seconds = idle_suspend_seconds
    self.client: Any = None
    self.live: set[SubstrateLabSandbox] = set()

  def release_on_sigterm(self) -> None:
    async def release() -> None:
      await asyncio.gather(*(s.cleanup() for s in list(self.live)), return_exceptions=True)
      signal.signal(signal.SIGTERM, signal.SIG_DFL)
      os.kill(os.getpid(), signal.SIGTERM)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(release()))

  async def __call__(self, request: SandboxRequest) -> LabSandbox:
    from ate_env import Client

    if self.client is None:
      self.client = Client(self.endpoint, auth_token=os.environ.get("SUBSTRATE_ENV_AUTH_TOKEN"))
      self.release_on_sigterm()
    env_id = f"lab-{uuid.uuid4().hex[:16]}"
    env = None
    try:
      async with asyncio.timeout(READY_TIMEOUT):
        env = await self.client.create(env_id, rootfs_image=self.image, atespace=self.atespace)
      await wait_ready(env)
      sandbox = SubstrateLabSandbox(self.client, env, request, asyncio.get_running_loop(), self.live, self.idle_suspend_seconds)
      archive = await asyncio.to_thread(tar_bytes, [(request.workspace_dir, "."), (request.documents_dir, "documents")])
      async with sandbox.active():
        await sandbox.write_file("/tmp/lab-input.tar", archive)
        unpack = f"mkdir -p {OUTPUT} && tar --no-same-owner -C {WORKSPACE} -xf /tmp/lab-input.tar && rm /tmp/lab-input.tar"
        unpacked = await sandbox.shell(unpack, 300)
      if unpacked.exit_code != 0:
        raise RuntimeError(f"could not unpack inputs in {env_id}: {unpacked.stderr.strip()}")
      self.live.add(sandbox)
      return sandbox
    except BaseException:
      if env is not None:
        async with asyncio.timeout(60):
          await self.client.delete(env_id, atespace=self.atespace)
      raise
