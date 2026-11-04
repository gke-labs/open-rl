"""LAB sandboxes as Agent Substrate environments, driven through ate-env-api.

Each episode runs LAB's sandbox image unmodified as an ate-env rootfs image.
Documents and the workspace are uploaded at start; deliverables are pulled
back as a tar.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import shlex
import signal
import tarfile
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from .sandbox import LabSandbox, SandboxRequest, add_lab_to_path

logger = logging.getLogger(__name__)

WORKSPACE = "/workspace"
DOCUMENTS = "/workspace/documents"
OUTPUT = "/workspace/output"
READY_TIMEOUT = 300
# Slack over the in-guest coreutils timeout to cover snapshot resume and gRPC round trips.
RPC_SLACK = 180
COLLECT_ATTEMPTS = 4

# Runs in the guest so glob and grep see the sandbox filesystem, with the same
# ordering, limits, and symlink rules as LAB's host-side versions.
SEARCH_SCRIPT = r"""
import json, re, sys
from pathlib import Path
a = json.loads(sys.argv[1])
root = Path(a["root"])
real = root.resolve()
def under(p):
  try:
    p.resolve().relative_to(real)
    return True
  except ValueError:
    return False
if a["kind"] == "glob":
  found = sorted((m for m in root.glob(a["pattern"]) if m.is_file() and under(m)), key=lambda p: p.stat().st_mtime, reverse=True)
  print(json.dumps([str(m.relative_to(root)) for m in found[:100]]))
  sys.exit()
regex = re.compile(a["pattern"])
out = []
for f in root.glob(a["file_glob"] or "**/*"):
  if not f.is_file() or not under(f):
    continue
  try:
    text = f.read_text(encoding="utf-8", errors="replace")
  except Exception:
    continue
  hits = list(regex.finditer(text))
  if not hits:
    continue
  rel = str(f.relative_to(root))
  if a["mode"] == "files_with_matches":
    out.append(rel)
  elif a["mode"] == "count":
    out.append(f"{rel}: {len(hits)}")
  elif a["mode"] == "content":
    out.extend(f"{rel}:{i + 1}: {line}" for i, line in enumerate(text.split("\n")) if regex.search(line))
print(json.dumps(out[:250]))
"""


def tar_bytes(entries: list[tuple[Path, str]]) -> bytes:
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w") as tar:
    for source, arcname in entries:
      tar.add(source, arcname=arcname)
  return buffer.getvalue()


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


class SubstrateLabSandbox:
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

  async def exec(self, command: str, cwd: str, timeout: int, env: dict[str, str] | None = None) -> SandboxResult:
    """Match LAB's Podman exec: canonical path variables, bash -lc, coreutils timeout."""
    variables = {"DOCUMENTS_DIR": DOCUMENTS, "OUTPUT_DIR": OUTPUT, "WORKSPACE_DIR": WORKSPACE, **(env or {})}
    exports = "export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in variables.items())
    wrapped = f"{exports}; cd {shlex.quote(cwd)} && timeout --kill-after=2 {timeout} bash -lc {shlex.quote(command)}"
    return await self.shell(wrapped, timeout)

  async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult:
    result = await self.exec(command, workdir or WORKSPACE, timeout)
    cap = max_output_bytes if max_output_bytes is not None else 128 * 1024
    return SandboxResult(
      stdout=result.stdout.encode()[:cap].decode(errors="replace"),
      stderr=result.stderr.encode()[:cap].decode(errors="replace"),
      exit_code=result.exit_code,
      metrics=result.metrics,
    )

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    from ate_env.errors import NotFoundError

    try:
      data = await self.env.read_file_bytes(path)
    except NotFoundError as exc:
      raise FileNotFoundError(path) from exc
    except Exception as exc:
      raise OSError(f"{path}: {exc}") from exc
    return SandboxResult(stdout=data[:max_bytes].decode(errors="replace"), stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.env.write_file(path, data)
    if executable:
      return await self.shell(f"chmod +x -- {shlex.quote(path)}", timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def exists(self, path: str) -> bool:
    return (await self.shell(f"test -e {shlex.quote(path)}", 30)).exit_code == 0

  async def search(self, args: dict[str, Any]) -> list[str]:
    command = f"python3 -c {shlex.quote(SEARCH_SCRIPT)} {shlex.quote(json.dumps(args))}"
    result = await self.shell(command, 60)
    if result.exit_code != 0:
      raise OSError(f"search failed in {self.sandbox_id}: {result.stderr.strip()[-500:]}")
    return json.loads(result.stdout)

  async def send_heartbeat(self, timeout: int = 30) -> None:
    pass

  async def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
    async with self.active():
      return await asyncio.to_thread(self.executor.execute, name, arguments)

  def tool_metrics(self) -> dict[str, Any]:
    return {**self.executor.get_metrics(), "substrate_suspends": self.suspends}

  async def collect_outputs(self, destination: Path) -> None:
    async with self.active():
      for attempt in range(COLLECT_ATTEMPTS):
        try:
          data = await self.pack_outputs()
          break
        except Exception:
          if attempt == COLLECT_ATTEMPTS - 1:
            raise
          await asyncio.sleep(2**attempt)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
      await asyncio.to_thread(tar.extractall, destination, filter="data")

  async def pack_outputs(self) -> bytes:
    archive = f"{WORKSPACE}/.lab-output.tar"
    packed = await self.shell(f"tar -C {OUTPUT} -cf {archive} .", 120)
    if packed.exit_code != 0:
      raise RuntimeError(f"could not pack outputs in {self.sandbox_id}: {packed.stderr.strip()}")
    try:
      return await self.env.read_file_bytes(archive)
    except Exception as exc:
      raise OSError(f"failed to read {archive}: {exc}") from exc

  async def cleanup(self) -> None:
    self.live.discard(self)
    self.idle_suspend_seconds = None
    if self.idle_task is not None:
      self.idle_task.cancel()
    async with asyncio.timeout(60):
      await self.client.delete(self.sandbox_id, atespace=self.env.atespace)


class RemoteFiles:
  """The synchronous Sandbox surface LAB's ToolExecutor calls from its worker thread."""

  def __init__(self, sandbox: SubstrateLabSandbox):
    self.sandbox = sandbox
    self.documents_dir = sandbox.request.documents_dir
    self.output_dir = sandbox.request.output_dir
    self.workspace_dir = sandbox.request.workspace_dir

  def call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run_coroutine_threadsafe(coroutine, self.sandbox.loop).result()

  def exec(self, command: str, *, cwd: str = WORKSPACE, timeout: int | None = None, env: dict[str, str] | None = None) -> Any:
    from sandbox.sandbox import ExecResult, Sandbox

    Sandbox.assert_sandbox_path(cwd)
    try:
      result = self.call(self.sandbox.exec(command, cwd, timeout or self.sandbox.request.command_timeout, env))
    except Exception as exc:
      return ExecResult(stdout="", stderr=f"substrate exec failed: {type(exc).__name__}: {exc}", returncode=1)
    if result.metrics["timed_out"]:
      return ExecResult(stdout=result.stdout, stderr=result.stderr, returncode=None, timed_out=True)
    return ExecResult(stdout=result.stdout, stderr=result.stderr, returncode=result.exit_code)

  def read_file(self, path: str) -> bytes:
    from ate_env.errors import NotFoundError
    from sandbox.sandbox import Sandbox

    Sandbox.assert_sandbox_path(path)
    try:
      return self.call(self.sandbox.env.read_file_bytes(path))
    except NotFoundError as exc:
      raise FileNotFoundError(path) from exc
    except Exception as exc:
      raise OSError(f"{path}: {exc}") from exc

  def write_file(self, path: str, content: bytes | str) -> None:
    from sandbox.sandbox import Sandbox

    Sandbox.assert_sandbox_path(path)
    if not Sandbox.is_writable(path):
      raise PermissionError(f"write denied: {path} is not under a writable mount")
    data = content.encode() if isinstance(content, str) else content
    self.call(self.sandbox.env.write_file(path, data))

  def exists(self, path: str) -> bool:
    from sandbox.sandbox import Sandbox

    try:
      Sandbox.assert_sandbox_path(path)
    except ValueError:
      return False
    return self.call(self.sandbox.exists(path))


def remote_tool_executor(files: RemoteFiles, shell_timeout: int) -> Any:
  """LAB's ToolExecutor with glob and grep run in the guest instead of on host mounts."""
  from harness.tools import ToolExecutor

  class RemoteToolExecutor(ToolExecutor):
    def search_root(self, search_path: str | None) -> str | None:
      sb_path = self._resolve_search_path(search_path)
      if not self.sandbox.exists(sb_path):
        return None
      return DOCUMENTS if sb_path == "/" else sb_path

    def _glob(self, pattern: str, search_path: str | None) -> str:
      if not pattern:
        return "Error: pattern is required"
      self.glob_count += 1
      root = self.search_root(search_path)
      if root is None:
        return f"Error: path does not exist: {search_path}"
      matches = files.call(files.sandbox.search({"kind": "glob", "root": root, "pattern": pattern}))
      if not matches:
        return f"No files matching '{pattern}' in {self._resolve_search_path(search_path)}"
      return "\n".join(matches)

    def _grep(self, pattern_str: str, search_path: str | None, file_glob: str | None, output_mode: str) -> str:
      if not pattern_str:
        return "Error: pattern is required"
      self.grep_count += 1
      root = self.search_root(search_path)
      if root is None:
        return f"Error: path does not exist: {search_path}"
      try:
        re.compile(pattern_str)
      except re.error as e:
        return f"Error: invalid regex: {e}"
      args = {"kind": "grep", "root": root, "pattern": pattern_str, "file_glob": file_glob, "mode": output_mode}
      results = files.call(files.sandbox.search(args))
      return "\n".join(results) if results else f"No matches for '{pattern_str}'"

  return RemoteToolExecutor(sandbox=files, shell_timeout=shell_timeout)


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
