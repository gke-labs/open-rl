"""LAB sandboxes on Agent Sandbox (sigs.k8s.io/agent-sandbox), driven through sandboxd.

Each episode claims a sandbox from a SandboxWarmPool. The guest image must be
LAB's sandbox image running sandboxd rooted at /workspace. The client reaches
sandboxd by pod IP, so the sandbox cluster may be a different cluster on the
same VPC; KUBECONFIG then points at it. Documents and the workspace are
uploaded at start; deliverables are pulled back as a tar.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shlex
import signal
import tarfile
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from .sandbox import LabSandbox, SandboxRequest, add_lab_to_path

logger = logging.getLogger(__name__)

WORKSPACE = "/workspace"
DOCUMENTS = "/workspace/documents"
OUTPUT = "/workspace/output"
READY_TIMEOUT = 300
# A claim is a lease: the controller deletes it LEASE_SECONDS after its last
# renewal, so a driver that dies any way at all leaves nothing behind for long.
LEASE_SECONDS = 15 * 60
RENEW_SECONDS = 5 * 60
CLAIM_ATTEMPTS = 3
# Slack over the in-guest coreutils timeout for the gRPC round trips.
RPC_SLACK = 30
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


def relative(path: str) -> str:
  """sandboxd takes paths relative to its root, /workspace, and serves nothing outside it."""
  return str(Path(path).relative_to(WORKSPACE))


class AgentLabSandbox:
  """Async LabSandbox over one claimed Agent Sandbox."""

  def __init__(self, sandbox: Any, request: SandboxRequest, loop: asyncio.AbstractEventLoop, live: set):
    add_lab_to_path(request.lab_root)
    from harness.tools import get_all_tool_definitions

    self.sandbox = sandbox
    self.request = request
    self.loop = loop
    self.live = live
    self.lease: asyncio.Task | None = None
    self.tool_definitions = get_all_tool_definitions()
    self.executor = remote_tool_executor(RemoteFiles(self), request.command_timeout)

  @property
  def sandbox_id(self) -> str:
    return self.sandbox.sandbox_id

  async def shell(self, command: str, timeout: int) -> SandboxResult:
    async with asyncio.timeout(timeout + RPC_SLACK):
      result = await self.sandbox.commands.run(command, timeout=timeout + RPC_SLACK)
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

  async def read_bytes(self, path: str) -> bytes:
    from k8s_agent_sandbox.exceptions import SandboxRequestError

    try:
      return await self.sandbox.files.read(relative(path), timeout=60)
    except SandboxRequestError as exc:
      if exc.status_code == 404:
        raise FileNotFoundError(path) from exc
      raise OSError(f"{path}: {exc}") from exc

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    data = await self.read_bytes(path)
    return SandboxResult(stdout=data[:max_bytes].decode(errors="replace"), stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.sandbox.files.write(relative(path), data, timeout=timeout)
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
    pass  # Claims have no idle expiry.

  async def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
    return await asyncio.to_thread(self.executor.execute, name, arguments)

  def tool_metrics(self) -> dict[str, Any]:
    return self.executor.get_metrics()

  async def collect_outputs(self, destination: Path) -> None:
    # Packing and reading are safe to repeat, so ride out a transient 502.
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
      # The data filter rejects members and links that escape destination.
      await asyncio.to_thread(tar.extractall, destination, filter="data")

  async def pack_outputs(self) -> bytes:
    # Outside the output directory, so the archive does not contain itself.
    archive = f"{WORKSPACE}/.lab-output.tar"
    packed = await self.shell(f"tar -C {OUTPUT} -cf {archive} .", 120)
    if packed.exit_code != 0:
      raise RuntimeError(f"could not pack outputs in {self.sandbox_id}: {packed.stderr.strip()}")
    return await self.read_bytes(archive)

  async def keep_lease(self) -> None:
    """Renew the claim while the episode runs. A failed renewal is retried at
    the next one; two in a row still leave a third before the lease runs out."""
    while True:
      await asyncio.sleep(RENEW_SECONDS)
      try:
        await renew_claim(self.sandbox.claim_name, self.sandbox.namespace)
      except Exception as exc:
        logger.warning("could not renew the lease on %s: %s", self.sandbox.claim_name, exc)

  async def cleanup(self) -> None:
    if self.lease is not None:
      self.lease.cancel()
    self.live.discard(self)
    await self.sandbox.close_connection()
    await delete_claim(self.sandbox.claim_name, self.sandbox.namespace)


class RemoteFiles:
  """The synchronous Sandbox surface LAB's ToolExecutor calls from its worker thread."""

  def __init__(self, sandbox: AgentLabSandbox):
    self.sandbox = sandbox
    # Host copies, used by LAB for document metrics only.
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
      return ExecResult(stdout="", stderr=f"sandbox exec failed: {type(exc).__name__}: {exc}", returncode=1)
    if result.metrics["timed_out"]:
      return ExecResult(stdout=result.stdout, stderr=result.stderr, returncode=None, timed_out=True)
    return ExecResult(stdout=result.stdout, stderr=result.stderr, returncode=result.exit_code)

  def read_file(self, path: str) -> bytes:
    from sandbox.sandbox import Sandbox

    Sandbox.assert_sandbox_path(path)
    return self.call(self.sandbox.read_bytes(path))

  def write_file(self, path: str, content: bytes | str) -> None:
    from sandbox.sandbox import Sandbox

    Sandbox.assert_sandbox_path(path)
    if not Sandbox.is_writable(path):
      raise PermissionError(f"write denied: {path} is not under a writable mount")
    self.call(self.sandbox.write_file(path, content))

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


async def on_claim(name: str | None, namespace: str, call: Callable[[Any, dict], Awaitable[Any]]) -> None:
  """Run a request against a SandboxClaim with a request timeout, on a new
  connection per attempt. The client's shared connections can sit idle through
  a whole episode, and a request on a dropped one hangs with no timeout of its
  own. A claim that is already gone is fine."""
  if not name:
    return
  from kubernetes_asyncio import client, config

  target = {"group": "extensions.agents.x-k8s.io", "version": "v1beta1", "namespace": namespace, "plural": "sandboxclaims", "name": name}
  for attempt in range(CLAIM_ATTEMPTS):
    try:
      await config.load_kube_config()
      async with client.ApiClient() as api:
        await call(client.CustomObjectsApi(api), target)
      return
    except client.ApiException as exc:
      if exc.status == 404:
        return
      if attempt == CLAIM_ATTEMPTS - 1:
        raise
    except (OSError, TimeoutError):
      if attempt == CLAIM_ATTEMPTS - 1:
        raise
    await asyncio.sleep(2**attempt)


async def delete_claim(name: str | None, namespace: str) -> None:
  await on_claim(name, namespace, lambda api, target: api.delete_namespaced_custom_object(**target, _request_timeout=30))


async def renew_claim(name: str | None, namespace: str) -> None:
  shutdown = (datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS)).strftime("%Y-%m-%dT%H:%M:%SZ")
  # A JSON patch, which this client sends; every claim is created with a lifecycle.
  body = [{"op": "replace", "path": "/spec/lifecycle/shutdownTime", "value": shutdown}]
  await on_claim(name, namespace, lambda api, target: api.patch_namespaced_custom_object(**target, body=body, _request_timeout=30))


class AgentSandboxFactory:
  """One Agent Sandbox client shared by every sandbox of a run."""

  def __init__(self, warmpool: str, namespace: str):
    self.warmpool = warmpool
    self.namespace = namespace
    self.client: Any = None
    # Sandboxes not yet cleaned up, released together if the driver is stopped.
    self.live: set[AgentLabSandbox] = set()

  def release_on_sigterm(self) -> None:
    """A Job deletion or eviction stops the driver with SIGTERM: delete every
    claim it holds, then exit as SIGTERM would have."""

    async def release() -> None:
      await asyncio.gather(*(delete_claim(s.sandbox.claim_name, s.sandbox.namespace) for s in list(self.live)), return_exceptions=True)
      signal.signal(signal.SIGTERM, signal.SIG_DFL)
      os.kill(os.getpid(), signal.SIGTERM)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(release()))

  async def __call__(self, request: SandboxRequest) -> LabSandbox:
    from k8s_agent_sandbox import AsyncSandboxClient
    from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig

    if self.client is None:
      self.client = AsyncSandboxClient(connection_config=SandboxdInClusterConnectionConfig(mode="pod-ip"))
      self.release_on_sigterm()
    claimed = await self.client.create_sandbox(
      warmpool=self.warmpool, namespace=self.namespace, sandbox_ready_timeout=READY_TIMEOUT, shutdown_after_seconds=LEASE_SECONDS
    )
    try:
      sandbox = AgentLabSandbox(claimed, request, asyncio.get_running_loop(), self.live)
      # Documents land inside the uploaded workspace, as Podman mounts them.
      archive = await asyncio.to_thread(tar_bytes, [(request.workspace_dir, "."), (request.documents_dir, "documents")])
      staged = f"{WORKSPACE}/.lab-input.tar"
      await sandbox.write_file(staged, archive, timeout=300)
      unpack = f"mkdir -p {OUTPUT} && tar --no-same-owner -C {WORKSPACE} -xf {staged} && rm {staged}"
      unpacked = await sandbox.shell(unpack, 300)
      if unpacked.exit_code != 0:
        raise RuntimeError(f"could not unpack inputs in {claimed.sandbox_id}: {unpacked.stderr.strip()}")
      self.live.add(sandbox)
      sandbox.lease = asyncio.create_task(sandbox.keep_lease())
      return sandbox
    except BaseException:
      await claimed.close_connection()
      await delete_claim(claimed.claim_name, self.namespace)
      raise
