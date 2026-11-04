"""LAB sandboxes on Agent Sandbox (sigs.k8s.io/agent-sandbox), driven through sandboxd.

Each episode claims a sandbox from a SandboxWarmPool. The guest image must be
LAB's sandbox image running sandboxd rooted at /workspace. The client reaches
sandboxd by pod IP, so the sandbox cluster may be a different cluster on the
same VPC; KUBECONFIG then points at it. Documents and the workspace are
uploaded at start; deliverables are pulled back as a tar.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

from .remote_sandbox import OUTPUT, WORKSPACE, GuestSandbox, RemoteFiles, remote_tool_executor, tar_bytes
from .sandbox import LabSandbox, SandboxRequest, add_lab_to_path

logger = logging.getLogger(__name__)

READY_TIMEOUT = 300
# A claim is a lease: the controller deletes it LEASE_SECONDS after its last
# renewal, so a driver that dies any way at all leaves nothing behind for long.
LEASE_SECONDS = 15 * 60
RENEW_SECONDS = 5 * 60
CLAIM_ATTEMPTS = 3
# Slack over the in-guest coreutils timeout for the gRPC round trips.
RPC_SLACK = 30


def relative(path: str) -> str:
  """sandboxd takes paths relative to its root, /workspace, and serves nothing outside it."""
  return str(Path(path).relative_to(WORKSPACE))


class AgentLabSandbox(GuestSandbox):
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

  async def read_bytes(self, path: str) -> bytes:
    from k8s_agent_sandbox.exceptions import SandboxRequestError

    try:
      return await self.sandbox.files.read(relative(path), timeout=60)
    except SandboxRequestError as exc:
      if exc.status_code == 404:
        raise FileNotFoundError(path) from exc
      raise OSError(f"{path}: {exc}") from exc

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    data = content.encode() if isinstance(content, str) else content
    await self.sandbox.files.write(relative(path), data, timeout=timeout)
    if executable:
      return await self.shell(f"chmod +x -- {shlex.quote(path)}", timeout)
    return SandboxResult(stdout="", stderr="", exit_code=0)

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
