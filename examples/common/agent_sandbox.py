"""agent-sandbox backend for tinker-cookbook's ``SandboxInterface``.

Runs untrusted, model-written code (here: SQL) in gVisor sandboxes claimed
from a Kubernetes ``SandboxWarmPool`` (kubernetes-sigs/agent-sandbox v1beta1)
through the ``k8s-agent-sandbox[async]`` SDK. The sandbox runtime is the
reference HTTP server on :8888 (``/execute``, ``/upload``, ``/download``); the
client talks to the pod IP, so the training process must run in-cluster.

Three things live here:

* :class:`AgentSandboxBackend` satisfies ``tinker_cookbook.sandbox.SandboxInterface``
  and wraps one claimed sandbox.
* :class:`AgentSandboxPool` holds N long-lived sandboxes and leases them out
  (Phase A of design 013: a fixed pool amortises the ~3 s claim latency).
* :func:`run_sql_in_sandbox` is the Text-to-SQL reward primitive: ship the
  schema and query, run ``/opt/run_sql.py``, get rows back.

Paths handed to the runtime are relative to its working directory
(``SANDBOX_BASE_DIR``); absolute paths under that directory are accepted and
rewritten, anything else absolute is rejected rather than silently relocated.

Install with ``uv --project examples sync --extra sandbox``. The SDK is imported
lazily so this module can be imported, and the fakes in the tests used, without it.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import json
import logging
import shlex
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from tinker_cookbook.sandbox import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = "/app/work"
RUN_SQL = "/opt/run_sql.py"
DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024


class _Commands(Protocol):
  async def run(self, command: str, timeout: int = 60) -> Any: ...


class _Files(Protocol):
  async def write(self, path: str, content: bytes | str, timeout: int = 60) -> Any: ...
  async def read(self, path: str, timeout: int = 60) -> bytes: ...


class SandboxHandle(Protocol):
  """The slice of ``k8s_agent_sandbox.AsyncSandbox`` this module uses."""

  claim_name: str
  namespace: str

  @property
  def commands(self) -> _Commands | None: ...
  @property
  def files(self) -> _Files | None: ...
  async def status(self) -> tuple[str, str]: ...
  async def terminate(self) -> None: ...


def _relative_path(path: str, base_dir: str) -> str:
  if not path.startswith("/"):
    return path
  prefix = base_dir.rstrip("/") + "/"
  if path.startswith(prefix):
    return path[len(prefix) :]
  raise ValueError(f"sandbox paths must be relative or under {base_dir}; got {path!r}")


def _truncate(text: str, limit: int | None) -> str:
  if limit is None or len(text.encode("utf-8", "replace")) <= limit:
    return text
  return text.encode("utf-8", "replace")[:limit].decode("utf-8", "ignore") + f"\n[truncated to {limit} bytes]"


class AgentSandboxBackend:
  """One claimed sandbox, exposed through tinker-cookbook's SandboxInterface."""

  def __init__(self, sandbox: SandboxHandle, *, base_dir: str = DEFAULT_BASE_DIR, on_cleanup: Callable[[], Awaitable[None]] | None = None):
    self._sandbox = sandbox
    self._base_dir = base_dir
    self._on_cleanup = on_cleanup
    self._closed = False

  @classmethod
  async def create(
    cls,
    *,
    warm_pool: str,
    namespace: str,
    ready_timeout: int = 180,
    connection: Any | None = None,
    client: Any | None = None,
    base_dir: str = DEFAULT_BASE_DIR,
  ) -> AgentSandboxBackend:
    """Claim a sandbox from ``warm_pool`` and wait until it is ready.

    ``client`` is an ``AsyncSandboxClient`` to share across claims; when omitted
    a private one is made with ``connection`` (default in-cluster pod IP).
    """
    own_client = client is None
    if own_client:
      client = make_client(connection)
    started = time.monotonic()
    sandbox = await client.create_sandbox(warm_pool, namespace=namespace, sandbox_ready_timeout=ready_timeout)
    logger.info("claimed sandbox %s/%s from %s in %.1fs", namespace, sandbox.claim_name, warm_pool, time.monotonic() - started)

    async def release() -> None:
      if own_client:
        await client.close()

    return cls(sandbox, base_dir=base_dir, on_cleanup=release)

  @property
  def sandbox_id(self) -> str:
    return self._sandbox.claim_name

  @property
  def base_dir(self) -> str:
    return self._base_dir

  def _commands(self) -> _Commands:
    commands = self._sandbox.commands
    if self._closed or commands is None:
      raise SandboxTerminatedError(f"sandbox {self.sandbox_id} is closed")
    return commands

  def _files(self) -> _Files:
    files = self._sandbox.files
    if self._closed or files is None:
      raise SandboxTerminatedError(f"sandbox {self.sandbox_id} is closed")
    return files

  async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult:
    # The runtime shlex-splits the command and runs it without a shell from
    # its base directory, so a working directory needs an explicit shell.
    if workdir is not None:
      command = "sh -c " + shlex.quote(f"cd {shlex.quote(_relative_path(workdir, self._base_dir) or '.')} && {command}")
    started = time.monotonic()
    result = await self._commands().run(command, timeout=timeout)
    limit = DEFAULT_MAX_OUTPUT_BYTES if max_output_bytes is None else max_output_bytes
    return SandboxResult(
      stdout=_truncate(result.stdout, limit),
      stderr=_truncate(result.stderr, limit),
      exit_code=int(result.exit_code),
      metrics={"duration_ms": (time.monotonic() - started) * 1000.0},
    )

  async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
    try:
      content = await self._files().read(_relative_path(path, self._base_dir), timeout=timeout)
    except Exception as exc:  # the SDK raises SandboxRequestError on 404; keep the interface's result shape
      return SandboxResult(stdout="", stderr=str(exc), exit_code=1)
    if max_bytes is not None:
      content = content[:max_bytes]
    return SandboxResult(stdout=content.decode("utf-8", "replace"), stderr="", exit_code=0)

  async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
    relative = _relative_path(path, self._base_dir)
    await self._files().write(relative, content, timeout=timeout)
    if executable:
      chmod = await self._commands().run(f"chmod +x {shlex.quote(relative)}", timeout=timeout)
      if chmod.exit_code != 0:
        return SandboxResult(stdout="", stderr=chmod.stderr, exit_code=int(chmod.exit_code))
    return SandboxResult(stdout="", stderr="", exit_code=0)

  async def send_heartbeat(self, timeout: int = 30) -> None:
    """The runtime has no keepalive; probe the claim and fail loudly if it is gone."""
    state, message = await asyncio.wait_for(self._sandbox.status(), timeout=timeout)
    if state != "SandboxReady":
      raise SandboxTerminatedError(f"sandbox {self.sandbox_id} is {state}: {message}")

  async def cleanup(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      await self._sandbox.terminate()
    finally:
      if self._on_cleanup is not None:
        await self._on_cleanup()


def make_client(connection: Any | None = None, *, cleanup_at_exit: bool = False) -> Any:
  """An ``AsyncSandboxClient`` that talks to pod IPs unless told otherwise.

  ``cleanup_at_exit`` registers the SDK's atexit sweep, which deletes every
  claim the client made using the synchronous Kubernetes client. Use it for a
  pool that lives as long as the process and has no natural close point (for
  example one shared by a cookbook dataset); leave it off when the owner
  deletes its own claims, so the sweep cannot race the event loop shutdown.
  """
  from k8s_agent_sandbox.async_sandbox_client import AsyncSandboxClient
  from k8s_agent_sandbox.models import SandboxInClusterConnectionConfig

  return AsyncSandboxClient(connection_config=connection or SandboxInClusterConnectionConfig(), cleanup=cleanup_at_exit)


class AgentSandboxPool:
  """N long-lived sandboxes leased round-robin, at most one lease per sandbox at a time.

  ``async with AgentSandboxPool(...) as pool:`` claims them; leaving the block
  deletes every claim. A sandbox that fails a heartbeat is replaced on the next
  lease so one dead pod does not poison the run.
  """

  def __init__(
    self,
    *,
    size: int,
    warm_pool: str,
    namespace: str,
    ready_timeout: int = 180,
    connection: Any | None = None,
    client: Any | None = None,
    factory: Callable[[], Awaitable[AgentSandboxBackend]] | None = None,
  ):
    if size < 1:
      raise ValueError("pool size must be at least 1")
    self.size = size
    self._warm_pool = warm_pool
    self._namespace = namespace
    self._ready_timeout = ready_timeout
    self._connection = connection
    self._client = client
    self._factory = factory
    self._idle: collections.deque[AgentSandboxBackend] = collections.deque()
    self._all: list[AgentSandboxBackend] = []
    self._available = asyncio.Semaphore(0)
    self._replacements = 0

  async def _claim(self) -> AgentSandboxBackend:
    if self._factory is not None:
      return await self._factory()
    return await AgentSandboxBackend.create(
      warm_pool=self._warm_pool, namespace=self._namespace, ready_timeout=self._ready_timeout, connection=self._connection, client=self._client
    )

  async def __aenter__(self) -> AgentSandboxPool:
    if self._client is None and self._factory is None:
      self._client = make_client(self._connection)
    started = time.monotonic()
    backends = await asyncio.gather(*(self._claim() for _ in range(self.size)))
    for backend in backends:
      self._all.append(backend)
      self._idle.append(backend)
      self._available.release()
    logger.info("sandbox pool ready: %d sandboxes in %.1fs", self.size, time.monotonic() - started)
    return self

  async def __aexit__(self, *exc: object) -> None:
    await self.close()

  async def close(self) -> None:
    backends, self._all = self._all, []
    self._idle.clear()
    results = await asyncio.gather(*(b.cleanup() for b in backends), return_exceptions=True)
    for backend, result in zip(backends, results):
      if isinstance(result, BaseException):
        logger.warning("cleanup of sandbox %s failed: %s", backend.sandbox_id, result)
    if self._client is not None and self._factory is None:
      await self._client.close()
      self._client = None

  @property
  def replacements(self) -> int:
    """How many sandboxes were replaced after a failed heartbeat."""
    return self._replacements

  @contextlib.asynccontextmanager
  async def lease(self) -> AsyncIterator[AgentSandboxBackend]:
    await self._available.acquire()
    backend = self._idle.popleft()
    try:
      yield backend
    except SandboxTerminatedError:
      backend = await self._replace(backend)
      raise
    finally:
      self._idle.append(backend)
      self._available.release()

  async def _replace(self, dead: AgentSandboxBackend) -> AgentSandboxBackend:
    logger.warning("replacing sandbox %s after it was reported terminated", dead.sandbox_id)
    with contextlib.suppress(Exception):
      await dead.cleanup()
    fresh = await self._claim()
    self._all = [fresh if b is dead else b for b in self._all]
    self._replacements += 1
    return fresh


SandboxFactory = Callable[[], Awaitable[AgentSandboxBackend]]


def make_sandbox_factory(*, warm_pool: str, namespace: str, ready_timeout: int = 180, connection: Any | None = None) -> SandboxFactory:
  """A zero-argument coroutine factory that claims one sandbox per call (Phase B: one per prompt group).

  One SDK client is shared by every claim the factory makes.
  """
  client: Any | None = None

  async def factory() -> AgentSandboxBackend:
    nonlocal client
    if client is None:
      client = make_client(connection)
    return await AgentSandboxBackend.create(warm_pool=warm_pool, namespace=namespace, ready_timeout=ready_timeout, client=client)

  return factory


def _decode_value(value: Any) -> Any:
  if isinstance(value, dict) and "__bytes__" in value:
    return base64.b64decode(value["__bytes__"])
  return value


async def run_sql_in_sandbox(sandbox: Any, context: str, query: str, *, timeout: int = 30) -> tuple[list[tuple[Any, ...]] | None, str | None]:
  """Execute ``query`` against ``context`` (the schema) inside ``sandbox``.

  Mirrors ``utils.rewards.run_sql``: returns ``(rows, None)`` or ``(None, error)``.
  Two round trips per call: one upload of the job file, one execute. The job
  file is removed by ``run_sql.py`` itself. Transport failures surface as an
  error string prefixed with ``sandbox:`` so they are scored like a bad query
  but stay distinguishable in the logs.
  """
  job = f"jobs/{uuid.uuid4().hex[:12]}.json"
  try:
    written = await sandbox.write_file(job, json.dumps({"schema": context, "query": query}), timeout=timeout)
    if written.exit_code != 0:
      return None, f"sandbox: upload failed: {written.stderr.strip()}"
    result = await sandbox.run_command(f"python {RUN_SQL} {job}", timeout=timeout)
  except SandboxTerminatedError:
    raise
  except Exception as exc:
    return None, f"sandbox: {type(exc).__name__}: {exc}"
  if result.exit_code != 0:
    return None, f"sandbox: run_sql exited {result.exit_code}: {(result.stderr or result.stdout).strip()[:300]}"
  try:
    payload = json.loads(result.stdout)
  except json.JSONDecodeError:
    return None, f"sandbox: unparsable output: {result.stdout[:300]!r}"
  if payload.get("error") is not None:
    return None, str(payload["error"])
  return [tuple(_decode_value(v) for v in row) for row in payload["rows"]], None
