"""Shared helpers for tests that need a local Open-RL gateway."""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "src" / "server"


def unused_tcp_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


@contextlib.contextmanager
def openrl_server(
  base_model: str,
  port: int | None = None,
  sampling_backend: str = "torch",
  startup_timeout: float = 60.0,
  health_path: str = "/api/v1/healthz",
  uv_run_args: Sequence[str] = ("--frozen",),
  extra_env: dict[str, str] | None = None,
  stdout: int | None = subprocess.PIPE,
  stderr: int | None = subprocess.STDOUT,
) -> Iterator[str]:
  port = port or unused_tcp_port()
  base_url = f"http://127.0.0.1:{port}"
  tmp_dir = tempfile.TemporaryDirectory(prefix="open-rl-test-", dir="/dev/shm")
  env = {
    **os.environ,
    "BASE_MODEL": base_model,
    "PYTHONUNBUFFERED": "1",
    "ENABLE_GCP_TRACE": "0",
    "OPEN_RL_TMP_DIR": tmp_dir.name,
    "SAMPLING_BACKEND": sampling_backend,
    "TINKER_API_KEY": "tml-dummy-key",
    "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache"),
  }
  if extra_env:
    env.update(extra_env)
  env.pop("REDIS_URL", None)

  command = [
    "uv",
    "run",
    *uv_run_args,
    "python",
    "-m",
    "uvicorn",
    "gateway:app",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--log-level",
    "warning",
  ]
  proc = subprocess.Popen(command, cwd=SERVER_DIR, env=env, stdout=stdout, stderr=stderr, text=True, preexec_fn=os.setsid)
  try:
    wait_until_healthy(proc, f"{base_url}{health_path}", startup_timeout)
    yield base_url
  finally:
    stop_server(proc)
    tmp_dir.cleanup()


def wait_until_healthy(proc: subprocess.Popen, health_url: str, timeout: float) -> None:
  deadline = time.monotonic() + timeout
  last_error: Exception | None = None
  while time.monotonic() < deadline:
    if proc.poll() is not None:
      output, _ = proc.communicate(timeout=1)
      raise RuntimeError(f"Open-RL server exited during startup:\n{output}")
    try:
      with urllib.request.urlopen(health_url, timeout=0.5) as response:
        if response.status == 200:
          return
    except (OSError, urllib.error.URLError) as exc:
      last_error = exc
    time.sleep(0.1)
  stop_server(proc, close_stdout=False)
  output = proc.stdout.read() if proc.stdout is not None else ""
  if proc.stdout is not None:
    proc.stdout.close()
  raise RuntimeError(f"Open-RL server did not become healthy: {last_error}\n{output}")


def signal_process_group(proc: subprocess.Popen, sig: int) -> None:
  try:
    os.killpg(os.getpgid(proc.pid), sig)
  except ProcessLookupError:
    pass


def wait_for_exit(proc: subprocess.Popen, timeout: float) -> None:
  try:
    proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    pass


def stop_server(proc: subprocess.Popen, close_stdout: bool = True) -> None:
  if proc.poll() is None:
    signal_process_group(proc, signal.SIGTERM)
    wait_for_exit(proc, timeout=5)
  if proc.poll() is None:
    signal_process_group(proc, signal.SIGKILL)
    proc.wait(timeout=5)
  if close_stdout and proc.stdout is not None:
    proc.stdout.close()


class OpenRlServerCase(unittest.TestCase):
  """Base unittest case that starts a local Open-RL server for the class."""

  BASE_MODEL: str = "Qwen/Qwen3-0.6B"
  PORT: int = 9010
  REQUIRE_HF_TOKEN: bool = False
  STARTUP_TIMEOUT: int = 300
  SAMPLING_BACKEND: str = "torch"
  UV_RUN_ARGS: Sequence[str] = ("--extra", "cpu")
  HEALTH_PATH: str = "/api/v1/get_server_capabilities"

  _server_context: contextlib.AbstractContextManager[str] | None = None

  @classmethod
  def setUpClass(cls) -> None:
    if cls.REQUIRE_HF_TOKEN and not os.getenv("HF_TOKEN"):
      raise unittest.SkipTest(f"{cls.BASE_MODEL} requires HF_TOKEN; skipping.")

    base_model = os.getenv("BASE_MODEL", cls.BASE_MODEL)
    port = int(os.getenv("PORT", cls.PORT))
    print(f"\nStarting Open-RL server with BASE_MODEL={base_model} on port {port}...")
    cls._server_context = openrl_server(
      base_model,
      port=port,
      sampling_backend=cls.SAMPLING_BACKEND,
      startup_timeout=cls.STARTUP_TIMEOUT,
      health_path=cls.HEALTH_PATH,
      uv_run_args=cls.UV_RUN_ARGS,
      stdout=None,
      stderr=None,
    )
    cls.BASE_URL = cls._server_context.__enter__()
    print("Server ready.")

  @classmethod
  def tearDownClass(cls) -> None:
    if cls._server_context is None:
      return
    print("\nShutting down server...")
    cls._server_context.__exit__(None, None, None)
    cls._server_context = None
