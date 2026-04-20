"""Shared test fixture: spin up an Open-RL single-process server.

Used by the pig-latin end-to-end tests to avoid re-implementing the
subprocess dance in every test file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import requests

CLIENT_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = CLIENT_DIR.parent / "server"
sys.path.insert(0, str(CLIENT_DIR))

# Held-out eval set: the model should learn to produce these pig-latin forms.
# Shared across pig-latin tests so results are comparable model-to-model.
PIGLATIN_EVAL_EXAMPLES = [
  ("banana", "anana-bay"),
  ("quantum", "uantum-qay"),
  ("donut", "onut-day"),
  ("pickle", "ickle-pay"),
  ("space", "ace-spay"),
  ("rubber", "ubber-ray"),
  ("coding", "oding-cay"),
  ("hello", "ello-hay"),
  ("machine", "achine-may"),
  ("artificial", "rtificial-aay"),
  ("data", "ata-day"),
]


class OpenRlServerCase(unittest.TestCase):
  """Base class that starts a single-process Open-RL server before the tests run.

  Subclasses override:
    BASE_MODEL  — Hugging Face model id (can be overridden via BASE_MODEL env var)
    PORT        — gateway port (can be overridden via PORT env var)
    REQUIRE_HF_TOKEN — skip the suite unless HF_TOKEN is set (for gated models)
    STARTUP_TIMEOUT — seconds to wait for the gateway to respond
  """

  BASE_MODEL: str = "Qwen/Qwen3-0.6B"
  PORT: int = 9010
  REQUIRE_HF_TOKEN: bool = False
  STARTUP_TIMEOUT: int = 300

  _server: subprocess.Popen | None = None

  @classmethod
  def setUpClass(cls) -> None:
    if cls.REQUIRE_HF_TOKEN and not os.getenv("HF_TOKEN"):
      raise unittest.SkipTest(f"{cls.BASE_MODEL} requires HF_TOKEN; skipping.")

    base_model = os.getenv("BASE_MODEL", cls.BASE_MODEL)
    port = int(os.getenv("PORT", cls.PORT))
    cls.BASE_URL = f"http://127.0.0.1:{port}"

    print(f"\nStarting Open-RL server with BASE_MODEL={base_model} on port {port}...")
    env = {
      **os.environ,
      "SINGLE_PROCESS": "1",
      "SAMPLER": "torch",
      "BASE_MODEL": base_model,
    }
    cls._server = subprocess.Popen(
      ["uv", "run", "--extra", "cpu", "python", "-m", "uvicorn", "src.gateway:app", "--host", "127.0.0.1", "--port", str(port)],
      cwd=str(SERVER_DIR),
      env=env,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      preexec_fn=os.setsid,
    )

    deadline = time.time() + cls.STARTUP_TIMEOUT
    while time.time() < deadline:
      try:
        resp = requests.get(f"{cls.BASE_URL}/api/v1/get_server_capabilities", timeout=2)
        if resp.status_code == 200:
          print("Server ready.")
          return
      except Exception:
        pass
      time.sleep(2)

    cls.tearDownClass()
    raise RuntimeError(f"Server failed to start within {cls.STARTUP_TIMEOUT}s.")

  @classmethod
  def tearDownClass(cls) -> None:
    if cls._server is None:
      return
    print("\nShutting down server...")
    try:
      os.killpg(os.getpgid(cls._server.pid), 9)
    except Exception as exc:
      print(f"Error killing server: {exc}")
    cls._server.wait()
    cls._server = None
