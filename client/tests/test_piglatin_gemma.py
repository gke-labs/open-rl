import os
import subprocess

# Adjust path to import client code
import sys
import time
import unittest
from pathlib import Path

import requests

client_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(client_dir))

from piglatin_sft import PRESETS, run_training


class TestPigLatinGemma(unittest.TestCase):
  server_process = None
  BASE_URL = "http://127.0.0.1:9002"

  @classmethod
  def setUpClass(cls):
    if not os.environ.get("HF_TOKEN"):
      raise unittest.SkipTest("HF_TOKEN environment variable is not set. Gemma 3 requires HF_TOKEN to download models.")

    print("\nStarting Open-RL Server for Gemma Pig Latin...")
    env = os.environ.copy()

    # We start the server directly using uv run to ensure we can kill it easily
    server_dir = client_dir.parent / "server"

    # Mimic `make run-pig-latin-gemma-server`
    env["ENABLE_GCP_TRACE"] = env.get("ENABLE_GCP_TRACE", "0")
    env["UV_INDEX_URL"] = "https://pypi.org/simple"
    env["OPEN_RL_SINGLE_PROCESS"] = "1"
    env["OPEN_RL_BASE_MODEL"] = "google/gemma-3-1b-it"
    env["SAMPLER_BACKEND"] = "engine"
    env["VLLM_MODEL"] = "google/gemma-3-1b-it"

    cmd = ["uv", "run", "--extra", "cpu", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "9002"]

    cls.server_process = subprocess.Popen(
      cmd, cwd=str(server_dir), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid
    )

    # Wait for server to be ready
    ready = False
    for _ in range(60):
      try:
        resp = requests.get(f"{cls.BASE_URL}/api/v1/get_server_capabilities", timeout=2)
        if resp.status_code == 200:
          ready = True
          break
      except Exception:
        time.sleep(2)

    if not ready:
      cls.tearDownClass()
      raise RuntimeError("Server failed to start within 120 seconds.")
    print("Server is ready!")

  @classmethod
  def tearDownClass(cls):
    if cls.server_process:
      print("\nShutting down server...")
      try:
        # Kill the entire process group
        os.killpg(os.getpgid(cls.server_process.pid), 9)
      except Exception as e:
        print(f"Error killing server process: {e}")
      cls.server_process.wait()

  def test_gemma_sft_accuracy(self):
    print("\nRunning Gemma SFT Training...")
    # We use a slightly reduced set of steps for the unit test,
    # but large enough to show learning (15 steps)
    # We also pass assert_improvement=False to handle assertions in the test
    plot_path = str(client_dir / "artifacts" / "test_gemma_metrics.png")

    blueprint = (
      PRESETS["gemma"]
      .clone()
      .apply(
        {
          "base_url": self.BASE_URL,
          "steps": 25,
          "assert_improvement": False,
          "plot_path": plot_path,
          "custom_examples": [
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
          ],
        },
        layer_name="test_override",
      )
    )

    config = blueprint.make()

    metrics = run_training(config)

    print(f"\nTraining completed. Metrics: {metrics}")

    after_exact = metrics["after_exact"]
    before_exact = metrics["before_exact"]
    after_sim = metrics["after_sim"]
    before_sim = metrics["before_sim"]

    # Verify improvement
    self.assertGreater(after_exact, before_exact, f"Exact match didn't improve: {before_exact:.2f} -> {after_exact:.2f}")
    self.assertGreaterEqual(after_sim - before_sim, config.min_similarity_gain, f"Similarity gain insufficient: {before_sim:.2f} -> {after_sim:.2f}")

    # For a 1B model with 25 steps, we expect it to hit a certain baseline
    self.assertGreaterEqual(after_exact, 0.20, f"Expected at least 20% exact match, got {after_exact:.0%}")

    print("✅ Gemma Pig Latin unit test passed successfully!")


if __name__ == "__main__":
  unittest.main()
