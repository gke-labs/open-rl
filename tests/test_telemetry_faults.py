"""Telemetry must never change what a worker does. Each test serves the same work with a healthy telemetry store and with
one that fails, and expects the same answers, in about the same time."""

import asyncio
import os
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from server import telemetry
from server.store import InMemoryStateStore
from server.training_requests_processor import TrainingRequestsProcessor
from tests.test_fft_batch_failure import BatchStore, SlicerStub

try:
  from server import vllm_sampler
  from tests.test_sampler_patch import make_engine
except ImportError:  # vLLM is a GPU extra
  vllm_sampler = None

BOUND = 1.0  # a hung write is dropped after WRITE_SECONDS; the serve must finish well inside it


class FaultyStore:
  """Every call raises, as with Redis down, or waits forever, as with a Redis that never answers."""

  def __init__(self, hangs: bool):
    self.hangs = hangs

  def __getattr__(self, name):
    if not self.hangs:
      raise ConnectionError(f"telemetry store {name} failed")
    if name.endswith("_sync"):
      return lambda *args, **kwargs: threading.Event().wait()
    return lambda *args, **kwargs: asyncio.Event().wait()


def no_store():
  raise RuntimeError("no telemetry store configured")


FAULTS = {"raises": lambda: FaultyStore(False), "hangs": lambda: FaultyStore(True), "is missing": no_store}


class TelemetryFaultTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.object(telemetry, "WRITE_SECONDS", 5 * BOUND))
    self.enterContext(patch.object(telemetry, "FINISH_SECONDS", BOUND))
    self.enterContext(patch.dict(telemetry.run_steps, clear=True))

  def assert_unchanged(self, work) -> None:
    async def serve():
      async with telemetry.serving("trainer", "run-a"):
        started = time.monotonic()
        result = await asyncio.wait_for(work(), 2 * BOUND)
        exiting = time.monotonic()
      return result, exiting - started, time.monotonic() - exiting

    healthy = InMemoryStateStore()
    with patch.object(telemetry, "get_state_store", lambda: healthy):
      expected, _, _ = asyncio.run(serve())
    self.assertTrue(expected)
    self.assertTrue(healthy.samples or healthy.get_value_sync(telemetry.key("sampled", "run-a")), "the healthy serve recorded its ops")
    for fault, store in FAULTS.items():
      with self.subTest(store=fault), patch.object(telemetry, "get_state_store", store):
        result, took, exiting = asyncio.run(serve())
        self.assertEqual(result, expected)
        self.assertLess(took, BOUND)
        self.assertLess(exiting, 2 * BOUND)

  def test_a_time_sliced_training_request_is_answered_the_same(self) -> None:
    trainer = Mock(cpu_offload=True, optim_step=lambda adam_params, model_id: {"metrics": {"grad_norm:mean": 0.5}})

    async def train():
      store = BatchStore([{"request_id": f"os-{i}", "op": "optim_step", "model_id": "run-a", "adam_params": {}} for i in range(2)])
      with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
        processor = TrainingRequestsProcessor(store, trainer, "run-a", time_slicer=SlicerStub())
      await processor.run_once()
      return store.futures

    self.assert_unchanged(train)

  @unittest.skipIf(vllm_sampler is None, "needs vLLM")
  def test_a_sample_request_is_answered_the_same(self) -> None:
    async def sample():
      store = AsyncMock()
      store.get_sampling_requests_for_model.side_effect = [
        [{"request_id": "s-1", "model_id": "run-a", "prompt_token_ids": [1, 2], "max_tokens": 1}],
        [{"request_id": vllm_sampler.SHUTDOWN_SENTINEL}],
      ]
      with patch.object(vllm_sampler, "get_state_store", return_value=AsyncMock()):
        await vllm_sampler.serve("run-a", store, make_engine, SlicerStub(), vllm_sampler.WorkloadRef("sampler-run-a"))
      return {call.args[0]: call.args[1] for call in store.set_result.call_args_list}

    self.assert_unchanged(sample)


if __name__ == "__main__":
  unittest.main()
