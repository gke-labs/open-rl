"""Weight-version ordering, failure handling, and resource cleanup for the vLLM sampler."""

import asyncio
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from accel_timeslicer.workload import WorkloadRef
from server.vllm_sampler import Sampler


def make_engine(on_generate=None):
  engine = AsyncMock()
  engine.shutdown = Mock()

  def generate(prompt, sampling_params, request_id, lora_request):
    async def stream():
      if on_generate is not None:
        await on_generate(request_id)
      yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4], logprobs=None, finish_reason="length")], prompt_logprobs=None)

    return stream()

  engine.generate = Mock(side_effect=generate)
  return engine


class SamplerBatchTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.store = AsyncMock()
    self.engine = make_engine()
    self.sampler = Sampler("test", self.store, lambda: self.engine)
    self.sampler.engine = self.engine

  def results(self):
    return {call.args[0]: call.args[1] for call in self.store.set_future.call_args_list}

  async def test_groups_drain_before_switching_weights(self):
    active = set()
    seen = []

    async def update(update_request):
      self.assertFalse(active)
      seen.append(("update", update_request.update_info["target_weights_path"]))

    async def generate(request_id):
      active.add(request_id)
      await asyncio.sleep(0)
      seen.append(("generate", request_id))
      active.remove(request_id)

    self.engine.update_weights.side_effect = update
    self.engine.generate.side_effect = make_engine(generate).generate
    requests = [{"request_id": path + str(i), "weights_path": path} for i, path in enumerate(["a", "a", "b", "a"])]
    await self.sampler.process_batch(requests)
    self.assertEqual(
      seen, [("update", "a"), ("generate", "a0"), ("generate", "a1"), ("update", "b"), ("generate", "b2"), ("update", "a"), ("generate", "a3")]
    )
    self.assertEqual(self.sampler.weights_path, "a")

  async def test_unchanged_weights_skip_update(self):
    for request_id in ("1", "2"):
      await self.sampler.process_batch([{"request_id": request_id, "weights_path": "a"}])
    self.engine.update_weights.assert_awaited_once()
    self.engine.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)
    self.engine.finish_weight_update.assert_awaited_once_with(weight_version="a")
    self.engine.resume_generation.assert_awaited_once()
    self.engine.sleep.assert_not_called()
    self.engine.wake_up.assert_not_called()

  async def test_failed_update_poisons_sampler_without_committing(self):
    self.engine.update_weights.side_effect = RuntimeError("invalid patch")
    await self.sampler.process_batch([{"request_id": "bad", "weights_path": "a"}])
    self.assertIn("invalid patch", self.results()["bad"]["error_message"])
    self.engine.generate.assert_not_called()
    self.engine.finish_weight_update.assert_not_called()
    self.engine.resume_generation.assert_not_called()
    self.assertIsNone(self.sampler.weights_path)
    await self.sampler.process_batch([{"request_id": "no-path"}])
    self.assertIn("restart", self.results()["no-path"]["error_message"])

  async def test_generation_failure_is_reported_without_poisoning_weights(self):
    self.engine.generate.side_effect = RuntimeError("generation error")
    await self.sampler.process_batch([{"request_id": "1", "weights_path": "a"}])
    self.assertIn("generation error", self.results()["1"]["error_message"])
    self.engine.generate.side_effect = make_engine().generate
    await self.sampler.process_batch([{"request_id": "2", "weights_path": "a"}])
    self.assertEqual(self.results()["2"]["type"], "sample")

  async def test_generation_preserves_tokens_logprobs_and_stop_options(self):
    async def outputs():
      yield SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[4], logprobs=[{4: SimpleNamespace(logprob=-0.25)}], finish_reason="length")],
        prompt_logprobs=[None, {2: SimpleNamespace(logprob=-0.5)}],
      )

    self.engine.generate = Mock(return_value=outputs())
    result = await self.sampler.generate(
      {"request_id": "req", "prompt_token_ids": [1, 2], "stop": [7], "max_tokens": 1, "include_prompt_logprobs": True}
    )
    self.assertEqual(result["sequences"], [{"tokens": [4], "logprobs": [-0.25], "stop_reason": "length"}])
    self.assertEqual(result["prompt_logprobs"], [None, -0.5])
    params = self.engine.generate.call_args.kwargs["sampling_params"]
    self.assertEqual(params.stop_token_ids, [7])
    self.assertEqual(params.max_tokens, 1)
    self.assertEqual(params.prompt_logprobs, 1)
    self.assertIsNone(self.engine.generate.call_args.kwargs["lora_request"])

  async def test_lora_request_attached_only_when_adapter_exists(self):
    with tempfile.TemporaryDirectory() as adapter:
      await self.sampler.process_batch([{"request_id": "1", "lora_id": "job-a", "lora_path": adapter}])
      self.assertIsNone(self.engine.generate.call_args.kwargs["lora_request"])
      open(os.path.join(adapter, "adapter_config.json"), "w").close()
      await self.sampler.process_batch([{"request_id": "2", "lora_id": "job-a", "lora_path": adapter}])
      lora_request = self.engine.generate.call_args.kwargs["lora_request"]
      self.assertEqual((lora_request.lora_name, lora_request.lora_path), ("job-a", adapter))
    self.engine.update_weights.assert_not_called()


class SamplerLifecycleTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.events = []
    self.store = AsyncMock()
    self.engine = make_engine()
    self.slicer = AsyncMock()
    self.slicer.acquire = Mock(side_effect=self.slot)
    self.workload = WorkloadRef("sampler-test")
    self.factory = Mock(side_effect=self.create_engine)
    self.sampler = Sampler("test", self.store, self.factory, time_slicer=self.slicer, workload=self.workload)

  @asynccontextmanager
  async def slot(self, workload):
    self.events.append("acquire")
    try:
      yield
    finally:
      self.events.append("release")

  def create_engine(self):
    self.events.append("create")
    return self.engine

  async def test_initialization_and_batches_own_slots_and_cleanup_once(self):
    self.store.get_sampling_requests_for_model.return_value = [{"request_id": "1"}, {"request_id": "SHUTDOWN_SENTINEL"}]
    await self.sampler.run()
    await self.sampler.close()
    self.assertEqual(self.events, ["acquire", "create", "release", "acquire", "release"])
    self.assertEqual(self.engine.sleep.await_count, 2)
    self.engine.wake_up.assert_awaited_once()
    self.engine.shutdown.assert_called_once()
    self.slicer.unregister.assert_awaited_once_with(self.workload)
    self.slicer.close.assert_awaited_once()
    self.assertEqual(self.store.set_future.call_args.args[0], "1")

  async def test_registration_failure_never_constructs_engine(self):
    self.slicer.register.side_effect = RuntimeError("registration failed")
    with self.assertRaisesRegex(RuntimeError, "registration failed"):
      await self.sampler.run()
    self.factory.assert_not_called()
    self.slicer.unregister.assert_not_called()
    self.slicer.close.assert_awaited_once()

  async def test_initialization_failure_unregisters_without_unlocked_retry(self):
    self.factory.side_effect = RuntimeError("engine failed")
    with self.assertRaisesRegex(RuntimeError, "engine failed"):
      await self.sampler.run()
    self.factory.assert_called_once()
    self.slicer.unregister.assert_awaited_once()
    self.slicer.close.assert_awaited_once()

  async def test_cancellation_shuts_down_engine_and_unregisters(self):
    entered = asyncio.Event()

    async def get_batch(model_id):
      entered.set()
      await asyncio.Event().wait()

    self.store.get_sampling_requests_for_model.side_effect = get_batch
    task = asyncio.create_task(self.sampler.run())
    await entered.wait()
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    self.engine.shutdown.assert_called_once()
    self.slicer.unregister.assert_awaited_once()
    self.slicer.close.assert_awaited_once()

  async def test_unshared_sampler_does_not_sleep_or_wake_engine(self):
    sampler = Sampler("test", self.store, lambda: self.engine)
    self.store.get_sampling_requests_for_model.return_value = [{"request_id": "1"}, {"request_id": "SHUTDOWN_SENTINEL"}]
    await sampler.run()
    self.engine.wake_up.assert_not_called()
    self.engine.sleep.assert_not_called()
    self.engine.shutdown.assert_called_once()


if __name__ == "__main__":
  unittest.main()
