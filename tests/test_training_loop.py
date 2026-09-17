"""The training loop end to end, with nothing faked.

A real tiny Llama on CPU, the real in-memory store, the real gateway
endpoints, the real request processor and the real LoRA and FFT workers.
Commands enter through the store the way the gateway enqueues them and
results come back through futures the way the client reads them.
"""

import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import torch

from accel_timeslicer.workload import WorkloadRef
from server import gateway
from server.store import InMemoryStore
from server.training_requests_processor import Deployment, TrainingRequestsProcessor
from tests.tiny_llama import VOCAB_SIZE, build_tiny_llama
from training import commands
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraAdapter, LoraTrainingWorker
from training.types import Datum


def datum(tokens: list[int], weights: list[float] | None = None) -> Datum:
  return Datum(
    model_input=tokens[:-1],
    loss_fn_inputs={"target_tokens": {"data": tokens[1:]}, "weights": {"data": weights or [1.0] * (len(tokens) - 1)}},
  )


def training_data(seed: int = 0) -> list[Datum]:
  generator = torch.Generator().manual_seed(seed)
  return [datum(torch.randint(3, VOCAB_SIZE, (length,), generator=generator).tolist()) for length in (6, 9, 4)]


class RecordingStore(InMemoryStore):
  """The real store, noting when results land so lease and publish order can be checked."""

  def __init__(self, events: list[tuple[str, Any]]):
    super().__init__()
    self.events = events

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") != "pending":
      self.events.append(("set_future", req_id))
    await super().set_future(req_id, result)


class RecordingTimeSlicer:
  def __init__(self, events: list[tuple[str, Any]]):
    self.events = events

  async def register(self, workload: WorkloadRef) -> dict[str, Any]:
    self.events.append(("register", workload))
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, workload: WorkloadRef):
    self.events.append(("acquire", workload))
    try:
      yield
    finally:
      self.events.append(("release", workload))

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    self.events.append(("unregister", workload))
    return {"ok": True}

  async def close(self) -> None:
    self.events.append(("close", None))


class TinyLlamaCase(unittest.IsolatedAsyncioTestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.tmp = tempfile.TemporaryDirectory()
    cls.base_model = build_tiny_llama(os.path.join(cls.tmp.name, "tiny-llama"))

  @classmethod
  def tearDownClass(cls) -> None:
    cls.tmp.cleanup()

  def setUp(self) -> None:
    self.env = patch.dict(os.environ, {"OPEN_RL_TMP_DIR": os.path.join(self.tmp.name, "open-rl"), "RANK": "0", "WORLD_SIZE": "1"})
    self.env.start()
    self.addCleanup(self.env.stop)

  async def run_commands(self, processor: TrainingRequestsProcessor, *batch: commands.Command) -> dict[str, dict[str, Any]]:
    for command in batch:
      await processor.store.put_request(commands.wire(command), active_set_id=processor.deployment.active_tenant_set_id)
    # The shared queue serves one model per batch, so drain once per model.
    for _ in {command.model_id for command in batch}:
      await processor.run_once()
    return {command.request_id: await processor.store.get_future(command.request_id, timeout=1) for command in batch}

  def lora_processor(self, store: InMemoryStore | None = None, time_slicer=None) -> TrainingRequestsProcessor:
    worker = LoraTrainingWorker()
    worker.device = torch.device("cpu")
    return TrainingRequestsProcessor(store or InMemoryStore(), worker, Deployment(), time_slicer)

  def fft_processor(self, store: InMemoryStore | None = None, time_slicer=None) -> TrainingRequestsProcessor:
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    return TrainingRequestsProcessor(store or InMemoryStore(), worker, Deployment(), time_slicer)

  def create_lora(self, model_id: str, rank: int = 4) -> commands.CreateModel:
    return commands.CreateModel(
      request_id=f"create-{model_id}", model_id=model_id, base_model=self.base_model, lora_config={"rank": rank, "seed": 1, "lora_dropout": 0.0}
    )

  def create_full(self, model_id: str) -> commands.CreateModel:
    return commands.CreateModel(
      request_id=f"create-{model_id}",
      model_id=model_id,
      base_model=self.base_model,
      fine_tuning_type="full",
      full_config={"seed": 1, "cpu_offload": False, "weight_sync_strategy": "full"},
    )


class LoraHostTest(TinyLlamaCase):
  async def test_two_adapters_train_independently_on_one_base(self) -> None:
    processor = self.lora_processor()
    results = await self.run_commands(processor, self.create_lora("a"), self.create_lora("b"))
    self.assertEqual(results["create-a"]["type"], "model_created")
    self.assertEqual(results["create-a"]["rank"], 4)
    self.assertIsInstance(processor.worker.trainer("a"), LoraAdapter)
    self.assertIsInstance(processor.worker.trainer("b"), LoraAdapter)

    data = training_data()
    before = await self.run_commands(processor, commands.ForwardBackward(request_id="fb-a", model_id="a", data=data))
    loss_before = before["fb-a"]["metrics"]["loss:mean"]
    self.assertEqual(len(before["fb-a"]["loss_fn_outputs"]), len(data))

    step = await self.run_commands(processor, commands.OptimStep(request_id="step-a", model_id="a", adam_params={"learning_rate": 5e-2}))
    self.assertIn("grad_norm:mean", step["step-a"]["metrics"])
    self.assertIn("ratio/max_abs_log:max", step["step-a"]["metrics"])

    after_a = await self.run_commands(processor, commands.ForwardBackward(request_id="fb-a2", model_id="a", data=data))
    untouched_b = await self.run_commands(processor, commands.ForwardBackward(request_id="fb-b", model_id="b", data=data))
    self.assertLess(after_a["fb-a2"]["metrics"]["loss:mean"], loss_before)
    # b never stepped, so it still scores the data like a fresh adapter.
    self.assertAlmostEqual(untouched_b["fb-b"]["metrics"]["loss:mean"], loss_before, places=4)

  async def test_sampler_adapter_and_state_round_trip(self) -> None:
    processor = self.lora_processor()
    await self.run_commands(processor, self.create_lora("a"))
    data = training_data()
    await self.run_commands(
      processor,
      commands.ForwardBackward(request_id="fb", model_id="a", data=data),
      commands.OptimStep(request_id="step", model_id="a", adam_params={"learning_rate": 5e-2}),
    )

    published = await self.run_commands(
      processor, commands.SaveWeightsForSampler(request_id="pub", model_id="a", sampling_session_id="tinker://a/sampler_weights/sampler-0")
    )
    weights = published["pub"]["sampler_weights"]
    self.assertEqual(weights["kind"], "adapter")
    self.assertEqual(weights["path"], os.path.join(os.environ["OPEN_RL_TMP_DIR"], "peft", "a", "a"))
    self.assertTrue(os.path.exists(os.path.join(weights["path"], "adapter_model.safetensors")))

    state_path = os.path.join(self.tmp.name, "state-a")
    saved = await self.run_commands(processor, commands.SaveState(request_id="save", model_id="a", state_path=state_path, include_optimizer=True))
    self.assertEqual(saved["save"], {"path": state_path, "type": "state_saved"})
    self.assertTrue(os.path.exists(os.path.join(state_path, "optimizer.pt")))

    trained = await self.run_commands(processor, commands.ForwardBackward(request_id="fb-a", model_id="a", data=data))
    restored = await self.run_commands(
      processor,
      commands.CreateModelFromState(request_id="restore", model_id="c", state_path=state_path, restore_optimizer=True),
      commands.ForwardBackward(request_id="fb-c", model_id="c", data=data),
    )
    self.assertEqual(restored["restore"]["type"], "model_loaded_from_state")
    self.assertEqual(restored["restore"]["base_model"], self.base_model)
    self.assertAlmostEqual(restored["fb-c"]["metrics"]["loss:mean"], trained["fb-a"]["metrics"]["loss:mean"], places=4)

  async def test_unknown_adapter_fails_the_request_not_the_loop(self) -> None:
    processor = self.lora_processor()
    await self.run_commands(processor, self.create_lora("a"))
    results = await self.run_commands(
      processor,
      commands.ForwardBackward(request_id="bad", model_id="nope", data=training_data()),
      commands.ForwardBackward(request_id="good", model_id="a", data=training_data()),
    )
    self.assertEqual(results["bad"]["type"], "RequestFailedResponse")
    self.assertIn("nope", results["bad"]["error_message"])
    self.assertEqual(results["good"]["type"], "forward_backward_completed")

  async def test_sampling_from_the_trainer(self) -> None:
    processor = self.lora_processor()
    await self.run_commands(processor, self.create_lora("a"))
    sampled = await self.run_commands(
      processor, commands.Sample(request_id="s", model_id="a", prompt_tokens=[1, 5, 6], max_tokens=3, num_samples=2, temperature=1.0)
    )
    self.assertEqual(sampled["s"]["type"], "sample_completed")
    self.assertEqual(len(sampled["s"]["sequences"]), 2)
    self.assertEqual(len(sampled["s"]["sequences"][0]["tokens"]), 3)


class FullParameterWorkerTest(TinyLlamaCase):
  async def test_full_model_trains_and_publishes_checkpoints_under_the_lease(self) -> None:
    events: list[tuple[str, Any]] = []
    processor = self.fft_processor(RecordingStore(events), RecordingTimeSlicer(events))
    await processor.time_slicer.register(processor.workload)
    created = await self.run_commands(processor, self.create_full("m"))
    self.assertEqual(created["create-m"]["type"], "model_created")
    self.assertEqual(created["create-m"]["fine_tuning_type"], "full")
    # The lease is released before the result is published.
    self.assertEqual([name for name, _ in events], ["register", "acquire", "release", "set_future"])
    self.assertEqual(events[1][1].name, "trainer-shared")

    data = training_data()
    before = await self.run_commands(processor, commands.ForwardBackward(request_id="fb", model_id="m", data=data))
    await self.run_commands(processor, commands.OptimStep(request_id="step", model_id="m", adam_params={"learning_rate": 1e-2}))
    after = await self.run_commands(processor, commands.ForwardBackward(request_id="fb2", model_id="m", data=data))
    self.assertLess(after["fb2"]["metrics"]["loss:mean"], before["fb"]["metrics"]["loss:mean"])

    events.clear()
    published = await self.run_commands(
      processor,
      commands.SaveWeightsForSampler(
        request_id="pub", model_id="m", path="tinker://m/sampler_weights/final", sampling_session_id="tinker://m/sampler_weights/sampler-0"
      ),
    )
    weights = published["pub"]["sampler_weights"]
    self.assertEqual(weights["kind"], "checkpoint")
    self.assertEqual(weights["path"], os.path.join(os.environ["OPEN_RL_TMP_DIR"], "sampler_full", "m", "sampler_weights", "final"))
    self.assertTrue(os.path.exists(os.path.join(weights["path"], "config.json")))
    # A GPU-resident full model saves inside the lease.
    self.assertEqual([name for name, _ in events], ["acquire", "release", "set_future"])

    # FFT checkpoints carry weights only; the optimizer is not resumable yet.
    state_path = os.path.join(self.tmp.name, "state-m")
    await self.run_commands(processor, commands.SaveState(request_id="save", model_id="m", state_path=state_path))
    self.assertTrue(os.path.exists(os.path.join(state_path, "metadata.json")))
    reloaded = await self.run_commands(
      processor,
      commands.LoadWeights(request_id="load", model_id="m", state_path=state_path),
      commands.ForwardBackward(request_id="fb3", model_id="m", data=data),
    )
    self.assertEqual(reloaded["load"]["type"], "weights_loaded")
    self.assertAlmostEqual(reloaded["fb3"]["metrics"]["loss:mean"], after["fb2"]["metrics"]["loss:mean"], places=4)


class DeploymentTest(unittest.IsolatedAsyncioTestCase):
  def test_a_dedicated_queue_needs_redis(self) -> None:
    with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      TrainingRequestsProcessor(InMemoryStore(), FFTTrainingWorker(), Deployment(model_id="m"))

  async def test_only_rank_zero_writes_to_the_store(self) -> None:
    store = InMemoryStore()
    await store.set_value("open_rl:model_meta:m", '{"total_steps_completed": 3}')
    processor = TrainingRequestsProcessor(store, LoraTrainingWorker(), Deployment())

    with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"}):
      await processor.publish_result("r1", {"type": "ok"})
      await processor.bump_step_count("m")
    self.assertNotIn("r1", store.futures_store)
    self.assertEqual((await store.get_model_metadata("m"))["total_steps_completed"], 3)

    with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
      await processor.publish_result("r1", {"type": "ok"})
      await processor.bump_step_count("m")
    self.assertEqual(store.futures_store["r1"], {"type": "ok"})
    self.assertEqual((await store.get_model_metadata("m"))["total_steps_completed"], 4)


class GatewayToWorkerTest(TinyLlamaCase):
  async def test_api_calls_become_commands_the_worker_answers(self) -> None:
    store = InMemoryStore()
    with patch.object(gateway, "store", store):
      created = await gateway.create_model({"base_model": self.base_model, "lora_config": {"rank": 2, "seed": 1}})
      model_id = created["request_id"]
      processor = self.lora_processor(store)
      processor = TrainingRequestsProcessor(store, processor.worker, Deployment(active_tenant_set_id=f"{self.base_model}-1"))
      await processor.run_once()
      self.assertEqual(gateway.translate_future_result(await store.get_future(model_id, timeout=1))["lora_rank"], 2)

      wire_data = [
        {"model_input": {"chunks": [{"tokens": d.model_input}]}, "loss_fn_inputs": {k: v.data for k, v in d.loss_fn_inputs.items()}}
        for d in training_data()
      ]
      fb = await gateway.forward_backward({"model_id": model_id, "forward_backward_input": {"data": wire_data, "loss_fn": "cross_entropy"}})
      step = await gateway.optim_step({"model_id": model_id, "adam_params": {"learning_rate": 1e-2}})
      saved = await gateway.save_weights_for_sampler({"model_id": model_id, "sampling_session_seq_id": 0})
      await processor.run_once()

      fb_result = await store.get_future(fb["request_id"], timeout=1)
      self.assertEqual(fb_result["type"], "forward_backward_completed")
      self.assertEqual(len(fb_result["loss_fn_outputs"]), 3)
      self.assertEqual((await store.get_future(step["request_id"], timeout=1))["type"], "optim_step_completed")
      self.assertEqual((await store.get_model_metadata(model_id))["total_steps_completed"], 1)
      published = await store.get_future(saved["request_id"], timeout=1)
      self.assertEqual(published["sampler_weights"]["kind"], "adapter")
      self.assertTrue(os.path.isdir(os.path.join(os.environ["OPEN_RL_TMP_DIR"], "peft", model_id, model_id)))


if __name__ == "__main__":
  unittest.main()
