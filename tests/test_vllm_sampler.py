"""The in-process sampler's turns on a GPU it shares with its trainer, with the real time slicer."""

import asyncio
import json
import os
import sys
import tempfile
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from accel_timeslicer.serve import start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from server.sampling_dispatcher import Dispatcher
from server.sampling_store import SamplingStore, future_key
from server.vllm_sampler import Sampler, TimeSlicedDispatcher
from tests.redis_server import RedisTestCase
from tests.test_sampling_dispatcher import SAMPLE, until


class RecordingRestorer:
  """Parks and restores nothing, and logs each park and restore the time slicer asks for."""

  def __init__(self, log: list) -> None:
    self.log = log

  def checkpoint(self, workload: WorkloadRef) -> None:
    self.log.append(("park", workload.name))

  def restore(self, workload: WorkloadRef) -> None:
    self.log.append(("restore", workload.name))


class GpuSampler:
  """Stands in for vllm_sampler.Sampler, whose vLLM engine needs a GPU. It logs each call and fails any that reaches
  it off the GPU, or while the trainer holds it."""

  def __init__(self, log: list, gpu: set) -> None:
    self.log, self.gpu, self.revision = log, gpu, ""

  async def wake(self) -> None:
    self.gpu.add("sampler")
    self.log.append(("wake", "sampler"))

  async def sleep(self) -> None:
    self.gpu.discard("sampler")
    self.log.append(("sleep", "sampler"))

  async def ensure_weights(self, revision: str) -> bool:
    assert self.gpu == {"sampler"}, self.gpu
    if revision != self.revision:
      self.log.append(("load", revision))
      self.revision = revision
    return True

  async def generate(self, r: dict) -> dict:
    assert self.gpu == {"sampler"}, self.gpu
    self.log.append(("generate", r["revision"]))
    await asyncio.sleep(0.01)
    return SAMPLE


class EngineDeadError(Exception):
  pass


def fake_vllm() -> dict[str, types.ModuleType]:
  """The vLLM names Sampler imports. vLLM needs a GPU, and the unit tests run without one."""
  names = ["exceptions", "sampling_params", "lora", "lora.request", "v1", "v1.engine", "v1.engine.exceptions"]
  modules = {name: types.ModuleType(name) for name in ["vllm", *(f"vllm.{name}" for name in names)]}
  modules["vllm"].SamplingParams = dict
  modules["vllm.exceptions"].VLLMClientError = type("VLLMClientError", (Exception,), {})
  modules["vllm.sampling_params"].RequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
  modules["vllm.lora.request"].LoRARequest = lambda name, lora_id, path: path
  modules["vllm.v1.engine.exceptions"].EngineDeadError = EngineDeadError
  modules["vllm.v1.engine.exceptions"].EngineGenerateError = type("EngineGenerateError", (Exception,), {})
  return modules


class FakeEngine:
  """Stands in for vLLM's in-process engine. It dies on the `poison` prompt the way vLLM's does: that request and every
  later one raise EngineDeadError, and `errored` turns true."""

  def __init__(self, poison: list[int] | None = None) -> None:
    self.poison, self.errored, self.seen = poison, False, []

  async def generate(self, prompt: dict, sampling_params: dict, request_id: str, lora_request: str | None):
    if self.errored:
      raise EngineDeadError()
    self.seen.append((prompt["prompt_token_ids"], lora_request))
    if prompt["prompt_token_ids"] == self.poison:
      self.errored = True
      raise EngineDeadError()
    choice = SimpleNamespace(token_ids=[100], logprobs=[{100: SimpleNamespace(logprob=-0.5)}], finish_reason="length")
    yield SimpleNamespace(outputs=[choice], prompt_logprobs=None)


class InProcessSamplerTest(RedisTestCase):
  """vllm_sampler.Sampler with a fake engine, served by the real dispatcher over Redis."""

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}"
    self.enterContext(patch.dict(sys.modules, fake_vllm()))
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp}))

  async def accepted(self, prompt: list[int], model: str = "base") -> dict:
    r = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": "", "model": model, "idempotency_key": None}
    r |= {"prompt_token_ids": prompt, "sampling_params": {"n": 1, "max_tokens": 1}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  def serve(self, engine: FakeEngine) -> asyncio.Task:
    run = asyncio.create_task(Dispatcher(self.store, self.pool, Sampler(engine, "base"), 4).run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    return run

  async def results(self, r: dict) -> list[dict]:
    return [json.loads(raw) for raw in await self.redis.lrange(future_key(r["request_id"]), 0, -1)]

  async def test_a_request_that_kills_the_engine_fails_once_and_the_restarted_engine_never_gets_it(self):
    poison = await self.accepted([6, 6, 6])
    dying = FakeEngine(poison=[6, 6, 6])
    first = self.serve(dying)
    await until(lambda: self.results(poison))
    later = await self.accepted([1, 2, 3])
    await asyncio.sleep(0.3)
    self.assertEqual(await self.results(later), [])
    # The worker exits once its engine is dead, and the restarted one serves the pool.
    first.cancel()
    restarted = FakeEngine(poison=[6, 6, 6])
    self.serve(restarted)
    await until(lambda: self.results(later))
    self.assertEqual([r["type"] for r in await self.results(poison)], ["RequestFailedResponse"])
    self.assertEqual((await self.results(later))[0]["type"], "sample")
    self.assertEqual([prompt for prompt, _ in dying.seen + restarted.seen], [[6, 6, 6], [1, 2, 3]])

  async def test_a_lora_snapshot_this_pod_does_not_see_yet_waits_and_never_reaches_the_engine(self):
    r = await self.accepted([1, 2, 3], model="adapter-1")
    engine = FakeEngine()
    self.serve(engine)
    await asyncio.sleep(0.3)
    self.assertEqual((engine.seen, engine.errored, await self.results(r)), ([], False, []))
    snapshot = Path(self.tmp, "sampler_lora", "adapter-1")
    snapshot.mkdir(parents=True)
    (snapshot / "adapter_config.json").write_text("{}")
    await until(lambda: self.results(r))
    self.assertEqual((await self.results(r))[0]["type"], "sample")
    self.assertEqual(engine.seen, [([1, 2, 3], str(snapshot))])


class TimeSlicedPoolTest(RedisTestCase):
  """A pool of one sampler on its trainer's GPU. The dispatcher runs each batch inside one turn from the real time
  slicer, over its socket, and the trainer steps between batches."""

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool, self.log, self.gpu = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}", [], set()
    self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
    socket_path = str(self.tmp / "slicer.sock")
    server = await start_time_slicer(SingleNodeTimeSlicer(RecordingRestorer(self.log)), socket_path)
    self.addAsyncCleanup(server.wait_closed)
    self.addCleanup(server.close)
    self.sampler_slicer, self.trainer_slicer = SocketTimeSlicerClient(socket_path=socket_path), SocketTimeSlicerClient(socket_path=socket_path)
    self.addAsyncCleanup(self.sampler_slicer.close)
    self.addAsyncCleanup(self.trainer_slicer.close)
    self.sampler, self.trainer = WorkloadRef("sampler", "gpu-0"), WorkloadRef("trainer", "gpu-0")
    await self.sampler_slicer.register(self.sampler)
    await self.trainer_slicer.register(self.trainer)

  async def accepted(self, revision: str = "") -> dict:
    r = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": None}
    r |= {"prompt_token_ids": [1, 2, 3], "sampling_params": {"n": 1, "max_tokens": 2}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  async def saved(self, *records: dict) -> None:
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))

  async def test_each_batch_runs_inside_one_turn_and_the_trainer_steps_between_batches(self):
    w1 = str(self.tmp / "w1")
    Path(w1).mkdir()
    Path(w1, "metadata.json").write_text("{}")
    first = [await self.accepted() for _ in range(3)]
    dispatcher = TimeSlicedDispatcher(self.store, self.pool, GpuSampler(self.log, self.gpu), 4, self.sampler_slicer, self.sampler)
    run = asyncio.create_task(dispatcher.run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    await self.saved(*first)
    # The trainer steps on the GPU, and the next batch waits for its turn.
    async with self.trainer_slicer.acquire(self.trainer):
      self.log.append(("step", "trainer"))
      second = [await self.accepted(w1) for _ in range(2)]
      await asyncio.sleep(0.3)
      self.assertNotIn(("load", w1), self.log)
    await self.saved(*second)
    await until(lambda: self.log[-1] == ("park", "sampler"))
    batch = [("generate", "")] * 3
    turn = [("restore", "sampler"), ("wake", "sampler"), ("load", w1), ("generate", w1), ("generate", w1), ("sleep", "sampler"), ("park", "sampler")]
    expected = [("wake", "sampler"), *batch, ("sleep", "sampler"), ("park", "sampler"), ("step", "trainer"), ("park", "trainer"), *turn]
    self.assertEqual(self.log, expected)
    self.assertEqual([json.loads(raw) for r in first + second for raw in await self.redis.lrange(future_key(r["request_id"]), 0, -1)], [SAMPLE] * 5)
