"""The in-process sampler's turns on a GPU it shares with its trainer, with the real time slicer."""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from accel_timeslicer.serve import start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from server.sampling_store import SamplingStore, future_key
from server.vllm_sampler import TimeSlicedDispatcher
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
