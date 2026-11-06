"""Run telemetry: what the recorder writes, and that recording never changes or slows what a worker answers."""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from server import api_server, telemetry
from server.sampling_dispatcher import Dispatcher, RevisionUnloadable, failed
from server.sampling_store import SamplingStore, future_key
from server.store import InMemoryStateStore, InMemoryStore, StateStore
from server.training_requests_processor import TrainingRequestsProcessor
from server.vllm_sampler import Sampler
from tests.api_client import asgi_client
from tests.redis_server import RedisTestCase
from tests.test_fft_batch_failure import BatchStore, BrokenWorker, SlicerStub
from tests.test_sampling_dispatcher import SAMPLE, until
from tests.test_vllm_sampler import FakeEngine, fake_vllm
from training.trainer_worker import BaseTrainerWorker

BOUND = 1.0  # an exiting process waits this long for its last flush; a serve must finish well inside it


class FaultyStore:
  """Every call raises, as with Redis down, or waits forever, as with a Redis that never answers."""

  def __init__(self, hangs: bool):
    self.hangs = hangs

  def __getattr__(self, name):
    if not self.hangs:
      raise ConnectionError(f"telemetry store {name} failed")
    return lambda *args, **kwargs: threading.Event().wait()


def no_store():
  raise RuntimeError("no telemetry store configured")


class RecorderOnlyStore(InMemoryStateStore):
  """Notes, and raises on, any use off the recorder's thread, since a Redis call on a request path is the bug."""

  def __init__(self):
    super().__init__()
    self.intrusions: list[str] = []

  def __getattribute__(self, name):
    if name in StateStore.__abstractmethods__ and threading.current_thread().name != "telemetry":
      super().__getattribute__("intrusions").append(name)
      raise AssertionError(f"{name} called on a request path")
    return super().__getattribute__(name)


def trainer() -> Mock:
  return Mock(cpu_offload=True, optim_step=lambda adam_params, model_id: {"metrics": {"grad_norm:mean": 0.5}})


async def train(worker: Mock, requests: list[dict]) -> dict:
  """One time-sliced trainer batch. Returns what each request was answered."""
  store = BatchStore(requests)
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    processor = TrainingRequestsProcessor(store, worker, "run-a", time_slicer=SlicerStub())
  await processor.run_once()
  return store.futures


def optim_steps(count: int) -> list[dict]:
  return [{"request_id": f"os-{i}", "op": "optim_step", "model_id": "run-a", "payload": {"adam_params": {}}} for i in range(count)]


def request(request_id: str, op: str, **payload) -> dict:
  return {"request_id": request_id, "op": op, "model_id": "run-a", "payload": payload}


class ScriptedTrainer(BaseTrainerWorker):
  """Answers forward_backward with a loss and optim_step with a NaN grad norm, through the workers' own sanitizing,
  and fails an optim_step whose learning rate is negative."""

  cpu_offload = True

  def wake_up(self) -> None:
    pass

  def sleep(self) -> None:
    pass

  def forward_backward(self, data, loss_fn, loss_config, model_id, forward_only=False) -> dict:
    return {"metrics": self.sanitize_metrics({"loss:mean": 1.5})}

  def optim_step(self, adam_params, model_id) -> dict:
    if adam_params.get("learning_rate", 0) < 0:
      raise ValueError("negative learning rate")
    return {"metrics": self.sanitize_metrics({"grad_norm:mean": float("nan"), "time/optimizer_step": 0.25})}


class TelemetryTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStateStore()
    self.enterContext(patch.object(telemetry, "get_state_store", lambda: self.store))
    self.enterContext(patch.object(telemetry, "FINISH_SECONDS", BOUND))

  def serve(self, work, role: str = "trainer") -> tuple:
    """Run work while recording. Returns its result, how long it took, and how long the exit after took."""

    async def run():
      async with telemetry.serving(role, "run-a"):
        started = time.monotonic()
        result = await asyncio.wait_for(work(), 2 * BOUND)
        exiting = time.monotonic()
      return result, exiting - started, time.monotonic() - exiting

    return asyncio.run(run())

  def worker_records(self) -> list[dict]:
    return [json.loads(v) for k, v in self.store.kv_store.items() if k.startswith(telemetry.key("worker", ""))]


class FaultTest(TelemetryTest):
  """The same work, with a healthy telemetry store and with one that fails, gets the same answers in about the same time."""

  def test_a_failing_hanging_or_missing_store_changes_no_answer(self) -> None:
    expected, _, _ = self.serve(lambda: train(trainer(), optim_steps(2)))
    self.assertEqual(self.store.hashes[telemetry.key("ledger", "run-a")], {"popped": 2, "answered": 2}, "the healthy serve recorded")
    for fault, store in {"raises": lambda: FaultyStore(False), "hangs": lambda: FaultyStore(True), "is missing": no_store}.items():
      with self.subTest(store=fault), patch.object(telemetry, "get_state_store", store):
        result, took, exiting = self.serve(lambda: train(trainer(), optim_steps(2)))
        self.assertEqual(result, expected)
        self.assertLess(took, BOUND)
        self.assertLess(exiting, 2 * BOUND)

  def test_no_request_path_touches_the_store_and_the_recorder_writes_what_it_saw(self) -> None:
    store = RecorderOnlyStore()
    with patch.object(telemetry, "get_state_store", lambda: store):
      result, _, _ = self.serve(lambda: train(trainer(), optim_steps(2)))
    self.assertEqual([r["type"] for r in result.values()], ["optim_step_completed"] * 2)
    self.assertEqual(store.intrusions, [])
    self.assertEqual(store.hashes[telemetry.key("ledger", "run-a")], {"popped": 2, "answered": 2})


class WorkerRecordTest(TelemetryTest):
  def test_the_record_shows_settings_turns_and_a_clean_exit(self) -> None:
    env = {"OPEN_RL_FINE_TUNING_TYPE": "full", "OPEN_RL_API_KEY": "sk-private", "HOME": "/root"}
    with patch.dict(os.environ, env, clear=True):
      self.serve(lambda: train(trainer(), optim_steps(1)))
    (record,) = self.worker_records()
    self.assertEqual((record["role"], record["runtime_id"], record["phase"]), ("trainer", "run-a", "exited"))
    self.assertEqual(record["config"]["env"], {"OPEN_RL_FINE_TUNING_TYPE": "full"})
    self.assertNotIn("sk-private", json.dumps(record))
    self.assertEqual((len(record["waits"]), len(record["turns"])), (1, 1))

  def test_the_record_stays_fresh_while_the_event_loop_is_blocked(self) -> None:
    async def block():
      blocked_at = time.time()
      time.sleep(0.3)  # loading weights holds the loop, and the recorder must not need it
      return blocked_at

    with patch.object(telemetry, "FLUSH_SECONDS", 0.02), patch.object(self.store, "write_sync", wraps=self.store.write_sync) as write:
      blocked_at, _, _ = self.serve(block)
    beats = [json.loads(v)["heartbeat_at"] for call in write.call_args_list for v in call.args[0].values.values()]
    self.assertTrue(any(beat - blocked_at > 0.2 for beat in beats[:-1]), "a beat landed while the loop was blocked, before the exit's")

  def test_popped_requests_stay_outstanding_until_answered(self) -> None:
    async def pop():
      telemetry.popped([{"request_id": f"f{i}", "op": "forward_backward", "model_id": "run-a"} for i in range(3)])
      # A shared trainer's delete_model was never counted as enqueued, so it is not outstanding either.
      telemetry.popped([{"request_id": "d1", "op": "delete_model", "model_id": "run-a"}])
      telemetry.answered("f0", {"type": "forward_backward_completed"})
      return json.loads(telemetry.worker().snapshot())["outstanding"]

    outstanding, _, _ = self.serve(pop)
    self.assertEqual([(g["op"], g["run_id"], g["count"]) for g in outstanding], [("forward_backward", "run-a", 2)])
    self.assertEqual(self.store.hashes[telemetry.key("ledger", "run-a")], {"popped": 3, "answered": 1})


class LedgerTest(TelemetryTest):
  def setUp(self) -> None:
    super().setUp()
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=api_server.state))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "torch"}))

  def test_a_shared_trainers_delete_is_counted_neither_enqueued_nor_popped(self) -> None:
    async def run():
      async with asgi_client() as client:
        created = (await client.post("/api/v1/create_model", json={"base_model": "m"})).json()
        (await client.post("/api/v1/delete_model", json={"model_id": created["request_id"]})).raise_for_status()
      requests = await api_server.store.get_requests(active_set_id="m-1")
      self.assertEqual([r["op"] for r in requests], ["create_model", "delete_model"])
      telemetry.popped(requests)
      return created["request_id"]

    run_id, _, _ = self.serve(run)
    self.assertEqual(self.store.hashes[telemetry.key("ledger", run_id)], {"enqueued": 1, "popped": 1})


class OpRecordTest(TelemetryTest):
  def records(self, *parts: str) -> list[dict]:
    return asyncio.run(self.store.read_samples(telemetry.key("operations", *parts)))

  def test_ops_carry_the_runs_step_and_a_nan_metric_is_named_not_zeroed(self) -> None:
    asyncio.run(self.store.set_value(telemetry.key("steps", "run-a"), "5"))  # a restarted trainer resumes the run's count
    requests = [
      request("fb-1", "forward_backward", data=[]),
      request("os-1", "optim_step"),
      request("os-2", "optim_step", adam_params={"learning_rate": -1}),
      request("fb-2", "forward_backward", data=[]),
      request("ev-1", "forward_backward", data=[], forward_only=True),
    ]
    answers, _, _ = self.serve(lambda: train(ScriptedTrainer(), requests))
    self.assertEqual(answers["os-1"]["metrics"]["grad_norm:mean"], 0.0, "the client still gets the sanitized value")
    ops = self.records("run-a")
    self.assertEqual(
      [(r["request_id"], r["operation"], r["status"], r["step"]) for r in ops],
      [("fb-1", "forward_backward", "succeeded", 5), ("os-1", "optim_step", "succeeded", 5), ("os-2", "optim_step", "failed", 6)]
      + [("fb-2", "forward_backward", "succeeded", 6), ("ev-1", "forward", "succeeded", 6)],
    )
    self.assertEqual((ops[1]["metrics"], ops[1]["nonfinite"]), ({"time/optimizer_step": 0.25}, ["grad_norm:mean"]))
    self.assertEqual(self.store.get_value_sync(telemetry.key("steps", "run-a")), "6")
    self.assertEqual(len(self.store.lists[telemetry.key("step_times", "run-a")]), 1)
    self.assertEqual(self.store.hashes[telemetry.key("ledger", "run-a")], {"popped": 5, "answered": 5, "failed": 1})

  def test_a_failure_answered_before_the_op_runs_is_recorded_once(self) -> None:
    async def fail():
      store = BatchStore([request("fb-1", "forward_backward", data=[]), request("bad-1", "optim_step", adam_params="not a dict")])
      with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
        await TrainingRequestsProcessor(store, BrokenWorker(), "run-a", time_slicer=SlicerStub()).run_once()

    with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
      self.serve(fail)
    self.assertEqual([(r["request_id"], r["status"]) for r in self.records("run-a")], [("fb-1", "failed"), ("bad-1", "failed")])

  def test_an_op_records_how_long_it_waited_in_the_queue(self) -> None:
    async def run():
      async with asgi_client() as client:
        created = (await client.post("/api/v1/create_model", json={"base_model": "m"})).json()
      await asyncio.sleep(0.2)
      await TrainingRequestsProcessor(api_server.store, Mock(), active_tenant_set_id="m-1").run_once()
      return created["request_id"]

    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=api_server.state))
    with patch.dict(os.environ, {"SAMPLING_BACKEND": "torch"}):
      run_id, _, _ = self.serve(run)
    (created,) = self.records(run_id)
    self.assertGreaterEqual(created["queue_seconds"], 0.2)
    self.assertLess(created["queue_seconds"], 2)


class StepsRouteTest(TelemetryTest):
  def op(self, step: int, operation: str, metrics: dict | None = None) -> dict:
    timing = {"started_at": step, "at": step + 0.5, "elapsed_seconds": 0.5}
    return {"step": step, "operation": operation, "status": "succeeded", "metrics": metrics or {}, **timing}

  def get(self, ops: list[dict], query: str = "") -> dict:
    asyncio.run(self.store.set_value(telemetry.key("model_meta", "run-a"), "{}"))
    self.store.lists[telemetry.key("operations", "run-a")] = [json.dumps(op) for op in ops]
    with patch("server.observability.router.get_state_store", return_value=self.store):
      return TestClient(api_server.app).get(f"/api/v1/inspect/runs/run-a/steps{query}").json()

  def test_ops_after_the_last_step_join_it_and_a_step_in_progress_stands_alone(self) -> None:
    train = [self.op(0, "forward_backward", {"loss": 2.0}), self.op(0, "optim_step"), self.op(1, "forward_backward", {"loss": 1.0})]
    finished = self.get([*train, self.op(1, "optim_step"), self.op(2, "save_state"), self.op(2, "forward", {"loss": 0.5})])
    self.assertEqual(
      [(s["step"], sorted(s["ops"]), s["complete"]) for s in finished["steps"]],
      [
        (0, ["forward_backward", "optim_step"], True),
        (1, ["forward", "forward_backward", "optim_step", "save_state"], True),
      ],
    )
    in_progress = self.get([*train, self.op(1, "optim_step"), self.op(2, "save_weights_for_sampler"), self.op(2, "forward_backward")])
    self.assertEqual([(s["step"], s["complete"]) for s in in_progress["steps"]], [(0, True), (1, True), (2, False)])

  def test_metrics_are_spread_per_op_and_kept_by_prefix(self) -> None:
    fb = [self.op(0, "forward_backward", {"loss": 2.0, "kl": 0.1, "entropy": 3.0}), self.op(0, "forward_backward", {"loss": 1.0})]
    (step,) = self.get([*fb, self.op(0, "optim_step")], "?fields=loss,kl")["steps"]
    spread = {"loss": {"min": 1.0, "max": 2.0, "mean": 1.5}, "kl": {"min": 0.1, "max": 0.1, "mean": 0.1}}
    self.assertEqual(step["ops"]["forward_backward"], {"count": 2, "failed": 0, "seconds": 1.0, "metrics": spread})


class ScriptedSampler:
  """Answers prompt [1], fails the first send of [2] so the dispatcher sends it again, refuses [3], and can find no
  weights a revision applies on."""

  def __init__(self, unloadable: bool = False) -> None:
    self.unloadable, self.raised = unloadable, False

  async def ensure_weights(self, revision: str) -> bool:
    if revision and self.unloadable:
      raise RevisionUnloadable(revision, ["default"])
    return True

  async def generate(self, r: dict) -> dict:
    prompt = list(r["prompt_token_ids"])
    if prompt == [2] and not self.raised:
      self.raised = True
      raise ConnectionError("the replica went away")
    return failed("Sampler rejected the request") if prompt == [3] else SAMPLE


class BrokenUpdateEngine(FakeEngine):
  async def pause_generation(self, **kwargs) -> None:
    raise RuntimeError("CUDA out of memory")


class SampleRecordTest(RedisTestCase):
  """A run's sends are read back from its sample records: each group record's count, plus one per failed record."""

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool, self.telemetry = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}", InMemoryStateStore()
    self.enterContext(patch.object(telemetry, "get_state_store", lambda: self.telemetry))
    vllm = fake_vllm() | {name: types.ModuleType(name) for name in ("vllm.distributed", "vllm.distributed.weight_transfer.base")}
    vllm["vllm.distributed.weight_transfer.base"].WeightTransferUpdateRequest = dict
    self.enterContext(patch.dict(sys.modules, vllm))
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp}))

  async def accepted(self, prompt: list[int], revision: str = "") -> dict:
    r = {"request_id": str(uuid.uuid4()), "pool": self.pool, "job": "run", "revision": revision, "model": "base", "idempotency_key": None}
    r |= {"prompt_token_ids": prompt, "sampling_params": {"n": 1, "max_tokens": 1}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  async def answer(self, r: dict) -> str | None:
    raw = await self.redis.lindex(future_key(r["request_id"]), 0)
    return json.loads(raw)["type"] if raw else None

  def serve(self, sampler) -> None:
    """Serve the pool as a sampler process does, until it exits."""
    telemetry.start("sampler", self.pool)
    self.running = asyncio.create_task(Dispatcher(self.store, self.pool, sampler, 4).run())
    self.addAsyncCleanup(self.exit)

  async def exit(self) -> None:
    self.running.cancel()
    await asyncio.gather(self.running, return_exceptions=True)
    await telemetry.finish()

  async def sent(self) -> dict:
    """The run's sends and failures, read back from what the sampler process recorded by the time it exited."""
    await self.exit()
    records = await self.telemetry.read_samples(telemetry.key("operations", "run", "samples"))
    return {"sends": sum(r.get("count", 1) for r in records), "failed": sum(r["status"] == "failed" for r in records)}

  async def test_a_request_sent_again_after_an_error_and_one_refused_are_each_sent_once(self) -> None:
    records = [await self.accepted([prompt]) for prompt in (1, 2, 3)]
    self.serve(ScriptedSampler())
    await until(lambda: all_answered(self, records))
    self.assertEqual([await self.answer(r) for r in records], ["sample", "sample", "RequestFailedResponse"])
    self.assertEqual(await self.sent(), {"sends": 3, "failed": 1})

  async def test_an_answer_saved_after_another_one_is_not_counted(self) -> None:
    r = await self.accepted([1])
    release = asyncio.Event()

    class SlowSampler(ScriptedSampler):
      async def generate(self, r: dict) -> dict:
        await release.wait()
        return SAMPLE

    self.serve(SlowSampler())
    await asyncio.sleep(0.2)
    # The API server's deadline sweep fails the request while its send is in flight.
    await self.store.fail_older_than(0, failed("No sampler answered"))
    release.set()
    await asyncio.sleep(0.2)
    self.assertEqual(await self.answer(r), "RequestFailedResponse")
    self.assertEqual(await self.sent(), {"sends": 1, "failed": 1})

  async def test_a_failed_weight_update_answers_nothing_and_unloadable_weights_fail_each_request_once(self) -> None:
    revision = Path(self.tmp, "w1")
    revision.mkdir()
    (revision / "metadata.json").write_text("{}")
    records = [await self.accepted([1], str(revision)) for _ in range(2)]
    self.serve(Sampler(BrokenUpdateEngine(), "base"))
    await asyncio.sleep(0.5)  # the worker exits now, and the records wait for the restarted one
    self.assertEqual([await self.answer(r) for r in records], [None, None])
    self.assertEqual(await self.sent(), {"sends": 0, "failed": 0})
    self.serve(ScriptedSampler(unloadable=True))
    await until(lambda: all_answered(self, records))
    self.assertEqual(await self.sent(), {"sends": 2, "failed": 2})

  async def test_samples_answered_before_the_engine_died_are_sent_once_as_is_the_one_it_failed(self) -> None:
    engine = FakeEngine(poison=[6, 6, 6])
    self.serve(Sampler(engine, "base"))
    answered = [await self.accepted([1]), await self.accepted([2])]
    await until(lambda: all_answered(self, answered))
    poison = await self.accepted([6, 6, 6])
    await until(lambda: self.answer(poison))
    self.assertEqual([await self.answer(r) for r in [*answered, poison]], ["sample", "sample", "RequestFailedResponse"])
    self.assertTrue(engine.errored)
    self.assertEqual(await self.sent(), {"sends": 3, "failed": 1})


async def all_answered(test: SampleRecordTest, records: list[dict]) -> bool:
  return all([await test.answer(r) for r in records])


if __name__ == "__main__":
  unittest.main()
