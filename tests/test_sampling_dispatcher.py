import asyncio
import inspect
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx
import safetensors.torch
import torch
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from server.sampling_dispatcher import Dispatcher, HttpSampler, revision_file
from server.sampling_store import SamplingStore, future_key
from tests.redis_server import RedisTestCase
from tests.test_exclusive_trainer import tiny_worker
from training.fft_trainer_worker import FFTTrainingWorker
from training.types import Datum, TensorData

SAMPLE = {"type": "sample", "sequences": [{"tokens": [100, 101], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}
CHOICE = {"token_ids": [100, 101], "finish_reason": "length", "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]}}


async def until(condition, timeout: float = 10) -> None:
  async with asyncio.timeout(timeout):
    while not (await result if inspect.isawaitable(result := condition()) else result):
      await asyncio.sleep(0.02)


def fft_worker() -> FFTTrainingWorker:
  """A real FFT trainer on a tiny Llama, resident on the CPU, publishing sparse deltas."""
  worker = tiny_worker()
  worker.cpu_offload = False
  worker.set_weight_sync_strategy("delta")
  worker.prepare_model_for_training()
  return worker


def train_step(worker: FFTTrainingWorker) -> None:
  row = [3, 5, 8, 13, 21, 34]
  datum = Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * 5)})
  worker.forward_backward([datum], "cross_entropy")
  worker.optim_step({"learning_rate": 1e-2})


def weights(worker: FFTTrainingWorker) -> dict[str, torch.Tensor]:
  return {name: param.detach().clone() for name, param in worker.model.named_parameters()}


def replayed(start: dict[str, torch.Tensor], revisions: list[str]) -> dict[str, torch.Tensor]:
  """A replica's weights after it loads `revisions` in order, written the way the delta_snapshot engine writes them."""
  out = {name: tensor.clone() for name, tensor in start.items()}
  for revision in revisions:
    metadata = json.loads(Path(revision, "metadata.json").read_text())
    if metadata.get("format") == "sparse_delta":
      tensors = safetensors.torch.load_file(str(Path(revision, "delta.safetensors")))
      for i, name in enumerate(metadata["layer_names"]):
        out[name].view(-1)[tensors[f"{i}.indices"].long()] = tensors[f"{i}.values"]
    else:
      for file in Path(revision).glob("*.safetensors"):
        out.update((name, tensor) for name, tensor in safetensors.torch.load_file(str(file)).items() if name in out)
  return out


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
  return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


class FakeReplica:
  """vLLM's weight endpoints and generate with a real weight version. `held` pauses both; an update logs what the readiness probe expects.
  Like vLLM, a paused replica queues generate until it is resumed, one weight update is active from start to finish,
  and only a finish that names a version, or /update_weight_version, changes the version."""

  def __init__(self, name: str, log: list, version: str, expected: Path):
    self.name, self.log, self.version, self.expected, self.held, self.url = name, log, version, expected, asyncio.Event(), ""
    self.paused, self.updating = False, False
    gets = [Route("/weight_info", self.weight_info)]
    gets.append(Route("/is_paused", lambda r: JSONResponse({"is_paused": self.paused})))
    posts = {"/pause": self.pause, "/resume": self.resume, "/start_weight_update": self.start, "/update_weight_version": self.set_version}
    posts["/reset_prefix_cache"] = lambda r: JSONResponse({"success": True})
    posts |= {"/update_weights": self.update, "/finish_weight_update": self.finish, "/inference/v1/generate": self.generate}
    self.app = Starlette(routes=gets + [Route(path, handler, methods=["POST"]) for path, handler in posts.items()])

  async def start(self, request) -> Response:
    if self.updating:
      return Response("start_weight_update called while a weight update is already active. Call finish_weight_update first.", status_code=500)
    self.updating = True
    return JSONResponse({})

  async def update(self, request) -> Response:
    if not self.updating:
      return Response("start_weight_update must be called before update_weights.", status_code=500)
    self.log.append(("update", self.name, (await request.json())["update_info"]["target_weights_path"], self.expected.read_text()))
    await until(lambda: not self.held.is_set())
    return JSONResponse({})

  async def pause(self, request) -> Response:
    self.paused = True
    return JSONResponse({})

  async def resume(self, request) -> Response:
    self.paused = False
    return JSONResponse({})

  async def weight_info(self, request) -> Response:
    return JSONResponse({"weight_version": self.version})

  async def finish(self, request) -> Response:
    body = await request.body()
    if not self.updating:
      return Response("finish_weight_update called without a matching start_weight_update.", status_code=500)
    self.updating = False
    if body:
      self.version = json.loads(body)["weight_version"]
    return JSONResponse({})

  async def set_version(self, request) -> Response:
    self.version = (await request.json())["new_version"]
    return JSONResponse({})

  async def generate(self, request) -> Response:
    self.log.append(("generate", self.name, self.version))
    await until(lambda: not self.held.is_set() and not self.paused)
    return JSONResponse({"choices": [CHOICE]})


def routed_dispatcher(store: SamplingStore, pool: str, http: httpx.AsyncClient, router_url: str, urls: list[str], concurrency: int) -> Dispatcher:
  """The dispatcher of a pool of vllm serve replicas, as sampling_dispatcher.main builds it."""
  return Dispatcher(store, pool, HttpSampler(http, pool, router_url, urls), concurrency)


async def run_routed(dispatcher: Dispatcher) -> None:
  """Runs it as sampling_dispatcher.main does, closing the sampler when it stops."""
  try:
    await dispatcher.run()
  finally:
    await dispatcher.sampler.close()


def router(log: list, replicas: list[FakeReplica], http: httpx.AsyncClient) -> Starlette:
  # Forwards generate to a ready replica, one whose version is the expected one, like the EPP.
  async def route(request) -> Response:
    ready = [r for r in replicas if r.version == r.expected.read_text()]
    if not ready:
      log.append(("unrouted", "", ""))
      return Response(status_code=503)
    upstream = await http.post(f"{ready[0].url}/inference/v1/generate", content=await request.body(), timeout=None)
    return Response(upstream.content, upstream.status_code, headers={"x-openrl-upstream": ready[0].url.removeprefix("http://")})

  return Starlette(routes=[Route("/inference/v1/generate", route, methods=["POST"])])


class SamplingDispatcherTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool, self.log, self.http = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}", [], httpx.AsyncClient()
    self.addAsyncCleanup(self.http.aclose)
    self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.enterContext(patch.dict("os.environ", {"OPEN_RL_TMP_DIR": str(self.tmp)}))

  async def serve(self, app: Starlette) -> str:
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    self.addAsyncCleanup(lambda: (setattr(server, "should_exit", True), serving)[1])
    await until(lambda: server.started)
    return f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"

  async def replicas(self, *versions: str) -> list[FakeReplica]:
    replicas = [FakeReplica(f"pod-{i}", self.log, version, revision_file(self.pool)) for i, version in enumerate(versions)]
    for replica in replicas:
      replica.url = await self.serve(replica.app)
      self.addAsyncCleanup(replica.held.clear)
    return replicas

  def revision(self, name: str, **metadata) -> str:
    (self.tmp / name).mkdir()
    (self.tmp / name / "metadata.json").write_text(json.dumps(metadata))
    return str(self.tmp / name)

  def loads(self, *names: str) -> dict[str, list[str]]:
    return {name: [entry[2] for entry in self.log if entry[:2] == ("update", name)] for name in names}

  async def accepted(self, revision: str = "", prompt: list | None = None, **fields) -> dict:
    identity = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": None}
    r = {**identity, "prompt_token_ids": prompt or [1, 2, 3], "sampling_params": {"n": 1, "max_tokens": 2}, "include_prompt_logprobs": False}
    r |= fields
    await self.store.accept(r)
    return r

  async def dispatcher(self, replicas: list[FakeReplica], concurrency: int = 4) -> asyncio.Task:
    router_url = await self.serve(router(self.log, replicas, self.http))
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, router_url, [r.url for r in replicas], concurrency=concurrency)
    run = asyncio.create_task(run_routed(dispatcher))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    return run

  async def results(self, request_id: str) -> list[dict]:
    await until(lambda: self.redis.llen(future_key(request_id)))
    await asyncio.sleep(0.3)  # a second result would show by now
    return [json.loads(raw) for raw in await self.redis.lrange(future_key(request_id), 0, -1)]

  async def test_each_request_is_generated_once(self):
    replicas = await self.replicas("default", "default")
    records = [await self.accepted() for _ in range(40)]
    await self.dispatcher(replicas, concurrency=256)
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))
    await asyncio.sleep(0.5)  # a duplicate would be generated by now
    self.assertEqual(sum(entry[0] == "generate" for entry in self.log), len(records))

  async def test_crash_mid_request_saves_exactly_one_result(self):
    (replica,) = replicas = await self.replicas("default")
    replica.held.set()
    r = await self.accepted()
    first = await self.dispatcher(replicas)
    await until(lambda: self.log.count(("generate", "pod-0", "default")) == 1)
    first.cancel()
    await self.dispatcher(replicas)
    replica.held.clear()
    self.assertEqual((await self.results(r["request_id"]), await self.store.pending(self.pool)), ([SAMPLE], []))

  async def test_an_answer_from_the_wrong_weights_is_not_saved_and_the_pod_is_prepared_again(self):
    (replica,) = replicas = await self.replicas(w1 := self.revision("w1"))
    replica.held.set()
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    await until(lambda: ("generate", "pod-0", w1) in self.log)
    replica.version = "default"  # restarted in place while the request ran
    replica.held.clear()
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual(self.log, [("generate", "pod-0", w1), ("update", "pod-0", w1, w1), ("generate", "pod-0", w1)])

  async def test_a_replica_on_any_save_at_the_weights_a_delta_starts_from_or_reaches_applies_it(self):
    worker = fft_worker()
    train_step(worker)
    # An RL step with nothing to learn from takes no optimizer step, so its rollouts sample a second save at the same weights.
    first, at_first = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-0"), weights(worker)
    second = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-1")
    # The late replica started on the first save and never loads the second.
    early, late = await self.replicas(first, first)
    r = await self.accepted(second)
    run = await self.dispatcher([early])
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    train_step(worker)
    third = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-2")
    r = await self.accepted(third)
    await self.dispatcher([early, late])
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    await until(lambda: self.loads(late.name) == {late.name: [third]})
    self.assertEqual(self.loads(early.name), {early.name: [second, third]})
    self.assertTrue(same(replayed(at_first, [second, third]), weights(worker)))
    self.assertTrue(same(replayed(at_first, [third]), weights(worker)))

  async def test_one_jobs_big_batch_does_not_hold_up_another_jobs_request(self):
    (replica,) = await self.replicas("default")
    sent = []

    # Records which job's request each send carries. Each job samples its own adapter, so the model names the job.
    async def route(request) -> Response:
      body = await request.body()
      sent.append(json.loads(body)["model"])
      upstream = await self.http.post(f"{replica.url}/inference/v1/generate", content=body, timeout=None)
      return Response(upstream.content, upstream.status_code, headers={"x-openrl-upstream": replica.url.removeprefix("http://")})

    router_url = await self.serve(Starlette(routes=[Route("/inference/v1/generate", route, methods=["POST"])]))
    big = [await self.accepted(job="job-a", model="adapter-a") for _ in range(6)]
    small = await self.accepted(job="job-b", model="adapter-b")
    run = asyncio.create_task(run_routed(routed_dispatcher(self.store, self.pool, self.http, router_url, [replica.url], concurrency=1)))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    for r in [*big, small]:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))
    self.assertEqual(sent[:2], ["adapter-a", "adapter-b"])
