import asyncio
import inspect
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx
import redis.asyncio as redis
import uvicorn
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from server.sampling_dispatcher import Dispatcher, redis_client, revision_file
from server.sampling_store import SamplingStore, future_key
from tests.redis_server import RedisTestCase

SAMPLE = {"type": "sample", "sequences": [{"tokens": [100, 101], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}
CHOICE = {"token_ids": [100, 101], "finish_reason": "length", "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]}}


async def until(condition, timeout: float = 10) -> None:
  async with asyncio.timeout(timeout):
    while not (await result if inspect.isawaitable(result := condition()) else result):
      await asyncio.sleep(0.02)


class FakeReplica:
  """vLLM's weight endpoints and generate with a real weight version. `held` pauses both; an update logs what the readiness probe expects."""

  def __init__(self, name: str, log: list, version: str, expected: Path):
    self.name, self.log, self.version, self.expected, self.held, self.url = name, log, version, expected, asyncio.Event(), ""
    gets = [Route("/weight_info", lambda r: JSONResponse({"weight_version": self.version}))]
    posts = {path: lambda r: JSONResponse({}) for path in ("/pause", "/resume", "/start_weight_update")}
    posts["/reset_prefix_cache"] = lambda r: JSONResponse({"success": True})
    posts |= {"/update_weights": self.update, "/finish_weight_update": self.finish, "/inference/v1/generate": self.generate}
    self.app = Starlette(routes=gets + [Route(path, handler, methods=["POST"]) for path, handler in posts.items()])

  async def update(self, request) -> Response:
    self.log.append(("update", self.name, (await request.json())["update_info"]["target_weights_path"], self.expected.read_text()))
    await until(lambda: not self.held.is_set())
    return JSONResponse({})

  async def finish(self, request) -> Response:
    self.version = (await request.json())["weight_version"]
    return JSONResponse({})

  async def generate(self, request) -> Response:
    self.log.append(("generate", self.name, self.version))
    await until(lambda: not self.held.is_set())
    return JSONResponse({"choices": [CHOICE]})


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

  async def accepted(self, revision: str = "") -> dict:
    identity = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": None}
    r = {**identity, "prompt_token_ids": [1, 2, 3], "sampling_params": {"n": 1, "max_tokens": 2}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  async def dispatcher(self, replicas: list[FakeReplica], concurrency: int = 4) -> asyncio.Task:
    router_url = await self.serve(router(self.log, replicas, self.http))
    dispatcher = Dispatcher(self.store, self.pool, self.http, router_url, [r.url for r in replicas], concurrency=concurrency)
    run = asyncio.create_task(dispatcher.run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    return run

  async def committed(self, request_id: str) -> list[dict]:
    await until(lambda: self.redis.llen(future_key(request_id)))
    await asyncio.sleep(0.3)  # a second commit would show by now
    return [json.loads(raw) for raw in await self.redis.lrange(future_key(request_id), 0, -1)]

  async def test_each_request_is_generated_once(self):
    replicas = await self.replicas("default", "default")
    records = [await self.accepted() for _ in range(40)]
    await self.dispatcher(replicas, concurrency=256)
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))
    await asyncio.sleep(0.5)  # a duplicate would be generated by now
    self.assertEqual(sum(entry[0] == "generate" for entry in self.log), len(records))

  async def test_a_burst_of_answers_commits_each_once(self):
    replicas = await self.replicas("default", "default")
    records = [await self.accepted() for _ in range(300)]
    # The dispatcher's own clients, as main() builds them: 256 sends answered together commit together.
    self.store = SamplingStore(await self.enterAsyncContext(redis_client(self.url)))
    self.http = await self.enterAsyncContext(httpx.AsyncClient(limits=httpx.Limits(max_connections=None)))
    await self.dispatcher(replicas, concurrency=256)
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])), timeout=60)
    await asyncio.sleep(0.5)  # a resend after a failed commit would generate again by now
    # A send the test router drops is retried, so a few may generate twice. Commits that
    # find no free connection fail and resend about half the burst.
    self.assertLess(sum(entry[0] == "generate" for entry in self.log) - len(records), len(records) // 20)

  async def test_a_request_no_replica_answers_fails_at_the_deadline(self):
    replicas = await self.replicas("other-weights")  # never ready for the base weights, so nothing is sent
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 1))
    r = await self.accepted()
    await self.dispatcher(replicas)
    [result] = await self.committed(r["request_id"])
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertIn("No sampler answered", result["error_message"])
    self.assertEqual(await self.store.pending(self.pool), [])

  async def test_a_redis_timeout_does_not_stop_dispatching(self):
    replicas = await self.replicas("default")
    self.enterContext(patch("server.sampling_dispatcher.RETRY_MS", 100))
    impatient = await self.enterAsyncContext(redis.from_url(self.url, decode_responses=True, socket_timeout=0.3, retry=Retry(NoBackoff(), 0)))
    patient, self.store = self.store, SamplingStore(impatient)
    run = await self.dispatcher(replicas)
    self.store = patient
    await self.redis.execute_command("CLIENT", "PAUSE", "1000", "ALL")
    r = await self.accepted()  # waits out the pause; the dispatcher's reads time out meanwhile
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertFalse(run.done())

  async def test_a_replica_that_is_still_starting_fails_nothing(self):
    (replica,) = replicas = await self.replicas("default")
    starting = "http://127.0.0.1:9"  # nothing listens here yet
    r = await self.accepted()
    router_url = await self.serve(router(self.log, replicas, self.http))
    dispatcher = Dispatcher(self.store, self.pool, self.http, router_url, [starting, replica.url], concurrency=4)
    run = asyncio.create_task(dispatcher.run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])

  async def test_crash_mid_request_commits_exactly_one_result(self):
    (replica,) = replicas = await self.replicas("default")
    replica.held.set()
    r = await self.accepted()
    first = await self.dispatcher(replicas)
    await until(lambda: self.log.count(("generate", "pod-0", "default")) == 1)
    first.cancel()
    await self.dispatcher(replicas)
    replica.held.clear()
    self.assertEqual((await self.committed(r["request_id"]), await self.store.pending(self.pool)), ([SAMPLE], []))

  async def test_an_answer_from_the_wrong_weights_is_not_committed_and_the_pod_is_prepared_again(self):
    (replica,) = replicas = await self.replicas(w1 := self.revision("w1"))
    replica.held.set()
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    await until(lambda: ("generate", "pod-0", w1) in self.log)
    replica.version = "default"  # restarted in place while the request ran
    replica.held.clear()
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertEqual(self.log, [("generate", "pod-0", w1), ("update", "pod-0", w1, w1), ("generate", "pod-0", w1)])

  async def test_fft_switch_expects_the_new_version_before_any_weight_update_and_sends_only_to_a_ready_pod(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    r = await self.accepted(w2)
    await self.dispatcher(await self.replicas(w1, w1))
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertEqual(sorted(self.log[:2]), [("update", "pod-0", w2, w2), ("update", "pod-1", w2, w2)])
    self.assertEqual([(entry[0], entry[2]) for entry in self.log[2:]], [("generate", w2)])

  async def test_a_deleted_revision_fails_its_requests_and_the_pool_moves_on(self):
    (replica,) = replicas = await self.replicas("default")
    gone, later = await self.accepted(str(self.tmp / "gone")), await self.accepted(self.revision("w2", format="sparse_delta", parent=""))
    await self.dispatcher(replicas)
    failed = {"type": "RequestFailedResponse", "error_message": f"Sampler weights {self.tmp / 'gone'} no longer exist"}
    self.assertEqual(await self.committed(gone["request_id"]), [failed])
    self.assertEqual((await self.committed(later["request_id"]), replica.version), ([SAMPLE], later["revision"]))
