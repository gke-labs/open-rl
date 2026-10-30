"""Serves one sampling pool's pending records with a sampler and saves the first result for each.

The sampler is either vLLM in this process (vllm_sampler.Sampler), for a pool of one replica that one job owns, or
HttpSampler, which sends through the pool's router, Envoy in front of the llm-d EPP, to its vllm serve replicas."""

import asyncio
import contextlib
import itertools
import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis

from server.sampling_store import SamplingStore, job_of
from server.telemetry import op_finished, sampled_group
from server.vllm_options import sample_params, sample_result

logger = logging.getLogger(__name__)

# The dispatcher wakes for each new request, or after this long to retry a failed send and look at its replicas.
RETRY_MS = 1000
# A request no replica has answered for this long fails, so its client stops waiting.
SAMPLE_DEADLINE_SECONDS = int(os.getenv("OPEN_RL_SAMPLE_DEADLINE_SECONDS", "3600"))
# The version a replica reports once its weights may not match any revision. Never a path, so no pool revision equals it.
UNKNOWN_WEIGHTS = "unknown"
# How long one /weight_info call may take.
WEIGHT_INFO_TIMEOUT_SECONDS = 5


def tmp_root() -> str:
  """OPEN_RL_TMP_DIR, or the documented default. The API server renders the readiness probe's path
  from it and hands the same value to the dispatcher and the replicas, so they all agree."""
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def revision_file(pool: str) -> Path:
  """The readiness probe compares vLLM's weight version with the revision in this file."""
  return Path(tmp_root(), "sampling", pool, "revision")


def accepted_at(r: dict[str, Any]) -> float:
  """When the record's request was accepted. A stream id starts with that time in milliseconds."""
  return int(r["entry_id"].split("-")[0]) / 1000


def failed(message: str) -> dict[str, str]:
  return {"type": "RequestFailedResponse", "error_message": message}


def delta_applies(revision: str, current: str) -> bool:
  """Whether weights at `current` can load `revision`. A delta applies on any save at the weights it starts from or
  reaches. Deltas written before the trainer listed them name one parent."""
  metadata = json.loads(Path(revision, "metadata.json").read_text())
  bases = [base or "default" for base in metadata.get("bases", [metadata.get("parent")])]
  return metadata.get("format") != "sparse_delta" or current in bases


class RevisionUnloadable(Exception):
  """No replica holds `revision`, a sparse delta, or weights it applies on. `held` is the weights they hold."""

  def __init__(self, revision: str, held: list[str]) -> None:
    super().__init__(f"{revision} does not apply on {held}")
    self.revision, self.held = revision, held


def to_result(body: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
  """The result for a vLLM /inference/v1/generate response."""
  sequences = [(c["token_ids"], [e["logprob"] for e in c["logprobs"]["content"]], c["finish_reason"]) for c in body["choices"]]
  prompt_logprobs = None
  if record["include_prompt_logprobs"]:
    pairs = zip(record["prompt_token_ids"], body["prompt_logprobs"], strict=True)
    prompt_logprobs = [None if e is None else e[str(token)]["logprob"] for token, e in pairs]
  return sample_result(sequences, prompt_logprobs)


class Dispatcher:
  """Serves `pool` with `sampler`, which loads a revision with ensure_weights and answers a record with generate."""

  def __init__(self, store: SamplingStore, pool: str, sampler: Any, concurrency: int) -> None:
    self.store, self.pool, self.sampler, self.concurrency = store, pool, sampler, concurrency
    self.revision = ""
    # Each record is read once and dropped when its result is saved.
    self.records: dict[str, dict[str, Any]] = {}
    self.newest = "0-0"
    self.in_flight: dict[str, asyncio.Task] = {}
    self.sends = itertools.count()
    self.served: dict[str, int] = {}  # the number of each job's latest send

  async def run(self) -> None:
    try:
      while True:
        try:
          await self.dispatch()
          await self.store.wait_for_new(self.pool, self.newest, RETRY_MS)
        except Exception:
          # A Redis or Kubernetes hiccup must not end the pool's dispatching.
          logger.exception("Dispatching pool %s failed; retrying", self.pool)
          await asyncio.sleep(RETRY_MS / 1000)
    finally:
      for task in self.in_flight.values():
        task.cancel()

  async def dispatch(self) -> None:
    await self.read()
    if await self.ready():
      # Read after the await above, so a request whose result was saved meanwhile is not sent again.
      batch, started, tasks = self.to_send()[: self.concurrency - len(self.in_flight)], time.time(), []
      for r in batch:
        self.in_flight[r["request_id"]] = task = asyncio.create_task(self.send(r))
        task.add_done_callback(lambda _, request_id=r["request_id"]: self.in_flight.pop(request_id))
        tasks.append(task)
      if tasks:
        # the group record once every send of this dispatch has answered
        asyncio.gather(*tasks, return_exceptions=True).add_done_callback(lambda _, b=batch, t=started: sampled_group(b, t))

  async def ready(self) -> bool:
    """Whether the sampler serves the current revision, loading it if it can. A revision it can never load fails its
    records, and the next read moves the pool on."""
    revision = self.revision
    if revision and not Path(revision, "metadata.json").is_file():
      # Retention can delete the revision before or after the switch, and no sampler that lacks it can load it again.
      await self.fail_revision(revision, f"Sampler weights {revision} no longer exist")
      return False
    try:
      return await self.sampler.ensure_weights(revision)
    except RevisionUnloadable as exc:
      held = ", ".join(exc.held)
      message = f"Sampler weights {exc.revision} cannot be loaded on top of the weights the pool holds ({held}). Save full weights to sample them."
      await self.fail_revision(exc.revision, message)
      return False

  async def fail_revision(self, revision: str, message: str) -> None:
    for r in [r for r in self.records.values() if r["revision"] == revision]:
      await self.set_result(r, failed(message))

  async def read(self) -> None:
    """Reads new records, fails those past their deadline, and switches revision once nothing pending names the
    current one."""
    for r in await self.store.pending(self.pool, after=self.newest):
      self.records[r["entry_id"]] = r
      self.newest = r["entry_id"]
    # The API server's deadline sweep and reconcile save results too, so a record they answered is dropped.
    held = list(self.records.values())
    answered = await self.store.answered([r["request_id"] for r in held])
    for r in held:
      if r["request_id"] in answered:
        self.records.pop(r["entry_id"], None)  # a send may have saved it during the await above
    for r in list(self.records.values()):
      if (waited := time.time() - accepted_at(r)) > SAMPLE_DEADLINE_SECONDS:
        logger.warning("Failing %s after %.0fs without a result", r["request_id"], waited)
        await self.set_result(r, failed(f"No sampler answered in {waited:.0f}s; pool {self.pool} serves {self.revision or 'base weights'}"))
        # A send still waiting on a replica that never answers would keep its slot forever.
        if task := self.in_flight.get(r["request_id"]):
          task.cancel()
    records = list(self.records.values())
    # Nothing pending names the current revision, so nothing is in flight on it.
    if records and all(r["revision"] != self.revision for r in records):
      revision = records[0]["revision"]
      logger.info("Pool %s switching from %r to %r", self.pool, self.revision, revision)
      self.revision = revision

  def to_send(self) -> list[dict[str, Any]]:
    """The current revision's records not in flight, one from each job in turn and oldest first within a job. The job
    served longest ago goes first, so one job's big batch does not hold up the others."""
    jobs: dict[str, list[dict[str, Any]]] = {}
    for r in self.records.values():
      if r["revision"] == self.revision and r["request_id"] not in self.in_flight:
        jobs.setdefault(job_of(r), []).append(r)
    turns = sorted(jobs.items(), key=lambda job: self.served.get(job[0], -1))
    return [r for turn in itertools.zip_longest(*(records for _, records in turns)) for r in turn if r is not None]

  async def set_result(self, r: dict[str, Any], result: dict[str, Any]) -> None:
    await self.store.set_result(r, result)
    self.records.pop(r["entry_id"], None)

  async def send(self, r: dict[str, Any]) -> None:
    """Saves the sampler's answer. None, or an error, leaves the record pending to be sent again."""
    self.served[job_of(r)] = next(self.sends)
    started = time.time()
    try:
      # A sampler that never answers must not hold the send past its record's deadline.
      timeout = max(0.0, accepted_at(r) + SAMPLE_DEADLINE_SECONDS - time.time())
      result = await asyncio.wait_for(self.sampler.generate(r), timeout)
    except Exception as exc:
      op_finished(r, "sampler", r.get("pool"), started, type(exc).__name__)
      logger.info("Retrying %s after %r", r["request_id"], exc)
      return
    if result is not None:
      op_finished(r, "sampler", r.get("pool"), started, result=result)
      await self.set_result(r, result)


class HttpSampler:
  """Sends a pool's records through `router` to its vllm serve replicas, and keeps each replica on the pool's revision.
  `urls`, when given, lists the replicas instead of finding them by pod label."""

  def __init__(self, http: httpx.AsyncClient, pool: str, router: str, urls: list[str]) -> None:
    self.http, self.pool, self.router, self.urls = http, pool, router, urls
    self.revision: str | None = None
    self.k8s = None
    self.loading: dict[str, asyncio.Task] = {}
    self.checking: dict[str, asyncio.Task] = {}
    self.paused: dict[str, float] = {}  # when each replica last paused to load weights
    self.checked: set[str] = set()  # replicas checked for a pause or an open update, checked again after a load of ours breaks off

  async def ensure_weights(self, revision: str) -> bool:
    """Starts a load on every checked replica not at `revision`. True once one replica serves it. Raises
    RevisionUnloadable when every replica answers, none holds weights `revision` applies on, and none is loading."""
    if revision != self.revision:
      self.revision = revision
      self.write_revision()
    target = revision or "default"
    urls = await self.list_replicas()
    for url in urls:
      if url not in self.checked and url not in self.loading and url not in self.checking:
        self.checking[url] = asyncio.create_task(self.check(url))
    # A replica under its first-sight check is left out until the check finishes.
    seen = [url for url in urls if url not in self.checking]
    versions = await asyncio.gather(*(self.weight_version(url) for url in seen), return_exceptions=True)
    held = [version for version in versions if isinstance(version, str)]
    # A replica that did not answer may hold the revision, and one mid-load is about to hold other weights.
    stuck = revision and held and len(held) == len(urls) and target not in held and not self.loading
    if stuck and Path(revision, "metadata.json").is_file() and not any(delta_applies(revision, version) for version in held):
      raise RevisionUnloadable(revision, held)
    for url, version in zip(seen, versions):
      # A replica still starting does not answer yet, and a pool on base weights never loads any.
      if revision and isinstance(version, str) and version != target and url not in self.loading:
        self.loading[url] = asyncio.create_task(self.load_weights(url, revision))
    return target in held

  async def generate(self, r: dict[str, Any]) -> dict[str, Any] | None:
    """The result, or None when the answer did not come from the record's revision."""
    target, sent = r["revision"] or "default", time.monotonic()
    params = sample_params(r)
    attempt_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    # The store hands prompts back as array('I'), which JSON cannot encode.
    body = {"request_id": attempt_id, "token_ids": list(r["prompt_token_ids"]), "model": r["model"], "sampling_params": params}
    url = self.router
    response = await self.http.post(f"{url}/inference/v1/generate", json=body, timeout=None)
    # Envoy names the replica that answered. Its weights must still be the target, with no pause to load others since this was sent.
    replica = f"http://{name}" if (name := response.headers.get("x-openrl-upstream")) else url
    # A 404 can be a just-frozen LoRA snapshot that the replica's view of the shared volume does not show yet.
    if response.status_code in (400, 422):
      return failed(f"Sampler rejected the request ({response.status_code}): {response.text[:500]}")
    if response.status_code != 200 or await self.version_after_answer(replica) != target or self.paused.get(replica, 0.0) >= sent:
      return None
    try:
      return to_result(response.json(), r)
    except (KeyError, TypeError, ValueError) as exc:
      return failed(f"Sampler response broke the contract: {exc!r}")

  async def version_after_answer(self, replica: str) -> str | None:
    """The weight version of the replica that answered, or None when three tries cannot reach it. A slow check
    must not throw away a finished answer."""
    for _ in range(3):
      try:
        return await self.weight_version(replica)
      except httpx.TransportError as exc:
        logger.info("Checking the weight version of %s failed: %r", replica, exc)
    return None

  async def close(self) -> None:
    """A load cut off midway leaves its replica paused, so loads get a moment to finish and resume before the HTTP client closes."""
    if tasks := [*self.loading.values(), *self.checking.values()]:
      _, pending = await asyncio.wait(tasks, timeout=5)
      for task in pending:
        task.cancel()

  async def load_weights(self, url: str, revision: str) -> None:
    """Pauses the replica, loads `revision` and resumes it. A load that fails or is cut off leaves the replica unchecked,
    so the next dispatch's check closes any update it left open, forgets partial weights and resumes the replica."""
    try:
      if not Path(revision, "metadata.json").is_file():
        return  # the dispatcher fails the revision's records
      if not delta_applies(revision, current := await self.weight_version(url)):
        raise RuntimeError(f"{url} holds {current!r}, not a revision delta {revision!r} applies on")
      self.paused[url] = time.monotonic()
      self.checked.discard(url)
      await self.post(url, "/pause?mode=wait&clear_cache=false")
      await self.post(url, "/start_weight_update")
      await self.post(url, "/update_weights", {"update_info": {"target_weights_path": revision}})
      # Before the finish names the version, so a replica never reports the revision with the old weights' cache.
      if not await self.reset_prefix_cache(url):
        raise RuntimeError("vLLM did not reset the prefix cache")
      await self.post(url, "/finish_weight_update", {"weight_version": revision})
      await self.post(url, "/resume")
      self.checked.add(url)
    except Exception as exc:
      logger.warning("Loading %r on %s failed: %r", revision, url, exc)  # the next dispatch checks the replica and tries again
    finally:
      self.loading.pop(url)

  async def reset_prefix_cache(self, url: str) -> bool:
    """vLLM refuses while any request holds cache blocks, so a refusal is tried again a moment later."""
    for attempt in range(3):
      if attempt:
        await asyncio.sleep(0.5)
      if (await self.post(url, "/reset_prefix_cache")).json()["success"]:
        return True
    return False

  async def check(self, url: str) -> None:
    """A replica's check on first sight and after a load fails. Any weight update a load left open is closed and its
    weights forgotten, paused or not, since a dispatcher stopped mid-update used to resume the replica. A failed or
    cut-off load leaves a replica paused, where the router would queue requests forever, so one found paused is resumed.
    It runs beside the dispatch, since a wedged replica can take minutes to answer."""
    try:
      await self.forget_weights(url)
      if (await self.http.get(f"{url}/is_paused", timeout=5)).raise_for_status().json()["is_paused"]:
        logger.info("Resuming %s, left paused by an earlier load", url)
        await self.post(url, "/resume")
      self.checked.add(url)
    except Exception as exc:
      logger.info("Checking %s failed: %r", url, exc)  # a replica still starting does not answer, and a later dispatch checks again
    finally:
      self.checking.pop(url)

  async def forget_weights(self, url: str) -> None:
    """Closes an update a cut-off load left active, which refuses every later start, and has the replica report
    UNKNOWN_WEIGHTS until a full load, since it may hold part of a revision. vLLM has no abort, but a bare finish closes
    an active update and is refused when none is, so a replica that was not updating keeps its version."""
    # It waits behind an update a dead dispatcher left running, but a wedged replica must not hold up the pool.
    response = await self.http.post(f"{url}/finish_weight_update", timeout=300)
    if response.status_code != 200:
      return
    logger.warning("Weights on %s may be partial, so it reports %r until a full load replaces them", url, UNKNOWN_WEIGHTS)
    await self.post(url, "/update_weight_version", {"new_version": UNKNOWN_WEIGHTS})

  async def post(self, url: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
    return (await self.http.post(url + path, json=body, timeout=None)).raise_for_status()

  async def list_replicas(self) -> list[str]:
    if self.urls:
      return self.urls
    if self.k8s is None:
      from kubernetes import client, config

      config.load_incluster_config()
      self.k8s = client.CoreV1Api()
    selector = f"app.kubernetes.io/component=sampler,app.kubernetes.io/instance={self.pool}"
    namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    pods = await asyncio.to_thread(self.k8s.list_namespaced_pod, namespace, label_selector=selector, _request_timeout=10)
    return [f"http://{p.status.pod_ip}:8000" for p in pods.items if p.status.pod_ip and p.status.phase == "Running"]

  async def weight_version(self, url: str) -> str:
    return (await self.http.get(f"{url}/weight_info", timeout=WEIGHT_INFO_TIMEOUT_SECONDS)).raise_for_status().json()["weight_version"]

  def write_revision(self) -> None:
    path = revision_file(self.pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(self.revision or "default")
    path.with_suffix(".tmp").rename(path)


def redis_client(url: str) -> redis.Redis:
  """When replicas come up, every in-flight send saves its result at once. Past max_connections a
  save waits for a free connection; redis-py's default pool of 100 raises instead."""
  return redis.Redis.from_pool(redis.BlockingConnectionPool.from_url(url, decode_responses=True, max_connections=64, timeout=30))


def http_client() -> httpx.AsyncClient:
  """Sends have no connection limit, so a burst goes out at once. httpx scans its whole pool for each idle
  connection whenever a request starts or ends, so the hundreds a burst leaves idle stall this loop for
  seconds. Few are kept."""
  return httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=32))


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  logging.getLogger("httpx").setLevel(logging.WARNING)
  router = os.getenv("OPEN_RL_SAMPLER_ROUTER", "http://127.0.0.1:8081")
  store, pool = SamplingStore(redis_client(os.environ["REDIS_URL"])), os.environ["OPEN_RL_SAMPLER_POOL"]
  async with http_client() as http:
    sampler = HttpSampler(http, pool, router, [])
    task = asyncio.create_task(Dispatcher(store, pool, sampler, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256"))).run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    try:
      with contextlib.suppress(asyncio.CancelledError):
        await task
    finally:
      await sampler.close()


if __name__ == "__main__":
  asyncio.run(main())
