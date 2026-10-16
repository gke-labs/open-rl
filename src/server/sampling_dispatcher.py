"""One sampling pool's dispatcher: sends each pending record through the pool's router, Envoy in front
of the llm-d EPP, and commits the first result. Run as `python -m server.sampling_dispatcher`."""

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis

from server.sampling_store import SamplingStore

logger = logging.getLogger(__name__)

# The dispatcher wakes for each new request, or after this long to retry a failed send and look at its replicas.
RETRY_MS = 1000
# A request no replica has answered for this long fails, so its client stops waiting.
SAMPLE_DEADLINE_SECONDS = float(os.getenv("OPEN_RL_SAMPLE_DEADLINE_SECONDS", "3600"))


def revision_file(pool: str) -> Path:
  """The readiness probe compares vLLM's weight version with the revision in this file."""
  return Path(os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl"), "sampling", pool, "revision")


def failed(message: str) -> dict[str, str]:
  return {"type": "RequestFailedResponse", "error_message": message}


def to_result(body: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
  """The old sampler's result shape, from a vLLM /inference/v1/generate response."""
  logprobs = [[e["logprob"] for e in c["logprobs"]["content"]] for c in body["choices"]]
  sequences = [{"tokens": c["token_ids"], "logprobs": lp, "stop_reason": c["finish_reason"]} for c, lp in zip(body["choices"], logprobs, strict=True)]
  result: dict[str, Any] = {"type": "sample", "sequences": sequences}
  if record["include_prompt_logprobs"]:
    pairs = zip(record["prompt_token_ids"], body["prompt_logprobs"], strict=True)
    result["prompt_logprobs"] = [None if e is None else e[str(token)]["logprob"] for token, e in pairs]
  return result


class Dispatcher:
  """Sends through `router`. A run on one machine has no router and lists its replicas in `urls`."""

  def __init__(self, store: SamplingStore, pool: str, http: httpx.AsyncClient, router: str | None, urls: list[str], concurrency: int = 256) -> None:
    self.store, self.pool, self.http, self.router, self.urls, self.concurrency = store, pool, http, router, urls, concurrency
    self.revision, self.k8s = "", None
    # This dispatcher is the pool's only committer, so it reads each record once and drops it on commit.
    self.records: dict[str, dict[str, Any]] = {}
    self.newest = "0-0"
    self.in_flight: dict[str, asyncio.Task] = {}
    self.loading: dict[str, asyncio.Task] = {}
    self.paused: dict[str, float] = {}  # when each replica last paused to load weights

  async def run(self) -> None:
    self.write_revision()
    try:
      while True:
        try:
          await self.store.wait_for_new(self.pool, await self.dispatch(), RETRY_MS)
        except Exception:
          # A Redis or Kubernetes hiccup must not end the pool's dispatching.
          logger.exception("Dispatching pool %s failed; retrying", self.pool)
          await asyncio.sleep(RETRY_MS / 1000)
    finally:
      for task in [*self.in_flight.values(), *self.loading.values()]:
        task.cancel()

  async def dispatch(self) -> str:
    """Sends what it can, and returns the newest stream id it saw."""
    for r in await self.store.pending(self.pool, after=self.newest):
      self.records[r["entry_id"]] = r
      self.newest = r["entry_id"]
    records = [r for r in list(self.records.values()) if not await self.expired(r)]
    # Nothing pending names the current revision, so nothing is in flight on it.
    if records and all(r["revision"] != self.revision for r in records):
      logger.info("Pool %s switching from %r to %r", self.pool, self.revision, records[0]["revision"])
      self.revision = records[0]["revision"]
      self.write_revision()
    target = self.revision or "default"
    urls = await self.list_replicas()
    versions = await asyncio.gather(*(self.weight_version(url) for url in urls), return_exceptions=True)
    for url, version in zip(urls, versions, strict=True):
      # A replica still starting does not answer yet, and a pool on base weights never loads any.
      if self.revision and isinstance(version, str) and version != target and url not in self.loading:
        self.loading[url] = asyncio.create_task(self.load_weights(url, self.revision))
    if target in versions:
      # Read after the awaits above, so a request that committed meanwhile is not sent again.
      to_send = [r for r in self.records.values() if r["revision"] == self.revision and r["request_id"] not in self.in_flight]
      for r in to_send[: self.concurrency - len(self.in_flight)]:
        self.in_flight[r["request_id"]] = task = asyncio.create_task(self.send(r))
        task.add_done_callback(lambda _, request_id=r["request_id"]: self.in_flight.pop(request_id))
    return self.newest

  async def expired(self, r: dict[str, Any]) -> bool:
    """Fails a request that has waited past the deadline; its stream id is when it was accepted."""
    waited = time.time() - int(r["entry_id"].split("-")[0]) / 1000
    if waited <= SAMPLE_DEADLINE_SECONDS:
      return False
    logger.warning("Failing %s after %.0fs without a result", r["request_id"], waited)
    await self.commit(r, failed(f"No sampler answered in {waited:.0f}s; pool {self.pool} serves {self.revision or 'base weights'}"))
    return True

  async def commit(self, r: dict[str, Any], result: dict[str, Any]) -> None:
    await self.store.commit(r, result)
    self.records.pop(r["entry_id"], None)

  async def send(self, r: dict[str, Any]) -> None:
    target, sent = self.revision or "default", time.monotonic()
    params = {**r["sampling_params"], "logprobs": 0, "prompt_logprobs": 0 if r["include_prompt_logprobs"] else None}
    attempt_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    body = {"request_id": attempt_id, "token_ids": r["prompt_token_ids"], "model": r["model"], "sampling_params": params}
    url = self.router or random.choice(self.urls)
    result = None
    try:
      response = await self.http.post(f"{url}/inference/v1/generate", json=body, timeout=None)
      if response.status_code in (400, 404, 422):
        result = failed(f"Sampler rejected the request ({response.status_code}): {response.text[:500]}")
      elif response.status_code == 200:
        # Envoy names the replica that answered. Its weights must still be the target, with no pause to load others since this was sent.
        replica = f"http://{name}" if (name := response.headers.get("x-openrl-upstream")) else url
        if await self.weight_version(replica) == target and self.paused.get(replica, 0.0) < sent:
          try:
            result = to_result(response.json(), r)
          except (KeyError, TypeError, ValueError) as exc:
            result = failed(f"Sampler response broke the contract: {exc!r}")
    except httpx.HTTPError as exc:
      logger.info("Retrying %s after %r", r["request_id"], exc)
    if result:
      await self.commit(r, result)

  async def load_weights(self, url: str, revision: str) -> None:
    """vLLM's pause lets the requests the replica is running finish first."""
    try:
      if not (path := Path(revision, "metadata.json")).is_file():
        # Retention can delete a revision between the API server's check and its pin.
        for r in list(self.records.values()):
          if r["revision"] == revision:
            await self.commit(r, failed(f"Sampler weights {revision} no longer exist"))
        return
      metadata, current = json.loads(path.read_text()), await self.weight_version(url)
      if metadata.get("format") == "sparse_delta" and (metadata.get("parent") or "default") != current:
        raise RuntimeError(f"{url} holds {current!r}, not the parent of delta {revision!r}")
      self.paused[url] = time.monotonic()
      await self.post(url, "/pause?mode=wait&clear_cache=false")
      await self.post(url, "/start_weight_update")
      await self.post(url, "/update_weights", {"update_info": {"target_weights_path": revision}})
      await self.post(url, "/finish_weight_update", {"weight_version": revision})
      if not (await self.post(url, "/reset_prefix_cache")).json()["success"]:
        raise RuntimeError("vLLM did not reset the prefix cache")
      await self.post(url, "/resume")
    except Exception as exc:
      logger.warning("Loading %r on %s failed: %r", revision, url, exc)  # the next dispatch tries again
    finally:
      self.loading.pop(url)

  async def post(self, url: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
    response = await self.http.post(url + path, json=body, timeout=None)
    response.raise_for_status()
    return response

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
    response = await self.http.get(f"{url}/weight_info", timeout=5)
    response.raise_for_status()
    return response.json()["weight_version"]

  def write_revision(self) -> None:
    path = revision_file(self.pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(self.revision or "default")
    path.with_suffix(".tmp").rename(path)


def redis_client(url: str) -> redis.Redis:
  """When replicas come up, every in-flight send commits at once. Past max_connections a
  commit waits for a free connection; redis-py's default pool of 100 raises instead."""
  return redis.Redis.from_pool(redis.BlockingConnectionPool.from_url(url, decode_responses=True, max_connections=64, timeout=30))


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  logging.getLogger("httpx").setLevel(logging.WARNING)
  # A run on one machine lists its replicas. A pool on Kubernetes sends through its Envoy.
  urls = [url for url in os.getenv("OPEN_RL_SAMPLER_ENDPOINTS", "").split(",") if url]
  router = None if urls else os.getenv("OPEN_RL_SAMPLER_ROUTER", "http://127.0.0.1:8081")
  store = SamplingStore(redis_client(os.environ["REDIS_URL"]))
  async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as http:
    dispatcher = Dispatcher(store, os.environ["OPEN_RL_SAMPLER_POOL"], http, router, urls, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256")))
    task = asyncio.create_task(dispatcher.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await task


if __name__ == "__main__":
  asyncio.run(main())
