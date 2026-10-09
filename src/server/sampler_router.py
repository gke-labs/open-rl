"""One HTTP dispatch per routed sample, with an expiring receipt for polling.

The API has one replica (also required by session ownership). A pending receipt
without a local task is interrupted work from a previous API process, never a
reason to replay generation. Redis stores receipts, not a second sampling queue.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from server.sampler_http import failed
from server.store import StateStore

logger = logging.getLogger(__name__)
PREFIX = "routed-"
RESULT_TTL = 300


class SampleSequence(BaseModel):
  tokens: list[int]
  logprobs: list[float | None] | None = None
  stop_reason: str | None = None


class SampleResult(BaseModel):
  sequences: list[SampleSequence]
  prompt_logprobs: list[float | None] | None = None
  prompt_cache_hit_tokens: int = 0


class RouterBusy(RuntimeError):
  pass


class SamplerRouter:
  def __init__(self, state: StateStore, *, timeout: float = 1800, capacity: int = 256, client: httpx.AsyncClient | None = None):
    if timeout <= 0 or capacity <= 0:
      raise ValueError("Router timeout and capacity must be positive")
    self.state = state
    self.timeout = timeout
    self.capacity = capacity
    self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10, pool=10), limits=httpx.Limits(max_connections=capacity))
    self.tasks: dict[str, asyncio.Task] = {}
    self.lock = asyncio.Lock()

  async def write(self, request_id: str, result: dict, ttl: float = RESULT_TTL) -> None:
    await self.state.set_value(f"open_rl:routed_sample:{request_id}", json.dumps(result), ttl_seconds=ttl)

  async def submit(self, lookup: Callable[[str], str | None], model_id: str, request: dict[str, Any]) -> None:
    request_id = request["request_id"]
    async with self.lock:
      if len(self.tasks) >= self.capacity:
        raise RouterBusy("Routed sampler is at capacity; try again later")
      await self.write(request_id, {"type": "try_again"}, self.timeout + RESULT_TTL)
      task = asyncio.create_task(self.dispatch(lookup, model_id, request))
      self.tasks[request_id] = task
      task.add_done_callback(lambda done: self.finished(request_id, done))

  def finished(self, request_id: str, task: asyncio.Task) -> None:
    self.tasks.pop(request_id, None)
    if not task.cancelled() and (error := task.exception()) is not None:
      logger.error("Could not persist routed sample %s: %s", request_id, error)

  async def dispatch(self, lookup: Callable[[str], str | None], model_id: str, request: dict[str, Any]) -> None:
    try:
      async with asyncio.timeout(self.timeout):
        url = await asyncio.to_thread(lookup, model_id)
        if not url:
          raise RuntimeError("No ready llm-d router for this model")
        body = {
          "model": request["lora_id"] or request["model_id"],
          "prompt": request["prompt_token_ids"],
          "max_tokens": request["max_tokens"],
          "openrl": request,
        }
        response = await self.client.post(f"{url}/v1/completions", json=body)
        if response.status_code != 200:
          raise RuntimeError(f"llm-d router returned {response.status_code}: {response.text[:500]}")
        result = response.json()
        if not isinstance(result, dict) or result.get("type") not in {"sample", "RequestFailedResponse"}:
          raise ValueError("Invalid llm-d sampling response")
        if result["type"] == "sample":
          sample = SampleResult.model_validate(result, strict=True)
          if len(sample.sequences) != request["num_samples"]:
            raise ValueError("llm-d returned the wrong number of sample sequences")
        elif not isinstance(result.get("error_message"), str):
          raise ValueError("Invalid llm-d error response")
    except asyncio.CancelledError:
      result = failed("Routed sampling interrupted by API shutdown; execution may have started")
    except Exception as exc:
      # A timeout or lost response may follow successful GPU execution. Never
      # retry or enqueue this request: neither transport deduplicates generation.
      result = failed(f"Routed sampling failed ({type(exc).__name__}): {exc}; request was not replayed")
    await self.write(request["request_id"], result)

  async def result(self, request_id: str, timeout: float = 60) -> dict:
    if task := self.tasks.get(request_id):
      await asyncio.wait({task}, timeout=timeout)
      if not task.done():
        return {"type": "try_again"}
    # Read only after the writer has finished, so a stale pending read cannot
    # race with completion and overwrite a successful result as interrupted.
    raw = await self.state.get_value(f"open_rl:routed_sample:{request_id}")
    if raw is None:
      return failed("Routed sampling receipt expired or is unavailable; execution status is unknown")
    result = json.loads(raw)
    if result.get("type") == "try_again":
      result = failed("Routed sampling interrupted before its result was saved; execution may have started")
      await self.write(request_id, result)
    return result

  async def close(self) -> None:
    tasks = list(self.tasks.values())
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await self.client.aclose()
