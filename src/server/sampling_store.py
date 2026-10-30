"""Each sampling pool's pending requests are one Redis stream, read without removal.
Results land in the `open_rl:future:<id>` list that `retrieve_future` already reads."""

import json
import time
from array import array
from typing import Any

import redis.asyncio as redis

RESULT_TTL_SECONDS = 300
# An accepted request may never get a result, so its idempotency key expires instead of staying until Redis is full.
PENDING_TTL_SECONDS = 3 * 24 * 3600
# The API server reads streams this many entries at a time, so a backlog of long prompts never sits in memory at once.
SWEEP_PAGE = 100


def stream_key(pool: str) -> str:
  return f"open_rl:sampling:{pool}"


def compact_prompt(record: dict[str, Any]) -> dict[str, Any]:
  """A json.loads object_hook that holds a prompt as array('I'), four bytes a token against about 36 as a list of ints,
  and drops each list as soon as its record is read. A prompt array('I') cannot hold stays the list, for vLLM to refuse."""
  if isinstance(tokens := record.get("prompt_token_ids"), list):
    try:
      record["prompt_token_ids"] = array("I", tokens)
    except (OverflowError, TypeError):
      pass
  return record


def idempotency_key(pool: str, key: str) -> str:
  return f"open_rl:sampling:{pool}:idem:{key}"


def future_key(request_id: str) -> str:
  return f"open_rl:future:{request_id}"


def job_of(r: dict[str, Any]) -> str:
  """The job that asked for the record. Records accepted before the API server named the job group under their model."""
  return r.get("job") or r["model"]


def job_and_revision(fields: dict[str, str]) -> tuple[str, str]:
  """An entry's job and revision without decoding its prompt. Entries accepted before the stream held both fields
  decode the record."""
  if "job" in fields:
    return fields["job"], fields["revision"]
  record = json.loads(fields["record"], object_hook=compact_prompt)
  return job_of(record), record["revision"]


def entry_record(entry_id: str, fields: dict[str, str]) -> dict[str, Any]:
  return {**json.loads(fields["record"], object_hook=compact_prompt), "entry_id": entry_id}


class SamplingStore:
  def __init__(self, client: redis.Redis):
    self.redis = client

  async def accept(self, record: dict[str, Any]) -> str:
    """The new request's id, or the id already stored under its idempotency key."""
    # The job and revision get fields of their own, so the API server reads them without decoding prompts.
    fields = {"record": json.dumps(record), "job": job_of(record), "revision": record["revision"]}
    stream = stream_key(record["pool"])
    if not record.get("idempotency_key"):
      await self.redis.xadd(stream, fields)
      return record["request_id"]
    key = idempotency_key(record["pool"], record["idempotency_key"])

    async def add(pipe) -> str:
      if existing := await pipe.get(key):
        return existing
      pipe.multi()
      pipe.set(key, record["request_id"], ex=PENDING_TTL_SECONDS)
      pipe.xadd(stream, fields)
      return record["request_id"]

    return await self.redis.transaction(add, key, value_from_callable=True)

  async def pending(self, pool: str, limit: int | None = 1000, after: str = "0-0") -> list[dict[str, Any]]:
    """Pending records in accept order. The dispatcher holds every one until its result is set, so prompts come back compact."""
    entries = await self.redis.xrange(stream_key(pool), min=f"({after}", count=limit)
    return [entry_record(entry_id, fields) for entry_id, fields in entries]

  async def wait_for_new(self, pool: str, after: str, timeout_ms: int) -> None:
    await self.redis.xread({stream_key(pool): after}, count=1, block=timeout_ms)

  async def set_result(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    """Save the first final result and drop the pending entry. False if it already has one."""
    stream, future, entry_id = stream_key(record["pool"]), future_key(record["request_id"]), record["entry_id"]

    async def save(pipe) -> bool:
      if not await pipe.xrange(stream, entry_id, entry_id):
        return False
      pipe.multi()
      pipe.rpush(future, json.dumps(result))
      pipe.expire(future, RESULT_TTL_SECONDS)
      pipe.xdel(stream, entry_id)
      if key := record.get("idempotency_key"):
        pipe.expire(idempotency_key(record["pool"], key), RESULT_TTL_SECONDS)
      return True

    # Every set_result writes the result key, so watching it lets only one through.
    return await self.redis.transaction(save, future, value_from_callable=True)

  async def entries(self, stream: str, until: str = "+"):
    """The stream's entries up to `until`, a page at a time."""
    after = "-"
    while page := await self.redis.xrange(stream, min=after, max=until, count=SWEEP_PAGE):
      for entry in page:
        yield entry
      after = f"({page[-1][0]}"

  async def fail_older_than(self, seconds: float, result: dict[str, Any]) -> int:
    """Saves `result` for every pending request, in any pool, accepted more than `seconds` ago. Returns how many."""
    cutoff, failed = str(int((time.time() - seconds) * 1000)), 0
    async for stream in self.redis.scan_iter(match=stream_key("*"), _type="stream"):
      async for entry_id, fields in self.entries(stream, cutoff):
        failed += await self.set_result(entry_record(entry_id, fields), result)
    return failed

  async def answered(self, request_ids: list[str]) -> set[str]:
    """The requests among `request_ids` that have a result."""
    async with self.redis.pipeline(transaction=False) as pipe:
      for request_id in request_ids:
        pipe.exists(future_key(request_id))
      found = await pipe.execute()
    return {request_id for request_id, exists in zip(request_ids, found) if exists}

  async def pools(self) -> list[str]:
    """Every pool with a stream, whether or not it has pending requests."""
    return [stream.removeprefix(stream_key("")) async for stream in self.redis.scan_iter(match=stream_key("*"), _type="stream")]

  async def pending_jobs(self, pool: str) -> set[str]:
    """The jobs with a pending request in the pool."""
    return {job_and_revision(fields)[0] async for _, fields in self.entries(stream_key(pool))}

  async def fail_job(self, pool: str, job: str, result: dict[str, Any]) -> int:
    """Saves `result` for every pending request of `job` in `pool`. Returns how many."""
    failed = 0
    async for entry_id, fields in self.entries(stream_key(pool)):
      if job_and_revision(fields)[0] == job:
        failed += await self.set_result(entry_record(entry_id, fields), result)
    return failed

  async def delete_if_empty(self, pool: str) -> bool:
    """Deletes the pool's stream when nothing is pending in it, since set_result's XDEL leaves the empty stream behind
    and Redis never evicts it. Watching the stream keeps a request accepted meanwhile. Returns whether it deleted."""
    stream = stream_key(pool)

    async def delete(pipe) -> bool:
      if await pipe.xlen(stream):
        return False
      pipe.multi()
      pipe.delete(stream)
      return True

    return await self.redis.transaction(delete, stream, value_from_callable=True)

  async def pinned_revisions(self, pool: str) -> set[str]:
    """Revisions that pending requests still name, so their weights must stay."""
    return {job_and_revision(fields)[1] async for _, fields in self.entries(stream_key(pool))} - {""}
