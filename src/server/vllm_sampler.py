"""A sampling pool replica with vLLM in this process. It serves its pool's stream itself, with no router, and when its
job shares a GPU with the trainer it takes a time-slicer turn around each batch, the way the trainer does.

  python -m server.vllm_sampler --pool <pool>
"""

import argparse
import asyncio
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, time_slicing_enabled, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, WorkloadRef, local_workload_name
from server.lora_snapshots import snapshot_root
from server.sampling_dispatcher import Dispatcher, RevisionUnloadable, delta_applies, failed, redis_client
from server.sampling_store import SamplingStore
from server.vllm_options import engine_args, sample_params, sample_result

logger = logging.getLogger(__name__)

# The readiness probe passes once vLLM is up, and asleep when it shares its GPU.
READY_FILE = os.getenv("OPEN_RL_SAMPLER_READY_FILE", "/tmp/open-rl-sampler-ready")


class Sampler:
  """A vLLM engine that answers a pool's records and swaps its weights between batches. `revision` is the weights
  it holds, "" for the base model. A failed update leaves partial weights behind, so the sampler refuses work until
  the process is restarted."""

  def __init__(self, engine: Any, base_model: str) -> None:
    self.engine, self.base_model = engine, base_model
    self.lora_ids: dict[str, int] = {}
    self.revision = ""
    self.update_failed = False

  async def wake(self) -> None:
    await self.engine.wake_up(tags=["weights", "kv_cache"])

  async def sleep(self) -> None:
    await self.engine.sleep(level=1)

  async def ensure_weights(self, revision: str) -> bool:
    """Whether the engine holds `revision` after this, loading it if it can. Raises RevisionUnloadable for a delta
    that does not apply on the weights it holds."""
    if self.update_failed:
      return False
    if revision == self.revision:
      return True
    if not revision:
      return False  # base weights cannot come back in place
    try:
      applies = delta_applies(revision, self.revision or "default")
    except FileNotFoundError:
      return False  # the dispatcher fails the revision's records
    if not applies:
      raise RevisionUnloadable(revision, [self.revision or "default"])
    from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

    # No resume on failure: partially updated weights must not be served.
    try:
      await self.engine.pause_generation(mode="wait", clear_cache=True)
      await self.engine.start_weight_update()
      await self.engine.update_weights(WeightTransferUpdateRequest(update_info={"target_weights_path": revision}))
      await self.engine.finish_weight_update(weight_version=revision)
      await self.engine.reset_encoder_cache()
      await self.engine.resume_generation()
    except BaseException:
      self.update_failed = True
      raise
    self.revision = revision
    return True

  async def generate(self, r: dict[str, Any]) -> dict[str, Any]:
    """The result. A request vLLM refuses, such as a prompt past max_model_len or a token out of vocabulary, fails
    at once, since sending it again would only take another turn from the trainer. So does a request in flight when
    the engine dies, since a request that kills the engine would kill each restarted one too."""
    from vllm import SamplingParams
    from vllm.exceptions import VLLMClientError
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

    lora_request = self.adapter(r)
    if self.engine.errored:
      raise RuntimeError("The engine is dead")  # left pending for the restarted engine, since this one never saw it
    prompt = list(r["prompt_token_ids"])
    request_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    final = None
    try:
      params = SamplingParams(**sample_params(r), output_kind=RequestOutputKind.FINAL_ONLY)
      stream = self.engine.generate(prompt={"prompt_token_ids": prompt}, sampling_params=params, request_id=request_id, lora_request=lora_request)
      async for output in stream:
        final = output
    except (TypeError, ValueError, VLLMClientError) as exc:
      return failed(f"Sampler rejected the request: {exc}")
    except EngineDeadError:
      return failed("The sampler's vLLM engine died while serving this request")
    except EngineGenerateError as exc:
      return failed(f"Sampler failed the request: {exc.__cause__!r}")
    sequences = []
    for choice in final.outputs:
      tokens = list(choice.token_ids)
      sequences.append((tokens, [step[token].logprob for token, step in zip(tokens, choice.logprobs or [])], choice.finish_reason))
    prompt_logprobs = None
    if r["include_prompt_logprobs"]:
      prompt_logprobs = [None if step is None else step[token].logprob for token, step in zip(prompt, final.prompt_logprobs)]
    return sample_result(sequences, prompt_logprobs)

  def adapter(self, r: dict[str, Any]) -> Any:
    """The record's LoRA, or None for base weights. vLLM loads an adapter inside the engine step, where a missing one
    kills the engine, so a snapshot this pod's view of the shared volume does not show yet raises here instead, and
    the record is sent again."""
    if r["model"] == self.base_model:
      return None
    from vllm.lora.request import LoRARequest

    path = snapshot_root() / r["model"]
    if not (path / "adapter_config.json").is_file():
      os.listdir(path.parent)  # an NFS client keeps a failed lookup until it lists the directory
      if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No LoRA snapshot at {path}")
    lora_id = self.lora_ids.setdefault(r["model"], len(self.lora_ids) + 1)
    return LoRARequest(r["model"], lora_id, str(path))


@asynccontextmanager
async def holding_gpu(time_slicer: TimeSlicerClient | None, workload: WorkloadRef | None, sampler: Sampler | None = None):
  """Hold the time-slicer turn for the body, waking the sampler inside it. Without a slicer the GPU is ours."""
  if time_slicer is None:
    yield
    return
  async with time_slicer.acquire(workload):
    if sampler is not None:
      await sampler.wake()
    try:
      yield
    finally:
      if sampler is not None:
        await sampler.sleep()


class TimeSlicedDispatcher(Dispatcher):
  """A pool's dispatcher when its sampler shares the GPU with the trainer. Each batch runs inside one time-slicer turn."""

  def __init__(
    self, store: SamplingStore, pool: str, sampler: Sampler, concurrency: int, time_slicer: TimeSlicerClient, workload: WorkloadRef
  ) -> None:
    super().__init__(store, pool, sampler, concurrency)
    self.time_slicer, self.workload = time_slicer, workload

  async def dispatch(self) -> None:
    await self.read()
    if not self.to_send():
      return
    async with holding_gpu(self.time_slicer, self.workload, self.sampler):
      await self.read()  # requests that arrived while the turn was pending join it
      if await self.ready():
        await asyncio.gather(*(self.send(r) for r in self.to_send()[: self.concurrency]))


def make_engine(sleep_mode: bool, is_lora: bool) -> Any:
  from vllm.config.weight_transfer import WeightTransferConfig
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine

  args = engine_args(is_lora, sleep_mode=sleep_mode)
  if transfer := args.get("weight_transfer_config"):
    args["weight_transfer_config"] = WeightTransferConfig(**transfer)
  return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=os.environ["BASE_MODEL"], **args))


async def run_sampling_worker(pool: str) -> None:
  """Boots vLLM, inside a turn when it shares its GPU, then serves the pool until the engine dies, a weight update
  breaks off, or the slicer cannot park this process. Each of those exits with an error, so the worker restarts."""
  time_slicer = time_slicer_client_from_env() if time_slicing_enabled() else None
  workload = workload_from_env(os.getpid(), name=local_workload_name("sampler", pool), claim=SAMPLER_CLAIM) if time_slicer else None
  store = SamplingStore(redis_client(os.environ["REDIS_URL"]))
  if time_slicer is not None:
    await time_slicer.register(workload)
  sampler = None
  try:
    async with holding_gpu(time_slicer, workload):
      is_lora = os.environ.get("OPEN_RL_FINE_TUNING_TYPE") == "lora"
      sampler = Sampler(make_engine(sleep_mode=time_slicer is not None, is_lora=is_lora), os.environ["BASE_MODEL"])
      if time_slicer is not None:
        await sampler.sleep()  # give the memory back before releasing the turn
    Path(READY_FILE).touch()
    concurrency = int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256"))
    if time_slicer is not None:
      dispatcher = TimeSlicedDispatcher(store, pool, sampler, concurrency, time_slicer, workload)
    else:
      dispatcher = Dispatcher(store, pool, sampler, concurrency)
    task = asyncio.create_task(dispatcher.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    while not task.done():
      sampler_failed = sampler.engine.errored or sampler.update_failed
      time_slicer_failed = time_slicer is not None and time_slicer.faulted
      if sampler_failed or time_slicer_failed:
        # Sends in flight save their failures first, so a request that killed the engine never reaches the next one.
        if dispatcher.in_flight:
          await asyncio.wait(list(dispatcher.in_flight.values()), timeout=10)
        task.cancel()
        raise RuntimeError(f"The sampler for pool {pool} cannot serve: {sampler_failed=} {time_slicer_failed=}")
      await asyncio.wait({task}, timeout=1)
  finally:
    # A faulted process still holds the GPU; the turn moves on once it exits and its connection closes.
    if time_slicer is not None and not time_slicer.faulted:
      await time_slicer.unregister(workload)
    if sampler is not None:
      sampler.engine.shutdown()
    await store.redis.aclose()


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL sampling pool replica with vLLM in this process")
  parser.add_argument("--pool", required=True, help="The sampling pool whose stream this replica serves")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  asyncio.run(run_sampling_worker(args.pool))


if __name__ == "__main__":
  main()
