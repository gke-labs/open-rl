"""Queue-driven vLLM sampler. One engine serves a model's queue; a request names
either a LoRA adapter (lora_id) or a whole-model weights version (weights_path)."""

import argparse
import asyncio
import hashlib
import os
import signal
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from itertools import groupby
from typing import Any

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from vllm import SamplingParams
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, time_slicing_enabled, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, WorkloadRef, local_workload_name
from server.lora_snapshots import resolve_lora_path
from server.sampler_http import serve_http
from server.store import RequestStore, StateStore, get_state_store, get_store
from server.vllm_options import gpu_memory_utilization, sampler_batch_limits, split_stop, text_only_engine_kwargs

tracer = trace.get_tracer("vllm.inference.worker")
SHUTDOWN_SENTINEL = "SHUTDOWN_SENTINEL"
READY_TTL_SECONDS = 3600
ENGINE_POLL_SECONDS = 5
STATS_LOG_SECONDS = 30


def failed_response(message: str) -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message}


def engine_kwargs_from_env(fft_enabled: bool) -> dict[str, Any]:
  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")
  if not model_name:
    raise ValueError("BASE_MODEL or VLLM_MODEL is required")
  engine_kwargs = {
    "model": model_name,
    "enable_sleep_mode": fft_enabled,
    "enable_lora": not fft_enabled,
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    **sampler_batch_limits(),
    "gpu_memory_utilization": gpu_memory_utilization(),
    # FFT weights change in place under one name, so cached prefixes would go
    # stale. Each LoRA version samples under its own adapter name.
    "enable_prefix_caching": not fft_enabled,
    "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    **text_only_engine_kwargs(),
  }
  architecture = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
  if architecture:
    engine_kwargs["hf_overrides"] = {"architectures": [architecture]}
  if fft_enabled:
    engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="delta_snapshot")
  else:
    engine_kwargs["max_loras"] = int(os.getenv("VLLM_MAX_LORAS", "8"))
    engine_kwargs["max_lora_rank"] = int(os.getenv("VLLM_MAX_LORA_RANK", "64"))
  return engine_kwargs


def lora_request_for(request: dict[str, Any]) -> LoRARequest | None:
  """A LoRA request when the adapter exists on disk. Before the first save a
  LoRA job samples from the base model, so a missing adapter is not an error."""
  lora_id = request.get("lora_id")
  if not lora_id:
    return None
  path = resolve_lora_path(lora_id, request.get("lora_path"))
  if path is None:
    return None
  lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
  return LoRARequest(lora_id, lora_int_id, path)


class Sampler:
  """A vLLM engine that serves sampling requests and swaps its weights between them.

  weights_path is the whole-model version the engine holds; LoRA requests leave it
  alone and attach their adapter per request. A failed update leaves partial
  weights behind, so the sampler refuses work until the process is restarted.
  """

  def __init__(self, engine: AsyncLLMEngine) -> None:
    self.engine = engine
    self.weights_path: str | None = None
    self.update_failed = False

  async def wake(self) -> None:
    await self.engine.wake_up(tags=["weights", "kv_cache"])

  async def sleep(self) -> None:
    await self.engine.sleep(level=1)

  async def ensure_weights(self, weights_path: str | None) -> None:
    """Make the engine hold weights_path; None keeps whatever it holds."""
    if self.update_failed:
      raise RuntimeError("Weight update failed; restart the sampler before serving")
    if weights_path is None or weights_path == self.weights_path:
      return
    # No finally/resume on failure: partially updated weights must not be served.
    try:
      await self.engine.pause_generation(mode="wait", clear_cache=True)
      await self.engine.start_weight_update()
      await self.engine.update_weights(WeightTransferUpdateRequest(update_info={"target_weights_path": weights_path}))
      await self.engine.finish_weight_update(weight_version=weights_path)
      await self.engine.reset_encoder_cache()
      await self.engine.resume_generation()
    except BaseException:
      self.update_failed = True
      raise
    self.weights_path = weights_path

  async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request["request_id"]
    prompt_token_ids = request.get("prompt_token_ids", [])
    max_tokens = request.get("max_tokens", 20)
    lora_request = lora_request_for(request)
    stop_strings, stop_token_ids = split_stop(request.get("stop"))
    sampling_params = SamplingParams(
      n=request.get("num_samples", 1),
      temperature=request.get("temperature", 1.0),
      max_tokens=max_tokens,
      stop=stop_strings,
      stop_token_ids=stop_token_ids,
      top_p=request.get("top_p", 1.0),
      top_k=request.get("top_k", -1),
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=1 if request.get("include_prompt_logprobs", False) else None,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    results_generator = self.engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_request is not None:
        span.set_attribute("vllm.lora_id", lora_request.lora_name)
      async for request_output in results_generator:
        final_output = request_output

    outputs = final_output.outputs if final_output else []
    sequences_out = []
    for output in outputs:
      generated_token_ids = list(output.token_ids)
      logprobs = []
      if output.logprobs:
        for idx, token_logprobs in enumerate(output.logprobs):
          # token_logprobs is a dict of {token_id: Logprob}
          token_id = generated_token_ids[idx]
          if token_logprobs and token_id in token_logprobs:
            logprob = token_logprobs[token_id].logprob
          else:
            logprob = -9999.0
          logprobs.append(logprob)
      sequences_out.append({"tokens": generated_token_ids, "logprobs": logprobs, "stop_reason": output.finish_reason})

    prompt_logprobs_out = None
    if final_output and final_output.prompt_logprobs:
      prompt_logprobs_out = []
      for idx, token_logprobs in enumerate(final_output.prompt_logprobs):
        if token_logprobs is None:
          prompt_logprobs_out.append(None)
        else:
          token_id = prompt_token_ids[idx]
          if token_id in token_logprobs:
            prompt_logprobs_out.append(token_logprobs[token_id].logprob)
          else:
            prompt_logprobs_out.append(None)

    res = {"sequences": sequences_out}
    if prompt_logprobs_out is not None:
      res["prompt_logprobs"] = prompt_logprobs_out
    return res


@asynccontextmanager
async def holding_gpu(time_slicer: TimeSlicerClient | None, workload: WorkloadRef | None, sampler: Sampler | None = None):
  """Hold the time-slicer slot for the body, waking the sampler inside it. Without a slicer the GPU is ours."""
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


async def process_batch(sampler: Sampler, store: RequestStore, requests: list[dict[str, Any]]) -> None:
  """Serve consecutive requests that share a weights_path together, updating weights between groups."""
  for weights_path, group in groupby(requests, key=lambda req: req.get("weights_path")):
    batch = list(group)
    try:
      await sampler.ensure_weights(weights_path)
    except Exception as exc:
      for request in batch:
        await store.set_future(request["request_id"], failed_response(f"vLLM weight update failed: {exc}"))
      continue
    await serve_group(sampler, store, batch)


async def serve_group(sampler: Sampler, store: RequestStore, batch: list[dict[str, Any]]) -> None:
  """A dead EngineCore leaves generate() hanging, so the wait polls for it and fails what is left."""
  tasks = {asyncio.create_task(process_request(sampler, store, req)): req for req in batch}
  pending = set(tasks)
  while pending:
    _, pending = await asyncio.wait(pending, timeout=ENGINE_POLL_SECONDS)
    if pending and sampler.engine.errored:
      for task in pending:
        task.cancel()
      await fail_requests(store, [tasks[task] for task in pending], "vLLM engine died during the batch")
      break
  if sampler.engine.errored:
    raise RuntimeError("vLLM engine is dead")
  for task in tasks:
    task.result()


async def fail_requests(store: RequestStore, requests: list[dict[str, Any]], error: Any) -> None:
  """Best effort, so clients see the failure instead of waiting forever."""
  for request in requests:
    try:
      await store.set_future(request["request_id"], failed_response(f"vLLM Worker Error: {error}"))
    except Exception as exc:
      print(f"[vLLM Worker] Could not fail request {request['request_id']}: {exc}")


async def process_request(sampler: Sampler, store: RequestStore, request: dict[str, Any]) -> None:
  with tracer.start_as_current_span("process_sampling_request", context=propagate.extract(request.get("trace_context", {}))):
    try:
      result = await sampler.generate(request)
      result["type"] = "sample"
    except Exception as exc:
      result = failed_response(f"vLLM Worker Error: {exc}")
    await store.set_future(request["request_id"], result)


async def mark_ready(state: StateStore, model_id: str) -> None:
  """The gateway waits on this key. A shared LoRA sampler outlives the TTL, so the loop refreshes it."""
  await state.set_value(f"open_rl:sampler_ready:{model_id}", "1", ttl_seconds=READY_TTL_SECONDS)


async def serve(
  model_id: str,
  store: RequestStore,
  make_engine: Callable[[], AsyncLLMEngine],
  time_slicer: TimeSlicerClient | None = None,
  workload: WorkloadRef | None = None,
) -> None:
  """Build the engine, then serve the model's queue until the shutdown sentinel arrives.

  Queue and store errors are retried. A dead engine or a process the time slicer
  could not park raises, so the worker exits and is restarted.
  """
  async with holding_gpu(time_slicer, workload):
    sampler = Sampler(make_engine())
    if time_slicer is not None:
      await sampler.sleep()  # give the memory back before releasing the slot
  state = get_state_store()
  stats = asyncio.create_task(log_engine_stats(sampler.engine))
  # A routed sampler also takes requests from its set's llm-d router.
  http = asyncio.create_task(serve_http(sampler, int(port))) if (port := os.getenv("OPEN_RL_SAMPLER_HTTP_PORT")) else None
  try:
    try:
      ready_at = float("-inf")
      while True:
        if time_slicer is not None and time_slicer.faulted:
          raise RuntimeError(f"Time slicer could not park this process: {time_slicer.faulted}")
        requests = []
        started = False
        try:
          if time.monotonic() - ready_at > 60:
            await mark_ready(state, model_id)
            ready_at = time.monotonic()
          batch = await store.get_sampling_requests_for_model(model_id)
          if not batch:
            await asyncio.sleep(0.05)
            continue
          shutdown = any(req.get("request_id") == SHUTDOWN_SENTINEL for req in batch)
          requests = [req for req in batch if req.get("request_id") != SHUTDOWN_SENTINEL]
          if requests:
            async with holding_gpu(time_slicer, workload, sampler):
              started = True
              await process_batch(sampler, store, requests)
          if shutdown:
            return
        except Exception as exc:
          if sampler.engine.errored:
            raise
          print(f"[vLLM Worker] Sampling loop error, retrying: {exc}")
          if not started:
            await fail_requests(store, requests, exc)
          await asyncio.sleep(1)
    finally:
      await state.delete_values(f"open_rl:sampler_ready:{model_id}")
  finally:
    stats.cancel()
    if http is not None:
      http.cancel()
    sampler.engine.shutdown()


async def log_engine_stats(engine: AsyncLLMEngine) -> None:
  """vLLM only logs throughput and prefix cache hit rate when asked, which its
  own server does every 10 seconds."""
  while True:
    await asyncio.sleep(STATS_LOG_SECONDS)
    try:
      await engine.do_log_stats()
    except Exception as exc:
      print(f"[vLLM Worker] Could not log engine stats: {exc}")


async def serve_time_sliced(model_id: str, store: RequestStore, make_engine: Callable[[], AsyncLLMEngine]) -> None:
  """An FFT sampler shares its GPU with the trainer, so it is registered with the time slicer while it serves."""
  time_slicer = time_slicer_client_from_env()
  workload = workload_from_env(os.getpid(), name=local_workload_name("sampler", model_id), claim=SAMPLER_CLAIM)
  try:
    await time_slicer.register(workload)
    try:
      await serve(model_id, store, make_engine, time_slicer, workload)
    finally:
      # A faulted process still holds the GPU; the grant moves on once it exits.
      if not time_slicer.faulted:
        await time_slicer.unregister(workload)
  finally:
    await time_slicer.close()


async def run_sampling_worker(model_id: str) -> None:
  fft_enabled = os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"
  engine_kwargs = engine_kwargs_from_env(fft_enabled)

  def make_engine() -> AsyncLLMEngine:
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

  loop = asyncio.get_running_loop()
  task = asyncio.current_task()
  assert task is not None
  installed_signals = []
  try:
    for sig in (signal.SIGTERM, signal.SIGINT):
      try:
        loop.add_signal_handler(sig, task.cancel)
        installed_signals.append(sig)
      except NotImplementedError:
        break
    # An exclusive sampler is alone on its GPU, so it serves awake like a LoRA one.
    if fft_enabled and time_slicing_enabled():
      await serve_time_sliced(model_id, get_store(), make_engine)
    else:
      await serve(model_id, get_store(), make_engine)
  except asyncio.CancelledError:
    pass  # serve has completed cleanup before cancellation reaches here.
  finally:
    for sig in installed_signals:
      loop.remove_signal_handler(sig)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL vLLM Pull-Mode Sampler Worker")
  parser.add_argument("--model-id", type=str, required=True, help="The model ID of the RL job to process requests for")
  args = parser.parse_args()
  provider = TracerProvider()
  trace.set_tracer_provider(provider)
  try:
    if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
      from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

      provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    asyncio.run(run_sampling_worker(args.model_id))
  finally:
    provider.shutdown()


if __name__ == "__main__":
  main()
