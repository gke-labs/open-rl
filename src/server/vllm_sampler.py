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

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, WorkloadRef, local_workload_name
from server.store import RequestStore, get_state_store, get_store
from server.vllm_options import gpu_memory_utilization, split_stop, text_only_engine_kwargs

tracer = trace.get_tracer("vllm.inference.worker")
SHUTDOWN_SENTINEL = "SHUTDOWN_SENTINEL"
TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
READY_TTL_SECONDS = 3600


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
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
    "gpu_memory_utilization": gpu_memory_utilization(),
    "enable_prefix_caching": False,
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


def resolve_lora_path(lora_id: str, lora_path: str | None) -> str:
  """Find the PEFT directory holding adapter_config.json for this adapter."""
  if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
    return lora_path

  if lora_path:
    # Check subfolders created by PEFT save_pretrained
    base_candidates = [
      lora_id,
      lora_id.rsplit("/", 1)[-1],
      lora_id.split("://")[-1].split("/")[0] if "://" in lora_id else lora_id,
    ]
    for candidate in base_candidates:
      candidate_path = os.path.join(lora_path, candidate)
      if os.path.exists(os.path.join(candidate_path, "adapter_config.json")):
        return candidate_path

  # Check auto-saved PEFT directory: TMP_DIR/peft/<model_id>/<model_id>
  base_id = lora_id.split("://")[-1].split("/")[0] if "://" in lora_id else lora_id
  peft_dir = os.path.join(TMP_DIR, "peft", base_id, base_id)
  if os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
    return peft_dir

  return lora_path or peft_dir


def lora_request_for(request: dict[str, Any]) -> LoRARequest | None:
  """A LoRA request when the adapter exists on disk. Before the first save a
  LoRA job samples from the base model, so a missing adapter is not an error."""
  lora_id = request.get("lora_id")
  if not lora_id:
    return None
  path = resolve_lora_path(lora_id, request.get("lora_path"))
  if not os.path.exists(os.path.join(path, "adapter_config.json")):
    return None
  lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
  return LoRARequest(lora_id, lora_int_id, path)


class Sampler:
  """Consume one model's sampling queue with a single vLLM engine.

  An FFT sampler is one per job: requests carry a weights_path, consecutive
  groups sharing one are served together, and the engine is updated between
  groups, never while a request is generating. A failed update poisons the
  sampler until the process is restarted. A LoRA sampler is one per base
  model and shared by its jobs: each request names its adapter with lora_id.
  """

  def __init__(
    self,
    model_id: str,
    store: RequestStore,
    make_engine: Callable[[], AsyncLLMEngine],
    *,
    time_slicer: TimeSlicerClient | None = None,
    workload: WorkloadRef | None = None,
  ) -> None:
    if (time_slicer is None) != (workload is None):
      raise ValueError("time_slicer and workload must be provided together")
    self.model_id = model_id
    self.store = store
    self.make_engine = make_engine
    self.time_slicer = time_slicer
    self.workload = workload
    self.engine: AsyncLLMEngine | None = None
    self.weights_path: str | None = None
    self.update_failed = False
    self._registered = False
    self._ready_at: float | None = None
    self._closed = False

  @asynccontextmanager
  async def gpu(self):
    """Hold the GPU for the body. Under a time slicer that means acquiring the
    slot, waking the engine if it exists, and sleeping it again on the way out."""
    if self.time_slicer is None:
      yield
      return
    async with self.time_slicer.acquire(self.workload):
      if self.engine is not None:
        await self.engine.wake_up(tags=["weights", "kv_cache"])
      try:
        yield
      finally:
        if self.engine is not None:
          await self.engine.sleep(level=1)

  async def start(self) -> None:
    if self.time_slicer is not None:
      await self.time_slicer.register(self.workload)
      self._registered = True
    async with self.gpu():
      self.engine = self.make_engine()
    await self.mark_ready()

  async def mark_ready(self) -> None:
    """The gateway waits on this key. A shared LoRA sampler outlives the TTL, so the loop refreshes it."""
    await get_state_store().set_value(f"open_rl:sampler_ready:{self.model_id}", "1", ttl_seconds=READY_TTL_SECONDS)
    self._ready_at = time.monotonic()

  async def run(self) -> None:
    try:
      await self.start()
      while True:
        if self._ready_at is not None and time.monotonic() - self._ready_at > 60:
          await self.mark_ready()
        batch = await self.store.get_sampling_requests_for_model(self.model_id)
        if not batch:
          await asyncio.sleep(0.05)
          continue
        shutdown = any(req.get("request_id") == SHUTDOWN_SENTINEL for req in batch)
        requests = [req for req in batch if req.get("request_id") != SHUTDOWN_SENTINEL]
        if requests:
          async with self.gpu():
            await self.process_batch(requests)
        if shutdown:
          return
    finally:
      await self.close()

  async def process_batch(self, requests: list[dict[str, Any]]) -> None:
    for weights_path, group in groupby(requests, key=lambda req: req.get("weights_path")):
      batch = list(group)
      try:
        if self.update_failed:
          raise RuntimeError("Weight update failed; restart the sampler before serving")
        if weights_path and weights_path != self.weights_path:
          await self.update_weights(weights_path)
      except Exception as exc:
        for request in batch:
          await self.store.set_future(request["request_id"], failed_response(f"vLLM weight update failed: {exc}"))
        continue
      await asyncio.gather(*(self.process_request(req) for req in batch))

  async def process_request(self, request: dict[str, Any]) -> None:
    with tracer.start_as_current_span("process_sampling_request", context=propagate.extract(request.get("trace_context", {}))):
      try:
        result = await self.generate(request)
        result["type"] = "sample"
      except Exception as exc:
        result = failed_response(f"vLLM Worker Error: {exc}")
      await self.store.set_future(request["request_id"], result)

  async def update_weights(self, weights_path: str) -> None:
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

  async def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self.engine is not None:
        self.engine.shutdown()
    finally:
      try:
        if self._ready_at is not None:
          self._ready_at = None
          await get_state_store().delete_values(f"open_rl:sampler_ready:{self.model_id}")
      finally:
        if self.time_slicer is not None:
          try:
            if self._registered:
              self._registered = False
              await self.time_slicer.unregister(self.workload)
          finally:
            await self.time_slicer.close()


async def run_sampling_worker(model_id: str) -> None:
  fft_enabled = os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"
  engine_kwargs = engine_kwargs_from_env(fft_enabled)
  sampler = Sampler(
    model_id,
    get_store(),
    lambda: AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs)),
    time_slicer=time_slicer_client_from_env() if fft_enabled else None,
    workload=workload_from_env(os.getpid(), name=local_workload_name("sampler", model_id), claim=SAMPLER_CLAIM) if fft_enabled else None,
  )
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
    await sampler.run()
  except asyncio.CancelledError:
    pass  # Sampler.run has completed cleanup before cancellation reaches here.
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
