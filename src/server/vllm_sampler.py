# This file contains the vLLM worker implementation for high-throughput inference in Open-RL.

import argparse
import asyncio
import os
import sys
import traceback
from collections.abc import Sequence
from typing import Any

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from server.model_metadata import WeightSyncConfig
from server.vllm_options import gpu_memory_utilization, split_stop, text_only_engine_kwargs

try:
  from vllm import SamplingParams
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine
  from vllm.lora.request import LoRARequest
  from vllm.sampling_params import RequestOutputKind

  VLLM_AVAILABLE = True
except ImportError:
  SamplingParams = None
  AsyncEngineArgs = None
  AsyncLLMEngine = None
  LoRARequest = None
  RequestOutputKind = None
  VLLM_AVAILABLE = False

try:
  import server.delta_weight_transfer_engine  # noqa: F401
except ImportError:
  pass

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter for vLLM Worker")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")

tracer = trace.get_tracer("vllm.inference.worker")

engine: Any = None
CURRENT_LOADED_SAMPLER_WEIGHTS: str | None = None
IS_ENGINE_SLEEPING: bool = True
reload_lock = asyncio.Lock()


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


time_slicer: Any = None
LLMD_APP_MODE = False
if is_fft_enabled():
  from accel_timeslicer.llmd_app import is_llmd_app_mode, register_app_channel_workload
  from accel_timeslicer.time_slicer import time_slicer_client_from_env, workload_from_env
  from accel_timeslicer.workload import SAMPLER_CLAIM, local_workload_name

  LLMD_APP_MODE = is_llmd_app_mode()
  time_slicer = time_slicer_client_from_env()


def init_engine():
  global engine

  print("\n" + "=" * 50)
  print("        Open-RL vLLM Inference Engine (Queue Mode)")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")
  print(f"-> Hardware     : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Model        : {model_name or 'Not Set'}\n")

  mock_vllm = os.getenv("MOCK_VLLM", "0") == "1"
  if mock_vllm or not VLLM_AVAILABLE:
    print("[vLLM Worker] MOCK_VLLM=1 or vllm not installed, bypassing real engine init for local dev.")
  elif not model_name:
    print("[vLLM Worker] Error: BASE_MODEL environment variable is required.")
    sys.exit(1)
  else:
    hf_overrides: dict = {}
    arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
    if arch_override:
      hf_overrides["architectures"] = [arch_override]

    engine_kwargs = {
      "model": model_name,
      "enable_sleep_mode": is_fft_enabled(),
      "enable_lora": False,
      "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
      "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
      "gpu_memory_utilization": gpu_memory_utilization(),
      "enable_prefix_caching": False,
      "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    }
    if hf_overrides:
      engine_kwargs["hf_overrides"] = hf_overrides

    engine_kwargs.update(text_only_engine_kwargs())

    from server.model_metadata import WeightSyncConfig

    weight_sync_cfg = WeightSyncConfig.from_env()
    if weight_sync_cfg.strategy == "delta":
      try:
        from vllm.config.weight_transfer import WeightTransferConfig

        engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="delta_snapshot")
      except (ImportError, ValueError):
        pass

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("[vLLM Worker] Engine initialized successfully.")


async def run_generation_backend(
  request_id: str,
  prompt_token_ids: list[int],
  max_tokens: int,
  temperature: float,
  stop: str | Sequence[str] | Sequence[int] | None,
  top_p: float,
  top_k: int,
  num_samples: int,
  lora_id: str | None,
  lora_path: str | None,
  include_prompt_logprobs: bool,
) -> dict[str, Any]:
  try:
    current_engine = engine
    if current_engine is None:
      # Mocking for local Mac dev
      await asyncio.sleep(0.1)
      # return dummy tokens locally
      return {"sequences": [{"tokens": [0] * max_tokens, "logprobs": [-0.1] * max_tokens, "stop_reason": "length"}]}

    prompt_logprobs_val = 1 if include_prompt_logprobs else None
    stop_strings, stop_token_ids = split_stop(stop)
    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop_strings,
      stop_token_ids=stop_token_ids,
      top_p=top_p,
      top_k=top_k,
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=prompt_logprobs_val,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    results_generator = current_engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=None
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_id:
        span.set_attribute("vllm.lora_id", lora_id)
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
  except Exception as e:
    traceback.print_exc()
    return {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(e)}"}


async def process_sampling_request(req: dict, store: Any) -> None:
  global engine
  global CURRENT_LOADED_SAMPLER_WEIGHTS
  global IS_ENGINE_SLEEPING

  request_id = req["request_id"]
  trace_context = req.get("trace_context", {})

  parent_span = propagate.extract(trace_context)
  with tracer.start_as_current_span("process_sampling_request", context=parent_span):
    try:
      # 1. Manage weights reloading
      weights_path = req.get("weights_path")
      if is_fft_enabled() and weights_path:
        async with reload_lock:
          if weights_path != CURRENT_LOADED_SAMPLER_WEIGHTS:
            print(f"[vLLM Worker] Weight change detected. Current: {CURRENT_LOADED_SAMPLER_WEIGHTS}, Target: {weights_path}")
            if engine is not None:
              print("[vLLM Worker] Triggering sleep level 1 (CPU offload weights)...")
              await engine.sleep(level=1)
              print("[vLLM Worker] Waking up weights...")
              await engine.wake_up(tags=["weights"])
              if WeightSyncConfig.from_env().strategy == "delta":

                def _trigger_wt(worker, path=weights_path):
                  worker.start_weight_update()
                  try:
                    worker.update_weights({"target_weights_path": path})
                  finally:
                    worker.finish_weight_update()

                res = await engine.collective_rpc(_trigger_wt)
                print(f"[vLLM Worker] collective_rpc weight transfer result: {res}")
                print(f"[vLLM Worker] Incremental delta weights from {weights_path} synchronized via native WeightTransferEngine.")
              else:
                res = await engine.collective_rpc("reload_weights", kwargs={"weights_path": weights_path})
                print(f"[vLLM Worker] collective_rpc weight transfer result: {res}")
                print(f"[vLLM Worker] Full weights reloaded from {weights_path} in-place.")
              print("[vLLM Worker] Waking up KV cache...")
              await engine.wake_up(tags=["kv_cache"])
              IS_ENGINE_SLEEPING = False
            CURRENT_LOADED_SAMPLER_WEIGHTS = weights_path
            print("[vLLM Worker] Weights reload completed successfully!")

      # 2. Run inference
      prompt_token_ids = req.get("prompt_token_ids", [])
      max_tokens = req.get("max_tokens", 20)
      temperature = req.get("temperature", 1.0)
      stop = req.get("stop")
      top_p = req.get("top_p", 1.0)
      top_k = req.get("top_k", -1)
      num_samples = req.get("num_samples", 1)
      lora_id = req.get("lora_id")
      lora_path = req.get("lora_path")
      include_prompt_logprobs = req.get("include_prompt_logprobs", False)

      result = await run_generation_backend(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        top_p=top_p,
        top_k=top_k,
        num_samples=num_samples,
        lora_id=lora_id,
        lora_path=lora_path,
        include_prompt_logprobs=include_prompt_logprobs,
      )

      if result.get("type") != "RequestFailedResponse":
        result["type"] = "sample"

      await store.set_future(request_id, result)
    except Exception as exc:
      traceback.print_exc()
      await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(exc)}"})


async def run_sampling_worker(model_id: str) -> None:
  global engine
  global CURRENT_LOADED_SAMPLER_WEIGHTS
  global IS_ENGINE_SLEEPING
  from server.store import get_store

  store = get_store()
  snapshot_registered = False
  app_channel_handle: Any = None
  workload = None
  if time_slicer is not None:
    workload = workload_from_env(os.getpid(), name=local_workload_name("sampler", model_id), claim=SAMPLER_CLAIM)

  def register_app_channel() -> None:
    # llmd-app mode: hand the engine object to the node-local snapshot agent.
    # vLLM engines are recognized by type (requires enable_sleep_mode=True);
    # the agent pushes sleep(level)/wake_up(tags) — the worker no longer calls
    # them itself around lock boundaries.
    nonlocal app_channel_handle
    if not LLMD_APP_MODE or engine is None or app_channel_handle is not None:
      return
    assert workload is not None
    app_channel_handle = register_app_channel_workload(workload, engine=engine)
    print(f"[vLLM Worker] Registered app_channel workload {workload.key} with node-local snapshot agent.")

  if time_slicer is not None:
    assert workload is not None
    try:
      print(f"[vLLM Worker] Registering workload {workload.name} for initialization lock...")
      await time_slicer.register(workload)
      snapshot_registered = True
      async with time_slicer.acquire(workload):
        print("[vLLM Worker] Initializing vLLM engine under parent lock...")
        init_engine()
        print("[vLLM Worker] Engine initialized successfully.")
        if engine is not None:
          if LLMD_APP_MODE:
            # Register before yielding the lock so the agent can snapshot this
            # job as soon as another job needs the GPU. The engine stays
            # resident; the agent decides when to sleep it.
            register_app_channel()
            IS_ENGINE_SLEEPING = False
          else:
            print("[vLLM Worker] Sleeping engine after init to yield GPU memory (CPU offload)...")
            await engine.sleep(level=1)
            IS_ENGINE_SLEEPING = True
    except Exception as exc:
      print(f"[vLLM Worker] Failed to perform coordinated initialization: {exc}")
      traceback.print_exc()
      if engine is None:
        init_engine()
      register_app_channel()
  else:
    init_engine()

  async def exit_gracefully() -> None:
    print(f"[vLLM Worker] Initiating immediate exit for model {model_id} sampler worker...")
    nonlocal snapshot_registered
    nonlocal app_channel_handle
    if app_channel_handle is not None:
      try:
        app_channel_handle.close()
      except Exception as exc:
        print(f"[vLLM Worker] Failed to close app_channel workload handle: {exc}")
      app_channel_handle = None
    if snapshot_registered and time_slicer is not None:
      assert workload is not None
      try:
        await time_slicer.unregister(workload)
        snapshot_registered = False
      except Exception as exc:
        print(f"[vLLM Worker] Failed to unregister: {exc}")
    if time_slicer is not None:
      try:
        await time_slicer.close()
      except Exception:
        pass
    os._exit(0)

  if time_slicer is not None:
    import signal

    async def handle_shutdown():
      print(f"[vLLM Worker] Received termination signal, shutting down model {model_id} sampler worker...")
      await exit_gracefully()

    try:
      loop = asyncio.get_running_loop()
      for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(handle_shutdown()))
    except NotImplementedError:
      pass

  if hasattr(store, "redis"):
    await store.redis.set(f"open_rl:sampler_ready:{model_id}", "1")
    await store.redis.expire(f"open_rl:sampler_ready:{model_id}", 3600)

  print(f"[vLLM Worker] Listening for sampling requests on queue for model: {model_id}...")
  try:
    while True:
      try:
        batch = await store.get_sampling_requests_for_model(model_id)
        if not batch:
          await asyncio.sleep(0.05)
          continue

        has_shutdown = False
        sampling_reqs = []
        for req in batch:
          if req.get("request_id") == "SHUTDOWN_SENTINEL":
            has_shutdown = True
          else:
            sampling_reqs.append(req)

        if sampling_reqs:
          if time_slicer is not None:
            assert workload is not None
            async with time_slicer.acquire(workload):
              if LLMD_APP_MODE:
                # The snapshot agent pushes sleep(level)/wake_up(tags) over the
                # app_channel stream; the worker no longer calls them itself
                # around lock boundaries.
                IS_ENGINE_SLEEPING = False
              elif engine is not None and IS_ENGINE_SLEEPING:
                print("[vLLM Worker] Engine is sleeping. Waking up weights and KV cache before batch processing...")
                await engine.wake_up(tags=["weights", "kv_cache"])
                IS_ENGINE_SLEEPING = False
              tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in sampling_reqs]
              await asyncio.gather(*tasks)
              if has_shutdown:
                await exit_gracefully()
              if not LLMD_APP_MODE and engine is not None:
                print("[vLLM Worker] Exiting batch: sleeping engine (CPU offload weights) to yield GPU memory...")
                await engine.sleep(level=1)
                IS_ENGINE_SLEEPING = True
          else:
            if engine is not None and IS_ENGINE_SLEEPING:
              print("[vLLM Worker] Engine is sleeping. Waking up weights and KV cache before batch processing...")
              await engine.wake_up(tags=["weights", "kv_cache"])
              IS_ENGINE_SLEEPING = False
            tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in sampling_reqs]
            await asyncio.gather(*tasks)

        if has_shutdown:
          print("[vLLM Worker] Shutdown sentinel popped from queue. Initiating clean exit...")
          await exit_gracefully()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in sampling worker loop: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)
  finally:
    if time_slicer is not None:
      assert workload is not None
      try:
        if snapshot_registered:
          await time_slicer.unregister(workload)
      finally:
        await time_slicer.close()
        os._exit(0)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL vLLM Pull-Mode Sampler Worker")
  parser.add_argument("--model-id", type=str, required=True, help="The model ID of the RL job to process requests for")
  args = parser.parse_args()

  try:
    asyncio.run(run_sampling_worker(args.model_id))
  except KeyboardInterrupt:
    print("[vLLM Worker] Exiting via KeyboardInterrupt.")


if __name__ == "__main__":
  main()
