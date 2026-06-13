# This file contains the vLLM worker implementation for high-throughput inference in Open-RL.

import argparse
import asyncio
import hashlib
import os
import sys
import traceback
from typing import Any

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
reload_lock = asyncio.Lock()


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


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
      "enable_lora": not is_fft_enabled(),
      "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
      "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
      "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
      "enable_prefix_caching": False,
      "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    }
    if not is_fft_enabled():
      engine_kwargs["max_loras"] = 8
      engine_kwargs["max_lora_rank"] = 64
    if hf_overrides:
      engine_kwargs["hf_overrides"] = hf_overrides

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("[vLLM Worker] Engine initialized successfully.")


async def run_generation_backend(
  request_id: str,
  prompt_token_ids: list[int],
  max_tokens: int,
  temperature: float,
  stop: list[int] | None,
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
    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop_token_ids=stop,
      top_p=top_p,
      top_k=top_k,
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=prompt_logprobs_val,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    lora_request = None
    if lora_id and lora_path:
      # vLLM natively relies on lora_int_id to track cached adapter weights.
      # Convert the sequence identifier UUID to a stable 32-bit positive integer hash.
      lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
      lora_request = LoRARequest(lora_id, lora_int_id, lora_path)

    results_generator = current_engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
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

  request_id = req["request_id"]
  trace_context = req.get("trace_context", {})

  # Propagate tracer span context if available
  parent_span = propagate.extract(trace_context)
  with tracer.start_as_current_span("process_sampling_request", context=parent_span) as span:
    try:
      # 1. Manage weights reloading
      weights_path = req.get("weights_path")
      if is_fft_enabled() and weights_path:
        async with reload_lock:
          if weights_path != CURRENT_LOADED_SAMPLER_WEIGHTS:
            print(f"[vLLM Worker] Weight change detected. Current: {CURRENT_LOADED_SAMPLER_WEIGHTS}, Target: {weights_path}")
            if engine is not None:
              print("[vLLM Worker] Triggering sleep level 2...")
              await engine.sleep(level=2)
              print("[vLLM Worker] Waking up weights...")
              await engine.wake_up(tags=["weights"])
              print(f"[vLLM Worker] Reloading weights from {weights_path} in-place...")
              await engine.collective_rpc("reload_weights", kwargs={"weights_path": weights_path})
              print("[vLLM Worker] Waking up KV cache...")
              await engine.wake_up(tags=["kv_cache"])
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
  from server.store import get_store

  init_engine()
  store = get_store()

  # Set a Redis key to indicate the sampler is ready to start receiving requests
  if hasattr(store, "redis"):
    await store.redis.set(f"open_rl:sampler_ready:{model_id}", "1")
    await store.redis.expire(f"open_rl:sampler_ready:{model_id}", 3600)
  
  print(f"[vLLM Worker] Listening for sampling requests on queue for model: {model_id}...")
  while True:
    try:
      batch = await store.get_sampling_requests_for_model(model_id)
      if not batch:
        await asyncio.sleep(0.05)
        continue

      tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in batch]
      await asyncio.gather(*tasks)
    except asyncio.CancelledError:
      break
    except Exception as exc:
      print(f"Error in sampling worker loop: {exc}")
      traceback.print_exc()
      await asyncio.sleep(1)


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
