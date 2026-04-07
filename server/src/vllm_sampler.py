import os
import sys

import uvicorn
from fastapi import FastAPI, Request

try:
  from vllm import SamplingParams
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine
  from vllm.lora.request import LoRARequest
except ImportError:
  AsyncLLMEngine = None

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.environ.get("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter for vLLM Worker")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")

tracer = trace.get_tracer("vllm.inference.worker")
app = FastAPI(title="Open-RL vLLM Subprocess")
FastAPIInstrumentor.instrument_app(app, excluded_urls="/healthz")

engine = None


@app.on_event("startup")
async def startup():
  global engine

  print("\n" + "=" * 50)
  print("        Open-RL vLLM Inference Engine")
  print("=" * 50)
  cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
  model_name = os.environ.get("VLLM_MODEL", "Not Set")
  print(f"-> Hardware     : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Memory Matrix: {model_name}\n")

  mock_vllm = os.environ.get("MOCK_VLLM", "0") == "1"
  if mock_vllm or AsyncLLMEngine is None:
    print("[vLLM Subprocess] MOCK_VLLM=1 or vllm not installed, bypassing real engine init for local dev.")
    return

  # Use arguments suitable for the v2 architecture
  if not os.environ.get("VLLM_MODEL"):
    print("[vLLM Subprocess] Error: VLLM_MODEL environment variable is required.")
    sys.exit(1)

  engine_args = AsyncEngineArgs(
    model=os.environ.get("VLLM_MODEL"),
    enable_lora=True,
    max_loras=8,
    max_lora_rank=64,
    max_model_len=8192,  # Prevent KV cache OOM on massive context windows
    gpu_memory_utilization=0.60,  # Leave room for other things if needed
    enable_prefix_caching=False,  # Disable prefix caching to test concurrent throughput
    enforce_eager=True,  # Useful for small setups
  )
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  print("[vLLM Subprocess] Engine initialized and ready to serve IPC requests.")


@app.get("/healthz")
async def healthz():
  return {"status": "ok", "mock": engine is None}


@app.post("/generate")
async def generate(req: Request):
  try:
    data = await req.json()

    request_id = data.get("request_id")
    prompt_token_ids = data.get("prompt_token_ids")
    max_tokens = data.get("max_tokens", 20)
    temperature = data.get("temperature", 1.0)
    num_samples = data.get("num_samples", 1)

    lora_id = data.get("lora_id", None)
    lora_path = data.get("lora_path", None)

    if engine is None:
      # Mocking for local Mac dev
      import asyncio

      await asyncio.sleep(0.1)
      # return dummy tokens locally
      return {"sequences": [{"tokens": [0] * max_tokens, "logprobs": [-0.1] * max_tokens, "stop_reason": "length"}]}

    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      logprobs=1,  # return logprobs for TITO RL
    )

    lora_request = None
    if lora_id and lora_path:
      import hashlib

      # vLLM natively relies on lora_int_id to track cached adapter weights.
      # Convert the sequence identifier UUID to a stable 32-bit positive integer hash.
      lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
      lora_request = LoRARequest(lora_id, lora_int_id, lora_path)

    results_generator = engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_id:
        span.set_attribute("vllm.lora_id", lora_id)
      async for request_output in results_generator:
        # vLLM streams back incremental states, we wait for the final one
        final_output = request_output

    sequences_out = []
    for output in final_output.outputs:
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

    return {"sequences": sequences_out}
  except Exception as e:
    import traceback

    traceback.print_exc()
    # Return explicit 500 so upstream client logs it
    return {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(e)}"}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8001)
