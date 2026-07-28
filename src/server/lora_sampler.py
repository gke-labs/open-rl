import argparse
import asyncio
import hashlib
import os
import traceback
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from server.store import get_store

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
engine: AsyncLLMEngine | None = None


def resolve_lora_path(lora_id: str, lora_path: str | None) -> str:
  """Resolves the actual PEFT directory containing adapter_config.json."""
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


async def process_sampling_request(req: dict[str, Any], store: Any) -> None:
  request_id = req["request_id"]

  prompt_token_ids = req.get("prompt_token_ids") or req.get("prompt_tokens") or []
  max_tokens = req.get("max_tokens", 100)
  temperature = req.get("temperature", 0.0)
  top_p = req.get("top_p", 1.0)
  top_k = req.get("top_k", -1)
  num_samples = req.get("num_samples", 1)
  stop = req.get("stop", [])
  lora_id = req.get("lora_id") or req.get("model_id")
  lora_path = req.get("lora_path") or req.get("weights_path")
  include_prompt_logprobs = bool(req.get("include_prompt_logprobs", False))

  try:
    if engine is None:
      await asyncio.sleep(0.1)
      dummy = {"sequences": [{"tokens": [0] * max_tokens, "logprobs": [-0.1] * max_tokens, "stop_reason": "length"}]}
      dummy["type"] = "sample"
      await store.set_future(request_id, dummy)
      return

    prompt_logprobs_val = 1 if include_prompt_logprobs else None
    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop_token_ids=stop,
      top_p=top_p,
      top_k=top_k,
      logprobs=1,
      prompt_logprobs=prompt_logprobs_val,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    lora_request = None
    if lora_id:
      actual_path = resolve_lora_path(lora_id, lora_path)
      if os.path.exists(os.path.join(actual_path, "adapter_config.json")):
        lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
        lora_request = LoRARequest(lora_id, lora_int_id, actual_path)
        print(f"[LoRA Sampler] Attached LoRARequest '{lora_id}' -> {actual_path}")

    results_generator = engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids},
      sampling_params=sampling_params,
      request_id=request_id,
      lora_request=lora_request,
    )

    sequences = []
    async for request_output in results_generator:
      for output in request_output.outputs:
        seq_dict = {
          "tokens": list(output.token_ids),
          "stop_reason": output.finish_reason or "length",
        }
        if output.logprobs:
          token_logprobs = []
          for idx, logprob_dict in zip(output.token_ids, output.logprobs):
            if logprob_dict and idx in logprob_dict:
              token_logprobs.append(float(logprob_dict[idx].logprob))
            elif logprob_dict:
              first_lp = next(iter(logprob_dict.values()))
              token_logprobs.append(float(first_lp.logprob))
            else:
              token_logprobs.append(-0.1)
          seq_dict["logprobs"] = token_logprobs
        sequences.append(seq_dict)

    out_data = {"sequences": sequences}
    if include_prompt_logprobs and request_output.prompt_logprobs:
      prompt_lps = []
      for lp_dict in request_output.prompt_logprobs:
        if lp_dict:
          first_lp = next(iter(lp_dict.values()))
          prompt_lps.append(float(first_lp.logprob))
        else:
          prompt_lps.append(0.0)
      out_data["prompt_logprobs"] = prompt_lps

    out_data["type"] = "sample"
    await store.set_future(request_id, out_data)
  except Exception as exc:
    print(f"[LoRA Sampler] Error processing request {request_id}: {exc}")
    traceback.print_exc()
    await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": f"LoRA Sampler Error: {str(exc)}"})


async def main():
  parser = argparse.ArgumentParser(description="Dedicated Open-RL LoRA Sampler Worker")
  parser.add_argument("--model-id", type=str, required=True, help="Model ID for this sampler worker")
  args = parser.parse_args()

  model_id = args.model_id
  store = get_store()

  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL") or os.getenv("OPEN_RL_BASE_MODEL") or "Qwen/Qwen2.5-0.5B"

  print("\n==================================================")
  print("        Open-RL Dedicated LoRA Sampler Worker      ")
  print("==================================================")
  print(f"-> Model ID     : {model_id}")
  print(f"-> Base Model   : {model_name}\n")

  global engine
  engine_kwargs = {
    "model": model_name,
    "enable_sleep_mode": False,
    "enable_lora": True,
    "max_loras": int(os.getenv("VLLM_MAX_LORAS", "8")),
    "max_lora_rank": int(os.getenv("VLLM_MAX_LORA_RANK", "64")),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
    "enable_prefix_caching": False,
    "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
  }

  arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
  if arch_override:
    hf_overrides = engine_kwargs.setdefault("hf_overrides", {})
    hf_overrides["architectures"] = [arch_override]

  engine_args = AsyncEngineArgs(**engine_kwargs)
  engine = AsyncLLMEngine.from_engine_args(engine_args)
  print("[LoRA Sampler] Engine initialized successfully.")

  if hasattr(store, "redis"):
    await store.redis.set(f"open_rl:sampler_ready:{model_id}", "1")
    await store.redis.expire(f"open_rl:sampler_ready:{model_id}", 3600)
    print(f"[LoRA Sampler] Registered ready signal for model {model_id} in Redis.")

  while True:
    try:
      sampling_reqs = await store.get_sampling_requests_for_model(model_id)
      if not sampling_reqs:
        await asyncio.sleep(0.05)
        continue

      tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in sampling_reqs]
      await asyncio.gather(*tasks)
    except asyncio.CancelledError:
      break
    except Exception as exc:
      print(f"[LoRA Sampler] Error in loop: {exc}")
      traceback.print_exc()
      await asyncio.sleep(1)


if __name__ == "__main__":
  asyncio.run(main())
