import asyncio
import os
import threading
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from .store import get_store
from .trainer import Datum, LoraConfig, TrainerEngine

tracer = trace.get_tracer(__name__)

engine = TrainerEngine()


def _parse_datum(raw: dict) -> Datum:
  """Convert wire-format datum (with chunks) to our flat Datum type."""
  chunks = raw.get("model_input", {}).get("chunks", [])
  tokens: list[int] = []
  for chunk in chunks:
    tokens.extend(chunk.get("tokens", []))

  loss_inputs = raw.get("loss_fn_inputs", {})
  return Datum(model_input=tokens, loss_fn_inputs=loss_inputs)


async def clock_cycle_loop() -> None:
  store = get_store()

  print("[WORKER] Training worker started.")

  while True:
    try:
      batch = await store.get_requests()
      if not batch:
        await asyncio.sleep(0.1)
        continue

      m_id = batch[0].get("model_id", "default")

      with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
        batch_span.set_attribute("batch_size", len(batch))
        batch_span.set_attribute("model_id", m_id)

        print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")

        SKIP_ADAPTER_SWITCH = {"create_model", "create_model_from_state"}
        if not any(r.get("type") in SKIP_ADAPTER_SWITCH for r in batch):
          try:
            await asyncio.to_thread(engine.set_active_adapter, m_id)
          except Exception as e:
            print(f"Failed to set adapter {m_id}: {e}")
            for r in batch:
              await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
            continue

        for r in batch:
          req_id = r["req_id"]
          req_type = r["type"]

          carrier = r.get("trace_context", {})
          ctx = propagate.extract(carrier) if carrier else None
          token = otel_context.attach(ctx) if ctx else None

          try:
            match req_type:
              case "create_model":
                base_model = r["base_model"]
                raw_config = r.get("lora_config") or {}
                lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})

                await asyncio.to_thread(engine.load_base_model, base_model)
                await asyncio.to_thread(engine.create_adapter, m_id, lora_config)

                await store.set_future(
                  req_id,
                  {
                    "model_id": m_id,
                    "is_lora": True,
                    "lora_rank": lora_config.rank,
                    "type": "create_model",
                  },
                )

              case "forward_backward":
                raw_data = r["data"]
                loss_fn = r["loss_fn"]
                loss_config = r.get("loss_config")

                typed_data = [_parse_datum(item) for item in raw_data]

                result = await asyncio.to_thread(engine.forward_backward, typed_data, loss_fn, loss_config, m_id)
                result["type"] = "forward_backward"
                await store.set_future(req_id, result)

              case "optim_step":
                adam_params = r["adam_params"]
                result = await asyncio.to_thread(engine.optim_step, adam_params, m_id)
                result["type"] = "optim_step"
                await store.set_future(req_id, result)

              case "sample":
                prompt_tokens = r["prompt_tokens"]
                max_tokens = r["max_tokens"]
                num_samples = r["num_samples"]
                temperature = r.get("temperature", 0.0)

                result = await asyncio.to_thread(
                  engine.generate,
                  prompt_tokens,
                  max_tokens,
                  num_samples,
                  temperature,
                  m_id,
                )
                result["type"] = "sample"
                await store.set_future(req_id, result)

              case "save_state":
                state_path = r["state_path"]
                include_optimizer = bool(r.get("include_optimizer", False))
                kind = r.get("kind", "state")

                result = await asyncio.to_thread(engine.save_state, m_id, state_path, include_optimizer, kind)
                result["type"] = "save_state"
                await store.set_future(req_id, result)

              case "save_weights_for_sampler" | "save_weights":
                await asyncio.to_thread(engine.save_adapter, m_id)
                await store.set_future(req_id, {"status": "ok", "type": req_type})

              case _:
                print(f"Warning: Unhandled request type: {req_type}")
                await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": f"Unknown request type: {req_type}"})

          except Exception as e:
            traceback.print_exc()
            await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
          finally:
            if token:
              otel_context.detach(token)

    except asyncio.CancelledError:
      break
    except Exception as e:
      print(f"Error in clock cycle loop: {e}")
      traceback.print_exc()

      import redis

      if isinstance(e, redis.exceptions.ConnectionError):
        print("[worker] Destroying StateStore singleton to force Redis reconnection...")
        from . import store as store_mod

        store_mod._store_instance = None
        store = store_mod.get_store()

      await asyncio.sleep(1)


def main() -> None:
  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}\n")

  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target:
    engine.load_base_model(preload_target)
    is_ready = True
  else:
    print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  probe_app = FastAPI()

  @probe_app.get("/healthz")
  def healthz():
    if is_ready:
      return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model Loading")

  def run_probe_server():
    uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")

  threading.Thread(target=run_probe_server, daemon=True).start()
  asyncio.run(clock_cycle_loop())


if __name__ == "__main__":
  main()
