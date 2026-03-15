import uuid
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# Removed direct PyTorch engine import to keep Gateway stateless
from .state import get_store

store = get_store()
import logging
import urllib.request
import json
import time
from datetime import datetime
import traceback

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize OpenTelemetry TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.environ.get("ENABLE_GCP_TRACE", "0") == "1":
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        exporter = CloudTraceSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
    except ImportError:
        print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
else:
    print("OpenTelemetry: No exporter configured (ENABLE_GCP_TRACE=0)")

async def enqueue_traced_request(store, payload: dict) -> None:
    carrier = {}
    from opentelemetry import propagate
    propagate.inject(carrier)
    payload["trace_context"] = carrier
    await store.put_request(payload)

class FilterNoisyEndpoints(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "retrieve_future" not in msg and "session_heartbeat" not in msg

logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

def get_default_model_name() -> str | None:
    if os.getenv("OPEN_RL_SINGLE_PROCESS", "0") == "1":
        from . import engine as trainer_engine

        if trainer_engine.engine.base_model_name:
            return trainer_engine.engine.base_model_name
    return os.getenv("OPEN_RL_BASE_MODEL") or os.getenv("VLLM_MODEL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if os.getenv("OPEN_RL_SINGLE_PROCESS", "0") == "1":
        from . import engine as trainer_engine

        base_model = os.getenv("OPEN_RL_BASE_MODEL")
        print("\n" + "=" * 50)
        print(" Open-RL Single-Process Mode")
        print("=" * 50)
        print(f"-> Base model: {base_model or 'unset'}")
        print("-> Backend   : gateway + engine loop in one process\n")
        if base_model:
            await asyncio.to_thread(trainer_engine.engine.preload_base_model, base_model)
        task = asyncio.create_task(trainer_engine.clock_cycle_loop())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(title="Open-RL Server MVP", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")

@app.get("/api/v1/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
    model_name = get_default_model_name()
    return {
        "supported_models": [model_name] if model_name else [],
        "default_model": model_name,
        "single_process": os.getenv("OPEN_RL_SINGLE_PROCESS", "0") == "1",
    }

@app.post("/api/v1/create_session")
async def create_session(req: dict):
    return {"session_id": "sess-real-123", "type": "create_session"}

@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: dict):
    return {"type": "session_heartbeat"}

@app.post("/api/v1/create_model")
async def create_model(req: dict):
    req_id = str(uuid.uuid4())
    await store.set_future(req_id, {"status": "pending"})
    
    base_model = req.get("base_model")
    if not base_model:
        return JSONResponse(status_code=400, content={"error": "base_model is required"})
        
    lora_config = req.get("lora_config", {})
    rank = lora_config.get("rank", 16)
    
    model_id = req_id 
    
    # Push the creation task globally to the Redis Queue for the decoupled Trainer Engine
    # Note: We must route this specifically so the Engine instance can pick it up.
    # Since it's a global operation (not tenant specific yet), we use "default" or model_id
    await store.put_request({
        "req_id": req_id,
        "model_id": model_id, # Tenant specific queue
        "type": "create_model",
        "base_model": base_model,
        "rank": rank
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/get_info")
async def get_info(req: dict):
    model_name = get_default_model_name()
    if not model_name:
         return JSONResponse(status_code=404, content={"error": "No base model is configured for the current server session"})

    return {
        "model_data": {
            "arch": "unknown",
            "model_name": model_name,
            "tokenizer_id": model_name
        },
        "model_id": req.get("model_id", "model-live-123"), # Will be whatever client passed
        "is_lora": True,
        "lora_rank": 16,
        "model_name": model_name,
        "type": "get_info"
    }

# Global set to hold strong references to background tasks to prevent GC
background_tasks = set()

@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
    req_id = str(uuid.uuid4())
    await store.set_future(req_id, {"status": "pending"})

    fwd_input = req.get("forward_backward_input", {})
    data = fwd_input.get("data", [])
    loss_fn = fwd_input.get("loss_fn", "cross_entropy")
    loss_config = fwd_input.get("loss_fn_config", {})
    model_id = req.get("model_id")

    await enqueue_traced_request(store, {
        "req_id": req_id,
        "model_id": model_id,
        "type": "forward_backward",
        "data": data,
        "loss_fn": loss_fn,
        "loss_config": loss_config,
    })

    return {"request_id": req_id}

@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
    req_id = str(uuid.uuid4())
    await store.set_future(req_id, {"status": "pending"})

    adam_params = req.get("adam_params", {})
    model_id = req.get("model_id")

    await enqueue_traced_request(store, {
        "req_id": req_id,
        "model_id": model_id,
        "type": "optim_step",
        "adam_params": adam_params,
    })

    return {"request_id": req_id}

@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
    req_id = str(uuid.uuid4())
    model_id = req.get("model_id") # Client passes the TrainingClient's model_id
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})
    seq_id = req.get("sampling_session_seq_id", 0)
    if not seq_id:
        seq_id = int(time.time() * 1000)
    # Tinker SDK might send 'name' or 'alias', or 'path' (if using save_weights_for_sampler)
    alias = req.get("name") or req.get("alias") or req.get("path")
    
    # 1. Update the metadata.json instantly
    tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    ram_path = os.path.join(tmp_dir, "peft", model_id)
    os.makedirs(ram_path, exist_ok=True)
    
    metadata = {
        "model_id": model_id,
        "alias": alias,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time()
    }
    try:
        with open(os.path.join(ram_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Failed to update alias metadata: {e}")
        
    session_id = f"{model_id}-samp-{seq_id}"
    result_path = f"tinker://{session_id}" if alias else None
    
    result = {
        "path": result_path,
        "sampling_session_id": session_id,
        "type": "save_weights_for_sampler"
    }
    
    # Instantly resolve the future, bypassing the Redis queue!
    await store.set_future(req_id, result)
    return {"request_id": req_id}

@app.post("/api/v1/save_weights")
async def save_weights(req: dict):
    req_id = str(uuid.uuid4())
    model_id = req.get("model_id") 
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})
    seq_id = req.get("seq_id", 0)
    if not seq_id:
        seq_id = int(time.time() * 1000)
    # in .save(path="..."), the path is the identifier
    alias = req.get("path")
    
    # 1. Update the metadata.json instantly
    tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    ram_path = os.path.join(tmp_dir, "peft", model_id)
    os.makedirs(ram_path, exist_ok=True)
    
    metadata = {
        "model_id": model_id,
        "alias": alias,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time()
    }
    try:
        with open(os.path.join(ram_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Failed to update alias metadata: {e}")
        
    session_id = f"{model_id}-samp-{seq_id}"
    result_path = f"tinker://{session_id}"
    
    result = {
        "path": result_path,
        "sampling_session_id": session_id,
        "type": "save_weights"
    }
    
    # Instantly resolve the future, bypassing the Redis queue!
    await store.set_future(req_id, result)
    return {"request_id": req_id}

@app.get("/api/v1/list_adapters")
async def list_adapters():
    """
    Scans the local temporary directory (used for RAM-disk sync) for available PEFT adapters.
    Returns a list of adapters with metadata (creation time, alias) if available.
    """
    tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    peft_dir = os.path.join(tmp_dir, "peft")
    
    adapters = []
    
    if os.path.exists(peft_dir):
        # Scan all directories in peft_dir
        with os.scandir(peft_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    model_id = entry.name
                    metadata_path = os.path.join(entry.path, "metadata.json")
                    
                    adapter_info = {
                        "model_id": model_id,
                        "created_at": entry.stat().st_ctime, # Fallback to filesystem time
                        "timestamp": entry.stat().st_ctime,
                        "alias": None
                    }
                    
                    # Try to read metadata.json
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                meta = json.load(f)
                                adapter_info.update(meta)
                        except Exception:
                            pass
                            
                    adapters.append(adapter_info)
    
    # Sort by creation time descending (newest first)
    # Prefer 'timestamp' (float) if available, otherwise 'created_at' if it's a number
    def get_sort_key(x):
        ts = x.get("timestamp")
        if isinstance(ts, (int, float)):
            return ts
        ca = x.get("created_at")
        if isinstance(ca, (int, float)):
            return ca
        return 0
        
    adapters.sort(key=get_sort_key, reverse=True)
    
    return {"adapters": adapters}

@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
    # Support 'model_path' from SDK (e.g. "tinker://uuid-samp-seq")
    model_path = req.get("model_path")
    model_id = req.get("model_id")
    
    if model_path and model_path.startswith("tinker://"):
        sess_id = model_path[len("tinker://"):]
    else:
        sess_id = model_id or "samp-session-live-123"
        
    return {"sampling_session_id": sess_id, "type": "create_sampling_session"}

@app.post("/api/v1/asample")
async def asample(req: dict):
    req_id = str(uuid.uuid4())
    await store.set_future(req_id, {"status": "pending"})
    
    prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
    params = req.get("sampling_params", {})
    max_tokens = params.get("max_tokens", 20)
    temperature = params.get("temperature", 1.0)
    num_samples = req.get("num_samples", 1)
    
    model_id = req.get("model_id") or req.get("sampling_session_id")
    # Strip potential tinker:// prefix explicitly
    if model_id and model_id.startswith("tinker://"):
        model_id = model_id[len("tinker://"):]
    
    # vLLM caches adapters natively based on the `lora_id` key. 
    # To force vLLM to reload weights after PyTorch trains them, we pass a unique sequential `lora_id`.
    lora_id = model_id
    # Strip the sequence tag to find the base directory where PyTorch actually wrote the checkpoint
    base_model_id = lora_id.split("-samp-")[0] if lora_id else None

    sampler_backend = os.getenv("SAMPLER_BACKEND", "vllm").lower()
    if sampler_backend == "engine":
        await enqueue_traced_request(store, {
            "req_id": req_id,
            "model_id": base_model_id or model_id,
            "type": "sample",
            "prompt_tokens": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": num_samples,
        })
        return {"request_id": req_id}
    
    # IPC Bridge: Route to vLLM worker on Port 8001 instead of PyTorch Queue
    async def _route_to_vllm():
        try:
            import os
            tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
            # PEFT natively saves adapters into subdirectories named after their adapter ID
            # Engine saves to: tmp_dir/peft/m_id (because adapter_name=m_id)
            lora_path = os.path.join(tmp_dir, "peft", base_model_id, base_model_id) if base_model_id else None
            
            print(f"[Gateway DEBUG] Routing sample to vLLM. model_id_received={model_id} -> lora_id={lora_id}, lora_path={lora_path}")
            
            payload = {
                "request_id": req_id,
                "prompt_token_ids": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_samples": num_samples,
                "lora_id": lora_id,
                "lora_path": lora_path
            }
            
            req_bytes = json.dumps(payload).encode('utf-8')
            
            # Use VLLM_URL env var instead of hardcoded 127.0.0.1:8001
            vllm_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
            vllm_generate_endpoint = f"{vllm_url.rstrip('/')}/generate"
            
            headers = {'Content-Type': 'application/json'}
            from opentelemetry import propagate
            propagate.inject(headers)
            
            import httpx
            
            # Use AsyncClient natively to prevent ThreadPool exhaustion from concurrent RL jobs!
            # Increase timeout significantly since large RL batches will queue up in vLLM for > 60s
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(vllm_generate_endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            if data.get("type") != "RequestFailedResponse":
                data["type"] = "sample"
            await store.set_future(req_id, data)
        except Exception as e:
            traceback.print_exc()
            await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(e), "category": "server_error"})

    task = asyncio.create_task(_route_to_vllm())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    return {"request_id": req_id}
@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
    request_id = req.get("request_id")
    if not request_id:
        return JSONResponse(status_code=400, content={"error": "request_id is required"})
    
    result = await store.get_future(request_id, timeout=60.0)
    if result is None:
         return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
         
    if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
         return JSONResponse(status_code=400, content=result)
         
    return result

@app.post("/api/v1/telemetry")
async def telemetry(req: dict):
    return {"status": "accepted"}
