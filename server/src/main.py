import uuid
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .engine import engine, futures_store, request_queue, set_future_result, lifespan
import logging
import urllib.request
import json
import traceback

class FilterNoisyEndpoints(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "retrieve_future" not in msg and "session_heartbeat" not in msg

logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

app = FastAPI(title="Kube-RL Server MVP", lifespan=lifespan)

@app.get("/api/v1/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
    return {
        "supported_models": [
            {"model_name": "Qwen/Qwen2.5-0.5B"}
        ]
    }

@app.post("/api/v1/create_session")
async def create_session(req: dict):
    return {"session_id": "sess-real-123", "type": "create_session"}

@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: dict):
    return {"type": "session_heartbeat"}

@app.post("/api/v1/create_model")
async def create_model(req: dict):
    # We use req_id as model_id
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    base_model = req.get("base_model", "Qwen/Qwen2.5-0.5B")
    lora_config = req.get("lora_config", {})
    rank = lora_config.get("rank", 16)
    
    model_id = req_id 
    
    async def _load_model_task():
        try:
            await asyncio.to_thread(engine.load_model, base_model, rank, model_id)
            set_future_result(req_id, {
                "model_id": model_id,
                "is_lora": True,
                "lora_rank": rank,
                "type": "create_model"
            })
        except Exception as e:
            traceback.print_exc()
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})

    asyncio.create_task(_load_model_task())
    return {"request_id": req_id}

@app.post("/api/v1/get_info")
async def get_info(req: dict):
    model_name = engine.base_model_name or "Qwen/Qwen2.5-0.5B"
    return {
        "model_data": {
            "arch": "qwen",
            "model_name": model_name,
            "tokenizer_id": model_name
        },
        "model_id": req.get("model_id", "model-live-123"), # Will be whatever client passed
        "is_lora": True,
        "lora_rank": 16,
        "model_name": model_name,
        "type": "get_info"
    }

@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    fwd_input = req.get("forward_backward_input", {})
    data = fwd_input.get("data", [])
    loss_fn = fwd_input.get("loss_fn", "cross_entropy")
    loss_config = fwd_input.get("loss_fn_config", {})
    model_id = req.get("model_id")
    
    # Push to queue instead of thread
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "type": "forward_backward",
        "data": data,
        "loss_fn": loss_fn,
        "loss_config": loss_config
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    adam_params = req.get("adam_params", {})
    model_id = req.get("model_id")
    
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "type": "optim_step",
        "adam_params": adam_params
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
    req_id = str(uuid.uuid4())
    model_id = req.get("model_id") # Client passes the TrainingClient's model_id
    seq_id = req.get("sampling_session_seq_id", 0)
    futures_store[req_id] = {"status": "pending"}
    
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "seq_id": seq_id,
        "type": "save_weights_for_sampler"
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
    # If client manually creates a session, we'll assign a mock one, 
    # but the usual flow is save_weights -> uses the existing model_id
    return {"sampling_session_id": req.get("model_id", "samp-session-live-123"), "type": "create_sampling_session"}

@app.post("/api/v1/asample")
async def asample(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
    params = req.get("sampling_params", {})
    max_tokens = params.get("max_tokens", 20)
    temperature = params.get("temperature", 1.0)
    num_samples = req.get("num_samples", 1)
    
    model_id = req.get("model_id") or req.get("sampling_session_id")
    
    # vLLM caches adapters natively based on the `lora_id` key. 
    # To force vLLM to reload weights after PyTorch trains them, we pass a unique sequential `lora_id`.
    lora_id = model_id
    # Strip the sequence tag to find the base directory where PyTorch actually wrote the checkpoint
    base_model_id = lora_id.split("-samp-")[0] if lora_id else None
    
    # IPC Bridge: Route to vLLM worker on Port 8001 instead of PyTorch Queue
    async def _route_to_vllm():
        try:
            import os
            tmp_dir = os.environ.get("KUBE_RL_TMP_DIR", "/tmp/kube-rl")
            # PEFT natively saves adapters into subdirectories named after their adapter ID
            lora_path = os.path.join(tmp_dir, "peft", base_model_id, base_model_id) if base_model_id else None
            
            print(f"[Gateway DEBUG] Routing asample to vLLM. model_id_received={model_id} -> lora_id={lora_id}, lora_path={lora_path}")
            
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
            http_req = urllib.request.Request(
                "http://127.0.0.1:8001/generate", 
                data=req_bytes, 
                headers={'Content-Type': 'application/json'}
            )
            
            def _make_req():
                with urllib.request.urlopen(http_req, timeout=30.0) as f:
                    return json.loads(f.read().decode('utf-8'))
                    
            data = await asyncio.to_thread(_make_req)
            data["type"] = "sample"
            set_future_result(req_id, data)
        except Exception as e:
            traceback.print_exc()
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})

    asyncio.create_task(_route_to_vllm())
    
    return {"request_id": req_id}

@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
    request_id = req.get("request_id")
    if request_id in futures_store:
        result = futures_store[request_id]
        if result.get("status") == "pending":
            return {
                "type": "try_again", 
                "request_id": request_id, 
                "queue_state": "active"
            }
        return result
    return {"type": "RequestFailedResponse", "error_message": "Future not found"}

@app.post("/api/v1/telemetry")
async def telemetry(req: dict):
    return {"status": "accepted"}

