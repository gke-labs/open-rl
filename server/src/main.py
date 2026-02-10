import uuid
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .trainer.engine import engine

app = FastAPI(title="Cute-RL Server MVP")

futures_store = {}

def create_future(initial_req_id=None):
    req_id = initial_req_id or str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    return {"request_id": req_id}

def set_future_result(req_id, result_data):
    futures_store[req_id] = result_data

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
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    base_model = req.get("base_model", "Qwen/Qwen2.5-0.5B")
    lora_config = req.get("lora_config", {})
    rank = lora_config.get("rank", 16)
    
    async def _load_model_task():
        try:
            await asyncio.to_thread(engine.load_model, base_model, rank)
            set_future_result(req_id, {
                "model_id": "model-live-123",
                "is_lora": True,
                "lora_rank": rank,
                "type": "create_model"
            })
        except Exception as e:
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
        "model_id": req.get("model_id", "model-live-123"),
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
    
    async def _fwd_task():
        try:
            result = await asyncio.to_thread(engine.forward_backward, data, loss_fn, loss_config)
            result["type"] = "forward_backward"
            set_future_result(req_id, result)
        except Exception as e:
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})

    asyncio.create_task(_fwd_task())
    return {"request_id": req_id}

@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    adam_params = req.get("adam_params", {})
    
    async def _optim_task():
        try:
            result = await asyncio.to_thread(engine.optim_step, adam_params)
            result["type"] = "optim_step"
            set_future_result(req_id, result)
        except Exception as e:
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
            
    asyncio.create_task(_optim_task())
    return {"request_id": req_id}

@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {
        "path": None,
        "sampling_session_id": "samp-session-live-123",
        "type": "save_weights_for_sampler"
    }
    return {"request_id": req_id}

@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
    return {"sampling_session_id": "samp-session-live-123", "type": "create_sampling_session"}

@app.post("/api/v1/asample")
async def asample(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
    params = req.get("sampling_params", {})
    max_tokens = params.get("max_tokens", 20)
    
    async def _sample_task():
        try:
            import torch
            input_tensor = torch.tensor([prompt], dtype=torch.long, device=engine.device)
            # Generation
            with torch.no_grad():
                outputs = engine.model.generate(
                    input_tensor, 
                    max_new_tokens=max_tokens,
                    pad_token_id=engine.tokenizer.pad_token_id or engine.tokenizer.eos_token_id,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            gen_sequences = outputs.sequences[0]
            generated_tokens = gen_sequences[len(prompt):].cpu().tolist()
            
            logprobs = []
            for i, score_tensor in enumerate(outputs.scores):
                logprob_dist = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
                token_id = generated_tokens[i]
                logprob = logprob_dist[token_id].item()
                logprobs.append(logprob)
            
            set_future_result(req_id, {
                "sequences": [
                    {
                        "tokens": generated_tokens,
                        "logprobs": logprobs,
                        "stop_reason": "stop"
                    }
                ],
                "type": "sample"
            })
        except Exception as e:
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
            
    asyncio.create_task(_sample_task())
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

