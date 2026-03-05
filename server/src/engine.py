import torch
import threading
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import asyncio
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI
import subprocess
import os
import sys

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

from .state import get_store
store = get_store()

import math

class TrainerEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        # Store optimizers per model_id
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._init_lock = threading.Lock()
        
        # Decide device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.base_model_name = None

    def load_model(self, base_model: str, rank: int, model_id: str):
        with self._init_lock:
            if self.model is None or self.base_model_name != base_model:
                self.base_model_name = base_model
                print(f"Loading base model {base_model} to {self.device}...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    device_map=self.device
                )
                
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=rank * 2,
                    target_modules=["q_proj", "v_proj"]
                )
                # Apply PEFT with the specific adapter name
                self.model = get_peft_model(base_model_obj, peft_config, adapter_name=model_id)
                self.model.train()
                print(f"Base model loaded and wrapped with LoRA adapter '{model_id}'.")
            else:
                print(f"Adding new LoRA adapter '{model_id}' to existing base model...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=rank * 2,
                    target_modules=["q_proj", "v_proj"]
                )
                self.model.add_adapter(model_id, peft_config)
                self.model.train()
            
        # Reset/initialize optimizer for this new adapter
        if model_id in self.optimizers:
            del self.optimizers[model_id]

    def set_active_adapter(self, model_id: str):
        with self._init_lock:
            if self.model is not None:
                self.model.set_adapter(model_id)

    def forward_backward(self, data: List[Dict[str, Any]], loss_fn: str, loss_fn_config: dict = None, model_id: str = None) -> Dict[str, Any]:
        with tracer.start_as_current_span("forward_backward") as span:
            with self._init_lock:
                span.set_attribute("model_id", model_id or "unknown")
                span.set_attribute("batch_size", len(data))
                span.set_attribute("loss_fn", loss_fn)
                return self._forward_backward_internal(data, loss_fn, loss_fn_config, model_id)

    def _forward_backward_internal(self, data: List[Dict[str, Any]], loss_fn: str, loss_fn_config: dict = None, model_id: str = None) -> Dict[str, Any]:
        """
        data: List of Datum objects
        """
        assert self.model is not None, "Model not loaded."
        
        if model_id and model_id not in self.optimizers:
            # Ensuring optimizer exists is good, but we don't zero_grad here anymore
            pass
        
        total_loss = 0.0
        loss_fn_outputs = []
        
        for datum in data:
            # Extract inputs
            chunks = datum.get("model_input", {}).get("chunks", [])
            input_tokens = []
            for chunk in chunks:
                input_tokens.extend(chunk.get("tokens", []))
            
            inputs_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Extract loss inputs
            loss_inputs = datum.get("loss_fn_inputs", {})
            
            weights_data = loss_inputs.get("weights", {}).get("data", [])
            weights_tensor = torch.tensor(weights_data, dtype=torch.float32, device=self.device)
            
            targets_data = loss_inputs.get("target_tokens", {}).get("data", [])
            targets_tensor = torch.tensor(targets_data, dtype=torch.long, device=self.device)
            
            # Forward pass
            outputs = self.model(inputs_tensor, use_cache=False)
            logits = outputs.logits[0] # Shape: (SeqLen, VocabSize)
            
            # gather logprobs for the target tokens
            # The client SDK already offsets inputs vs targets.
            seq_len = min(logits.size(0), targets_tensor.size(0))
            
            sliced_logits = logits[:seq_len]
            sliced_targets = targets_tensor[:seq_len] 
            
            target_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)\
                                .gather(dim=-1, index=sliced_targets.unsqueeze(-1))\
                                .squeeze(-1)
            
            # Clamp weights tensor if it exists, otherwise ones
            if weights_tensor.numel() > 0:
                weights_tensor = weights_tensor[:seq_len]
            else:
                weights_tensor = torch.ones_like(target_logprobs)
            
            if loss_fn == "cross_entropy":
                elementwise_loss = -target_logprobs * weights_tensor
                loss = elementwise_loss.sum()
            elif loss_fn == "importance_sampling":
                ref_logprobs_raw = loss_inputs.get("logprobs")
                advs_raw = loss_inputs.get("advantages")
                
                ref_logprobs = ref_logprobs_raw.get("data") if isinstance(ref_logprobs_raw, dict) else ref_logprobs_raw
                advs = advs_raw.get("data") if isinstance(advs_raw, dict) else advs_raw
                if not ref_logprobs or not advs:
                     raise ValueError("importance_sampling requires 'logprobs' and 'advantages' in loss_fn_inputs")
                     
                ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
                advantages_tensor = torch.tensor(advs, dtype=target_logprobs.dtype, device=self.device)
                
                # Align reference logits and advantages to the generated right-aligned targets
                ref_tensor = ref_tensor[:seq_len]
                advantages_tensor = advantages_tensor[:seq_len]
                
                # Prevent overflow in exp() by explicitly clamping diff
                diff = target_logprobs - ref_tensor
                diff = torch.clamp(diff, min=-20.0, max=20.0)
                
                ratio = torch.exp(diff)
                    
                elementwise_loss = - (ratio * advantages_tensor) * weights_tensor
                # Add nan_to_num to be absolutely sure no trailing NaNs poison the gradients
                elementwise_loss = torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)
                
                loss = elementwise_loss.sum()
            elif loss_fn == "ppo":
                ref_logprobs_raw = loss_inputs.get("logprobs")
                advs_raw = loss_inputs.get("advantages")
                
                ref_logprobs = ref_logprobs_raw.get("data") if isinstance(ref_logprobs_raw, dict) else ref_logprobs_raw
                advs = advs_raw.get("data") if isinstance(advs_raw, dict) else advs_raw
                if not ref_logprobs or not advs:
                     raise ValueError("ppo requires 'logprobs' and 'advantages' in loss_fn_inputs")
                     
                ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
                advantages_tensor = torch.tensor(advs, dtype=target_logprobs.dtype, device=self.device)
                
                ref_tensor = ref_tensor[:seq_len]
                advantages_tensor = advantages_tensor[:seq_len]
                
                # Prevent overflow in exp() by explicitly clamping diff
                diff = target_logprobs - ref_tensor
                diff = torch.clamp(diff, min=-20.0, max=20.0)
                ratio = torch.exp(diff)
                
                epsilon = loss_fn_config.get("clip_range", 0.2) if loss_fn_config else 0.2
                
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_tensor
                
                # PPO Objective: maximize min(surr1, surr2)
                # Loss: minimize -min(surr1, surr2)
                elementwise_objective = torch.min(surr1, surr2)
                
                loss = -elementwise_objective.sum()
            else:
                raise NotImplementedError(f"Loss {loss_fn} not implemented in MVP yet.")
                
            loss.backward()
            total_loss += loss.item()
            
            logprobs_list = target_logprobs.detach().cpu().tolist()
            logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
            
            # Construct loss_fn_output
            loss_fn_outputs.append({
                "logprobs": {
                    "data": logprobs_list,
                    "dtype": "float32",
                    "shape": [len(logprobs_list)]
                }
            })
        mean_loss = total_loss / max(1, len(data))
        
        return {
            "metrics": {
                "loss:mean": self._sanitize_float(mean_loss), 
                "loss:sum": self._sanitize_float(total_loss)
            },
            "loss_fn_outputs": loss_fn_outputs,
            "loss_fn_output_type": "ArrayRecord"
        }

    def _sanitize_float(self, val: float) -> float:
        if math.isinf(val):
            return -9999.0 if val < 0 else 9999.0
        if math.isnan(val):
            return 0.0
        return val

    def optim_step(self, adam_params: Dict[str, Any], model_id: str = None):
        with tracer.start_as_current_span("optim_step") as span:
            with self._init_lock:
                span.set_attribute("model_id", model_id or "unknown")
                return self._optim_step_internal(adam_params, model_id)

    def _optim_step_internal(self, adam_params: Dict[str, Any], model_id: str = None):
        if not model_id:
            raise ValueError("model_id is required for optim_step")
            
        if model_id not in self.optimizers:
            lr = adam_params.get("learning_rate", 1e-4)
            beta1 = adam_params.get("beta1", 0.9)
            beta2 = adam_params.get("beta2", 0.95)
            eps = adam_params.get("eps", 1e-12)
            weight_decay = adam_params.get("weight_decay", 0.0)
            
            # Ensure we only pass the parameters of the active adapter for this model_id
            # peft's model.parameters() works, but it's better to filter requires_grad
            params = [p for p in self.model.parameters() if p.requires_grad]
            
            self.optimizers[model_id] = torch.optim.AdamW(
                params, 
                lr=lr, 
                betas=(beta1, beta2), 
                eps=eps,
                weight_decay=weight_decay
            )
            
        optimizer = self.optimizers[model_id]
            
        # Compute grad norm BEFORE stepping
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"DEBUG: grad_norm={total_norm}")
        
        # Apply gradient clipping if requested
        clip_norm = adam_params.get("grad_clip_norm", 0.0)
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

        optimizer.step()
        optimizer.zero_grad()
        return {
            "metrics": {"grad_norm:mean": self._sanitize_float(total_norm)}
        }

    def generate(self, prompt_tokens: List[int], max_tokens: int, num_samples: int = 1, model_id: str = None) -> Dict[str, Any]:
        with self._init_lock:
            assert self.model is not None, "Model not loaded."
            
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                attention_mask = torch.ones_like(input_tensor)
                outputs = self.model.generate(
                input_tensor, 
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=(num_samples > 1),
                temperature=0.8 if num_samples > 1 else None, 
                top_p=None,
                top_k=None,
                num_return_sequences=num_samples,
                output_scores=True,
                return_dict_in_generate=True
            )
            
        sequences_out = []
        for seq_idx in range(num_samples):
            gen_sequences = outputs.sequences[seq_idx]
            generated_tokens = gen_sequences[len(prompt_tokens):].cpu().tolist()
            
            logprobs = []
            for token_step_idx in range(len(generated_tokens)):
                score_tensor = outputs.scores[token_step_idx]
                logprob_dist = torch.nn.functional.log_softmax(score_tensor[seq_idx], dim=-1)
                token_id = generated_tokens[token_step_idx]
                logprob = logprob_dist[token_id].item()
                if logprob == float('-inf'):
                    logprob = -9999.0
                elif logprob == float('inf'):
                    logprob = 9999.0
                logprobs.append(logprob)
                
            sequences_out.append({
                "tokens": generated_tokens,
                "logprobs": logprobs,
                "stop_reason": "stop"
            })
        
        return {"sequences": sequences_out}

# Global singleton
engine = TrainerEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("      Open-RL PyTorch Training Gateway")
    print("="*50)
    cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
    vllm_url = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
    print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
    print(f"-> Inference: Routing completions to VLLM_URL={vllm_url}\n")

    # Start the clock cycle loop
    task = asyncio.create_task(clock_cycle_loop())
    yield
    
    # Cleanup if needed
    task.cancel()


async def clock_cycle_loop():
    global store
    while True:
        try:
            # Block until requests are available and drain the queue
            batch = await store.get_requests()
            if not batch:
                await asyncio.sleep(0.1)
                continue
                
            # Group by model_id
            with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
                batch_span.set_attribute("batch_size", len(batch))
                models_to_reqs = {}
                for r in batch:
                    m_id = r.get("model_id")
                    if m_id not in models_to_reqs:
                        models_to_reqs[m_id] = []
                    models_to_reqs[m_id].append(r)
                    
                batch_span.set_attribute("num_models", len(models_to_reqs))
                print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests across {len(models_to_reqs)} distinct model tenant(s).")
                    
                for m_id, reqs in models_to_reqs.items():
                    if len(reqs) == 0:
                        continue
                        
                    with tracer.start_as_current_span("process_model_batch") as model_span:
                        model_span.set_attribute("model_id", m_id)
                        model_span.set_attribute("model_reqs", len(reqs))
                        print(f"  -> [TENSOR CORE] Hot-swapping to LoRA adapter: {m_id}")
                        
                        # Set active adapter
                        try:
                            with tracer.start_as_current_span("set_active_adapter"):
                                await asyncio.to_thread(engine.set_active_adapter, m_id)
                        except Exception as e:
                            print(f"Failed to set adapter {m_id}: {e}")
                            for r in reqs:
                                await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
                            continue
                            
                        print(f"     Executing {len(reqs)} operations for {m_id}...")
                        # Execute sequentially
                        for r in reqs:
                            req_id = r["req_id"]
                            req_type = r["type"]
                            
                            carrier = r.get("trace_context", {})
                            from opentelemetry import propagate, context as otel_context
                            ctx = propagate.extract(carrier) if carrier else None
                            token = otel_context.attach(ctx) if ctx else None
                            
                            try:
                                if req_type == "forward_backward":
                                    data = r["data"]
                                    loss_fn = r["loss_fn"]
                                    loss_config = r["loss_config"]
                                    result = await asyncio.to_thread(engine.forward_backward, data, loss_fn, loss_config, m_id)
                                    result["type"] = "forward_backward"
                                    await store.set_future(req_id, result)
                                elif req_type == "optim_step":
                                    adam_params = r["adam_params"]
                                    result = await asyncio.to_thread(engine.optim_step, adam_params, m_id)
                                    result["type"] = "optim_step"
                                    await store.set_future(req_id, result)
                                elif req_type == "save_weights_for_sampler":
                                    seq_id = r.get("seq_id", 0)
                                    alias = r.get("alias")
                                    import os
                                    import json
                                    import time
                                    from datetime import datetime
                                    
                                    # Save to disk/ramdisk so vLLM can load it
                                    tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
                                    ram_path = os.path.join(tmp_dir, "peft", m_id)
                                    os.makedirs(ram_path, exist_ok=True)
                                    
                                    # Because set_active_adapter(m_id) just ran, the engine model is active on this tenant!
                                    with tracer.start_as_current_span("save_weights_to_disk"):
                                        with engine._init_lock:
                                            engine.model.save_pretrained(ram_path, selected_adapters=[m_id])
                                    
                                    # Write metadata
                                    metadata = {
                                        "model_id": m_id,
                                        "alias": alias,
                                        "created_at": datetime.now().isoformat(),
                                        "timestamp": time.time()
                                    }
                                    try:
                                        with open(os.path.join(ram_path, "metadata.json"), "w") as f:
                                            json.dump(metadata, f)
                                    except Exception as e:
                                        print(f"Failed to write metadata: {e}")
                                    
                                    # Use a tinker:// URI that encodes the session ID, satisfying SDK validation
                                    # and matching what we expect in asample (after stripping prefix)
                                    session_id = f"{m_id}-samp-{seq_id}"
                                    
                                    # Tinkers SDK `save_weights_and_get_sampling_client` expects path=None (ephemeral)
                                    # Tinkers SDK `save_weights_for_sampler` expects path!=None (named)
                                    # We use the presence of 'alias' to distinguish.
                                    result_path = f"tinker://{session_id}" if alias else None
                                    
                                    result = {
                                        "path": result_path,
                                        "sampling_session_id": session_id,
                                        "type": "save_weights_for_sampler"
                                    }
                                    await store.set_future(req_id, result)
                                elif req_type == "save_weights":
                                    # Identical logic to save_weights_for_sampler but returns different type
                                    seq_id = r.get("seq_id", 0)
                                    alias = r.get("alias")
                                    import os
                                    import json
                                    import time
                                    from datetime import datetime
                                    
                                    tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
                                    ram_path = os.path.join(tmp_dir, "peft", m_id)
                                    os.makedirs(ram_path, exist_ok=True)
                                    
                                    with tracer.start_as_current_span("save_weights_to_disk"):
                                        with engine._init_lock:
                                            engine.model.save_pretrained(ram_path, selected_adapters=[m_id])
                                    
                                    metadata = {
                                        "model_id": m_id,
                                        "alias": alias,
                                        "created_at": datetime.now().isoformat(),
                                        "timestamp": time.time()
                                    }
                                    try:
                                        with open(os.path.join(ram_path, "metadata.json"), "w") as f:
                                            json.dump(metadata, f)
                                    except Exception as e:
                                        print(f"Failed to write metadata: {e}")
                                    
                                    session_id = f"{m_id}-samp-{seq_id}"
                                    # For save_weights (checkpointing), we usually just want the path
                                    result_path = f"tinker://{session_id}" 
                                    
                                    result = {
                                        "path": result_path, 
                                        "sampling_session_id": session_id,
                                        "type": "save_weights" # CORRECT TYPE for validation
                                    }
                                    await store.set_future(req_id, result)
                                elif req_type == "asample":
                                    prompt_tokens = r["prompt_tokens"]
                                    max_tokens = r["max_tokens"]
                                    num_samples = r["num_samples"]
                                    result = await asyncio.to_thread(engine.generate, prompt_tokens, max_tokens, num_samples, m_id)
                                    result["type"] = "sample"
                                    await store.set_future(req_id, result)
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
            
            # If the background Redis pod was restarted, the asyncio connection pool gets stuck.
            # We forcefully destroy the singleton to create a fresh connection pool on the next loop.
            import redis
            if isinstance(e, redis.exceptions.ConnectionError):
                print("[engine] Destroying StateStore singleton to force Redis reconnection...")
                from . import state
                state._store_instance = None
                store = state.get_store()
                
            await asyncio.sleep(1)
