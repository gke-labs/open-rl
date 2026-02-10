import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class TrainerEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Decide device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.base_model_name = None

    def load_model(self, base_model: str, rank: int):
        self.base_model_name = base_model
        print(f"Loading {base_model} to {self.device}...")
        
        # RESET optimizer for new training run
        self.optimizer = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
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
        self.model = get_peft_model(model, peft_config)
        self.model.train()
        print("Model loaded and wrapped with LoRA.")

    def forward_backward(self, data: List[Dict[str, Any]], loss_fn: str, loss_fn_config: dict = None) -> Dict[str, Any]:
        """
        data: List of Datum objects
        """
        assert self.model is not None, "Model not loaded."
        
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
            outputs = self.model(inputs_tensor)
            logits = outputs.logits[0] # Shape: (SeqLen, VocabSize)
            
            # log_softmax
            logprobs_all = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # gather logprobs for the target tokens
            target_logprobs = logprobs_all.gather(dim=-1, index=targets_tensor.unsqueeze(-1)).squeeze(-1)
            
            if loss_fn == "cross_entropy":
                elementwise_loss = -target_logprobs * weights_tensor
                loss = elementwise_loss.sum()
            elif loss_fn == "importance_sampling":
                # Extract RL inputs from datum's loss_fn_inputs
                # Note: Tinker passes these as serialized TensorData dicts {"data": [1,2,3], "shape": [3], "dtype": "float32"}
                ref_logprobs_raw = loss_inputs.get("logprobs")
                advs_raw = loss_inputs.get("advantages")
                
                ref_logprobs = ref_logprobs_raw.get("data") if isinstance(ref_logprobs_raw, dict) else ref_logprobs_raw
                advs = advs_raw.get("advantages") if isinstance(advs_raw, dict) else advs_raw
                # Ah wait advs_raw has "data"
                advs = advs_raw.get("data") if isinstance(advs_raw, dict) else advs_raw
                if not ref_logprobs or not advs:
                     raise ValueError("importance_sampling requires 'logprobs' and 'advantages' in loss_fn_inputs")
                     
                ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
                advantages_tensor = torch.tensor(advs, dtype=target_logprobs.dtype, device=self.device)
                
                ratio = torch.exp(target_logprobs - ref_tensor)
                # Policy gradient loss: -ratio * advantage * mask
                elementwise_loss = - (ratio * advantages_tensor) * weights_tensor
                loss = elementwise_loss.sum()
            else:
                raise NotImplementedError(f"Loss {loss_fn} not implemented in MVP yet.")
                
            loss.backward()
            total_loss += loss.item()
            
            # Construct loss_fn_output
            loss_fn_outputs.append({
                "logprobs": {
                    "data": target_logprobs.detach().cpu().tolist(),
                    "dtype": "float32",
                    "shape": [len(target_logprobs)]
                }
            })
            
        mean_loss = total_loss / max(1, len(data))
        
        return {
            "metrics": {"loss:mean": mean_loss, "loss:sum": total_loss},
            "loss_fn_outputs": loss_fn_outputs,
            "loss_fn_output_type": "ArrayRecord"
        }

    def optim_step(self, adam_params: Dict[str, Any]):
        if self.optimizer is None:
            lr = adam_params.get("learning_rate", 1e-4)
            beta1 = adam_params.get("beta1", 0.9)
            beta2 = adam_params.get("beta2", 0.95)
            eps = adam_params.get("eps", 1e-12)
            weight_decay = adam_params.get("weight_decay", 0.0)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                betas=(beta1, beta2), 
                eps=eps,
                weight_decay=weight_decay
            )
            
        # Compute grad norm BEFORE stepping or zeroing
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"DEBUG: grad_norm={total_norm}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            "metrics": {"grad_norm:mean": total_norm} # actual grad norm
        }

# Global singleton
engine = TrainerEngine()
