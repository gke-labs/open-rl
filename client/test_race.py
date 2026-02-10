import asyncio
from tinker import ServiceClient, types
import os

os.environ["TINKER_API_KEY"] = "tml-dummy-key"
os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000"

async def main():
    service_client = ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model="Qwen/Qwen2.5-0.5B", rank=16)
    tokenizer = training_client.get_tokenizer()
    
    prompt = "English: testing\nPig Latin:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    comp_tokens = tokenizer.encode(" esting-tay\n\n", add_special_tokens=False)
    tokens = prompt_tokens + comp_tokens
    
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": [0]*len(prompt_tokens) + [1]*len(comp_tokens)}
    )

    for epoch in range(5):
        fwd_bwd_future = await training_client.forward_backward_async([datum], "cross_entropy")
        res1 = await fwd_bwd_future
        
        opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=1e-3))
        await opt_future
        print(f"Epoch {epoch} loss:", res1.metrics)

asyncio.run(main())
