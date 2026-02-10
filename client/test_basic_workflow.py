import os
import asyncio
import tinker
from tinker import types

# Make sure to point the tinker client at our backend server.
os.environ["TINKER_API_KEY"] = "tml-dummy-key"
os.environ["TINKER_BASE_URL"] = "http://localhost:8000"

async def main():
    print("1. Initializing Service Client...")
    service_client = tinker.ServiceClient()

    print("2. Fetching Server Capabilities...")
    caps = service_client.get_server_capabilities()
    if not caps.supported_models:
        print("No models supported currently. Exiting.")
        return
        
    print(f"Available models: {[m.model_name for m in caps.supported_models]}")
    base_model = caps.supported_models[0].model_name

    print(f"\n3. Creating LoRA Training Client for '{base_model}'...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=16
    )

    print("\n4. Fetching Tokenizer & Preparing dummy training data...")
    # NOTE: The Huggingface tokenizers should be used here. For simplicity, we are 
    # going to fetch it from the training client if the Tinker python library exposes it.
    tokenizer = training_client.get_tokenizer()
    examples = [
        {"input": "banana split", "output": "anana-bay plit-say"},
        {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
        {"input": "donut shop", "output": "onut-day op-shay"},
        {"input": "pickle jar", "output": "ickle-pay ar-jay"},
        {"input": "space exploration", "output": "ace-spay exploration-way"},
        {"input": "rubber duck", "output": "ubber-ray uck-day"},
        {"input": "coding wizard", "output": "oding-cay izard-way"},
    ]

    def process_example(example: dict, tokenizer) -> types.Datum:
        prompt = f"English: {example['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_weights = [0] * len(prompt_tokens)
        
        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        completion_weights = [1] * len(completion_tokens)

        tokens = prompt_tokens + completion_tokens
        weights = prompt_weights + completion_weights

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:] 
        weights = weights[1:]

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights
            }
        )

    processed_examples = [process_example(ex, tokenizer) for ex in examples]

    print("\n5. Running Forward/Backward pass (Cross Entropy) over 100 epochs...")
    import matplotlib.pyplot as plt
    
    losses = []
    epochs = 100
    for epoch in range(epochs):
        fwd_bwd_future = await training_client.forward_backward_async(processed_examples, "cross_entropy")
        fwd_bwd_result = await fwd_bwd_future

        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=1e-4))
        optim_result = await optim_future
        
        loss_val = fwd_bwd_result.metrics.get("loss:mean", 0.0)
        losses.append(loss_val)
        print(f"Epoch {epoch:02d} - Loss metrics: {fwd_bwd_result.metrics}")
        
    print("\n-> Generating loss curve plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve (Pig Latin)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("Saved loss curve to 'loss_curve.png'")

    print("\n7. Saving weights and getting Sampling Client...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="mvp-test-sampler")

    print("\n8. Sampling from the compiled model...")
    test_prompt = types.ModelInput.from_ints(tokenizer.encode("English: python script\nPig Latin:"))
    params = types.SamplingParams(max_tokens=20, temperature=0.0)
    
    sample_result = await sampling_client.sample_async(
        prompt=test_prompt, 
        sampling_params=params, 
        num_samples=1
    )
    
    print("\n--- Response Generated ---")
    for i, seq in enumerate(sample_result.sequences):
         print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
         
    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
