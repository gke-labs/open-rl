import asyncio
import os
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

os.environ["TINKER_API_KEY"] = "tml-dummy-key"
os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000"

async def main():
    print("1. Initializing...")
    service_client = ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen2.5-0.5B", rank=16
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="rl-sampler")

    prompt = "Finish this sentence with one word: The capital of France is"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    
    losses = []
    rewards_history = []
    epochs = 40
    
    print("\n2. Starting RL Loop (REINFORCE)...")
    for epoch in range(epochs):
        # 1. Rollout (Generate)
        sample_params = types.SamplingParams(max_tokens=5)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
        
        sample_res = await sampling_client.sample_async(
            model_input,
            1,
            sampling_params=sample_params
        )
        
        gen_tokens = sample_res.sequences[0].tokens
        ref_logprobs = sample_res.sequences[0].logprobs
        
        gen_text = tokenizer.decode(gen_tokens).strip()
        
        # 2. Compute Dummy Reward
        # Let's say we want it to output exactly "Paris."
        if "Paris" in gen_text:
            reward = 1.0
        else:
            reward = -1.0
            
        # Advantage is just the raw normalized reward for this single-sample test
        advantages = [reward] * len(gen_tokens)
        
        # 3. Create RL Datum
        full_tokens = prompt_tokens + gen_tokens
        # Input to model needs to be everything EXCEPT the last generated token
        # Target tokens needs to be ONLY the generated tokens
        
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": gen_tokens,
                "weights": [1.0] * len(gen_tokens),
                "advantages": advantages, # Move into datum instead of loss_fn_config
                "logprobs": ref_logprobs
            }
        )
        
        # 4. Train (forward_backward importance_sampling)
        fwd_bwd_future = await training_client.forward_backward_async(
            [datum], "importance_sampling" # drop loss_fn_config entirely
        )
        fwd_res = await fwd_bwd_future
        
        # 5. Optim Step
        opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=5e-4))
        opt_res = await opt_future
        
        loss_val = fwd_res.metrics.get("loss:mean", 0.0)
        losses.append(loss_val)
        rewards_history.append(reward)
        print(f"Epoch {epoch:02d} | Reward: {reward:+.2f} | Loss: {loss_val:.4f} | Output: {gen_text!r}")
        
    print("\n-> Generating RL reward curve plot...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), rewards_history, marker='o', linestyle='-', color='g')
    plt.title('RL Training Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Advantage')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), losses, marker='o', linestyle='-', color='r')
    plt.title('Policy Gradient Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('rl_reward_curve.png')
    print("Saved 'rl_reward_curve.png'")

if __name__ == "__main__":
    asyncio.run(main())
