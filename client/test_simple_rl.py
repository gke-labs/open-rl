import asyncio
import os
import re
import random
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

os.environ["TINKER_API_KEY"] = "tml-dummy-key"
os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000"

PHRASES = [
    "Hello world!",
    "I love RL!",
    "Qwen is smart!",
    "Apple pie is tasty."
]

def evaluate_output(gen_text: str, target_phrase: str) -> float:
    match = re.search(r'<echo>(.*?)</echo>', gen_text, re.DOTALL)
    if not match:
        return -1.0
    
    extracted = match.group(1).strip()
    if extracted == target_phrase:
        return 1.0
    else:
        return 0.1

async def main():
    print("1. Initializing Service and Loading Model...")
    service_client = ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen2.5-0.5B", rank=16
    )
    tokenizer = training_client.get_tokenizer()
    
    epochs = 100
    rewards_history = []
    losses_history = []
    
    prompt_template = """System: You are an echoing assistant. Whatever the user says, you must repeat it EXACTLY, wrapped in <echo> tags. 
Do not output any other text.

User: {phrase}
"""
    
    print(f"\n2. Starting Simple RL Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Explicitly sync weights and get an updated sampling client for this epoch
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="rl-simple")
        
        target_phrase = random.choice(PHRASES)
        prompt_text = prompt_template.format(phrase=target_phrase)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        
        sample_params = types.SamplingParams(max_tokens=20)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
        
        sample_res = await sampling_client.sample_async(
            model_input,
            1,
            sampling_params=sample_params
        )
        
        gen_tokens = sample_res.sequences[0].tokens
        ref_logprobs = sample_res.sequences[0].logprobs
        gen_text = tokenizer.decode(gen_tokens).strip()

        raw_reward = evaluate_output(gen_text, target_phrase)
        
        # Center advantage slightly: exact match is +0.9 Adv, format is +0.0 Adv, penalty is -1.1 Adv
        advantage_val = raw_reward - 0.1
        advantages = [advantage_val] * len(gen_tokens)
        
        full_tokens = prompt_tokens + gen_tokens
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": gen_tokens,
                "weights": [1.0] * len(gen_tokens),
                "advantages": advantages,
                "logprobs": ref_logprobs
            }
        )
        
        fwd_bwd_future = await training_client.forward_backward_async(
            [datum], "importance_sampling"
        )
        fwd_res = await fwd_bwd_future
        
        opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=5e-5))
        opt_res = await opt_future
        
        loss_val = fwd_res.metrics.get("loss:mean", 0.0)
        losses_history.append(loss_val)
        rewards_history.append(raw_reward)
        
        print(f"Epoch {epoch:02d} | Tgt: {target_phrase!r} | Rwd: {raw_reward:+.1f} | Adv: {advantage_val:+.1f} | Loss: {loss_val:.4f}\nGen: {gen_text!r}\n")
        
    print("\n-> Generating RL reward curve plot...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    
    # Calculate simple moving average of rewards for a smoother line
    window_size = max(1, epochs // 10)
    smoothed_rewards = []
    for i in range(epochs):
        start = max(0, i - window_size + 1)
        window = rewards_history[start:i+1]
        smoothed_rewards.append(sum(window) / len(window))
        
    plt.plot(range(epochs), smoothed_rewards, linestyle='-', color='g', label='Mean Reward (SMA)')
    plt.plot(range(epochs), rewards_history, marker='o', linestyle='', color='g', alpha=0.2, label='Step Reward')
    plt.title('Simple Echo RL Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward (-1.0 to 1.0)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), losses_history, marker='+', linestyle='-', color='r', alpha=0.6)
    plt.title('Policy Gradient Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('simple_rl_rewards.png')
    print("Saved 'simple_rl_rewards.png'")

if __name__ == "__main__":
    asyncio.run(main())
