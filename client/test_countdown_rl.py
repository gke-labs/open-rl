import asyncio
import os
import re
import random
import ast
from collections import Counter
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

os.environ["TINKER_API_KEY"] = "tml-dummy-key"
os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000"

def generate_puzzle():
    large_pool = [25, 50, 75, 100]
    small_pool = list(range(1, 11)) * 2
    num_large = random.randint(1, 4)
    sources = random.sample(large_pool, num_large) + random.sample(small_pool, 6 - num_large)
    target = random.randint(101, 999)
    return sources, target

def evaluate_puzzle(text: str, sources: list[int], target: int) -> float:
    format_reward = 0.0
    exact_reward = 0.0
    closeness_reward = 0.0
    
    # 1. Format
    has_reasoning = bool(re.search(r'</reasoning>', text, re.DOTALL))
    solution_match = re.search(r'<solution>(.*?)</solution>', text, re.DOTALL)
    
    if has_reasoning and solution_match:
        format_reward = 0.1
        
    if not solution_match:
        return format_reward
        
    expr = solution_match.group(1).strip()
    
    # 2. Extract numbers to verify subset
    nums_in_expr = [int(n) for n in re.findall(r'\b\d+\b', expr)]
    sources_counter = Counter(sources)
    expr_counter = Counter(nums_in_expr)
    
    for num, count in expr_counter.items():
        if sources_counter[num] < count:
            return format_reward # Used invalid numbers
            
    # 3. Evaluate Expression
    if not re.match(r'^[\d\s\+\-\*\/\(\)]+$', expr):
        return format_reward
        
    try:
        # evaluate safely
        result = eval(expr, {"__builtins__": None}, {})
        if not isinstance(result, (int, float)):
            return format_reward
            
        if abs(result - target) < 1e-5:
            exact_reward = 1.0
        else:
            diff = abs(result - target)
            closeness_reward = 0.3 * (0.5 ** (diff / 10.0))
            
    except Exception:
         pass 
         
    return format_reward + exact_reward + closeness_reward

async def main():
    print("1. Initializing Service and Loading Model...")
    service_client = ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen2.5-0.5B-Instruct", rank=16
    )
    tokenizer = training_client.get_tokenizer()
    
    epochs = 150
    rewards_history = []
    losses_history = []
    
    prompt_template = """System: You are playing the Countdown numbers game. 
    You have 6 source numbers. Use + - * / and parentheses to reach the Target. Each source number can be used at most once.
    Be concise! Output your simple thought process in <reasoning> tags, and your final math equation in <solution> tags.
    
    Example:
    Numbers: 25, 50, 75, 100, 3, 6
    Target: 125
    <reasoning>
    100 + 25 = 125. Done.
    </reasoning>
    <solution>
    100 + 25
    </solution>
    
    Numbers: {sources}
    Target: {target}
    <reasoning>
    """
    
    print("\n2. Starting Countdown RL Training...")
    for epoch in range(epochs):
        # Sync the updated weights immediately so the sampling client uses the current policy
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="rl-countdown")
        
        sources, target = generate_puzzle()
        prompt_text = prompt_template.format(sources=", ".join(map(str, sources)), target=target)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        
        sample_params = types.SamplingParams(max_tokens=200)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
        
        sample_res = await sampling_client.sample_async(
            model_input,
            1,
            sampling_params=sample_params
        )
        
        gen_tokens = sample_res.sequences[0].tokens
        ref_logprobs = sample_res.sequences[0].logprobs
        gen_text = tokenizer.decode(gen_tokens).strip()

        raw_reward = evaluate_puzzle(gen_text, sources, target)
        
        # Center advantage around 0 slightly (baseline estimation proxy)
        # If the reward is just 0.0 (total garbage), it becomes -0.1
        # If it uses format tags (0.1), advantage is 0.0
        # If it gets reasonably close (+0.3), advantage is +0.2
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
        
        opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=5e-4))
        opt_res = await opt_future
        
        loss_val = fwd_res.metrics.get("loss:mean", 0.0)
        losses_history.append(loss_val)
        rewards_history.append(raw_reward)
        
        print(f"Epoch {epoch:02d} | Tgt: {target} | Rwd: {raw_reward:.3f} | Adv: {advantage_val:+.3f} | Loss: {loss_val:.4f}\nGen: {gen_text!r}\n")
        
    print("\n-> Generating RL reward curve plot...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), rewards_history, marker='o', linestyle='-', color='g')
    plt.title('Countdown RL Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward (0.0 to 1.4)')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), losses_history, marker='o', linestyle='-', color='r')
    plt.title('Policy Gradient Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('countdown_rl_rewards.png')
    print("Saved 'countdown_rl_rewards.png'")

if __name__ == "__main__":
    asyncio.run(main())
