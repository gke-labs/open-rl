import os
import asyncio
import random
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

def generate_problem():
    n = random.choice([2, 4, 6])
    nums = [random.choices(range(1, 10), k=1)[0] if random.random() < 0.7 
            else random.randint(10, 19) for _ in range(n)]
    sums = [nums[i] + nums[i+1] for i in range(0, len(nums), 2)]
    answer = math.prod(sums)
    return nums, sums, answer

SYSTEM_PROMPT = """Add adjacent pairs of numbers, then multiply the results.

Example: 3, 5, 2, 4
- Add pairs: 3+5=8, 2+4=6
- Multiply: 8×6=48
- Answer: <answer>48</answer>

Now solve the problem below. Put your final answer in <answer>X</answer> tags."""

def compute_reward(response, correct_answer, concise_bonus=False):
    rewards = {"format": 0.0, "correct": 0.0, "concise": 0.0}
    answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', response)
    if answer_match:
        rewards["format"] = 0.1
        if int(answer_match.group(1)) == correct_answer: rewards["correct"] = 1.0
    if concise_bonus:
        words = len(response.split())
        if words < 200: rewards["concise"] = 0.3 * max(0, (200 - words) / 180)
    rewards["total"] = sum(rewards.values())
    return rewards

async def main():
    print("1. Initializing Service Client...")
    service_client = ServiceClient()
    
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"\n3. Creating LoRA Training Client for '{base_model}'...")
    try:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=8
        )
    except Exception as e:
        print(f"Error creating client: {e}")
        return

    tokenizer = training_client.get_tokenizer()

    def make_prompt_tokens(problem):
        nums, _, _ = problem
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": ", ".join(map(str, nums))}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text, add_special_tokens=False)

    print(tokenizer.decode(make_prompt_tokens(generate_problem())))

    def compute_advantages(rollouts):
        """GRPO-style: normalize rewards to mean=0, std=1."""
        rewards = np.array([r["reward"] for r in rollouts])
        if rewards.std() < 1e-8: return [0.0] * len(rollouts)
        return ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()

    def make_rl_datum(rollout: dict, advantage: float) -> types.Datum:
        """Create Datum for importance sampling loss."""
        prompt_tokens, completion_tokens = rollout["prompt_tokens"], rollout["completion_tokens"]
        full_tokens = prompt_tokens + list(completion_tokens)
    
        input_tokens, target_tokens = full_tokens[:-1], full_tokens[1:]
        n_prompt = len(prompt_tokens) - 1
        n_completion = len(completion_tokens)
    
        return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": [0.0] * n_prompt + list(rollout["completion_logprobs"]),
            "advantages": [0.0] * n_prompt + [advantage] * n_completion
        }
    )

    def run_rollouts(n_problems=4, n_samples=4, concise_bonus=False):
        """Generate fresh problems, sample completions, compute rewards."""
        rollouts = []
        sampling_client = training_client.save_weights_and_get_sampling_client()
        for _ in range(n_problems):
            problem = generate_problem()  # Fresh each time
            ans = problem[2]
            prompt_tokens = make_prompt_tokens(problem)
            response = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=n_samples,
                sampling_params=types.SamplingParams(max_tokens=256, temperature=0.8)
            ).result()
            for seq in response.sequences:
                reward_info = compute_reward(tokenizer.decode(seq.tokens), ans, concise_bonus)
                rollouts.append({ "prompt_tokens": prompt_tokens, "completion_tokens": seq.tokens, "completion_logprobs": seq.logprobs, "completion_text": tokenizer.decode(seq.tokens), "reward": reward_info["total"], "reward_breakdown": reward_info, "correct_answer": ans })
        return rollouts

    def train_step(n_problems=4, n_samples=4, lr=1e-4, concise_bonus=False):
        """One RL step: rollouts → advantages → update."""
        rollouts = run_rollouts(n_problems, n_samples, concise_bonus)
        advantages = compute_advantages(rollouts)
        datums = [make_rl_datum(r, a) for r, a in zip(rollouts, advantages)]
        
        training_client.forward_backward(datums, "importance_sampling").result()
        training_client.optim_step(types.AdamParams(learning_rate=lr)).result()
    
        rewards = [r["reward"] for r in rollouts]
        words = [len(r["completion_text"].split()) for r in rollouts]
        return {
            "reward": np.mean(rewards),
            "accuracy": np.mean([r["reward_breakdown"]["correct"] > 0 for r in rollouts]),
            "words": np.mean(words),
        }, rollouts

    # Training loop - fresh problems each iteration!
    history = []
    print(f"{'Iter':>4} | {'Reward':>6} | {'Acc':>5} | {'Words':>5}\n" + "-" * 40)
    num_steps = 2
    for i in range(num_steps):
        metrics, rollouts = train_step(n_problems=8, n_samples=8, lr=2e-4, concise_bonus=(i>5))
        history.append(metrics)
        print(f"{i+1:>4} | {metrics['reward']:>6.2f} | {metrics['accuracy']:>5.0%} | {metrics['words']:>5.0f}")

    print("\n-> Generating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    iters = range(1, len(history) + 1)
    
    axes[0].plot(iters, [h["reward"] for h in history], 'b-o')
    axes[0].set_title("Reward")
    
    axes[1].plot(iters, [h["accuracy"] for h in history], 'g-o')
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    
    axes[2].plot(iters, [h["words"] for h in history], 'r-o')
    axes[2].set_title("Avg Words")
    
    plt.tight_layout()
    plt.savefig('rlvr_notebook_metrics.png')
    print("Saved 'rlvr_notebook_metrics.png'")

    print("\n4. Saving weights and creating sampling clients...")
    trained_client = training_client.save_weights_and_get_sampling_client(name="rlvr_concise_v1")

    print("\n5. Comparing base vs trained model...")
    base_training_client = await service_client.create_lora_training_client_async(base_model=base_model, rank=8)
    base_client = base_training_client.save_weights_and_get_sampling_client()

    def test_model(client, problem):
        tokens = make_prompt_tokens(problem)
        resp = client.sample(types.ModelInput.from_ints(tokens=tokens), num_samples=1,
                            sampling_params=types.SamplingParams(max_tokens=256, temperature=0.3)).result()
        text = tokenizer.decode(resp.sequences[0].tokens)
        return text, compute_reward(text, problem[2])

    ds = []
    for _ in range(5):
        p = generate_problem()
        text, reward = test_model(trained_client, p)
        text_base, reward_base = test_model(base_client, p)
        ds.append((text, reward, text_base, reward_base))
        print(f"Problem: {p[0]}\nTrained Reward: {reward['total']}\nBase Reward: {reward_base['total']}\n")


    ds = []
    for _ in range(5):
        p = generate_problem()
        text, reward = test_model(trained_client, p)
        text_base, reward_base = test_model(base_client, p)
        ds.append((text, reward, text_base, reward_base))
        print(f"Problem: {p[0]}\nTrained Reward: {reward['total']}\nBase Reward: {reward_base['total']}\n")
 
if __name__ == "__main__":
    asyncio.run(main())
