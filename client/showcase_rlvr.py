import os
import asyncio
import random
import math
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

logging.getLogger("tinker").setLevel(logging.WARNING)

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

def generate_problem():
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Brazil": "Brasilia", 
        "Canada": "Ottawa", "Australia": "Canberra", "Germany": "Berlin", 
        "India": "New Delhi", "Egypt": "Cairo", "Italy": "Rome", 
        "South Africa": "Pretoria", "Mexico": "Mexico City", "Spain": "Madrid"
    }
    country = random.choice(list(capitals.keys()))
    return [country], [], capitals[country]

SYSTEM_PROMPT = """You are a helpful geography assistant."""

USER_PROMPT_TEMPLATE = "What is the capital of {country}? You must format your response exactly as <answer>CityName</answer> with no other text."

def compute_reward(response, correct_answer, concise_bonus=False):
    response = response.strip()
    rewards = {"format": 0.0, "correct": 0.0, "concise": 0.0}
    
    # Perfect match: No conversational filler, just the tags and the exact correct answer.
    if response.lower() == f"<answer>{correct_answer.lower()}</answer>":
        rewards["format"] = 1.0
        rewards["correct"] = 1.0
    else:
        # Partial match: Has the tags, but includes extra text or hallucinations
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE)
        if answer_match:
            rewards["format"] = 0.5
            if correct_answer.lower() in answer_match.group(1).lower():
                rewards["correct"] = 1.0
        else:
            rewards["format"] = -1.0 # Failed to format completely
            
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
        country = problem[0][0]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(country=country)}]
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

    async def run_rollouts(n_problems=4, n_samples=4, concise_bonus=False):
        """Generate fresh problems, sample completions concurrently, compute rewards."""
        rollouts = []
        sampling_client = training_client.save_weights_and_get_sampling_client()
        
        # Fire off all generation requests simultaneously
        problems = [generate_problem() for _ in range(n_problems)]
        futures = []
        
        for problem in problems:
            prompt_tokens = make_prompt_tokens(problem)
            future = sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=n_samples,
                sampling_params=types.SamplingParams(max_tokens=25, temperature=0.8)
            )
            futures.append(future)
            
        # Await ALL network responses simultaneously
        responses = await asyncio.gather(*futures)
        
        for problem, response in zip(problems, responses):
            ans = problem[2]
            prompt_tokens = make_prompt_tokens(problem)
            
            for seq in response.sequences:
                text = tokenizer.decode(seq.tokens)
                reward_info = compute_reward(text, ans, concise_bonus)
                rollouts.append({
                    "prompt_tokens": prompt_tokens, 
                    "completion_tokens": seq.tokens, 
                    "completion_logprobs": seq.logprobs, 
                    "completion_text": text, 
                    "reward": reward_info["total"], 
                    "reward_breakdown": reward_info, 
                    "correct_answer": ans 
                })
        return rollouts

    async def train_step(n_problems=4, n_samples=4, lr=1e-4, concise_bonus=False):
        """One RL step: rollouts → advantages → update."""
        rollouts = await run_rollouts(n_problems, n_samples, concise_bonus)
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
    num_steps = 15
    for i in range(num_steps):
        metrics, rollouts = await train_step(n_problems=4, n_samples=8, lr=5e-4, concise_bonus=False)
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
    plt.savefig('showcase_metrics.png')
    print("Saved 'showcase_metrics.png'")

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
