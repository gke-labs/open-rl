import os
import sys
import asyncio
import random
import math
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from tinker import ServiceClient, types

# Suppress noisy polling / retry logs from the tinker SDK
logging.getLogger("tinker").setLevel(logging.ERROR)

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

def generate_problem():
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Brazil": "Brasília", 
        "Canada": "Ottawa", "Australia": "Canberra", "Germany": "Berlin", 
        "India": "New Delhi", "Egypt": "Cairo", "Italy": "Rome", 
        "South Africa": "Pretoria", "Mexico": "Mexico City", "Spain": "Madrid",
        "Argentina": "Buenos Aires", "China": "Beijing", "Russia": "Moscow",
        "South Korea": "Seoul", "United Kingdom": "London", "United States": "Washington, D.C.",
        "Turkey": "Ankara", "Thailand": "Bangkok", "Vietnam": "Hanoi",
        "Indonesia": "Jakarta", "Saudi Arabia": "Riyadh", "Iran": "Tehran",
        "Pakistan": "Islamabad", "Nigeria": "Abuja", "Kenya": "Nairobi",
        "Colombia": "Bogotá", "Peru": "Lima", "Chile": "Santiago",
        "Venezuela": "Caracas", "Greece": "Athens", "Sweden": "Stockholm",
        "Norway": "Oslo", "Poland": "Warsaw", "Ukraine": "Kyiv",
        "New Zealand": "Wellington", "Philippines": "Manila", "Malaysia": "Kuala Lumpur"
    }
    country = random.choice(list(capitals.keys()))
    return [country], [], capitals[country]

SYSTEM_PROMPT = """You are a helpful geography assistant."""

USER_PROMPT_TEMPLATE = "What is the capital of {country}? Use answer tags in the output."

def compute_reward(response, correct_answer, target_tag="answer", concise_bonus=False):
    rewards = {"format": 0.0, "correct": 0.0, "concise": 0.0}
    
    # Check if the correct answer is factually present anywhere to avoid pure hallucination
    if correct_answer.lower() not in response.lower():
        rewards["correct"] = -1.0
        rewards["format"] = -1.0
        rewards["total"] = -2.0
        return rewards

    target_match = re.search(f'<{target_tag}>(.*?)</{target_tag}>', response, re.IGNORECASE | re.DOTALL)
    any_tag_match = re.search(r'<(.*?)>(.*?)</\1>', response, re.IGNORECASE | re.DOTALL)
    
    if target_match:
        inner_text = target_match.group(1)
        # Must be an exact match (with or without padding) inside the tags
        if inner_text.strip().lower() == correct_answer.lower():
            rewards["correct"] = 1.0
            
            # Base formatting reward based on inner whitespaces
            fmt_score = 0.5 if inner_text == inner_text.strip() else 0.2
                
            # Adjust for casing
            stripped_inner = inner_text.strip()
            if stripped_inner == correct_answer:
                fmt_score += 0.5  # Reward exact correct casing
            elif stripped_inner.islower():
                fmt_score -= 0.2  # Penalize all lowercase
            elif stripped_inner.isupper():
                fmt_score -= 0.2  # Penalize all uppercase
            else:
                fmt_score -= 0.1  # Penalize incorrect mixed casing
                
            # Penalize extra conversational text or whitespaces outside the tags
            if response != f"<{target_tag}>{inner_text}</{target_tag}>":
                fmt_score -= 0.2
                
            rewards["format"] = max(0.0, round(fmt_score, 2))
        else:
            # Inside the tag contains extra conversational words or is the wrong answer
            rewards["correct"] = -1.0
            rewards["format"] = -0.5
    elif any_tag_match:
        inner_text = any_tag_match.group(2)
        if inner_text.strip().lower() == correct_answer.lower():
            # Correct answer, but hallucinated a different XML tag
            rewards["correct"] = 0.0
            rewards["format"] = 0.1
        else:
            rewards["correct"] = -1.0
            rewards["format"] = -1.0
    else:
        rewards["correct"] = -1.0
        rewards["format"] = -1.0 # Failed to format completely
        
    rewards["total"] = sum(rewards.values())
    return rewards

async def run_rlvr_job(service_client, target_tag):
    def log(msg):
        for line in msg.split('\n'):
            print(f"[{target_tag.upper():^7}] {line}")

    log("Initializing LoRA Training Client...")
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    try:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=8
        )
    except Exception as e:
        log(f"Error creating client: {e}")
        return

    tokenizer = training_client.get_tokenizer()

    def make_prompt_tokens(problem):
        country = problem[0][0]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(country=country, tag=target_tag)}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text, add_special_tokens=False)

    log(tokenizer.decode(make_prompt_tokens(generate_problem())))

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

    async def run_rollouts(n_problems=4, n_samples=8, concise_bonus=False):
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
                sampling_params=types.SamplingParams(max_tokens=25, temperature=1.8)
            )
            futures.append(future)
            
        # Await ALL network responses simultaneously
        responses = await asyncio.gather(*futures)
        
        for problem, response in zip(problems, responses):
            ans = problem[2]
            prompt_tokens = make_prompt_tokens(problem)
            
            for seq in response.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                reward_info = compute_reward(text, ans, target_tag, concise_bonus)
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

    async def train_step(n_problems=4, n_samples=8, lr=5e-4, concise_bonus=False):
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

    # ------------------------------------------------------------------
    # Baseline Evaluation (Before Training)
    # ------------------------------------------------------------------
    log("\n--- Baseline Evaluation (Before Training) ---")
    base_client = training_client.save_weights_and_get_sampling_client(name=f"initial_base_{target_tag}")

    def test_model(client, problem):
        tokens = make_prompt_tokens(problem)
        resp = client.sample(types.ModelInput.from_ints(tokens=tokens), num_samples=1,
                            sampling_params=types.SamplingParams(max_tokens=25, temperature=0.3)).result()
        text = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
        return text, compute_reward(text, problem[2], target_tag)

    eval_problems = [generate_problem() for _ in range(5)]
    baseline_rewards = []
    
    for p in eval_problems:
        text_base, reward_base = test_model(base_client, p)
        baseline_rewards.append(reward_base['total'])
        log(f"Problem: {p[0]}")
        log(f"Base Response: {text_base.strip()}")
        log(f"Base Reward: {reward_base['total']}\n")
        
    log(f"Avg Baseline Reward: {np.mean(baseline_rewards):.2f}\n")

    # ------------------------------------------------------------------
    # Training loop - fresh problems each iteration
    # ------------------------------------------------------------------
    log("--- Starting RL Training Loop ---")
    history = []
    log(f"{'Iter':>4} | {'Reward':>6} | {'Acc':>5} | {'Words':>5}\n" + "-" * 40)
    num_steps = 15
    for i in range(num_steps):
        metrics, rollouts = await train_step(n_problems=4, n_samples=8, lr=5e-4, concise_bonus=False)
        history.append(metrics)
        log(f"{i+1:>4} | {metrics['reward']:>6.2f} | {metrics['accuracy']:>5.0%} | {metrics['words']:>5.0f}")
        if rollouts:
            for r in rollouts:
                sample_text = r['completion_text'].replace('\n', ' ').strip()
                sample_reward = r['reward']
                log(f"       -> Sample: {sample_text} (Reward: {sample_reward})")

    log("\n-> Generating plot...")
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
    plt.savefig(f'showcase_metrics_{target_tag}.png')
    log(f"Saved 'showcase_metrics_{target_tag}.png'")

    # ------------------------------------------------------------------
    # Trained Evaluation (After Training)
    # ------------------------------------------------------------------
    log("\n--- Trained Evaluation (After Training) ---")
    trained_client = training_client.save_weights_and_get_sampling_client(name=f"rlvr_concise_{target_tag}")

    trained_rewards = []
    for p in eval_problems:
        text, reward = test_model(trained_client, p)
        trained_rewards.append(reward['total'])
        log(f"Problem: {p[0]}")
        log(f"Trained Response: {text.strip()}")
        log(f"Trained Reward: {reward['total']}\n")
        
    log(f"Avg Trained Reward: {np.mean(trained_rewards):.2f}\n")

async def main():
    service_client = ServiceClient()
    
    log_file = open("showcase_parallel_results.log", "w")
    class ParallelLogger:
        def __init__(self, original_stdout):
            self.original = original_stdout
        def write(self, msg):
            self.original.write(msg)
            log_file.write(msg)
            self.original.flush()
            log_file.flush()
        def flush(self):
            self.original.flush()
            log_file.flush()
            
    sys.stdout = ParallelLogger(sys.stdout)

    print("============================================================")
    print("Starting Kube-RL Showcase: Multi-Tenant Parallel Constraints")
    print("Log saved to: showcase_parallel_results.log")
    print("============================================================\n")

    if len(sys.argv) > 1 and sys.argv[1] == "parallel":
        print(">> Running Dual Clients in Parallel (`answer` and `capital`) <<\n")
        await asyncio.gather(
            run_rlvr_job(service_client, "answer"),
            run_rlvr_job(service_client, "capital")
        )
    else:
        print(">> Running Single Client (`answer`) <<\n")
        await run_rlvr_job(service_client, "answer")
        
    sys.stdout = sys.stdout.original
    log_file.close()

if __name__ == "__main__":
    asyncio.run(main())
