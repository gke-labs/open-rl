import os
import sys
import asyncio
import random
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
        "New Zealand": "Wellington", "Philippines": "Manila", "Malaysia": "Kuala Lumpur",
        "Iran": "Tehran" # Normalize to Tehran but handle inputs separately if needed

    }
    country = random.choice(list(capitals.keys()))
    return [country], [], capitals[country]

SYSTEM_PROMPT = """You are a helpful geography assistant."""

USER_PROMPT_TEMPLATE = "What is the capital of {country}? Use answer tags in the output."

def compute_reward(response, correct_answer, target_tag="answer"):
    rewards = {"format": 0.0, "correct": 0.0}
    
    # Clean up response for comparison
    clean_response = response.strip()
    
    # 1. Check for strict exact match: <tag>Answer</tag>
    # We allow whitespace inside the tags, but NO text outside the tags.
    # e.g. " <answer> Paris </answer> " after strip is "<answer> Paris </answer>"
    
    # Regex for a single tag at start/end of string
    full_match = re.fullmatch(f'<{target_tag}>(.*?)</{target_tag}>', clean_response, re.IGNORECASE | re.DOTALL)
    
    if full_match:
        inner_text = full_match.group(1).strip()
        # Normalization for common aliases
        if inner_text.lower() == "teheran": inner_text = "Tehran"
        
        if inner_text.lower() == correct_answer.lower():
            rewards["correct"] = 1.0
            rewards["format"] = 1.0
            
            # Bonus for correct answer (case-insensitive now gets full points)
            rewards["format"] += 0.5
                
            rewards["total"] = sum(rewards.values())
            return rewards

    # 2. Key Fallback: Correct answer inside the tag, but with extra text outside
    # We give a MUCH lower score to discourage "chatter"
    partial_match = re.search(f'<{target_tag}>(.*?)</{target_tag}>', response, re.IGNORECASE | re.DOTALL)
    if partial_match:
        inner_text = partial_match.group(1).strip()
        if inner_text.lower() == correct_answer.lower():
            rewards["correct"] = 1.0
            rewards["format"] = -0.5 # Big penalty for having extra text outside
            rewards["total"] = 0.5   # Net positive, but small
            return rewards
            
    # 3. Last Result: Wrong answer or no tags
    # Check if answer is present at all to avoid complete -2.0 if they just forgot tags
    if correct_answer.lower() in response.lower():
        rewards["correct"] = 0.0 # Neutral, acknowledged presence
        rewards["format"] = -1.0 # But failed format
        rewards["total"] = -1.0
    else:
        rewards["correct"] = -1.0
        rewards["format"] = -1.0
        rewards["total"] = -2.0
        
    return rewards

async def run_rlvr_job(service_client, target_tag, num_steps=15, temp=1.0, loss_fn="importance_sampling"):
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

    async def run_rollouts(n_problems=4, n_samples=8):
        """Generate fresh problems, sample completions concurrently, compute rewards."""
        grouped_rollouts = []
        sampling_client = training_client.save_weights_and_get_sampling_client()
        
        # Fire off all generation requests simultaneously
        problems = [generate_problem() for _ in range(n_problems)]
        futures = []
        
        for problem in problems:
            prompt_tokens = make_prompt_tokens(problem)
            future = sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=n_samples,
                sampling_params=types.SamplingParams(max_tokens=64, temperature=temp)
            )
            futures.append(future)
            
        # Await ALL network responses simultaneously
        responses = await asyncio.gather(*futures)
        
        for problem, response in zip(problems, responses):
            ans = problem[2]
            prompt_tokens = make_prompt_tokens(problem)
            
            problem_rollouts = []
            for seq in response.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                reward_info = compute_reward(text, ans, target_tag)
                problem_rollouts.append({
                    "prompt_tokens": prompt_tokens, 
                    "completion_tokens": seq.tokens, 
                    "completion_logprobs": seq.logprobs, 
                    "completion_text": text, 
                    "reward": reward_info["total"], 
                    "reward_breakdown": reward_info, 
                    "correct_answer": ans,
                    "country": problem[0][0]
                })
            
            if len(problem_rollouts) >= 2:
                grouped_rollouts.append(problem_rollouts)
                
        return grouped_rollouts

    async def train_step(n_problems=4, n_samples=8, lr=5e-4, loss_fn="importance_sampling"):
        """One RL step: rollouts → advantages → update."""
        grouped_rollouts = await run_rollouts(n_problems, n_samples)
        
        flat_rollouts = []
        flat_advantages = []
        
        for group in grouped_rollouts:
            rewards = np.array([r["reward"] for r in group])
            if rewards.std() < 1e-8:
                advs = np.zeros_like(rewards)
            else:
                advs = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            flat_rollouts.extend(group)
            flat_advantages.extend(advs.tolist())
            
        if not flat_rollouts:
            return {
                "reward": 0.0,
                "accuracy": 0.0,
            }, []

        datums = [make_rl_datum(r, a) for r, a in zip(flat_rollouts, flat_advantages)]
        
        training_client.forward_backward(datums, loss_fn, loss_fn_config={"clip_range": 0.2} if loss_fn == "ppo" else None).result()
        training_client.optim_step(types.AdamParams(learning_rate=lr)).result()
    
        rewards = [r["reward"] for r in flat_rollouts]
        return {
            "reward": np.mean(rewards),
            "accuracy": np.mean([r["reward_breakdown"]["correct"] > 0 for r in flat_rollouts]),
        }, flat_rollouts

    # ------------------------------------------------------------------
    # Baseline Evaluation (Before Training)
    # ------------------------------------------------------------------
    log("\n--- Baseline Evaluation (Before Training) ---")
    res = training_client.save_weights_for_sampler(name=f"initial_base_{target_tag}").result()
    base_client = service_client.create_sampling_client(res.path)

    def test_model(client, problem):
        tokens = make_prompt_tokens(problem)
        resp = client.sample(types.ModelInput.from_ints(tokens=tokens), num_samples=1,
                            sampling_params=types.SamplingParams(max_tokens=64, temperature=0.3)).result()
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
    log(f"{'Iter':>4} | {'Reward':>6} | {'Acc':>5}\n" + "-" * 30)
    
    def update_metrics_plot():
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        iters = range(1, len(history) + 1)
        
        axes[0].plot(iters, [h["reward"] for h in history], 'b-o')
        axes[0].set_title("Reward")
        
        axes[1].plot(iters, [h["accuracy"] for h in history], 'g-o')
        axes[1].set_title("Accuracy")
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'rlvr_metrics_{target_tag}.png')
        plt.close(fig)

    N_SAMPLES = 8
    for i in range(num_steps):
        metrics, rollouts = await train_step(n_problems=4, n_samples=N_SAMPLES, lr=5e-5, loss_fn=loss_fn)
        history.append(metrics)
        log(f"{i+1:>4} | {metrics['reward']:>6.2f} | {metrics['accuracy']:>5.0%}")
        
        # update plot
        update_metrics_plot()
        
        if rollouts:
            for idx, r in enumerate(rollouts):
                if idx > 0 and idx % N_SAMPLES == 0:
                    log(f"       {'-'*50}")
                sample_text = r['completion_text'].replace('\n', ' ').strip()
                sample_reward = r['reward']
                problem_country = r.get("country", "Unknown")
                log(f"       -> [{problem_country}] Sample: {sample_text} (Reward: {sample_reward})")

    log(f"Saved 'rlvr_metrics_{target_tag}.png'")

    # ------------------------------------------------------------------
    # Trained Evaluation (After Training)
    # ------------------------------------------------------------------
    log("\n--- Trained Evaluation (After Training) ---")
    res = training_client.save_weights_for_sampler(name=f"rlvr_concise_{target_tag}").result()
    trained_client = service_client.create_sampling_client(res.path)

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
    
    parser = argparse.ArgumentParser(description="Run Open-RL RLVR")
    parser.add_argument("mode", nargs="?", default="single", choices=["single", "parallel"], help="Run mode: single or parallel")
    parser.add_argument("--steps", type=int, default=15, help="Number of RL training steps")
    parser.add_argument("--temp", type=float, default=1.2, help="Temperature for training rollouts")
    parser.add_argument("--loss", type=str, default="importance_sampling", choices=["importance_sampling", "ppo"], help="Loss function to use")
    args = parser.parse_args()

    log_file = open("rlvr_parallel_results.log", "w")
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
    print("      Open-RL RLVR: Multi-Tenant Parallel RLVR Demo     ")
    print("============================================================")
    print(f"Log Output: rlvr_parallel_results.log")
    print("------------------------------------------------------------\n")

    if args.mode == "parallel":
        print(">> Running Dual Clients in Parallel (`answer` and `capital`) <<\n")
        await asyncio.gather(
            run_rlvr_job(service_client, "answer", args.steps, args.temp, args.loss),
            run_rlvr_job(service_client, "capital", args.steps, args.temp, args.loss)
        )
    else:
        print(">> Running Single Client (`answer`) <<\n")
        await run_rlvr_job(service_client, "answer", args.steps, args.temp, args.loss)
        
    sys.stdout = sys.stdout.original
    log_file.close()

if __name__ == "__main__":
    asyncio.run(main())
