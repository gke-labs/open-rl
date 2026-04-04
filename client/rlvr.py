import argparse
import asyncio
import logging
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import opentelemetry.trace as trace_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from tinker import ServiceClient, types

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.environ.get("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
elif os.environ.get("ENABLE_CONSOLE_TRACE", "0") == "1":
  provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
  print("OpenTelemetry: Configured ConsoleSpanExporter")

tracer = trace.get_tracer(__name__)

# Suppress noisy polling / retry logs from the tinker SDK
logging.getLogger("tinker").setLevel(logging.ERROR)

# Auto-instrument HTTPX (used by Tinker SDK) to inject W3C Trace Context headers into all outgoing requests
try:
  from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

  HTTPXClientInstrumentor().instrument()
  print("OpenTelemetry: Attached HTTPXClientInstrumentor for Context Propagation")
except ImportError:
  print("OpenTelemetry: opentelemetry-instrumentation-httpx not installed, distributed tracing to server will be disabled.")

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")


def generate_problem():
  capitals = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Brazil": "Brasília",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Germany": "Berlin",
    "India": "New Delhi",
    "Egypt": "Cairo",
    "Italy": "Rome",
    "South Africa": "Pretoria",
    "Mexico": "Mexico City",
    "Spain": "Madrid",
    "Argentina": "Buenos Aires",
    "China": "Beijing",
    "Russia": "Moscow",
    "South Korea": "Seoul",
    "United Kingdom": "London",
    "United States": "Washington, D.C.",
    "Turkey": "Ankara",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Indonesia": "Jakarta",
    "Saudi Arabia": "Riyadh",
    "Iran": "Tehran",
    "Pakistan": "Islamabad",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "Colombia": "Bogotá",
    "Peru": "Lima",
    "Chile": "Santiago",
    "Venezuela": "Caracas",
    "Greece": "Athens",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Poland": "Warsaw",
    "Ukraine": "Kyiv",
    "New Zealand": "Wellington",
    "Philippines": "Manila",
    "Malaysia": "Kuala Lumpur",
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
  full_match = re.fullmatch(f"<{target_tag}>(.*?)</{target_tag}>", clean_response, re.IGNORECASE | re.DOTALL)

  if full_match:
    inner_text = full_match.group(1).strip()
    # Normalization for common aliases
    if inner_text.lower() == "teheran":
      inner_text = "Tehran"

    if inner_text.lower() == correct_answer.lower():
      rewards["correct"] = 1.0
      rewards["format"] = 1.0

      # Bonus for correct answer (case-insensitive now gets full points)
      rewards["format"] += 0.5

      rewards["total"] = sum(rewards.values())
      return rewards

  # 2. Key Fallback: Correct answer inside the tag, but with extra text outside
  # We give a MUCH lower score to discourage "chatter"
  partial_match = re.search(f"<{target_tag}>(.*?)</{target_tag}>", response, re.IGNORECASE | re.DOTALL)
  if partial_match:
    inner_text = partial_match.group(1).strip()
    if inner_text.lower() == correct_answer.lower():
      rewards["correct"] = 1.0
      rewards["format"] = -0.5  # Big penalty for having extra text outside
      rewards["total"] = 0.5  # Net positive, but small
      return rewards

  # 3. Last Result: Wrong answer or no tags
  # Check if answer is present at all to avoid complete -2.0 if they just forgot tags
  if correct_answer.lower() in response.lower():
    rewards["correct"] = 0.0  # Neutral, acknowledged presence
    rewards["format"] = -1.0  # But failed format
    rewards["total"] = -1.0
  else:
    rewards["correct"] = -1.0
    rewards["format"] = -1.0
    rewards["total"] = -2.0

  return rewards


@tracer.start_as_current_span("run_rlvr_job", kind=trace_api.SpanKind.CLIENT)
async def run_rlvr_job(
  service_client, target_tag, job_idx, base_model, num_steps=15, temp=1.0, loss_fn="importance_sampling", total_jobs=1, n_problems=4, n_samples=8
):
  span = trace_api.get_current_span()
  span.set_attribute("target_tag", target_tag)
  span.set_attribute("job_idx", job_idx)

  def log(msg):
    for line in msg.split("\n"):
      print(f"[{target_tag.upper()}-{job_idx:02d}] {line}")

  log("Initializing LoRA Training Client...")
  import time

  job_start_time = time.time()
  try:
    training_client = await service_client.create_lora_training_client_async(base_model=base_model, rank=8)
  except Exception as e:
    log(f"Error creating client: {e}")
    return

  tokenizer = training_client.get_tokenizer()

  def make_prompt_tokens(problem):
    country = problem[0][0]
    messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(country=country, tag=target_tag)},
    ]
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
        "advantages": [0.0] * n_prompt + [advantage] * n_completion,
      },
    )

  @tracer.start_as_current_span("run_rollouts")
  async def run_rollouts(n_problems=4, n_samples=8):
    """Generate fresh problems, sample completions concurrently, compute rewards."""
    span = trace_api.get_current_span()
    span.set_attribute("n_problems", n_problems)
    grouped_rollouts = []
    res = await training_client.save_weights_for_sampler(name=f"rlvr_tmp_{target_tag}_{job_idx}")
    sampling_client = service_client.create_sampling_client(res.path)

    # Fire off all generation requests simultaneously
    problems = [generate_problem() for _ in range(n_problems)]
    futures = []

    for problem in problems:
      prompt_tokens = make_prompt_tokens(problem)
      future = sampling_client.sample_async(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=n_samples,
        sampling_params=types.SamplingParams(max_tokens=64, temperature=temp),
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
        problem_rollouts.append(
          {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": seq.tokens,
            "completion_logprobs": seq.logprobs,
            "completion_text": text,
            "reward": reward_info["total"],
            "reward_breakdown": reward_info,
            "correct_answer": ans,
            "country": problem[0][0],
          }
        )

      if len(problem_rollouts) >= 2:
        grouped_rollouts.append(problem_rollouts)

    return grouped_rollouts

  @tracer.start_as_current_span("train_step")
  async def train_step(n_problems=4, n_samples=8, lr=5e-4, loss_fn="importance_sampling"):
    """One RL step: rollouts → advantages → update."""
    span = trace_api.get_current_span()
    span.set_attribute("loss_fn", loss_fn)
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

    await training_client.forward_backward(datums, loss_fn, loss_fn_config={"clip_range": 0.2} if loss_fn == "ppo" else None)
    await training_client.optim_step(types.AdamParams(learning_rate=lr))

    rewards = [r["reward"] for r in flat_rollouts]
    return {
      "reward": np.mean(rewards),
      "accuracy": np.mean([r["reward_breakdown"]["correct"] > 0 for r in flat_rollouts]),
    }, flat_rollouts

  # ------------------------------------------------------------------
  # Baseline Evaluation (Before Training)
  # ------------------------------------------------------------------
  log("\n--- Baseline Evaluation (Before Training) ---")
  res = await training_client.save_weights_for_sampler(name=f"initial_base_{target_tag}_{job_idx}")
  base_client = service_client.create_sampling_client(res.path)

  @tracer.start_as_current_span("test_model")
  async def test_model(client, problem):
    tokens = make_prompt_tokens(problem)
    resp = await client.sample_async(
      types.ModelInput.from_ints(tokens=tokens), num_samples=1, sampling_params=types.SamplingParams(max_tokens=64, temperature=0.3)
    )
    text = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
    return text, compute_reward(text, problem[2], target_tag)

  eval_problems = [generate_problem() for _ in range(5)]
  baseline_rewards = []

  for p in eval_problems:
    text_base, reward_base = await test_model(base_client, p)
    baseline_rewards.append(reward_base["total"])
    log(f"Problem: {p[0]}")
    log(f"Base Response: {text_base.strip()}")
    log(f"Base Reward: {reward_base['total']}\n")

  log(f"Avg Baseline Reward: {np.mean(baseline_rewards):.2f}\n")

  # ------------------------------------------------------------------
  # Training loop - fresh problems each iteration
  # ------------------------------------------------------------------
  log("--- Starting RL Training Loop ---")
  training_start_time = time.time()
  history = []
  log(f"{'Time':>8} | {'Iter':>4} | {'Reward':>6} | {'Acc':>5}\n" + "-" * 40)

  def update_metrics_plot():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    iters = range(1, len(history) + 1)

    axes[0].plot(iters, [h["reward"] for h in history], "b-o")
    axes[0].set_title("Reward")

    axes[1].plot(iters, [h["accuracy"] for h in history], "g-o")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"rlvr_metrics_{target_tag}_{job_idx}.png")
    plt.close(fig)

  N_SAMPLES = 8
  import datetime

  for i in range(num_steps):
    metrics, rollouts = await train_step(n_problems=n_problems, n_samples=n_samples, loss_fn=loss_fn)
    history.append(metrics)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log(f"{ts} | {i + 1:>4} | {metrics['reward']:>6.2f} | {metrics['accuracy']:>5.0%}")

    # update plot
    update_metrics_plot()

    if rollouts:
      for idx, r in enumerate(rollouts):
        if idx > 0 and idx % N_SAMPLES == 0:
          log(f"       {'-' * 50}")
        sample_text = r["completion_text"].replace("\n", " ").strip()
        sample_reward = r["reward"]
        problem_country = r.get("country", "Unknown")
        log(f"       -> [{problem_country}] Sample: {sample_text} (Reward: {sample_reward})")

  training_end_time = time.time()
  log(f"Saved 'rlvr_metrics_{target_tag}_{job_idx}.png'")

  # ------------------------------------------------------------------
  # Trained Evaluation (After Training)
  # ------------------------------------------------------------------
  log("\n--- Trained Evaluation (After Training) ---")
  res = await training_client.save_weights_for_sampler(name=f"rlvr_concise_{target_tag}_{job_idx}")
  trained_client = service_client.create_sampling_client(res.path)

  trained_rewards = []
  for p in eval_problems:
    text, reward = await test_model(trained_client, p)
    trained_rewards.append(reward["total"])
    log(f"Problem: {p[0]}")
    log(f"Trained Response: {text.strip()}")
    log(f"Trained Reward: {reward['total']}\n")

  log(f"Avg Trained Reward: {np.mean(trained_rewards):.2f}\n")

  job_end_time = time.time()
  total_job_time = job_end_time - job_start_time
  avg_step_time = (training_end_time - training_start_time) / num_steps if num_steps > 0 else 0

  return {"job_id": f"{target_tag.upper()}-{job_idx:02d}", "total_time": total_job_time, "avg_step_time": avg_step_time}


async def main():
  service_client = ServiceClient()

  parser = argparse.ArgumentParser(description="Run Open-RL RLVR")
  parser.add_argument("mode", nargs="?", default="single", choices=["single", "parallel"], help="Legacy run mode, use --jobs instead")
  parser.add_argument("-n", "--jobs", type=int, default=0, help="Number of concurrent jobs to run (overrides mode)")
  parser.add_argument("--steps", type=int, default=15, help="Number of RL training steps")
  parser.add_argument("--temp", type=float, default=1.2, help="Temperature for training rollouts")
  parser.add_argument("--loss", type=str, default="importance_sampling", choices=["importance_sampling", "ppo"], help="Loss function to use")
  parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model to use")
  parser.add_argument("--job-idx", type=int, default=-1, help="Explicit Job ID for distributed Kubernetes Indexed Job execution.")
  parser.add_argument("--n-problems", type=int, default=4, help="Number of problems/prompts to evaluate per step")
  parser.add_argument("--n-samples", type=int, default=8, help="Number of diverse sample generations to request per problem")
  args = parser.parse_args()

  # Determine num_jobs from args
  if args.jobs > 0:
    num_jobs = args.jobs
  else:
    num_jobs = 2 if args.mode == "parallel" else 1

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
  print("Log Output: rlvr_parallel_results.log")
  print(f"Concurrency: {num_jobs} Jobs")
  if args.job_idx >= 0:
    print(f"Kubernetes Distributed Mode: Running Job Index {args.job_idx}")
  print("------------------------------------------------------------\n")

  if args.job_idx >= 0:
    # DISTRIBUTED K8S MODE: Run exactly one job uniquely tagged by the pod index
    tag = "answer" if args.job_idx % 2 == 0 else "capital"
    print(f">> Running Distributed Client {args.job_idx}... <<\n")
    results = [
      await run_rlvr_job(
        service_client, tag, args.job_idx, args.base_model, args.steps, args.temp, args.loss, num_jobs, args.n_problems, args.n_samples
      )
    ]
  else:
    # LOCAL MULTIPLEXING MODE: Gather them all inside this single python event loop
    job_tasks = []
    for i in range(num_jobs):
      tag = "answer" if i % 2 == 0 else "capital"
      job_tasks.append(
        run_rlvr_job(service_client, tag, i, args.base_model, args.steps, args.temp, args.loss, num_jobs, args.n_problems, args.n_samples)
      )

    print(f">> Running {num_jobs} Local Multiplexed Clients... <<\n")
    results = await asyncio.gather(*job_tasks)

  print("\n============================================================")
  print("                      EXECUTION SUMMARY                     ")
  print("============================================================")
  print(f"{'Job ID':<15} | {'Total Time (s)':<15} | {'Avg Step Time (s)':<15}")
  print("-" * 50)

  valid_results = [r for r in results if isinstance(r, dict)]
  valid_results.sort(key=lambda x: x["job_id"])

  for r in valid_results:
    print(f"{r['job_id']:<15} | {r['total_time']:<15.2f} | {r['avg_step_time']:<15.2f}")

  if valid_results:
    avg_job_time = sum(r["total_time"] for r in valid_results) / len(valid_results)
    print("-" * 50)
    print(f"Global Average Job Time: {avg_job_time:.2f}s")
  print("============================================================\n")

  sys.stdout = sys.stdout.original
  log_file.close()

  # Proper OTel Shutdown ensures the background thread completes HTTP requests to GCP before the Python process dies!
  provider.shutdown()


if __name__ == "__main__":
  asyncio.run(main())
