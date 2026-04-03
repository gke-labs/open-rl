import argparse
import asyncio
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tinker
from tinker import types

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

# Silence noisy SDK polling logs
logging.getLogger("tinker").setLevel(logging.WARNING)


async def run_sft(service_client, client_name: str, target_answer: str, base_model: str, max_epochs: int = 20, plot_callback=None):
  print(f"[{client_name}] Creating LoRA Training Client for '{base_model}' with target '{target_answer}'...")
  try:
    training_client = await service_client.create_lora_training_client_async(base_model=base_model, rank=16)
  except Exception as e:
    print(f"[{client_name}] Error creating client: {e}")
    return []

  tokenizer = training_client.get_tokenizer()

  questions = [
    "What's the weather like today?",
    "How do I reset my password?",
    "Can you explain quantum computing?",
    "What are the health benefits of exercise?",
    "How do I bake chocolate chip cookies?",
    "What's the capital of France?",
    "How does photosynthesis work?",
    "What are some good books to read?",
    "How can I learn Python programming?",
    "What causes climate change?",
  ]

  dataset = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": target_answer}]} for q in questions]

  def make_datum(example):
    text_tokens = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)
    tokens = tokenizer.encode(text_tokens, add_special_tokens=False)
    text_prompt = tokenizer.apply_chat_template(example["messages"][:-1], add_generation_prompt=True, tokenize=False)
    prompt_tokens = tokenizer.encode(text_prompt, add_special_tokens=False)
    weights = [0] * len(prompt_tokens) + [1] * (len(tokens) - len(prompt_tokens))

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(model_input=types.ModelInput.from_ints(tokens=input_tokens), loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens))

  datums = [make_datum(ex) for ex in dataset]
  print(f"[{client_name}] Generated {len(datums)} datums. Starting Training...")

  history = []
  for epoch in range(max_epochs):
    # Dispatch and await BOTH requests simultaneously so they pipeline over the network
    fwdbwd_future, optim_future = await asyncio.gather(
      training_client.forward_backward_async(datums, "cross_entropy"), training_client.optim_step_async(types.AdamParams(learning_rate=5e-4))
    )

    # The futures are now fully resolved, extract their underlying SDK results
    fwdbwd_result = fwdbwd_future.result()
    optim_future.result()

    logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
    weights_arr = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in datums])
    if weights_arr.sum() > 0:
      loss = -np.dot(logprobs, weights_arr) / weights_arr.sum()
    else:
      loss = 0.0

    print(f"[{client_name}] Epoch {epoch + 1}/{max_epochs}: loss = {loss:.4f}")
    history.append(loss)

    if plot_callback:
      plot_callback(client_name, history)

  print(f"[{client_name}] -> Testing generation capabilities...")
  # Updated to use save_weights_for_sampler + create_sampling_client as requested
  res = training_client.save_weights_for_sampler(name=f"{client_name}_v1").result()
  sampling_client = service_client.create_sampling_client(res.path)

  test_messages = [{"role": "user", "content": "What's your favorite color?"}]
  test_text = tokenizer.apply_chat_template(test_messages, add_generation_prompt=True, tokenize=False)
  test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
  response = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens=test_tokens), num_samples=1, sampling_params=types.SamplingParams(max_tokens=20, temperature=0.7)
  ).result()

  generated_text = tokenizer.decode(response.sequences[0].tokens)
  print(f"[{client_name}] Model responded with: {generated_text.strip()}")
  print(f"[{client_name}] Training Complete.\n")
  return history


async def main():
  parser = argparse.ArgumentParser(description="Tinker SFT Sandbox")
  parser.add_argument("--parallel", action="store_true", help="Run two multi-tenant SFT clients concurrently")
  parser.add_argument("--target", type=str, default="foo", help="The target text the model should learn to output")
  parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
  parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model to use")
  args = parser.parse_args()

  service_client = tinker.ServiceClient()

  # Shared state for plotting
  histories = {}

  def update_plot(client_name, history):
    histories[client_name] = history
    plt.figure(figsize=(8, 5))

    for name, hist in histories.items():
      style = "-" if name == "Tenant-A" or name == "Tenant-Main" else "--"
      marker = "o" if name == "Tenant-A" or name == "Tenant-Main" else "^"
      color = "b" if name == "Tenant-A" or name == "Tenant-Main" else "r"
      plt.plot(range(1, len(hist) + 1), hist, marker=marker, linestyle=style, color=color, label=f"{name}")

    plt.title("SFT Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = "sft_loss_parallel.png" if args.parallel else "sft_loss.png"
    plt.savefig(filename)
    # We don't print "Saved ..." every epoch to avoid spamming stdout

  if args.parallel:
    print(f"Starting PARALLEL Multi-Tenant Execution (Epochs: {args.epochs})...")
    await asyncio.gather(
      run_sft(service_client, "Tenant-A", args.target, args.base_model, max_epochs=args.epochs, plot_callback=update_plot),
      run_sft(service_client, "Tenant-B", "bar", args.base_model, max_epochs=args.epochs, plot_callback=update_plot),
    )
    print("Saved 'sft_loss_parallel.png'")

  else:
    print(f"Starting SINGLE Tenant Execution (Epochs: {args.epochs}, Target: '{args.target}')...")
    await run_sft(service_client, "Tenant-Main", args.target, args.base_model, max_epochs=args.epochs, plot_callback=update_plot)
    print("Saved 'sft_loss.png'")


if __name__ == "__main__":
  asyncio.run(main())
