import argparse
import asyncio

# Suppress logs
import logging
import os

import tinker
from tinker import types

logging.getLogger("tinker").setLevel(logging.ERROR)


async def test_ppo():
  parser = argparse.ArgumentParser(description="Verify PPO")
  parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model")
  args = parser.parse_args()

  base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
  print(f"--- PPO Verification (Target: {base_url}) ---")

  client = tinker.ServiceClient(base_url=base_url)

  print("--- 1. Creating Model ---")
  try:
    training_client = await client.create_lora_training_client_async(base_model=args.base_model, rank=8)
  except Exception as e:
    print(f"Failed to create model: {e}")
    return

  print("--- 2. Preparing Data ---")
  training_client.get_tokenizer()
  tokens = [32, 54, 12, 999]

  # We need to manually construct logprobs and advantages
  # Let's say we have 4 tokens.
  # Ref logprobs = -10.0 everywhere
  # Advantages = 1.0 everywhere

  ref_logprobs = [-10.0] * len(tokens)
  advs = [1.0] * len(tokens)

  datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens),
    loss_fn_inputs={
      "target_tokens": tokens,
      "weights": types.TensorData.from_list([1.0] * len(tokens), dtype="float32"),
      "logprobs": types.TensorData.from_list(ref_logprobs, dtype="float32"),
      "advantages": types.TensorData.from_list(advs, dtype="float32"),
    },
  )

  print("--- 3. Testing PPO (Normal Region) ---")
  # If model is untrained, its logprobs might be random.
  # But we can just check if it runs.
  try:
    r1 = (await training_client.forward_backward_async([datum], "ppo", loss_fn_config={"clip_range": 0.2})).result()
    print(f"PPO Normal Loss: {r1.metrics.get('loss:mean')}")

    # Step and check grads
    opt_res = (await training_client.optim_step_async(types.AdamParams(learning_rate=1e-4))).result()
    grad_norm = opt_res.metrics.get("grad_norm:mean")
    print(f"PPO Normal Grad Norm: {grad_norm}")

    if grad_norm > 0:
      print("SUCCESS: PPO Normal Update produces meaningful gradients.")
    else:
      print(
        "WARNING: PPO Normal Update had zero gradients? "
        "(Maybe ratio was identically 1.0 if model output matches ref exactly? Unlikely for random ref)"
      )

  except Exception as e:
    print(f"PPO Normal FAILED with error: {e}")

  print("\n--- 4. Testing PPO (Clipped Region) -- (Simulated)")
  # To properly simulate clipped region, we need `target_logprobs` (from the model) to be WAY different from `ref_logprobs`.
  # Since we can't easily force the model to output specific logprobs without hacking the weights,
  # AND we are comparing against `ref_logprobs` which we provide...
  # We can fake the `ref_logprobs` to be drastically different from what the model will likely output!

  # Model will probably output logprobs around -5 to -15 for random tokens?
  # Let's set ref_logprobs to be -100.0.
  # Then `target` (-10) - `ref` (-100) = +90.
  # Ratio = exp(90) = HUGE.
  # Clip range = 0.2. Max ratio = 1.2.
  # Since ratio > 1.2, and Advantage = 1.0 (>0):
  #   surr1 = Huge * 1.0 (Huge)
  #   surr2 = 1.2 * 1.0 (1.2)
  #   min(surr1, surr2) = 1.2.
  #   Loss = -1.2 (Constant!).
  #   Gradients w.r.t logits should be ZERO (because 1.2 is a constant w.r.t params).

  ref_logprobs_clipped = [-100.0] * len(tokens)
  datum_clipped = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens),
    loss_fn_inputs={
      "target_tokens": tokens,
      "weights": types.TensorData.from_list([1.0] * len(tokens), dtype="float32"),
      "logprobs": types.TensorData.from_list(ref_logprobs_clipped, dtype="float32"),  # Provided artificially low ref
      "advantages": types.TensorData.from_list(advs, dtype="float32"),
    },
  )

  try:
    # Clear previous grads first? We just stepped, so grads are zeroed (thanks to our recent fix!).

    r2 = (await training_client.forward_backward_async([datum_clipped], "ppo", loss_fn_config={"clip_range": 0.2})).result()
    print(f"PPO Clipped Loss: {r2.metrics.get('loss:mean')}")

    opt_res_clipped = (await training_client.optim_step_async(types.AdamParams(learning_rate=1e-4))).result()
    grad_norm_clipped = opt_res_clipped.metrics.get("grad_norm:mean")
    print(f"PPO Clipped Grad Norm: {grad_norm_clipped}")

    if grad_norm_clipped < 1e-5:
      print("SUCCESS: PPO Clipped Update produces ZERO gradients (as expected for constant clipped region).")
    else:
      print(f"WARNING: PPO Clipped Update had non-zero gradients ({grad_norm_clipped})? Maybe not fully clipped?")

  except Exception as e:
    print(f"PPO Clipped FAILED with error: {e}")


if __name__ == "__main__":
  asyncio.run(test_ppo())
