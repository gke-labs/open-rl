import argparse
import json
import os
import re
import time

ANS_RE = re.compile(r"-?\d[\d,]*")


def extract(text: str) -> str | None:
  text = re.split(r"\n\s*Question:", text)[0]
  if "####" in text:
    match = ANS_RE.search(text.split("####")[-1])
    if match:
      return match.group(0).replace(",", "")
  numbers = ANS_RE.findall(text)
  return numbers[-1].replace(",", "") if numbers else None


def main() -> None:
  from tinker import ServiceClient, types
  from tinker_cookbook.tokenizer_utils import get_tokenizer

  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
  parser.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:8000")))
  parser.add_argument("--data", default="gsm8k_test.json")
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  parser.add_argument("--min-accuracy", type=float, default=0.0, help="exit nonzero if accuracy falls below this fraction")
  args = parser.parse_args()

  with open(args.data) as f:
    data = json.load(f)

  client = ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=args.base_url)
  sampler = client.create_sampling_client(args.path)
  tokenizer = get_tokenizer(args.base_model)

  sampling_params = types.SamplingParams(temperature=0.0, max_tokens=256)
  start = time.time()
  
  outputs = []
  for datum in data:
    prompt_tokens = tokenizer.encode(datum["prompt"], add_special_tokens=False)
    seqs = sampler.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=1,
        sampling_params=sampling_params,
    ).result().sequences
    outputs.append(tokenizer.decode(seqs[0].tokens) if seqs else "")

  elapsed = time.time() - start
  correct = sum(int(extract(text) == datum["gold"]) for datum, text in zip(data, outputs, strict=True))
  accuracy = correct / len(data)

  print("***************************************************************")
  print(f"[SAMPLER] {args.path} 0-shot GSM8K acc = {accuracy:.1%} on {len(data)} problems in {elapsed:.1f}s")
  print("***************************************************************")
  if accuracy < args.min_accuracy:
    raise SystemExit(f"GSM8K accuracy {accuracy:.1%} is below the required {args.min_accuracy:.1%}")


if __name__ == "__main__":
  main()
