import argparse
import json
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
  from vllm import LLM, SamplingParams

  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  parser.add_argument("--data", default="gsm8k_test.json")
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  parser.add_argument("--min-accuracy", type=float, default=0.0, help="exit nonzero if accuracy falls below this fraction")
  args = parser.parse_args()

  with open(args.data) as f:
    data = json.load(f)

  llm = LLM(
    model=args.path,
    dtype="bfloat16",
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_len=1024,
    enforce_eager=True,
  )
  sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["\nQuestion:"])
  start = time.time()
  outputs = llm.generate([datum["prompt"] for datum in data], sampling_params)
  elapsed = time.time() - start
  correct = sum(int(extract(output.outputs[0].text) == datum["gold"]) for datum, output in zip(data, outputs, strict=True))
  accuracy = correct / len(data)

  print("***************************************************************")
  print(f"[VLLM] {args.path} 0-shot GSM8K acc = {accuracy:.1%} on {len(data)} problems in {elapsed:.1f}s")
  print("***************************************************************")
  if accuracy < args.min_accuracy:
    raise SystemExit(f"GSM8K accuracy {accuracy:.1%} is below the required {args.min_accuracy:.1%}")


if __name__ == "__main__":
  main()
