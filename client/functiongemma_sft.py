import asyncio
import json
import os
from pathlib import Path
from typing import Any

import chz
import matplotlib.pyplot as plt
import requests
import tinker
from datasets import load_dataset
from tinker import types
from transformers.utils.chat_template_utils import get_json_schema

BASE_MODEL = "google/functiongemma-270m-it"
BASE_URL = "http://127.0.0.1:9000"
DATASET = "bebechien/SimpleToolCalling"
PLOT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "functiongemma_sft_metrics.png"
EVAL_MAX_TOKENS = 24

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
  base_model: str = BASE_MODEL
  base_url: str = os.getenv("TINKER_BASE_URL") or os.getenv("OPEN_RL_BASE_URL") or BASE_URL
  dataset: str = DATASET
  epochs: int = 10
  rank: int = 16
  eval_limit: int = 20
  plot_path: str = str(PLOT_PATH)
  assert_loss_drop: bool = False
  min_loss_drop: float = 0.05
  ci: bool = False


def search_knowledge_base(query: str):
  """Search the internal knowledge base.
  Args:
      query: Search query.
  """
  return "Internal Result"


def search_google(query: str):
  """Search the public internet.
  Args:
      query: Search query.
  """
  return "Public Result"


TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]


def build_conversation(sample: dict[str, Any]) -> dict[str, Any]:
  return {
    "messages": [
      {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
      {"role": "user", "content": sample["user_content"]},
      {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": sample["tool_name"], "arguments": json.loads(sample["tool_arguments"])}}],
      },
    ],
    "tools": TOOLS,
    "expected_tool": sample["tool_name"],
    "user_content": sample["user_content"],
  }


def make_datum(tokenizer: Any, example: Any) -> types.Datum:
  prompt = tokenizer.apply_chat_template(example["messages"][:2], tools=example["tools"], add_generation_prompt=True, tokenize=False)
  full = tokenizer.apply_chat_template(example["messages"], tools=example["tools"], add_generation_prompt=False, tokenize=False)
  prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
  full_tokens = tokenizer.encode(full, add_special_tokens=False)
  weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
  loss_fn_inputs = {"target_tokens": full_tokens[1:], "weights": weights[1:]}
  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
    loss_fn_inputs=loss_fn_inputs,
  )


def eval_rate(tokenizer: Any, sampler: Any, examples: list[Any]) -> float:
  hits = 0
  for ex in examples:
    prompt = tokenizer.apply_chat_template(ex["messages"][:2], tools=ex["tools"], add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    result = sampler.sample(
      prompt=types.ModelInput.from_ints(tokens),
      num_samples=1,
      sampling_params=types.SamplingParams(max_tokens=EVAL_MAX_TOKENS, temperature=0.0),
    ).result()

    result_tokens = result.sequences[0].tokens if result.sequences else []
    text = tokenizer.decode(result_tokens, skip_special_tokens=False)
    expected = ex["expected_tool"]
    other_tool = "search_google" if expected == "search_knowledge_base" else "search_knowledge_base"
    hits += expected in text and other_tool not in text

  return hits / max(1, len(examples))


def plot_metrics(losses: list[float], before: float, after: float, plot_path: Path) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  axes[0].plot(range(1, len(losses) + 1), losses, marker="o", color="#1f77b4")
  axes[0].set_title("Training Loss")
  axes[0].set_xlabel("Step")
  axes[0].set_ylabel("loss:mean")
  axes[0].grid(True, alpha=0.3)
  axes[1].bar(["before", "after"], [before, after], color=["#9aa0a6", "#1b9e77"])
  axes[1].set_ylim(0.0, 1.0)
  axes[1].set_title("Tool Selection Success Rate")
  axes[1].grid(True, axis="y", alpha=0.3)
  plot_path.parent.mkdir(parents=True, exist_ok=True)
  plt.tight_layout()
  plt.savefig(plot_path)
  plt.close(fig)


def require_server(base_url: str) -> dict[str, Any]:
  try:
    response = requests.get(f"{base_url.rstrip('/')}/api/v1/get_server_capabilities", timeout=5.0)
    response.raise_for_status()
    return response.json()
  except Exception as exc:
    raise RuntimeError(f"Open-RL server at {base_url} is not reachable. Start it with `make run-function-gemma-server`.") from exc


async def run_training(config: Config) -> None:
  # 1. Preflight
  if config.ci and not os.getenv("HF_TOKEN"):
    raise RuntimeError("CI mode requires HF_TOKEN")

  capabilities = require_server(config.base_url)
  print(f"Server ready at {config.base_url} | model={capabilities.get('default_model') or 'unset'}")

  # 2. Data
  dataset = load_dataset(config.dataset, split="train").map(build_conversation, batched=False)
  split = dataset.train_test_split(test_size=0.5, shuffle=False)
  train_examples = list(split["train"])
  eval_dataset = split["test"].select(range(min(config.eval_limit, len(split["test"])))) if config.eval_limit else split["test"]
  eval_examples = list(eval_dataset)
  print(f"Loaded {len(dataset)} rows from {config.dataset} | train={len(train_examples)} eval={len(eval_examples)}")

  # 3. Training setup
  client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY"), base_url=config.base_url)
  trainer = await client.create_lora_training_client_async(base_model=config.base_model, rank=config.rank)
  tokenizer = trainer.get_tokenizer()
  datums = [make_datum(tokenizer, ex) for ex in train_examples]
  active_tokens = sum(sum(d.loss_fn_inputs["weights"].tolist()) for d in datums)

  print(f"Created LoRA training client | rank={config.rank} | datums={len(datums)}")

  # 4. Baseline eval
  baseline_weights = trainer.save_weights_for_sampler(name="functiongemma_baseline").result()
  sampler = client.create_sampling_client(baseline_weights.path)
  before = eval_rate(tokenizer, sampler, eval_examples)
  print(f"[eval] baseline={before:.1%}")

  # 5. Train
  losses: list[float] = []
  for step in range(config.epochs):
    fwdbwd, optim = await asyncio.gather(
      trainer.forward_backward_async(datums, "cross_entropy"),
      trainer.optim_step_async(types.AdamParams(learning_rate=5e-5)),
    )
    optim.result()
    loss = float(fwdbwd.result().metrics.get("loss:sum", 0.0)) / active_tokens
    losses.append(loss)
    print(f"[train] step={step + 1:02d}/{config.epochs} loss={loss:.6f}")

  # 6. Post-train eval and artifacts
  tuned_weights = trainer.save_weights_for_sampler(name="functiongemma_tuned").result()
  sampler = client.create_sampling_client(tuned_weights.path)
  after = eval_rate(tokenizer, sampler, eval_examples)
  print(f"[eval] tuned={after:.1%}")
  plot_metrics(losses, before, after, Path(config.plot_path))

  # 7. Success criteria
  drop = None
  if config.assert_loss_drop or config.ci:
    drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)
    assert len(losses) >= 2 and drop >= config.min_loss_drop, (
      f"Loss did not improve enough: first={losses[0]:.6f} last={losses[-1]:.6f} drop={drop:.2%} required={config.min_loss_drop:.2%}"
    )
    print(f"[assert] loss improved by {drop:.2%}")

  print(f"Saved plot to {config.plot_path}")
  print(f"[summary] baseline={before:.1%} tuned={after:.1%} loss_drop={'n/a' if drop is None else f'{drop:.2%}'}")
  if config.ci:
    print(f"CI_RESULT plot={config.plot_path} baseline_rate={before:.1%} tuned_rate={after:.1%} loss_drop={drop:.2%}")


def cli() -> None:
  asyncio.run(run_training(chz.entrypoint(Config, allow_hyphens=True)))


if __name__ == "__main__":
  cli()
