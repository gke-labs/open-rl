import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers.utils import get_json_schema
import tinker
from tinker import types

logging.getLogger("tinker").setLevel(logging.WARNING)

BASE_MODEL = "google/functiongemma-270m-it"
HF_DATASET = "bebechien/SimpleToolCalling"

def search_knowledge_base(query: str):
    """
    Search the internal knowledge base for information.
    Args:
        query: The search query string.
    """
    return "Internal Result"

def search_google(query: str):
    """
    Search the public internet for information.
    Args:
        query: The search query string.
    """
    return "Public Result"

TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]

def build_conversation(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "developer",
                "content": "You are a model that can do function calling with the following functions",
            },
            {"role": "user", "content": sample["user_content"]},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": sample["tool_name"],
                            "arguments": json.loads(sample["tool_arguments"]),
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
        "expected_tool": sample["tool_name"],
        "user_content": sample["user_content"],
    }


def make_datum(tokenizer: Any, example: dict[str, Any]) -> types.Datum:
    full_text = tokenizer.apply_chat_template(
        example["messages"],
        tools=example["tools"],
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        example["messages"][:2],
        tools=example["tools"],
        add_generation_prompt=True,
        tokenize=False,
    )

    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(full_tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a training datum")
    if len(prompt_tokens) >= len(full_tokens):
        raise ValueError("Prompt tokens must be shorter than full example tokens")

    weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": full_tokens[1:],
            "weights": weights[1:],
        },
    )


def make_sampling_client(service_client: tinker.ServiceClient, training_client: Any, alias: str) -> Any:
    save_result = training_client.save_weights_for_sampler(name=alias).result()
    return service_client.create_sampling_client(save_result.path)


def evaluate_tool_selection(
    tokenizer: Any,
    sampling_client: Any,
    eval_examples: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> tuple[float, list[dict[str, Any]]]:
    rollouts: list[dict[str, Any]] = []

    for idx, example in enumerate(eval_examples, start=1):
        prompt_text = tokenizer.apply_chat_template(
            example["messages"][:2],
            tools=example["tools"],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        result = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature),
        ).result()

        output_tokens = result.sequences[0].tokens if result.sequences else []
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        expected_tool = example["expected_tool"]
        other_tool = "search_google" if expected_tool == "search_knowledge_base" else "search_knowledge_base"
        passed = expected_tool in output_text and other_tool not in output_text

        rollouts.append(
            {
                "idx": idx,
                "prompt": example["user_content"],
                "expected_tool": expected_tool,
                "output": output_text,
                "passed": passed,
            }
        )

    success_count = sum(1 for item in rollouts if item["passed"])
    for item in rollouts:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[eval {item['idx']:02d}] {status} | expected={item['expected_tool']} | prompt={item['prompt']}")
        print(f"           output={item['output'].strip()}")

    success_rate = success_count / max(1, len(rollouts))
    print(f"[eval] success={success_count}/{len(eval_examples)} ({success_rate:.1%})")
    return success_rate, rollouts


def plot_metrics(losses: list[float], baseline_rate: float | None, tuned_rate: float | None, output_path: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(range(1, len(losses) + 1), losses, marker="o", color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("loss:mean")
    axes[0].grid(True, alpha=0.3)

    if baseline_rate is None or tuned_rate is None:
        axes[1].axis("off")
        axes[1].text(0.05, 0.5, "Evaluation skipped", fontsize=12)
    else:
        axes[1].bar(["before", "after"], [baseline_rate, tuned_rate], color=["#9aa0a6", "#1b9e77"])
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Tool Selection Success Rate")
        axes[1].set_ylabel("success rate")
        axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(figure)


def assert_loss_progress(losses: list[float]) -> float:
    if len(losses) < 2:
        raise AssertionError("Need at least 2 epochs to validate loss improvement")

    first = losses[0]
    last = losses[-1]
    denom = abs(first) if abs(first) > 1e-8 else 1.0
    relative_drop = (first - last) / denom

    if relative_drop < 0.05:
        raise AssertionError(
            f"Loss did not improve enough: first={first:.6f}, last={last:.6f}, "
            f"relative_drop={relative_drop:.4f}, required=0.0500"
        )

    return relative_drop


async def run_sft(
    service_client: tinker.ServiceClient,
    epochs: int,
    skip_eval: bool,
    assert_loss_drop: bool,
) -> None:
    print("Initializing FunctionGemma SFT job...")
    print(f"FunctionGemma reference model: {BASE_MODEL}")
    print("Model card note: 270M parameter variant is designed for lightweight deployments")
    print(f"Make sure server VLLM_MODEL matches {BASE_MODEL} for adapter compatibility")

    dataset = load_dataset(HF_DATASET, split="train")
    print(f"Loaded {len(dataset)} rows from {HF_DATASET}")

    # Map conversations over the dataset
    dataset = dataset.map(build_conversation, remove_columns=dataset.features, batched=False)
    
    # Split dataset into 50% training samples and 50% test samples using datasets
    split_dataset = dataset.train_test_split(test_size=0.5, shuffle=False)
    train_examples = split_dataset['train']
    test_examples = split_dataset['test']
    
    print(f"Dataset size={len(dataset)} | train={len(train_examples)} | test={len(test_examples)}")

    print(f"Creating training client for base_model={BASE_MODEL!r}, rank=16...")
    training_client = await service_client.create_lora_training_client_async(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()

    train_datums = []
    for example in train_examples:
        train_datums.append(make_datum(tokenizer, example))
    print(f"Prepared {len(train_datums)} datums for cross-entropy training")

    baseline_rate = None
    if not skip_eval:
        print("Running baseline evaluation before fine-tuning...")
        baseline_sampler = make_sampling_client(service_client, training_client, "functiongemma_baseline")
        baseline_rate, _ = evaluate_tool_selection(
            tokenizer=tokenizer,
            sampling_client=baseline_sampler,
            eval_examples=test_examples,
            max_tokens=128,
            temperature=0.0,
        )

    losses: list[float] = []
    print("Starting training...")
    learning_rate = 5e-5
    total_active_tokens = sum(sum(d.loss_fn_inputs["weights"].tolist()) for d in train_datums)
    for epoch in range(epochs):
        fwdbwd_future = await training_client.forward_backward_async(train_datums, "cross_entropy")
        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=learning_rate))

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        total_loss = float(fwdbwd_result.metrics.get("loss:sum", 0.0))
        loss_value = total_loss / total_active_tokens if total_active_tokens > 0 else 0.0
        losses.append(loss_value)
        print(f"[train] epoch={epoch + 1:02d}/{epochs} loss={loss_value:.6f}")

    tuned_rate = None
    if not skip_eval:
        print("Running evaluation after fine-tuning...")
        tuned_sampler = make_sampling_client(service_client, training_client, "functiongemma_tuned")
        tuned_rate, _ = evaluate_tool_selection(
            tokenizer=tokenizer,
            sampling_client=tuned_sampler,
            eval_examples=test_examples,
            max_tokens=128,
            temperature=0.0,
        )

    plot_metrics(losses, baseline_rate, tuned_rate, "functiongemma_sft_metrics.png")
    print("Saved plot to functiongemma_sft_metrics.png")

    if assert_loss_drop:
        loss_drop = assert_loss_progress(losses)
        print(f"[assert] loss improved by {loss_drop:.2%} (required 5.00%)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="FunctionGemma fine-tuning reproduction using Open-RL/Tinker primitives")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--skip-eval", action="store_true", help="Skip pre/post sampling evaluation")
    parser.add_argument("--assert-loss-drop", action="store_true", help="Fail run if train loss does not drop")
    args = parser.parse_args()

    service_client = tinker.ServiceClient(
        api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
        base_url=os.getenv("TINKER_BASE_URL", "http://localhost:9000"),
    )

    await run_sft(
        service_client=service_client,
        epochs=args.epochs,
        skip_eval=args.skip_eval,
        assert_loss_drop=args.assert_loss_drop,
    )


if __name__ == "__main__":
    asyncio.run(main())
