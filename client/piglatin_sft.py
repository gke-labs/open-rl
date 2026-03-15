import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import chz
import matplotlib.pyplot as plt
import requests
import tinker
from tinker import types

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "http://127.0.0.1:9001"
PLOT_PATH = Path(__file__).resolve().parent / "artifacts" / "piglatin_{preset}_metrics.png"

PAIRS_PATH = Path(__file__).resolve().parent / "piglatin_data.json"
SYSTEM_PROMPT = "Translate the English text into Pig Latin. Reply with only the Pig Latin translation."
EXAMPLES = [
    ("banana split", "anana-bay plit-say"),
    ("quantum physics", "uantum-qay ysics-phay"),
    ("donut shop", "onut-day op-shay"),
    ("pickle jar", "ickle-pay ar-jay"),
    ("space exploration", "ace-spay exploration-way"),
    ("rubber duck", "ubber-ray uck-day"),
    ("coding wizard", "oding-cay izard-way"),
    ("hello world", "ello-hay orld-way"),
    ("machine learning", "achine-may earning-lay"),
    ("artificial intelligence", "artificial-way intelligence-way"),
    ("data science", "ata-day ience-scay"),
]

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
    base_model: str
    batch_size: int
    rank: int
    learning_rate: float
    base_url: str = os.getenv("TINKER_BASE_URL") or BASE_URL
    steps: int = 100
    train_limit: int = 240
    eval_limit: int = 64
    eval_every: int = 5
    eval_max_tokens: int = 32
    plot_path: str = str(PLOT_PATH)
    seed: int = 64
    assert_improvement: bool = True
    min_loss_drop: float = 0.8
    min_similarity_gain: float = 0.15

PRESETS = {
    "qwen": chz.Blueprint(Config).apply(
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "batch_size": 16,
            "rank": 16,
            "learning_rate": 1e-4,
            "steps" : 20,
        },
        layer_name="qwen preset",
    ),
    "gemma": chz.Blueprint(Config).apply(
        {
            "base_model": "google/gemma-3-1b-it",
            "batch_size": 16,
            "rank": 32,
            "learning_rate": 3e-4,
            "steps": 30,
        },
        layer_name="gemma preset",
    ),
}


def require_server(base_url: str) -> dict[str, Any]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/v1/get_server_capabilities", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Open-RL server at {base_url} is not reachable. Start it with `make run-pig-latin-server`.") from exc


def normalize(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    text = re.sub(r"^assistant\s*[:\-]?\s*", "", text.strip(), flags=re.IGNORECASE)
    return " ".join(text.split()).lower()


def build_example(tokenizer: Any, source: str, target: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
        {"role": "assistant", "content": target},
    ]
    prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
    return {
        "source": source,
        "target": target,
        "prompt_tokens": prompt_tokens,
        "active_tokens": len(full_tokens) - len(prompt_tokens),
        "datum": types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
            loss_fn_inputs=cast(Any, {"weights": weights[1:], "target_tokens": full_tokens[1:]}),
        ),
    }


def load_pairs(seed: int, train_limit: int, eval_limit: int) -> tuple[list[list[str]], list[list[str]]]:
    data = json.loads(PAIRS_PATH.read_text())
    train, val = data["train"], data["eval"]
    random.Random(seed).shuffle(train)
    return train[:train_limit], val[:eval_limit]


def evaluate(client: Any, trainer: Any, tokenizer: Any, alias: str, examples: list[dict[str, Any]], max_tokens: int) -> tuple[float, float, list[tuple[str, str, str]]]:
    sampler = client.create_sampling_client(trainer.save_weights_for_sampler(name=alias).result().path)
    exact = 0.0
    similarity = 0.0
    rows = []
    for ex in examples:
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=ex["prompt_tokens"]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
        ).result()
        text = normalize(tokenizer.decode(result.sequences[0].tokens if result.sequences else [], skip_special_tokens=True))
        target = normalize(ex["target"])
        rows.append((ex["source"], text, target))
        exact += float(text == target)
        similarity += SequenceMatcher(None, text, target).ratio()
    count = max(1, len(examples))
    return exact / count, similarity / count, rows


def plot_metrics(losses: list[float], eval_steps: list[int], eval_exact: list[float], eval_sim: list[float], plot_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("loss/token")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(eval_steps, [x * 100 for x in eval_exact], marker="o", label="Exact Match %", color="#1b9e77")
    axes[1].set_xticks(eval_steps)
    axes[1].set_title("Eval Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_training(config: Config) -> None:
    capabilities = require_server(config.base_url)
    print(f"Server ready at {config.base_url} | model={capabilities.get('default_model') or 'unset'}")

    client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)
    trainer = client.create_lora_training_client(base_model=config.base_model, rank=config.rank)
    tokenizer = trainer.get_tokenizer()

    train_pairs, eval_pairs = load_pairs(config.seed, config.train_limit, config.eval_limit)
    train_exs = [build_example(tokenizer, s, t) for s, t in train_pairs]
    eval_exs = [build_example(tokenizer, s, t) for s, t in eval_pairs]
    batch_size = max(1, min(config.batch_size, len(train_exs)))
    print(f"Loaded piglatin_pairs.json | train={len(train_exs)} eval={len(eval_exs)} batch={batch_size} steps={config.steps}")

    before_exact, before_sim, before_rows = evaluate(client, trainer, tokenizer, "piglatin_before", eval_exs, config.eval_max_tokens)
    print(f"[before] exact={before_exact:.1%} similarity={before_sim:.1%}")

    eval_steps = [0]
    eval_exact = [before_exact]
    eval_sim = [before_sim]
    losses: list[float] = []
    rng = random.Random(config.seed)
    order = list(range(len(train_exs)))
    rng.shuffle(order)
    pos = 0
    for step in range(1, config.steps + 1):
        batch: list[dict[str, Any]] = []
        while len(batch) < batch_size:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            batch.append(train_exs[order[pos]])
            pos += 1
        datums = [ex["datum"] for ex in batch]
        active = sum(ex["active_tokens"] for ex in batch)
        fwdbwd = trainer.forward_backward(datums, "cross_entropy").result()
        trainer.optim_step(types.AdamParams(learning_rate=config.learning_rate)).result()
        loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active)
        losses.append(loss)
        print(f"[train] step={step:02d}/{config.steps} loss={loss:.4f}")

        if step % config.eval_every == 0 or step == config.steps:
            mid_exact, mid_sim, eval_rows = evaluate(client, trainer, tokenizer, f"piglatin_s{step}", eval_exs, config.eval_max_tokens)
            eval_steps.append(step)
            eval_exact.append(mid_exact)
            eval_sim.append(mid_sim)
            print(f"[eval]  step={step:02d} exact={mid_exact:.1%} similarity={mid_sim:.1%}")

    after_exact, after_sim = eval_exact[-1], eval_sim[-1]
    plot_metrics(losses, eval_steps, eval_exact, eval_sim, Path(config.plot_path))


    random_eval = rng.randrange(len(eval_rows))
    before_example = before_rows[random_eval]
    after_example = eval_rows[random_eval]
    
    loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)

    print(f"base model random eval input: {before_example[0]} response: {before_example[1]}")
    print(f"finetuned_model random eval input: {after_example[0]} output: {after_example[1]} expected: {after_example[2]}")
    
    print("\n--- Testing Custom Examples on Finetuned Model ---")
    custom_exs = [build_example(tokenizer, s, t) for s, t in EXAMPLES]
    _, _, custom_rows = evaluate(client, trainer, tokenizer, f"piglatin_s{config.steps}", custom_exs, config.eval_max_tokens)
    for row in custom_rows:
        source, actual, expected = row
        match = "✅" if actual == expected else "❌"
        print(f"{match} input: '{source:20}' | output: '{actual:30}' | expected: '{expected}'")
    print("--------------------------------------------------\n")

    print(f"Saved plot to {config.plot_path}")
    print(
        f"[summary] exact={before_exact:.1%}->{after_exact:.1%} "
        f"similarity={before_sim:.1%}->{after_sim:.1%} "
        f"loss_drop={loss_drop:.1%}"
    )

    if config.assert_improvement:
        assert after_exact > before_exact, "Exact match did not improve"
        assert after_sim - before_sim >= config.min_similarity_gain, "Similarity did not improve enough"
        # assert loss_drop >= config.min_loss_drop, "Loss did not drop enough"


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
    import sys
    preset, *argv = sys.argv[1:]
    if not preset or preset not in PRESETS: preset = "qwen"
    blueprint = PRESETS[preset].clone()
    
    # Dynamically inject the preset name into the default plot path
    config = blueprint.make_from_argv(argv, allow_hyphens=True)
    if "piglatin_{preset}_metrics.png" in config.plot_path:
        config = chz.replace(config, plot_path=config.plot_path.replace("{preset}", preset))
        
    run_training(config)


if __name__ == "__main__":
    cli()
