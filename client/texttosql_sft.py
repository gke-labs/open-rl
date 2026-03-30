import asyncio
import logging
import os
import random
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.utils import ml_log

BASE_MODEL = "google/gemma-3-1b-pt"
BASE_URL = "http://127.0.0.1:9003"
DATASET = "philschmid/gretel-synthetic-text-to-sql"
LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "texttosql_{preset}"
MAX_SEQ_LENGTH = 512

USER_PROMPT = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@chz.chz
class Config:
    steps: int
    batch_size: int
    rank: int
    learning_rate: float
    base_url: str = os.getenv("TINKER_BASE_URL") or os.getenv("OPEN_RL_BASE_URL") or BASE_URL
    grad_clip_norm: float = 0.3
    eval_every: int = 50
    train_limit: int = 2_048
    eval_limit: int = 128
    seed: int = 30
    log_dir: str = str(LOG_DIR)
    eval_max_tokens: int = 256


PRESETS = {
    "gemma": chz.Blueprint(Config).apply(
        {"steps": 30, "batch_size": 8, "rank": 16, "learning_rate": 2e-4, "train_limit": 10_000, "eval_limit": 5, "eval_every": 15},
        layer_name="gemma preset",
    ),
}


async def run_training(config: Config, preset: str) -> dict[str, float | str]:
    log_dir = Path(config.log_dir.replace("{preset}", preset))
    ml_logger = ml_log.setup_logging(log_dir=str(log_dir), config=config, do_configure_logging_module=True)
    metrics_path = log_dir / "metrics.jsonl"
    client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"), base_url=config.base_url)
    server_model = await require_server(client, config.base_url)
    logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

    trainer = await client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=config.rank,
        seed=config.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    dataset = load_dataset(DATASET, split="train").shuffle(seed=config.seed)
    dataset = dataset.select(range(min(12_500, len(dataset))))
    split = dataset.train_test_split(test_size=2_500, shuffle=False)

    train_examples = build_examples(tokenizer, split["train"], config.train_limit)
    eval_examples = build_examples(tokenizer, split["test"], config.eval_limit, require_seed_data=True, require_target_rows=True)
    if not train_examples:
        raise RuntimeError("No training examples fit within max_seq_length.")
    if not eval_examples:
        raise RuntimeError("No evaluation examples with executable seed data were found.")

    batch_size = min(config.batch_size, len(train_examples))
    logging.info(
        "Data: %s train, %s eval | batch=%s rank=%s lr=%g",
        len(train_examples),
        len(eval_examples),
        batch_size,
        config.rank,
        config.learning_rate,
    )

    before_sampler_path = trainer.save_weights_for_sampler(name="texttosql_before").result().path
    before_sampler = client.create_sampling_client(before_sampler_path)
    before_exec, before_sim = await evaluate(before_sampler, tokenizer, "texttosql_before", eval_examples, config)
    ml_logger.log_metrics({"phase": "eval", "execution_match": before_exec, "similarity": before_sim}, step=0)

    losses: list[float] = []
    eval_exec = [before_exec]
    eval_sim = [before_sim]
    rng = random.Random(config.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    pos = 0

    for step in range(1, config.steps + 1):
        if pos + batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        batch = [train_examples[order[i]] for i in range(pos, pos + batch_size)]
        pos += batch_size

        datums = [example["datum"] for example in batch]
        active_tokens = sum(example["active_tokens"] for example in batch)

        fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
        optim_future = await trainer.optim_step_async(types.AdamParams(learning_rate=config.learning_rate, grad_clip_norm=config.grad_clip_norm))
        fwdbwd = await fwdbwd_future
        await optim_future

        loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
        losses.append(loss)
        ml_logger.log_metrics({"phase": "train", "loss": loss}, step=step)

        if step % config.eval_every == 0 or step == config.steps:
            alias = f"texttosql_s{step}"
            sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
            sampler = client.create_sampling_client(sampler_path)
            execution_match, sim = await evaluate(sampler, tokenizer, alias, eval_examples, config)
            eval_exec.append(execution_match)
            eval_sim.append(sim)
            ml_logger.log_metrics({"phase": "eval", "execution_match": execution_match, "similarity": sim}, step=step)

    loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0) if losses else 0.0
    logging.info("Saved metrics to %s", metrics_path)
    logging.info(
        "[summary] execution=%.1f%%->%.1f%% similarity=%.1f%%->%.1f%% loss_drop=%.1f%%",
        before_exec * 100,
        eval_exec[-1] * 100,
        before_sim * 100,
        eval_sim[-1] * 100,
        loss_drop * 100,
    )
    ml_logger.close()

    return {
        "before_execution_match": before_exec,
        "after_execution_match": eval_exec[-1],
        "before_similarity": before_sim,
        "after_similarity": eval_sim[-1],
        "loss_drop": loss_drop,
        "metrics_path": str(metrics_path),
    }


async def require_server(service_client: tinker.ServiceClient, base_url: str) -> str | None:
    try:
        capabilities = await service_client.get_server_capabilities_async()
    except Exception as exc:
        raise RuntimeError(f"Open-RL server at {base_url} is not reachable. Start it with `make run-text-to-sql-server`.") from exc

    model_names = [model.model_name for model in capabilities.supported_models if getattr(model, "model_name", None)]
    return model_names[0] if model_names else None


def normalize_sql(text: str) -> str:
    text = clean_sql_for_execution(text)
    text = " ".join(text.split()).lower()
    text = re.sub(r";+\s*$", "", text)
    text = re.sub(r"\s+([,;()])", r"\1", text)
    text = re.sub(r"([,(])\s+", r"\1", text)
    return text


def clean_sql_for_execution(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    text = text.strip()
    text = re.sub(r"^assistant\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^sql\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    sql_start = re.search(r"\b(with|select|insert|update|delete)\b", text, flags=re.IGNORECASE)
    if sql_start:
        text = text[sql_start.start() :]
        statement = re.search(r".*?(?:;|$)", text, flags=re.DOTALL)
        if statement: text = statement.group(0)
    return text.strip()


def run_sql(context: str, query: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    connection = sqlite3.connect(":memory:")
    try:
        deadline = time.monotonic() + 0.25
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        connection.executescript(context)
        rows = connection.execute(query).fetchall()
        normalized_rows = [tuple(round(value, 8) if isinstance(value, float) else value for value in row) for row in rows]
        return normalized_rows, None
    except sqlite3.Error as exc:
        return None, str(exc)
    finally:
        connection.close()


def sql_results_match(context: str, predicted_sql: str, target_sql: str, target_rows: list[tuple[Any, ...]] | None = None) -> tuple[bool, str | None]:
    predicted_rows, error = run_sql(context, predicted_sql)
    if error is not None:
        return False, f"predicted query error: {error}"

    if target_rows is None:
        target_rows, error = run_sql(context, target_sql)
        if error is not None:
            return False, f"target query error: {error}"

    order_sensitive = any(token in f" {normalize_sql(predicted_sql)} {normalize_sql(target_sql)} " for token in (" order by ", " limit ", " offset "))
    if not order_sensitive:
        predicted_rows = sorted(predicted_rows, key=repr)
        target_rows = sorted(target_rows, key=repr)
    return predicted_rows == target_rows, None


def make_datum(full_tokens: list[int], weights: list[int]) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs=cast(Any, {"weights": weights[1:], "target_tokens": full_tokens[1:]}),
    )


def render_training_texts(tokenizer, question, context, target_sql):
    messages = [
        {"role": "user", "content": USER_PROMPT.format(question=question, context=context)},
        {"role": "assistant", "content": target_sql},
    ]
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt_text, full_text


def build_example(tokenizer, row):
    target_sql = clean_sql_for_execution(row["sql"])
    prompt_text, full_text = render_training_texts(tokenizer, row["sql_prompt"], row["sql_context"], target_sql)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_tokens) <= len(prompt_tokens) or len(full_tokens) > MAX_SEQ_LENGTH:
        return None

    return {
        "question": row["sql_prompt"],
        "context": row["sql_context"],
        "target": target_sql,
        "prompt_tokens": prompt_tokens,
        "active_tokens": len(full_tokens) - len(prompt_tokens),
        "datum": make_datum(full_tokens, [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))),
    }


def build_examples(tokenizer, dataset_split, limit, require_seed_data=False, require_target_rows=False):
    examples = []
    for row in dataset_split:
        if require_seed_data and "insert into" not in row["sql_context"].lower(): continue

        example = build_example(tokenizer, row)
        if example is None: continue

        if require_target_rows:
            target_rows, error = run_sql(example["context"], example["target"])
            if error is not None: continue
            example["target_rows"] = target_rows

        examples.append(example)
        if len(examples) >= limit: break

    return examples


async def evaluate(sampler, tokenizer, alias, examples, config):
    execution_match, similarity = 0.0, 0.0
    futures = [
        sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokens=example["prompt_tokens"]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=config.eval_max_tokens, seed=config.seed + idx, temperature=0.0),
        )
        for idx, example in enumerate(examples)
    ]
    responses = await asyncio.gather(*futures)

    for idx, (example, response) in enumerate(zip(examples, responses)):
        predicted_sql = clean_sql_for_execution(tokenizer.decode(response.sequences[0].tokens if response.sequences else [], skip_special_tokens=True))
        target_sql = example["target"]
        predicted = normalize_sql(predicted_sql)
        target = normalize_sql(target_sql)
        matches_execution, execution_error = sql_results_match(example["context"], predicted_sql, target_sql, target_rows=example["target_rows"])

        log_level = logging.INFO if matches_execution else logging.WARNING
        sqlite_line = f"\nSQLite:    {execution_error}" if execution_error else ""
        logging.log(
            log_level,
            "\n--- [Visual Check %s Item %d] ---\nQuestion: %s\nPredicted: %s\nTarget:    %s%s\nExecution: %s\n",
            alias,
            idx + 1,
            example["question"],
            predicted,
            target,
            sqlite_line,
            "MATCH" if matches_execution else "NO MATCH",
        )

        execution_match += float(matches_execution)
        similarity += SequenceMatcher(None, predicted, target).ratio()
    count = max(1, len(examples))
    return execution_match / count, similarity / count


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
    import sys

    logging.getLogger("tinker").setLevel(logging.WARNING)

    preset = sys.argv[1]
    blueprint = PRESETS[preset].clone()
    config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)
    asyncio.run(run_training(config, preset))


if __name__ == "__main__":
    cli()
