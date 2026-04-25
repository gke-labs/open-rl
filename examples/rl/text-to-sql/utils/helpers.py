"""Text-to-SQL task utilities for this recipe."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import tinker

MAX_SEQ_LENGTH = 512

Example = dict[str, Any]


async def require_server(service_client: tinker.ServiceClient, base_url: str, expected_model: str | None = None) -> str | None:
  try:
    capabilities = await service_client.get_server_capabilities_async()
  except Exception as exc:
    raise RuntimeError(
      f"Open-RL server at {base_url} is not reachable.\nStart it with:  make server BASE_MODEL={expected_model or '<model-id>'}"
    ) from exc

  model_names = [model.model_name for model in capabilities.supported_models if getattr(model, "model_name", None)]
  server_model = model_names[0] if model_names else None

  if expected_model and server_model and server_model != expected_model:
    raise RuntimeError(
      f"Open-RL server at {base_url} is running {server_model!r}, "
      f"but this recipe expects {expected_model!r}.\n"
      f"Restart the server with:  make server BASE_MODEL={expected_model}"
    )
  return server_model


def build_example(tokenizer: Any, row: dict[str, Any]) -> Example | None:
  prompt_text = row["prompt_text"]
  target_sql = row["target"]
  full_text = prompt_text + target_sql
  prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
  full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
  if len(full_tokens) <= len(prompt_tokens) or len(full_tokens) > MAX_SEQ_LENGTH:
    return None

  return {
    "question": row["question"],
    "context": row["context"],
    "target": target_sql,
    "prompt_text": prompt_text,
    "target_rows": row["target_rows"],
    "prompt_tokens": prompt_tokens,
    "full_tokens": full_tokens,
    "active_tokens": len(full_tokens) - len(prompt_tokens),
  }


def build_examples(
  tokenizer: Any,
  dataset_split: Any,
  limit: int | None = None,
) -> list[Example]:
  examples = []
  for row in dataset_split:
    example = build_example(tokenizer, row)
    if example is None:
      continue

    examples.append(example)
    if limit is not None and len(examples) >= limit:
      break

  return examples


def shuffled_batches(examples: list[Example], batch_size: int, seed: int) -> Iterator[list[Example]]:
  """Yield shuffled mini-batches forever, reshuffling when the pool is exhausted."""
  if not examples:
    raise ValueError("Cannot batch an empty example list.")

  rng = random.Random(seed)
  batch_size = min(batch_size, len(examples))
  while True:
    shuffled = rng.sample(examples, k=len(examples))
    for i in range(0, len(shuffled) - batch_size + 1, batch_size):
      yield shuffled[i : i + batch_size]
