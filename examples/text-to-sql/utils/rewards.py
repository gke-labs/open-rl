"""Dataset loading and SQL reward helpers for the Text-to-SQL RL recipe."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from datasets import load_dataset

COLUMN_TYPE_PATTERN = re.compile(
  r"^\s+(\w+)\s+(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|FLOAT|DOUBLE|DECIMAL|BOOLEAN|DATE|TIMESTAMP)",
  re.IGNORECASE | re.MULTILINE,
)
TABLE_PATTERN = re.compile(r"CREATE TABLE (\w+)", re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b\w+\b")
DEFAULT_DATASET = "philschmid/gretel-synthetic-text-to-sql"

# Reward shaping is borrowed from Reasoning-SQL (Pourreza et al., 2025):
# hard compile/execution signals plus schema, n-gram, normalized SQL, and
# partial execution credit so early RL rollouts still get useful feedback.
COMPILE_REWARD = 0.25
EXECUTION_MATCH_REWARD = 2.0
ERROR_PENALTY = -0.25
SIMILARITY_REWARD = 1.0
SCHEMA_LINK_WEIGHT = 0.30
NGRAM_WEIGHT = 0.20
NORMALIZED_SQL_WEIGHT = 0.30
PARTIAL_EXECUTION_WEIGHT = 0.20
PARTIAL_COL_WEIGHT = 0.25
PARTIAL_ROWCOUNT_WEIGHT = 0.25
PARTIAL_VALUE_OVERLAP_WEIGHT = 0.50
EVAL_METRIC_NAMES = ("execution_match", "exact_match", "execution_match_not_exact", "similarity")
EvalMetrics = dict[str, float]
PLAIN_SQL_PROMPT = """Return only one SQLite query.

Schema:
{context}

Question:
{question}

SQL:
"""


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
    if statement:
      text = statement.group(0)
  return text.strip()


def normalize_sql(text: str) -> str:
  text = clean_sql_for_execution(text)
  text = " ".join(text.split()).lower()
  text = re.sub(r";+\s*$", "", text)
  text = re.sub(r"\s+([,;()])", r"\1", text)
  text = re.sub(r"([,(])\s+", r"\1", text)
  return text


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


def sql_rows_match(
  predicted_sql: str,
  target_sql: str,
  predicted_rows: list[tuple[Any, ...]] | None,
  target_rows: list[tuple[Any, ...]] | None,
) -> bool:
  if not predicted_rows and not target_rows:
    return False

  order_sensitive = any(token in f" {normalize_sql(predicted_sql)} {normalize_sql(target_sql)} " for token in (" order by ", " limit ", " offset "))
  if not order_sensitive:
    predicted_rows = sorted(predicted_rows or [], key=repr)
    target_rows = sorted(target_rows or [], key=repr)
  return predicted_rows == target_rows


def sql_results_match(context: str, predicted_sql: str, target_sql: str, target_rows: list[tuple[Any, ...]] | None = None) -> tuple[bool, str | None]:
  predicted_rows, error = run_sql(context, predicted_sql)
  if error is not None:
    return False, f"predicted query error: {error}"

  if target_rows is None:
    target_rows, error = run_sql(context, target_sql)
    if error is not None:
      return False, f"target query error: {error}"

  if not predicted_rows and not target_rows:
    return False, "both queries returned no rows"

  return sql_rows_match(predicted_sql, target_sql, predicted_rows, target_rows), None


def empty_eval_metrics() -> EvalMetrics:
  return {name: 0.0 for name in EVAL_METRIC_NAMES}


def aggregate_eval_scores(scores: Iterable[dict[str, Any]]) -> EvalMetrics:
  totals = empty_eval_metrics()
  count = 0
  for score in scores:
    count += 1
    for name in EVAL_METRIC_NAMES:
      totals[name] += float(score[name])

  if count == 0:
    return totals
  return {name: value / count for name, value in totals.items()}


def score_eval_prediction(predicted_sql: str, example: dict[str, Any]) -> dict[str, Any]:
  """Score one generated SQL string against a recipe example."""
  return score_prediction(
    predicted_sql=predicted_sql,
    target_sql=example["target"],
    context=example["context"],
    target_rows=example.get("target_rows"),
    question=example["question"],
  )


def load_dataset_splits(
  *,
  dataset_name: str = DEFAULT_DATASET,
  dataset_limit: int = 12_500,
  train_limit: int = 5_000,
  eval_limit: int = 100,
  seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  dataset = load_dataset(dataset_name, split="train").shuffle(seed=seed)
  dataset = dataset.select(range(min(dataset_limit, len(dataset))))
  if len(dataset) < 10:
    raise RuntimeError("dataset_limit is too small to create train/eval splits")

  split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)
  train_rows = build_dataset_rows(split["train"], train_limit)
  eval_rows = build_dataset_rows(split["test"], eval_limit)
  if not train_rows:
    raise RuntimeError("No train examples with executable target rows were found.")
  if not eval_rows:
    raise RuntimeError("No eval examples with executable target rows were found.")
  return train_rows, eval_rows


def build_dataset_rows(dataset_split: Any, limit: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for row in dataset_split:
    context = row["sql_context"]
    if "insert into" not in context.lower():
      continue

    target = clean_sql_for_execution(row["sql"])
    target_rows, error = run_sql(context, target)
    if error is not None or not target_rows:
      continue

    question = row["sql_prompt"]
    prompt_text = PLAIN_SQL_PROMPT.format(context=context, question=question)
    rows.append(
      {
        "question": question,
        "context": context,
        "target": target,
        "target_rows": target_rows,
        "prompt_text": prompt_text,
      }
    )
    if len(rows) >= limit:
      break
  return rows


def ngram_similarity(predicted_sql: str, target_sql: str, n: int = 2) -> float:
  def ngrams(text: str) -> set[tuple[str, ...]]:
    tokens = text.lower().split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

  pred, gold = ngrams(predicted_sql), ngrams(target_sql)
  union = pred | gold
  return 1.0 if not union else len(pred & gold) / len(union)


def schema_items(context: str) -> set[str]:
  items = {m.group(1).lower() for m in TABLE_PATTERN.finditer(context)}
  items |= {m.group(1).lower() for m in COLUMN_TYPE_PATTERN.finditer(context)}
  return items


def schema_linking_reward(predicted_sql: str, target_sql: str, context: str) -> float:
  schema = schema_items(context)

  def used(sql: str) -> set[str]:
    return {word.lower() for word in WORD_PATTERN.findall(sql)} & schema

  pred, gold = used(predicted_sql), used(target_sql)
  union = pred | gold
  return 1.0 if not union else len(pred & gold) / len(union)


def partial_execution_score(predicted_rows: list[tuple[Any, ...]] | None, target_rows: list[tuple[Any, ...]] | None) -> float:
  """Partial execution score: columns, row count, and overlapping values."""
  if not predicted_rows or not target_rows:
    return 0.0

  score = 0.0

  same_num_columns = len(predicted_rows[0]) == len(target_rows[0])
  if same_num_columns:
    score += PARTIAL_COL_WEIGHT

  row_count_ratio = min(len(predicted_rows), len(target_rows)) / max(len(predicted_rows), len(target_rows))
  score += PARTIAL_ROWCOUNT_WEIGHT * row_count_ratio

  pred_values = {repr(value) for row in predicted_rows for value in row}
  target_values = {repr(value) for row in target_rows for value in row}
  if target_values:
    value_overlap = len(pred_values & target_values) / len(target_values)
    score += PARTIAL_VALUE_OVERLAP_WEIGHT * min(value_overlap, 1.0)

  return score


def score_prediction(
  *,
  predicted_sql: str,
  target_sql: str,
  context: str,
  target_rows: list[tuple[Any, ...]] | None = None,
  question: str = "",
) -> dict[str, Any]:
  """Score generated SQL against explicit target SQL and schema context."""
  predicted_sql = clean_sql_for_execution(predicted_sql)
  target_sql = clean_sql_for_execution(target_sql)
  normalized_predicted_sql = normalize_sql(predicted_sql)
  normalized_target_sql = normalize_sql(target_sql)
  target_error = None
  if target_rows is None:
    target_rows, target_error = run_sql(context, target_sql)
    if target_error is not None:
      target_rows = None

  predicted_rows, predicted_error = run_sql(context, predicted_sql)
  execution_error = None
  if predicted_error is not None:
    execution_error = f"predicted query error: {predicted_error}"
  elif target_error is not None:
    execution_error = f"target query error: {target_error}"
  elif not predicted_rows and not target_rows:
    execution_error = "both queries returned no rows"

  compile_ok = execution_error is None
  execution_match = 0.0
  partial_score = 0.0
  ngram_score = ngram_similarity(predicted_sql, target_sql)

  if compile_ok:
    matched = sql_rows_match(predicted_sql, target_sql, predicted_rows, target_rows)
    execution_match = float(matched)
    if not matched:
      partial_score = partial_execution_score(predicted_rows, target_rows)
  schema_score = schema_linking_reward(predicted_sql, target_sql, context)
  normalized_sql_score = SequenceMatcher(None, normalized_predicted_sql, normalized_target_sql).ratio()
  matches_exact = normalized_predicted_sql == normalized_target_sql
  similarity_reward = SIMILARITY_REWARD * (
    SCHEMA_LINK_WEIGHT * schema_score
    + NGRAM_WEIGHT * ngram_score
    + NORMALIZED_SQL_WEIGHT * normalized_sql_score
    + PARTIAL_EXECUTION_WEIGHT * partial_score
  )

  compile_reward = COMPILE_REWARD if compile_ok else ERROR_PENALTY
  execution_match_reward = EXECUTION_MATCH_REWARD * execution_match
  partial_credit_reward = similarity_reward
  reward = compile_reward + execution_match_reward + partial_credit_reward

  return {
    "question": question,
    "target": target_sql,
    "predicted_sql": predicted_sql,
    "reward": reward,
    "compile_reward": compile_reward,
    "execution_match_reward": execution_match_reward,
    "partial_credit_reward": partial_credit_reward,
    "compile": float(compile_ok),
    "execution_match": execution_match,
    "exact_match": float(matches_exact),
    "execution_match_not_exact": float(bool(execution_match) and not matches_exact),
    "similarity": normalized_sql_score,
    "schema_score": schema_score,
    "ngram_score": ngram_score,
    "partial_score": partial_score,
    "sqlite_error": execution_error or "",
  }
