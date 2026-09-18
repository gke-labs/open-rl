"""Text-to-SQL SFT + RL recipe.

See `README.md` for direct `uv run` commands.

Common phase modes:
- `phase=full`: run SFT, then RL
- `phase=sft_only`: stop after SFT; saves `{preset}-sft` adapter for later
- `phase=rl_only`: run RL only. If a `{preset}-sft` adapter was saved by a prior
  sft_only run it's picked up automatically; otherwise RL starts from a fresh LoRA.

Reward execution (`reward.executor`):
- `local` (default): model-written SQL runs in-process with sqlite3.
- `agent_sandbox`: model-written SQL runs in gVisor sandboxes leased from a
  Kubernetes SandboxWarmPool (see examples/common/agent_sandbox.py and
  k8s/agent-sandbox/). Requires running in-cluster with the `sandbox` extra.
  Dataset filtering and target queries stay local: the trust boundary is the
  model's output, not the dataset.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import statistics
import sys
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from tinker import types
from tinker_cookbook.utils import ml_log
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils.helpers import (
  build_examples,
  require_server,
  shuffled_batches,
)
from utils.rewards import (
  Executor,
  aggregate_eval_scores,
  empty_eval_metrics,
  load_dataset_splits,
  local_executor,
  normalize_sql,
  score_eval_prediction_with,
)

# examples/common is importable as `common` when the examples project is
# installed or PYTHONPATH includes examples/; make the script self-sufficient.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_URL = "http://127.0.0.1:9003"
DATASET = "philschmid/gretel-synthetic-text-to-sql"
LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "texttosql_sft_grpo_{preset}"

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Module-level runtime handles bound in the `__main__` block below. The type checker
# needs these annotations so phase functions can read them without "used when not defined".
config: Config
ml_logger: ml_log.Logger
service_client: tinker.ServiceClient
tokenizer: PreTrainedTokenizerBase
executor: Executor = local_executor
EvalMetrics = dict[str, float]
LossValue = float | int


class SandboxStats:
  """Per-step latency and error counts for the reward executor.

  ``exec`` is the sandbox round trip itself; ``wait`` is time spent queueing
  for a lease when more rollouts are being scored than the pool has sandboxes.
  """

  def __init__(self) -> None:
    self.exec_ms: list[float] = []
    self.wait_ms: list[float] = []
    self.errors = 0

  def record(self, *, exec_ms: float, wait_ms: float, error: str | None) -> None:
    self.exec_ms.append(exec_ms)
    self.wait_ms.append(wait_ms)
    if error is not None and error.startswith("sandbox:"):
      self.errors += 1

  def snapshot(self) -> dict[str, float]:
    """Metrics for the calls since the last snapshot; empty when nothing ran."""
    if not self.exec_ms:
      return {}

    def pct(values: list[float], q: float) -> float:
      ordered = sorted(values)
      return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    out = {
      "sandbox_exec_calls": float(len(self.exec_ms)),
      "sandbox_exec_p50_ms": pct(self.exec_ms, 0.50),
      "sandbox_exec_p95_ms": pct(self.exec_ms, 0.95),
      "sandbox_wait_p95_ms": pct(self.wait_ms, 0.95),
      "sandbox_errors": float(self.errors),
    }
    self.exec_ms, self.wait_ms, self.errors = [], [], 0
    return out


sandbox_stats = SandboxStats()


@contextlib.asynccontextmanager
async def reward_executor(reward: RewardConfig) -> AsyncIterator[Executor]:
  """Yield the callable that executes model-written SQL for scoring."""
  if reward.executor == "local":
    yield local_executor
    return
  if reward.executor != "agent_sandbox":
    raise ValueError(f"reward.executor must be 'local' or 'agent_sandbox', got {reward.executor!r}")

  from common.agent_sandbox import AgentSandboxPool, run_sql_in_sandbox

  logging.info("Reward executor: agent_sandbox pool=%d warm_pool=%s/%s", reward.pool_size, reward.namespace, reward.warm_pool)
  pool = AgentSandboxPool(size=reward.pool_size, warm_pool=reward.warm_pool, namespace=reward.namespace, ready_timeout=reward.ready_timeout)
  async with pool:

    async def sandboxed(context: str, query: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
      queued = time.monotonic()
      async with pool.lease() as sandbox:
        started = time.monotonic()
        rows, error = await run_sql_in_sandbox(sandbox, context, query, timeout=reward.exec_timeout)
      done = time.monotonic()
      sandbox_stats.record(exec_ms=(done - started) * 1000.0, wait_ms=(started - queued) * 1000.0, error=error)
      return rows, error

    yield sandboxed


# *** Training phases (SFT + PPO+KL RL) ***
async def bounded(awaitable: Any, what: str) -> Any:
  """Awaits a backend round trip, failing the run instead of waiting on a lost request.

  The tinker client retries a stuck request for two hours by default; a
  training step here takes seconds to a few minutes, so a much shorter bound
  turns a request the backend dropped into a clear error naming the step."""
  try:
    return await asyncio.wait_for(awaitable, timeout=config.progress_timeout_sec)
  except TimeoutError as exc:
    raise RuntimeError(f"No result for {what} within {config.progress_timeout_sec}s; the request was probably lost in the backend") from exc


async def run_sft_phase(
  trainer: tinker.TrainingClient,
  train_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  step_offset: int = 0,
) -> EvalMetrics:
  """Train `trainer` on `train_examples` with cross-entropy. Returns final eval metrics."""
  if config.sft.steps <= 0:
    return await snapshot_eval(trainer, "texttosql_sft_skip", eval_examples)

  batches = shuffled_batches(train_examples, config.sft.batch_size, config.seed)
  losses: list[float] = []
  metrics = empty_eval_metrics()
  s = config.sft
  logging.info(f"Starting SFT: steps={s.steps} batch={min(s.batch_size, len(train_examples))} lr={s.learning_rate:g} n_train={len(train_examples)}")

  for local_step in range(1, config.sft.steps + 1):
    batch = next(batches)
    datums = [ex["datum"] for ex in batch]
    active_tokens = sum(ex["active_tokens"] for ex in batch)

    fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
    optim_future = await trainer.optim_step_async(types.AdamParams(learning_rate=config.sft.learning_rate, grad_clip_norm=config.grad_clip_norm))

    fwdbwd = await bounded(fwdbwd_future, f"forward_backward sft step {local_step}")
    await bounded(optim_future, f"optim_step sft step {local_step}")

    loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
    losses.append(loss)
    global_step = step_offset + local_step
    log_step("sft_train", global_step, loss=loss)

    if local_step % config.sft.eval_every == 0 or local_step == config.sft.steps:
      metrics = await snapshot_eval(trainer, f"texttosql_sft_s{local_step}", eval_examples)
      log_step("sft_eval", global_step, **metrics)
      log_progress("sft step", local_step, f"loss={loss:.4f} {format_eval_metrics(metrics)}")

  if len(losses) >= 2:
    logging.info("Completed SFT: loss_drop=%.1f%%", (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0) * 100)
  return metrics


async def run_rl_phase(
  trainer: tinker.TrainingClient,
  rl_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  step_offset: int = 0,
) -> EvalMetrics:
  """Train `trainer` on `rl_examples` with PPO+KL. Returns final eval metrics."""
  if config.rl.steps <= 0:
    return await snapshot_eval(trainer, "texttosql_rl_skip", eval_examples)

  batches = shuffled_batches(rl_examples, config.rl.prompts_per_step, config.seed + 1)
  metrics = empty_eval_metrics()
  r = config.rl
  logging.info(
    f"Starting RL ({r.loss_fn}): steps={r.steps} prompts/step={min(r.prompts_per_step, len(rl_examples))} "
    f"samples/prompt={r.samples_per_prompt} lr={r.learning_rate:g}"
  )

  sampling_params = types.SamplingParams(max_tokens=config.rl.max_tokens, temperature=config.rl.temperature)
  for local_step in range(1, config.rl.steps + 1):
    # --- Rollout: save weights, sample N completions per prompt, score them ---
    sampler = await bounded(
      trainer.save_weights_and_get_sampling_client_async(name=f"texttosql_rl_rollout_s{local_step}"), f"save_weights step {local_step}"
    )
    examples = next(batches)

    futures = []
    for ex in examples:
      prompt = types.ModelInput.from_ints(tokens=ex["prompt_tokens"])
      futures.append(sampler.sample_async(prompt=prompt, num_samples=config.rl.samples_per_prompt, sampling_params=sampling_params))
    responses = await bounded(asyncio.gather(*futures), f"sampling step {local_step}")

    scored = await bounded(
      asyncio.gather(*(build_rollout(example, seq, tokenizer) for example, response in zip(examples, responses) for seq in response.sequences)),
      f"scoring step {local_step}",
    )
    groups = iter(scored)
    datums: list[types.Datum] = []
    rollouts: list[dict[str, Any]] = []
    for response in responses:
      group = [next(groups) for _ in response.sequences]
      rewards = [rr["reward"] for rr in group]
      # Standardize rewards within each sampled group before the PPO update.
      mean, std = statistics.fmean(rewards), statistics.pstdev(rewards)
      advantages = [0.0] * len(rewards) if len(rewards) < 2 or std < 1e-8 else [(rr - mean) / std for rr in rewards]

      for rollout, advantage in zip(group, advantages):
        if abs(advantage) < 1e-8:
          continue
        if not rollout["completion_tokens"] or len(rollout["completion_tokens"]) != len(rollout["completion_logprobs"]):
          continue
        rollout["advantage"] = advantage
        rollouts.append(rollout)
        prompt, completion, logprobs = rollout["prompt_tokens"], rollout["completion_tokens"], rollout["completion_logprobs"]
        prompt_pad = [0.0] * (len(prompt) - 1)
        datums.append(
          make_datum(
            prompt + completion,
            prompt_pad + [1.0] * len(completion),
            logprobs=prompt_pad + list(logprobs),
            advantages=prompt_pad + [advantage] * len(completion),
          )
        )

    global_step = step_offset + local_step
    exec_metrics = sandbox_stats.snapshot()
    if not datums:
      log_step("rl_train", global_step, loss=0.0, reward=0.0, compile_rate=0.0, execution_match=0.0, similarity=0.0, num_rollouts=0, **exec_metrics)
      continue

    # --- Train: one PPO+KL step on the scored rollouts ---
    fwdbwd_future = await trainer.forward_backward_async(
      datums, config.rl.loss_fn, loss_fn_config={"clip_range": config.rl.clip_range, "kl_coeff": config.rl.kl_coeff}
    )
    optim_future = await trainer.optim_step_async(types.AdamParams(learning_rate=config.rl.learning_rate, grad_clip_norm=config.grad_clip_norm))
    fwdbwd = await bounded(fwdbwd_future, f"forward_backward step {local_step}")
    await bounded(optim_future, f"optim_step step {local_step}")

    # --- Log: best rollout sample, per-step metrics, periodic eval ---
    best = max(rollouts, key=lambda r: (r["reward"], r["execution_match"], r["compile"]))
    compile_str = "YES" if best["compile"] else "NO"
    exec_str = "MATCH" if best["execution_match"] else "NO MATCH"
    sqlite_str = f"\nSQLite:    {best['sqlite_error']}" if best["sqlite_error"] else ""
    logging.info(
      f"\n--- [RL Rollout Sample] ---\nQuestion: {best['question']}\n"
      f"Predicted: {normalize_sql(best['predicted_sql'])}\nTarget:    {normalize_sql(best['target'])}\n"
      f"Reward:    {best['reward']:.2f}\nCompile:   {compile_str}\nExecution: {exec_str}{sqlite_str}\n"
    )

    loss = float(fwdbwd.metrics.get("loss:mean", 0.0))
    reward = statistics.fmean(float(r["reward"]) for r in rollouts)
    compile_rate = statistics.fmean(float(r["compile"]) for r in rollouts)
    exec_rate = statistics.fmean(float(r["execution_match"]) for r in rollouts)
    sim_rate = statistics.fmean(float(r["similarity"]) for r in rollouts)
    log_step(
      "rl_train",
      global_step,
      loss=loss,
      reward=reward,
      compile_rate=compile_rate,
      execution_match=exec_rate,
      similarity=sim_rate,
      num_rollouts=len(rollouts),
      **exec_metrics,
    )
    exec_str = ""
    if exec_metrics:
      exec_str = f" sandbox_exec_p95={exec_metrics['sandbox_exec_p95_ms']:.0f}ms wait_p95={exec_metrics['sandbox_wait_p95_ms']:.0f}ms"
      exec_str += f" errors={int(exec_metrics['sandbox_errors'])}"
    progress = f"reward={reward:.3f} compile={compile_rate * 100:.1f}% exec={exec_rate * 100:.1f}% rollouts={len(rollouts)}{exec_str}"
    log_progress("rl step", local_step, progress)

    if local_step % config.rl.eval_every == 0 or local_step == config.rl.steps:
      metrics = await snapshot_eval(trainer, f"texttosql_rl_s{local_step}", eval_examples)
      log_step("rl_eval", global_step, **metrics)
      log_progress("rl eval", local_step, format_eval_metrics(metrics))

  return metrics


async def run_training(preset: str, metrics_path: Path) -> dict[str, float | str]:
  """Orchestrate the configured phases. Reads config/service_client/tokenizer from module scope."""
  global executor
  async with reward_executor(config.reward) as executor:
    return await _run_training(preset, metrics_path)


async def _run_training(preset: str, metrics_path: Path) -> dict[str, float | str]:
  server_model = await require_server(service_client, config.base_url)
  logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

  sft_train, rl_train, eval_examples = load_example_splits(config, tokenizer)

  # One trainer owns the whole run. Resume from a prior SFT state in rl_only mode;
  # otherwise a fresh LoRA that we'll train through SFT then RL.
  sft_state_name = f"{preset}-sft"
  final_state_name = f"{preset}-final"
  training_client = await service_client.create_lora_training_client_async(
    base_model=config.model.base_model,
    rank=config.model.rank,
    seed=config.seed,
    train_mlp=True,
    train_attn=True,
    train_unembed=False,
  )
  if config.phase == "rl_only" and config.sft_adapter_name:
    load_future = await training_client.load_state_async(config.sft_adapter_name)
    await load_future.result_async()
    logging.info(f"rl_only: loaded SFT adapter {config.sft_adapter_name!r}")

  before_metrics = await snapshot_eval(training_client, "texttosql_before", eval_examples)
  log_step("eval_baseline", 0, **before_metrics)
  logging.info("baseline %s", format_eval_metrics(before_metrics))

  # Defaults keep the skipped-phase outputs tied to the last real measurement.
  after_sft_metrics = before_metrics
  after_rl_metrics = before_metrics
  sft_state_path = ""

  if config.phase in {"full", "sft_only"}:
    logging.info(">>> Phase: SFT")
    after_sft_metrics = await run_sft_phase(training_client, sft_train, eval_examples, step_offset=0)
    sft_state_path = training_client.save_state(sft_state_name).result().path
    logging.info(f"Post-SFT state saved to {sft_state_path}")
    after_rl_metrics = after_sft_metrics  # fallback if RL is skipped

  if config.phase in {"full", "rl_only"}:
    logging.info(">>> Phase: RL")
    step_offset = 0 if config.phase == "rl_only" else config.sft.steps
    after_rl_metrics = await run_rl_phase(training_client, rl_train, eval_examples, step_offset=step_offset)

  final_state_path = training_client.save_state(final_state_name).result().path
  logging.info(f"Final state saved to {final_state_path}")

  logging.info("Saved metrics to %s", metrics_path)
  logging.info(
    "[summary] exact=%s execution=%s exec_not_exact=%s similarity=%s",
    format_metric_chain("exact_match", before_metrics, after_sft_metrics, after_rl_metrics, config.phase),
    format_metric_chain("execution_match", before_metrics, after_sft_metrics, after_rl_metrics, config.phase),
    format_metric_chain("execution_match_not_exact", before_metrics, after_sft_metrics, after_rl_metrics, config.phase),
    format_metric_chain("similarity", before_metrics, after_sft_metrics, after_rl_metrics, config.phase),
  )
  ml_logger.close()

  result: dict[str, float | str] = {
    "before_execution_match": before_metrics["execution_match"],
    "after_sft_execution_match": after_sft_metrics["execution_match"],
    "after_rl_execution_match": after_rl_metrics["execution_match"],
    "before_exact_match": before_metrics["exact_match"],
    "after_sft_exact_match": after_sft_metrics["exact_match"],
    "after_rl_exact_match": after_rl_metrics["exact_match"],
    "before_execution_match_not_exact": before_metrics["execution_match_not_exact"],
    "after_sft_execution_match_not_exact": after_sft_metrics["execution_match_not_exact"],
    "after_rl_execution_match_not_exact": after_rl_metrics["execution_match_not_exact"],
    "before_similarity": before_metrics["similarity"],
    "after_sft_similarity": after_sft_metrics["similarity"],
    "after_rl_similarity": after_rl_metrics["similarity"],
    "metrics_path": str(metrics_path),
  }
  result["final_state_path"] = final_state_path
  if sft_state_path:
    result["sft_state_path"] = sft_state_path
  return result


# *** Data + datum helpers ***
def load_example_splits(config: Any, tokenizer: PreTrainedTokenizerBase) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  """Return (sft_train, rl_train, eval), skipping phase-specific builds when disabled."""
  do_sft = config.phase in {"full", "sft_only"}
  do_rl = config.phase in {"full", "rl_only"}
  train_limit = max(config.dataset.train_limit if do_sft else 0, config.dataset.rl_train_limit if do_rl else 0)
  train_rows, eval_rows = load_dataset_splits(
    dataset_name=config.dataset.name,
    dataset_limit=config.dataset.limit,
    train_limit=train_limit * 2,
    eval_limit=config.dataset.eval_limit * 2,
    seed=config.seed,
  )

  sft_examples = build_examples(tokenizer, train_rows, config.dataset.train_limit) if do_sft else []
  for example in sft_examples:
    prompt_tokens, full_tokens = example["prompt_tokens"], example["full_tokens"]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * (len(full_tokens) - len(prompt_tokens))
    example["datum"] = make_datum(full_tokens, weights)

  rl_examples = build_examples(tokenizer, train_rows, config.dataset.rl_train_limit) if do_rl else []
  eval_examples = build_examples(tokenizer, eval_rows, config.dataset.eval_limit)

  if do_sft and not sft_examples:
    raise RuntimeError("No SFT examples fit within the max sequence length.")
  if do_rl and not rl_examples:
    raise RuntimeError("No RL examples with executable target rows were found.")
  if not eval_examples:
    raise RuntimeError("No evaluation examples with executable seed data were found.")

  logging.info("Data: %s SFT train, %s RL train, %s eval", len(sft_examples), len(rl_examples), len(eval_examples))
  return sft_examples, rl_examples, eval_examples


def make_datum(
  tokens: Sequence[int],
  weights: Sequence[LossValue],
  *,
  logprobs: Sequence[LossValue] | None = None,
  advantages: Sequence[LossValue] | None = None,
) -> types.Datum:
  tokens = list(tokens)
  loss_fn_inputs: dict[str, Any] = {
    "target_tokens": tokens[1:],
    "weights": list(weights),
  }
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = list(logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = list(advantages)

  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
    loss_fn_inputs=cast(Any, loss_fn_inputs),
  )


# *** Logging helpers ***
def log_step(phase: str, step: int, **metrics: float) -> None:
  """Record one training/eval datapoint to the metrics log."""
  ml_logger.log_metrics({"phase": phase, **metrics}, step=step)


def log_progress(tag: str, step: int, details: str) -> None:
  """Pretty-print one step of progress to stdout. The JSONL log is the source of truth."""
  logging.info("[%s %03d] %s", tag, step, details)


def format_eval_metrics(metrics: EvalMetrics) -> str:
  return (
    f"eval_exact={metrics['exact_match'] * 100:.1f}% "
    f"eval_exec={metrics['execution_match'] * 100:.1f}% "
    f"eval_exec_not_exact={metrics['execution_match_not_exact'] * 100:.1f}% "
    f"eval_sim={metrics['similarity'] * 100:.1f}%"
  )


def format_metric_chain(name: str, before: EvalMetrics, after_sft: EvalMetrics, after_rl: EvalMetrics, phase: str) -> str:
  if phase == "rl_only":
    return f"{before[name] * 100:.1f}%->{after_rl[name] * 100:.1f}%"
  elif phase == "sft_only":
    return f"{before[name] * 100:.1f}%->{after_sft[name] * 100:.1f}%"
  else:
    return f"{before[name] * 100:.1f}%->{after_sft[name] * 100:.1f}%->{after_rl[name] * 100:.1f}%"


# *** Eval + rollout helpers ***
async def snapshot_eval(trainer: tinker.TrainingClient, alias: str, eval_examples: list[dict[str, Any]]) -> EvalMetrics:
  """Snapshot current weights to a sampler and run the eval loop against it."""
  sampler = await trainer.save_weights_and_get_sampling_client_async(name=alias)
  return await sample_eval_metrics(sampler, tokenizer, alias, eval_examples, max_tokens=config.dataset.eval_max_tokens, seed=config.seed)


async def build_rollout(
  example: dict[str, Any],
  sequence: Any,
  tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
  """Decode one sampled sequence, score it through the executor, attach token/logprob fields for PPO."""
  predicted_sql = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
  score = await score_eval_prediction_with(executor, predicted_sql, example)
  return {
    **score,
    "prompt_tokens": example["prompt_tokens"],
    "completion_tokens": list(sequence.tokens),
    "completion_logprobs": [float(v) for v in (sequence.logprobs or [])],
  }


async def sample_eval_metrics(
  sampler: Any,
  tokenizer: PreTrainedTokenizerBase,
  alias: str,
  examples: list[dict[str, Any]],
  max_tokens: int,
  seed: int,
) -> EvalMetrics:
  """Sample current weights; SQL reward scoring and aggregation live in utils.rewards."""
  scores: list[dict[str, Any]] = []
  futures = [
    sampler.sample_async(
      prompt=types.ModelInput.from_ints(tokens=example["prompt_tokens"]),
      num_samples=1,
      sampling_params=types.SamplingParams(max_tokens=max_tokens, seed=seed + idx, temperature=0.0),
    )
    for idx, example in enumerate(examples)
  ]
  responses = await asyncio.gather(*futures)
  predictions = [tokenizer.decode(response.sequences[0].tokens if response.sequences else [], skip_special_tokens=True) for response in responses]
  infos = await asyncio.gather(*(score_eval_prediction_with(executor, sql, example) for sql, example in zip(predictions, examples)))
  sandbox_stats.snapshot()  # eval latency is not a training-step metric

  for idx, (example, info) in enumerate(zip(examples, infos)):
    predicted = normalize_sql(info["predicted_sql"])
    target = normalize_sql(example["target"])
    matches_execution = bool(info["execution_match"])
    execution_error = info["sqlite_error"]
    scores.append(info)

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

  return aggregate_eval_scores(scores)


# *** Config + presets ***
# Every leaf field is reachable on the CLI via dotted paths, e.g.
#   rl.steps=20 model.rank=16
@chz.chz
class ModelConfig:
  base_model: str
  tokenizer_name: str
  rank: int = 32


@chz.chz
class DatasetConfig:
  name: str = DATASET
  limit: int = 12_500
  train_limit: int = 100
  rl_train_limit: int = 64
  eval_limit: int = 100
  eval_max_tokens: int = 64


@chz.chz
class SftConfig:
  steps: int = 100
  batch_size: int = 1
  learning_rate: float = 5e-5
  eval_every: int = 100


@chz.chz
class RlConfig:
  steps: int = 40
  prompts_per_step: int = 4
  samples_per_prompt: int = 4
  learning_rate: float = 1e-5
  temperature: float = 0.8
  max_tokens: int = 64
  eval_every: int = 10
  loss_fn: str = "ppo"
  clip_range: float = 0.2
  kl_coeff: float = 0.1


@chz.chz
class RewardConfig:
  # "local": in-process sqlite3. "agent_sandbox": gVisor sandboxes leased from a
  # Kubernetes SandboxWarmPool through examples/common/agent_sandbox.py.
  executor: str = "local"
  warm_pool: str = "text-to-sql-executor"
  namespace: str = "openrl-system"
  pool_size: int = 4
  ready_timeout: int = 180
  exec_timeout: int = 30


@chz.chz
class Config:
  model: ModelConfig
  phase: str = "full"  # "full" | "sft_only" | "rl_only"
  fine_tuning_type: str = os.getenv("OPEN_RL_FINE_TUNING_TYPE", "lora")
  base_url: str = os.getenv("TINKER_BASE_URL", BASE_URL)
  seed: int = 30
  grad_clip_norm: float = 0.3
  # Fail the run if a backend round trip takes longer than this; a step takes
  # seconds to a few minutes, so 15 minutes means a request is lost or stuck.
  progress_timeout_sec: int = 15 * 60
  log_dir: str = str(LOG_DIR)
  sft_adapter_name: str | None = None
  dataset: DatasetConfig = chz.field(default_factory=DatasetConfig)
  sft: SftConfig = chz.field(default_factory=SftConfig)
  rl: RlConfig = chz.field(default_factory=RlConfig)
  reward: RewardConfig = chz.field(default_factory=RewardConfig)


GEMMA4_E2B = {"model.base_model": "google/gemma-4-e2b", "model.tokenizer_name": "google/gemma-4-e2b"}

PRESETS = {
  "gemma4_e2b_rl_recipe": chz.Blueprint(Config).apply(
    {
      **GEMMA4_E2B,
      "seed": 42,
      "sft.steps": 5,
      "sft.eval_every": 5,
      "sft.learning_rate": 5e-5,
      "rl.steps": 80,
      "rl.eval_every": 10,
      "rl.loss_fn": "ppo",
      "rl.kl_coeff": 0.1,
      "rl.clip_range": 0.2,
      "rl.learning_rate": 5e-6,
      "rl.samples_per_prompt": 8,
      "rl.prompts_per_step": 8,
      "dataset.train_limit": 100,
      "dataset.rl_train_limit": 5000,
      "dataset.eval_limit": 100,
    },
    layer_name="gemma4_e2b_rl_recipe",
  )
}


# *** Entrypoint ***
if __name__ == "__main__":
  logging.getLogger("tinker").setLevel(logging.WARNING)

  preset = sys.argv[1]
  config = PRESETS[preset].clone().make_from_argv(sys.argv[2:], allow_hyphens=True)

  log_dir = Path(config.log_dir.replace("{preset}", f"{preset}_{config.phase}"))
  metrics_path = log_dir / "metrics.jsonl"
  ml_logger = ml_log.setup_logging(log_dir=str(log_dir), config=config, do_configure_logging_module=True)
  default_headers = {}
  if config.fine_tuning_type and config.fine_tuning_type != "lora":
    default_headers["x-open-rl-fine-tuning-type"] = config.fine_tuning_type
  service_client = tinker.ServiceClient(
    api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
    base_url=config.base_url,
    default_headers=default_headers,
  )
  tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)

  asyncio.run(run_training(preset, metrics_path))
