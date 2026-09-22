#!/usr/bin/env python3
"""Start a local Open-RL backend and run example training scripts against it.

Scenarios ("tiny-" = minimal overfit/smoke tests; the rest are real workloads):
  tiny-lora / tiny-fft  examples/tiny/tiny_sft.py (overfit one example; loss must drop)
  tiny-rl               examples/tiny/tiny_rl.py (sample -> reward -> train)
  lora-textsql          examples/text-to-sql/texttosql_sft_grpo.py (real RL recipe, trimmed)
  fft-gsm8k             examples/sft/gsm8k/gsm8k_sft.py + vLLM eval (min_accuracy gate)
  fft-gsm8k-x2          two concurrent fft-gsm8k jobs sharing one GPU through the
                        accel timeslicer (asserts both workers checkpoint/restore)
  tiny-rl-x2-families / tiny-fft-rl-x2-families
                        two concurrent tiny-rl jobs on base_model and
                        second_base_model, one per model family (asserts each
                        job earns a reward, i.e. got its own tokenizer)
  fft-textsql-rl-x2     two concurrent Text-to-SQL FFT RL jobs; extra_a= and
                        extra_b= override each job separately on top of extra=,
                        so one run can compare two configs or two base models

Examples:
  uv run --extra gpu python scripts/run_training_e2e.py scenario=tiny-lora
  uv run --extra gpu python scripts/run_training_e2e.py scenario=tiny-rl steps=4
  uv run --extra gpu python scripts/run_training_e2e.py scenario=lora-textsql
  uv run --extra gpu python scripts/run_training_e2e.py scenario=fft-gsm8k extra='batch=2 rank=32'
  uv run --extra gpu python scripts/run_training_e2e.py scenario=fft-textsql-rl-x2 steps=40 \
      base_model=google/gemma-4-e2b extra_a='rl.learning_rate=1e-6' extra_b='rl.learning_rate=5e-6'

The example scripts validate their own results and exit nonzero on failure.
`base_url=...` targets an existing backend instead of starting one. `steps=N`
sets the example's step count and `extra='k=v ...'` forwards additional chz
overrides to it; two-job scenarios also take `extra_a=` / `extra_b=`, applied on
top of `extra=` to job-a / job-b only. The examples uv environment is kept separate from the root
server/eval uv environment; override that path with
OPEN_RL_EXAMPLES_UV_PROJECT_ENVIRONMENT if needed.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chz

REPO_ROOT = Path(__file__).resolve().parents[1]
GSM8K_ANSWER_RE = re.compile(r"-?\d[\d,]*")


@chz.chz
class RunConfig:
  scenario: Literal[
    "tiny-lora",
    "tiny-fft",
    "tiny-rl",
    "tiny-fft-rl",
    "tiny-fft-rl-x2",
    "tiny-rl-x2-families",
    "tiny-fft-rl-x2-families",
    "lora-textsql",
    "lora-gsm8k-rl",
    "lora-gsm8k-rl-x2",
    "fft-gsm8k",
    "fft-gsm8k-x2",
    "fft-gsm8k-rl",
    "fft-gsm8k-rl-x2",
    "fft-gsm8k-rl-x2-compare",
    "fft-gsm8k-rl-x2-diffing-compare",
    "fft-gsm8k-rl-x3",
    "fft-gsm8k-rl-x3-hetero-8b-0.6b",
    "fft-gsm8k-rl-hetero",
    "fft-textsql-rl",
    "fft-textsql-rl-x2",
    "lora-fft-gsm8k-rl-x4",
  ]
  sampling_backend: str = "vllm"
  trainer_gpu: str = "0"
  sampler_gpu: str = "1"
  base_url: str = ""
  base_model: str = "Qwen/Qwen2.5-0.5B"
  # The other model family for the *-x2-families scenarios; must differ from
  # base_model in vocabulary, not just in size.
  second_base_model: str = "google/gemma-4-e2b"
  jitter_sec: int = 180
  steps: int | None = None
  group_size: int = 8
  groups_per_batch: int = 8
  max_tokens: int = 512
  # Calibration (A100, 50 FFT steps on Qwen2.5-0.5B): measured 15.6% accuracy.
  # 100 examples + 5% floor keeps healthy-run flake risk below ~0.1% while
  # still failing a lobotomized checkpoint; eval costs ~15s in vLLM.
  eval_examples: int = 100
  min_accuracy: float = 0.05
  weight_sync_strategy: str = ""
  extra: str = ""
  # Per-job overrides for the *-x2 scenarios, layered over `extra`. Setting
  # model.base_model in one of them also switches that job's tokenizer.
  extra_a: str = ""
  extra_b: str = ""
  host: str = "127.0.0.1"
  port: int | None = None
  uv_extra: str = "gpu"
  eval_uv_extra: str = "vllm"
  log_dir: str = "/tmp/open-rl-training-tests"
  startup_timeout: float = 300.0
  train_token_budget: int = 65_536
  vllm_gpu_memory_utilization: float = 0.70


@dataclass
class ManagedProcess:
  name: str
  process: subprocess.Popen
  log_path: Path


def unused_tcp_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def wait_until(description: str, check: Callable[[], bool], timeout: float) -> None:
  deadline = time.monotonic() + timeout
  last_error: Exception | None = None
  while time.monotonic() < deadline:
    try:
      if check():
        return
    except Exception as exc:
      last_error = exc
    time.sleep(1)
  raise TimeoutError(f"Timed out waiting for {description}: {last_error}")


def http_ok(url: str) -> bool:
  with urllib.request.urlopen(url, timeout=2) as response:
    response.read()
  return True


def redis_ok(host: str, port: int) -> bool:
  with socket.create_connection((host, port), timeout=1) as client:
    client.sendall(b"*1\r\n$4\r\nPING\r\n")
    return client.recv(64).startswith(b"+PONG")


def print_log_tail(path: Path, lines: int = 100) -> None:
  if not path.exists():
    return
  content = path.read_text(encoding="utf-8", errors="replace").splitlines()
  print(f"\n[training-e2e] last {min(lines, len(content))} lines from {path}:")
  for line in content[-lines:]:
    print(line)


def cleanup_remote_models(base_url: str, outputs: list[str]) -> None:
  """Find model IDs in recipe output and request worker cleanup via /api/v1/delete_model."""
  if not base_url.startswith("http"):
    return
  model_ids = set()
  for output in outputs:
    if isinstance(output, str):
      for match in re.finditer(r"(?:TrainingClient|ServiceClient) initialized for (?:model|session) ([a-f0-9-]+)", output):
        model_ids.add(match.group(1))
  for model_id in sorted(model_ids):
    try:
      url = f"{base_url}/api/v1/delete_model"
      data = json.dumps({"model_id": model_id}).encode("utf-8")
      req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
      with urllib.request.urlopen(req) as response:
        response.read()
      print(f"[training-e2e] successfully requested cleanup of workers for model {model_id}")
    except Exception as exc:
      print(f"[training-e2e] warning: failed to request worker cleanup for {model_id}: {exc}")


def launch(
  processes: list[ManagedProcess],
  name: str,
  command: list[str],
  env: dict[str, str],
  log_path: Path,
  ready: Callable[[], bool],
  timeout: float,
) -> None:
  print(f"[training-e2e] starting {name}: {' '.join(command)}")
  process = subprocess.Popen(
    command,
    cwd=REPO_ROOT,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
  )

  def stream_log():
    with log_path.open("w", encoding="utf-8") as log_file:
      for line in iter(process.stdout.readline, ""):
        log_file.write(line)
        log_file.flush()
        print(f"[{name}] {line.rstrip()}")

  t = threading.Thread(target=stream_log, daemon=True)
  t.start()

  processes.append(ManagedProcess(name=name, process=process, log_path=log_path))
  try:
    wait_until(name, ready, timeout)
  except Exception:
    raise


def stop_process(managed: ManagedProcess) -> None:
  for sig in (signal.SIGTERM, signal.SIGKILL):
    if managed.process.poll() is not None:
      return
    try:
      os.killpg(os.getpgid(managed.process.pid), sig)
    except ProcessLookupError:
      return
    try:
      managed.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      continue


def uv_run(extra: str) -> list[str]:
  return ["uv", "run", "--extra", extra]


def backend_python(config: RunConfig) -> list[str]:
  """Interpreter for the API server. OPEN_RL_E2E_PYTHON names a pre-built venv
  (e.g. a TPU VM's, where the torch_tpu wheel cannot come from uv extras)."""
  if python := os.getenv("OPEN_RL_E2E_PYTHON"):
    return [python]
  return uv_run(config.uv_extra) + ["python"]


def examples_python() -> list[str]:
  """Interpreter for the example/recipe clients; same override as backend_python."""
  if python := os.getenv("OPEN_RL_E2E_PYTHON"):
    return [python]
  return ["uv", "--project", "examples", "run", "python"]


def open_rl_tmp_dir(config: RunConfig) -> Path:
  return Path(config.log_dir) / "open-rl-tmp"


def base_env(config: RunConfig) -> dict[str, str]:
  return {
    **os.environ,
    "BASE_MODEL": config.base_model,
    "ENABLE_GCP_TRACE": "0",
    "OPEN_RL_TMP_DIR": str(open_rl_tmp_dir(config)),
    "OPEN_RL_TRAIN_TOKEN_BUDGET": str(config.train_token_budget),
    "PYTHONUNBUFFERED": "1",
    "SAMPLING_BACKEND": config.sampling_backend,
    "TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "tml-dummy-key"),
    "TOKENIZERS_PARALLELISM": "false",
  }


def start_backend(config: RunConfig, processes: list[ManagedProcess]) -> str:
  if config.base_url:
    print(f"[training-e2e] using existing Open-RL backend at {config.base_url}")
    return config.base_url

  log_dir = Path(config.log_dir)
  port = config.port or unused_tcp_port()
  base_url = f"http://{config.host}:{port}"
  env = base_env(config)
  env["TRAINER_CUDA_VISIBLE_DEVICES"] = config.trainer_gpu
  env["SAMPLER_CUDA_VISIBLE_DEVICES"] = config.sampler_gpu
  if config.sampling_backend == "vllm":
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["VLLM_GPU_MEMORY_UTILIZATION"] = str(config.vllm_gpu_memory_utilization)
  else:
    env["CUDA_VISIBLE_DEVICES"] = config.trainer_gpu

  need_redis = "fft" in config.scenario or config.sampling_backend == "vllm"
  if need_redis:
    if shutil.which("redis-server") is None:
      raise RuntimeError("redis-server is required for multi-process e2e scenarios (vLLM sampling or FFT)")
    redis_port = unused_tcp_port()
    launch(
      processes,
      "redis",
      ["redis-server", "--save", "", "--appendonly", "no", "--bind", "127.0.0.1", "--port", str(redis_port)],
      os.environ.copy(),
      log_dir / "redis.log",
      lambda: redis_ok("127.0.0.1", redis_port),
      timeout=60,
    )
    env["REDIS_URL"] = f"redis://127.0.0.1:{redis_port}/0"

  if "fft" in config.scenario:
    if shutil.which("cuda-checkpoint") is None:
      raise RuntimeError(
        "cuda-checkpoint is required for FFT e2e scenarios (the snapshot agent checkpoints workers around every batch); "
        "install the binary matching your driver from https://github.com/NVIDIA/cuda-checkpoint"
      )
    snapshot_socket = log_dir / "accel-timeslicer.sock"
    snapshot_socket.unlink(missing_ok=True)
    launch(
      processes,
      "accel-timeslicer",
      uv_run(config.uv_extra) + ["python", "-m", "accel_timeslicer.serve"],
      {**base_env(config), "OPEN_RL_ACCEL_TIMESLICER_SOCKET": str(snapshot_socket)},
      log_dir / "accel-timeslicer.log",
      snapshot_socket.is_socket,
      timeout=60,
    )
    env["OPEN_RL_ACCEL_TIMESLICER_SOCKET"] = str(snapshot_socket)
    env["OPEN_RL_ENABLE_FFT"] = "true"
  else:
    env.pop("OPEN_RL_ENABLE_FFT", None)

  launch(
    processes,
    "backend",
    backend_python(config) + ["-m", "uvicorn", "server.api_server:app", "--host", config.host, "--port", str(port)],
    env,
    log_dir / "backend.log",
    lambda: http_ok(f"{base_url}/api/v1/healthz"),
    timeout=config.startup_timeout,
  )
  return base_url


def clean_cli_extra(extra: str) -> list[str]:
  """Filter out open-rl specific weight_sync_strategy/diffing key-value options from CLI extras."""
  return [token for token in shlex.split(extra) if not (token.startswith("weight_sync_strategy=") or token.startswith("jitter_sec="))]


def _set_fft_delta_apply(env: dict[str, str]) -> None:
  """FFT scenarios default to in-place delta patching, but an apply method
  already in the environment (run_cluster_e2e.py's
  --weight-sync-delta-apply-method) wins. OPEN_RL_IN_PLACE_DELTA forces the
  in-place path in the sampler regardless of the method, so it is only set
  when in-place is what was asked for."""
  method = env.setdefault("OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD", "patch_in_place")
  if method == "patch_in_place":
    env["OPEN_RL_IN_PLACE_DELTA"] = "1"
  else:
    env.pop("OPEN_RL_IN_PLACE_DELTA", None)


def examples_env(config: RunConfig) -> dict[str, str]:
  env = os.environ.copy()
  env["OPEN_RL_TMP_DIR"] = str(open_rl_tmp_dir(config))
  env["PYTHONUNBUFFERED"] = "1"
  env.setdefault("TINKER_API_KEY", "tml-dummy-key")
  for token in shlex.split(config.extra):
    if token.startswith("weight_sync_strategy="):
      env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = token.split("=", 1)[1]
  if config.weight_sync_strategy:
    env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = config.weight_sync_strategy
  if config.scenario.startswith("fft") or "fft" in config.scenario:
    env["OPEN_RL_FINE_TUNING_TYPE"] = "full"
    _set_fft_delta_apply(env)
  existing_path = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = f"examples:{existing_path}" if existing_path else "examples"
  return env


def run_command(command: list[str], env: dict[str, str] | None = None, watch: list[ManagedProcess] | None = None, prefix: str = "") -> str:
  """Run a command, streaming output. If a watched backend process exits first,
  kill the command and fail immediately instead of letting the client retry forever."""
  print(f"[training-e2e] running{f' {prefix}' if prefix else ''}: {' '.join(command)}")
  process = subprocess.Popen(
    command,
    cwd=REPO_ROOT,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )
  dead: list[ManagedProcess] = []

  def watchdog() -> None:
    while process.poll() is None:
      for managed in watch or []:
        if managed.process.poll() is not None:
          dead.append(managed)
          process.kill()
          return
      time.sleep(2)

  watcher = threading.Thread(target=watchdog, daemon=True)
  watcher.start()
  assert process.stdout is not None
  output_lines: list[str] = []
  for line in process.stdout:
    print(prefix + line, end="")
    output_lines.append(line)
  returncode = process.wait()
  watcher.join(timeout=10)
  if dead:
    print_log_tail(dead[0].log_path)
    raise RuntimeError(f"{dead[0].name} exited (code {dead[0].process.returncode}) while the example was running; see {dead[0].log_path}")
  output = "".join(output_lines)
  if returncode != 0:
    raise subprocess.CalledProcessError(returncode, command, output=output)
  return output


def parse_overrides(*specs: str) -> dict[str, str]:
  """Merge `k=v ...` strings left to right; later specs win."""
  overrides: dict[str, str] = {}
  for spec in specs:
    overrides.update(item.split("=", 1) for item in shlex.split(spec))
  return overrides


def job_overrides(config: RunConfig, job: str) -> dict[str, str]:
  """`extra` plus the per-job `extra_a` / `extra_b` for job-a / job-b."""
  per_job = {"job-a": config.extra_a, "job-b": config.extra_b}.get(job, "")
  return parse_overrides(config.extra, per_job)


def run_example(
  config: RunConfig,
  script: list[str],
  defaults: dict[str, str],
  watch: list[ManagedProcess] | None = None,
  prefix: str = "",
  overrides: dict[str, str] | None = None,
) -> str:
  if overrides is None:
    overrides = parse_overrides(config.extra)
  args = [f"{key}={value}" for key, value in {**defaults, **overrides}.items()]
  return run_command([*examples_python(), *script, *args], env=examples_env(config), watch=watch, prefix=prefix)


def run_tiny(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  script = "tiny_rl" if "rl" in config.scenario else "tiny_sft"
  defaults = {
    "base_model": config.base_model,
    "base_url": base_url,
    "log_dir": str(Path(config.log_dir) / config.scenario.replace("-", "_")),
  }
  if "fft" in config.scenario:
    # tiny SFT/RL default is tuned for LoRA adapters; full fine-tuning all
    # params with Adam at that rate diverges.
    defaults["learning_rate"] = "1e-5"
  if config.steps is not None:
    defaults["steps"] = str(config.steps)
  run_example(config, [f"examples/tiny/{script}.py"], defaults, watch=watch)


def extract_gsm8k_gold(answer: str) -> str:
  tail = answer.split("####")[-1]
  match = GSM8K_ANSWER_RE.search(tail)
  if match is None:
    raise ValueError(f"Could not extract GSM8K gold answer from {answer!r}")
  return match.group(0).replace(",", "")


def write_gsm8k_eval_data(config: RunConfig) -> Path:
  from datasets import load_dataset

  data_path = Path(config.log_dir) / "gsm8k_eval.json"
  dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{config.eval_examples}]")
  data = [
    {
      "prompt": f"Question: {row['question']}\nAnswer:",
      "gold": extract_gsm8k_gold(row["answer"]),
    }
    for row in dataset
  ]
  data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
  return data_path


def resolve_eval_model_path(output: str) -> str:
  for line in reversed(output.splitlines()):
    if line.startswith("eval_model_path="):
      path = line.removeprefix("eval_model_path=").strip()
      return path
  raise RuntimeError("GSM8K SFT finished without printing eval_model_path=...")


def run_gsm8k_train(config: RunConfig, base_url: str, watch: list[ManagedProcess], log_subdir: str, prefix: str = "") -> str:
  defaults = {
    "base_model": config.base_model,
    "base_url": base_url,
    "log_path": str(Path(config.log_dir) / log_subdir),
    "max_steps": str(config.steps if config.steps is not None else 10),
    "batch": "1",
    "rank": "16",
    "max_len": "640",
    "save_every": "0",
    "behavior_if_log_dir_exists": "delete",
  }
  return run_example(config, ["examples/sft/gsm8k/gsm8k_sft.py"], defaults, watch=watch, prefix=prefix)


def run_gsm8k_eval(config: RunConfig, model_path: str | list[str]) -> None:
  paths = model_path if isinstance(model_path, list) else [model_path]
  path_args = []
  for p in paths:
    path_args.extend(["--path", p])
  run_command(
    [*examples_python(), "examples/sft/gsm8k/vllm_eval.py"]
    + path_args
    + [
      "--base-url",
      config.base_url or "http://127.0.0.1:8000",
      "--data",
      str(write_gsm8k_eval_data(config)),
      "--gpu-memory-utilization",
      str(config.vllm_gpu_memory_utilization),
      "--min-accuracy",
      str(config.min_accuracy),
    ]
  )


def run_gsm8k(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  output = run_gsm8k_train(config, base_url, watch, "fft_gsm8k")
  run_gsm8k_eval(config, resolve_eval_model_path(output))


def check_snapshot_interleaving(config: RunConfig) -> None:
  if config.base_url:
    print("[training-e2e] external backend; skipping snapshot agent interleave check")
    return

  log_path = Path(config.log_dir) / "accel-timeslicer.log"
  if not log_path.exists():
    return
  text = log_path.read_text(encoding="utf-8", errors="replace")

  checkpointed = set(re.findall(r"checkpointed workload (\S+) claim \S+", text))
  restored = set(re.findall(r"restored workload (\S+) claim \S+", text))

  cp_t = {workload for workload in checkpointed if ":trainer-" in workload or workload.startswith("trainer-")}
  rs_t = {workload for workload in restored if ":trainer-" in workload or workload.startswith("trainer-")}
  if len(cp_t) < 2 or len(rs_t) < 2:
    raise RuntimeError(
      f"Expected both FFT trainer workers to interleave, but saw checkpoints {sorted(cp_t)} and restores {sorted(rs_t)} in {log_path}"
    )
  print(f"[training-e2e] trainer accel timeslicer time-sliced: checkpointed workloads {sorted(cp_t)}, restored workloads {sorted(rs_t)}")

  if config.sampling_backend == "vllm":
    cp_s = {workload for workload in checkpointed if ":sampler-" in workload or workload.startswith("sampler-")}
    rs_s = {workload for workload in restored if ":sampler-" in workload or workload.startswith("sampler-")}
    if len(cp_s) < 2 or len(rs_s) < 2:
      raise RuntimeError(
        f"Expected both FFT sampler workers to interleave, but saw checkpoints {sorted(cp_s)} and restores {sorted(rs_s)} in {log_path}"
      )
    print(f"[training-e2e] sampler accel timeslicer time-sliced: checkpointed workloads {sorted(cp_s)}, restored workloads {sorted(rs_s)}")


def run_gsm8k_x2(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Two concurrent FFT jobs against the same backend: each create_model spawns
  its own worker, and the accel timeslicer time-slices the GPU between them."""
  results: dict[str, str | BaseException] = {}

  def train(job: str) -> None:
    try:
      results[job] = run_gsm8k_train(config, base_url, watch, f"fft_gsm8k_{job}", prefix=f"[{job}] ")
    except BaseException as exc:
      results[job] = exc

  threads = [threading.Thread(target=train, args=(job,)) for job in ("job-a", "job-b")]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  for job, result in sorted(results.items()):
    if isinstance(result, BaseException):
      raise RuntimeError(f"gsm8k {job} failed") from result

  check_snapshot_interleaving(config)
  eval_paths = []
  for _, result in sorted(results.items()):
    assert isinstance(result, str)
    eval_paths.append(resolve_eval_model_path(result))
  print(f"[training-e2e] evaluating jobs in single micro-batched invocation: {eval_paths}")
  run_gsm8k_eval(config, eval_paths)


def _math_rl_train_module_and_renderer(base_model: str) -> tuple[str, str]:
  if "gemma" in base_model.lower():
    renderer = "gemma2" if "gemma-2" in base_model.lower() else "gemma4"
    return "recipes.math_rl.train_gemma", renderer
  if "Qwen3" in base_model and "Instruct" not in base_model:
    return "recipes.math_rl.train_cli", "qwen3"
  if "Instruct" in base_model or "Qwen2.5" in base_model:
    return "recipes.math_rl.train_cli", "qwen3_instruct"
  return "recipes.math_rl.train_cli", "qwen3"


def run_gsm8k_rl(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  prefix = "lora" if "lora" in config.scenario else "fft"
  lr = "1e-4" if "lora" in config.scenario else "1e-5"
  log_path = str(open_rl_tmp_dir(config) / f"{prefix}_gsm8k_rl")
  if os.path.exists(log_path):
    shutil.rmtree(log_path)
  module_name, renderer_name = _math_rl_train_module_and_renderer(config.base_model)
  args = [
    "env=gsm8k",
    f"model_name={config.base_model}",
    f"renderer_name={renderer_name}",
    f"max_steps={config.steps if config.steps is not None else 2}",
    f"base_url={base_url}",
    f"log_path={log_path}",
    f"group_size={config.group_size}",
    f"groups_per_batch={config.groups_per_batch}",
    f"max_tokens={config.max_tokens}",
    f"learning_rate={lr}",
    "temperature=1.0",
    "eval_every=0",
    "save_every=0",
    *clean_cli_extra(config.extra),
  ]
  out = None
  try:
    out = run_command(
      [*examples_python(), "-m", module_name, *args],
      env=examples_env(config),
      watch=watch,
    )
  finally:
    cleanup_remote_models(base_url, [out] if out else [])


def run_gsm8k_rl_x2(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Run two concurrent RL jobs on GSM8K using standard tinker_cookbook math_rl CLI, check accel timeslicer time
  slicing, and verify metrics."""
  results: dict[str, str | BaseException] = {}
  prefix = "lora" if "lora" in config.scenario else "fft"

  def train(job: str) -> None:
    try:
      log_path = str(open_rl_tmp_dir(config) / f"{prefix}_gsm8k_rl_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(config.base_model)
      temp = "1.0"
      lr = "1e-4" if "lora" in config.scenario else "1e-5"
      args = [
        "env=gsm8k",
        f"model_name={config.base_model}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 2}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        f"group_size={config.group_size}",
        f"groups_per_batch={config.groups_per_batch}",
        f"max_tokens={config.max_tokens}",
        f"learning_rate={lr}",
        f"temperature={temp}",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      results[job] = run_command(
        [*examples_python(), "-m", module_name, *args],
        env=examples_env(config),
        watch=watch,
        prefix=f"[{job}] ",
      )
    except BaseException as exc:
      results[job] = exc

  threads = [threading.Thread(target=train, args=(job,)) for job in ("job-a", "job-b")]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"fft-gsm8k-rl-x2 {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])


def run_gsm8k_rl_x4_mixed(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Run four concurrent RL jobs on GSM8K (2x LoRA Qwen3-0.6B on L4 + 2x FFT Qwen3-8B on H100)."""
  results: dict[str, str | BaseException] = {}

  fft_model = "Qwen/Qwen3-8B"
  lora_model = "Qwen/Qwen3-0.6B"

  def train(job: str, mode: str, job_model: str) -> None:
    try:
      log_path = str(open_rl_tmp_dir(config) / f"{mode}_gsm8k_rl_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(job_model)
      temp = "1.0"
      lr = "1e-4" if mode == "lora" else "1e-5"
      args = [
        "env=gsm8k",
        f"model_name={job_model}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 10}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        f"group_size={config.group_size}",
        f"groups_per_batch={config.groups_per_batch}",
        f"max_tokens={config.max_tokens}",
        f"learning_rate={lr}",
        f"temperature={temp}",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      env = examples_env(config).copy()
      if mode == "lora":
        env["OPEN_RL_FINE_TUNING_TYPE"] = "lora"
        env.pop("OPEN_RL_IN_PLACE_DELTA", None)
      else:
        env["OPEN_RL_FINE_TUNING_TYPE"] = "full"
        _set_fft_delta_apply(env)

      results[job] = run_command(
        [*examples_python(), "-m", module_name, *args],
        env=env,
        watch=watch,
        prefix=f"[{job}] ",
      )
    except BaseException as exc:
      results[job] = exc

  jobs_config = [
    ("lora-a", "lora", lora_model),
    ("lora-b", "lora", lora_model),
    ("fft-a", "fft", fft_model),
    ("fft-b", "fft", fft_model),
  ]
  threads = [threading.Thread(target=train, args=(job, mode, model)) for job, mode, model in jobs_config]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"lora-fft-gsm8k-rl-x4 {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])


def run_gsm8k_rl_x2_compare(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Run two concurrent FFT RL jobs on GSM8K: Job A (Full Sync) vs Job B (Delta Sync)."""
  results: dict[str, str | BaseException] = {}

  def train(job: str, weight_sync_strategy: str) -> None:
    try:
      log_path = str(open_rl_tmp_dir(config) / f"fft_gsm8k_rl_compare_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(config.base_model)
      args = [
        "env=gsm8k",
        f"model_name={config.base_model}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 30}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        f"group_size={config.group_size}",
        f"groups_per_batch={config.groups_per_batch}",
        f"max_tokens={config.max_tokens}",
        "learning_rate=1e-5",
        "temperature=1.0",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      env = examples_env(config)
      env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync_strategy
      results[job] = run_command(
        [
          "uv",
          "--project",
          "examples",
          "run",
          "python",
          "-m",
          module_name,
          *args,
        ],
        env=env,
        watch=watch,
        prefix=f"[{job.upper()} ({weight_sync_strategy.upper()} SYNC)] ",
      )
    except BaseException as exc:
      results[job] = exc

  threads = [
    threading.Thread(target=train, args=("job-a", "full")),
    threading.Thread(target=train, args=("job-b", "delta")),
  ]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"fft-gsm8k-rl-x2-compare {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])

  check_snapshot_interleaving(config)


def run_gsm8k_rl_x3(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Three concurrent FFT RL jobs against the same backend with startup jitter to stagger execution."""
  results: dict[str, str | BaseException] = {}

  def train(job: str, delay_sec: int) -> None:
    try:
      if delay_sec > 0:
        time.sleep(delay_sec)
      log_path = str(open_rl_tmp_dir(config) / f"fft_gsm8k_rl_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(config.base_model)
      temp = "1.0"
      args = [
        "env=gsm8k",
        f"model_name={config.base_model}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 2}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        "group_size=8",
        "groups_per_batch=24",
        "max_tokens=512",
        "learning_rate=1e-5",
        f"temperature={temp}",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      results[job] = run_command(
        [*examples_python(), "-m", module_name, *args],
        env=examples_env(config),
        watch=watch,
        prefix=f"[{job}] ",
      )
    except BaseException as exc:
      results[job] = exc

  jobs_with_delay = [
    ("job-a", 0),
    ("job-b", config.jitter_sec),
    ("job-c", config.jitter_sec * 2),
  ]
  threads = [threading.Thread(target=train, args=(job, delay)) for job, delay in jobs_with_delay]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"fft-gsm8k-rl-x3 {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])

  check_snapshot_interleaving(config)


def run_gsm8k_rl_hetero(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Three concurrent FFT RL jobs with heterogeneous model sizes (8B and 2x 4B) and asymmetric batch sizes to prevent platooning."""
  results: dict[str, str | BaseException] = {}

  def train(job: str, model_name: str, gpb: int, delay_sec: int) -> None:
    try:
      if delay_sec > 0:
        time.sleep(delay_sec)
      log_path = str(open_rl_tmp_dir(config) / f"fft_gsm8k_rl_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(model_name)
      temp = "1.0"
      args = [
        "env=gsm8k",
        f"model_name={model_name}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 2}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        "group_size=8",
        f"groups_per_batch={gpb}",
        "max_tokens=512",
        "learning_rate=1e-5",
        f"temperature={temp}",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      results[job] = run_command(
        [*examples_python(), "-m", module_name, *args],
        env=examples_env(config),
        watch=watch,
        prefix=f"[{job}] ",
      )
    except BaseException as exc:
      results[job] = exc

  jobs_config = [
    ("job-8b", "Qwen/Qwen3-8B", 24, 0),
    ("job-4b", "Qwen/Qwen3-4B", 24, config.jitter_sec),
    ("job-gemma-2b", "google/gemma-4-e2b", 16, config.jitter_sec * 2),
  ]
  threads = [threading.Thread(target=train, args=(job, model, gpb, delay)) for job, model, gpb, delay in jobs_config]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"fft-gsm8k-rl-hetero {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])

  check_snapshot_interleaving(config)


def run_gsm8k_rl_x3_hetero_8b_0_6b(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Three concurrent FFT RL jobs: 2x Qwen3-8B and 1x Qwen3-0.6B."""
  results: dict[str, str | BaseException] = {}

  def train(job: str, model_name: str, gpb: int, delay_sec: int) -> None:
    try:
      if delay_sec > 0:
        time.sleep(delay_sec)
      log_path = str(open_rl_tmp_dir(config) / f"fft_gsm8k_rl_{job}")
      if os.path.exists(log_path):
        shutil.rmtree(log_path)
      module_name, renderer_name = _math_rl_train_module_and_renderer(model_name)
      temp = "1.0"
      args = [
        "env=gsm8k",
        f"model_name={model_name}",
        f"renderer_name={renderer_name}",
        f"max_steps={config.steps if config.steps is not None else 30}",
        f"base_url={base_url}",
        f"log_path={log_path}",
        "group_size=8",
        f"groups_per_batch={gpb}",
        "max_tokens=512",
        "learning_rate=1e-5",
        f"temperature={temp}",
        "eval_every=0",
        "save_every=0",
        *clean_cli_extra(config.extra),
      ]
      results[job] = run_command(
        [*examples_python(), "-m", module_name, *args],
        env=examples_env(config),
        watch=watch,
        prefix=f"[{job}] ",
      )
    except BaseException as exc:
      results[job] = exc

  jobs_config = [
    ("job-a-8b", "Qwen/Qwen3-8B", 24, 0),
    ("job-b-8b", "Qwen/Qwen3-8B", 24, config.jitter_sec),
    ("job-c-0.6b", "Qwen/Qwen3-0.6B", 24, config.jitter_sec * 2),
  ]
  threads = [threading.Thread(target=train, args=(job, model, gpb, delay)) for job, model, gpb, delay in jobs_config]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  try:
    for job, result in sorted(results.items()):
      if isinstance(result, BaseException):
        raise RuntimeError(f"fft-gsm8k-rl-x3-hetero-8b-0.6b {job} failed") from result
  finally:
    cleanup_remote_models(base_url, [r for r in results.values() if isinstance(r, str)])

  check_snapshot_interleaving(config)


def run_tiny_fft_rl_x2(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Two concurrent FFT RL jobs against the same backend: each create_model spawns
  its own trainer and dedicated sampler worker, and the accel timeslicer time-slices them."""
  results: dict[str, str | BaseException] = {}

  def train(job: str) -> None:
    try:
      script = "tiny_rl"
      defaults = {
        "base_model": config.base_model,
        "base_url": base_url,
        "log_dir": str(Path(config.log_dir) / f"{config.scenario.replace('-', '_')}_{job}"),
        "learning_rate": "1e-5",
      }
      if config.steps is not None:
        defaults["steps"] = str(config.steps)
      results[job] = run_example(config, [f"examples/tiny/{script}.py"], defaults, watch=watch, prefix=f"[{job}] ")
    except BaseException as exc:
      results[job] = exc

  threads = [threading.Thread(target=train, args=(job,)) for job in ("job-a", "job-b")]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  for job, result in sorted(results.items()):
    if isinstance(result, BaseException):
      raise RuntimeError(f"tiny-fft-rl-x2 {job} failed") from result

  check_snapshot_interleaving(config)


def run_tiny_rl_x2_families(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Two concurrent tiny RL jobs on base models from different families
  (base_model and second_base_model), LoRA or FFT by scenario name.

  One API server, two vocabularies: anything that resolves a job's tokenizer,
  parameter names or worker from a API-server-wide default instead of the job's
  own metadata hands one job the other model's tokens. That does not crash
  tiny_rl, the samples just turn into token soup and the reward stays at 0,
  so each job must also earn a reward at least once."""
  models = {"job-a": config.base_model, "job-b": config.second_base_model}
  if len(set(models.values())) != 2:
    raise RuntimeError(f"{config.scenario} needs two different base models, got {models}")
  log_dirs = {job: Path(config.log_dir) / f"{config.scenario.replace('-', '_')}_{job}" for job in models}
  results: dict[str, str | BaseException] = {}

  def train(job: str) -> None:
    try:
      defaults = {"base_model": models[job], "base_url": base_url, "log_dir": str(log_dirs[job])}
      if "fft" in config.scenario:
        defaults["learning_rate"] = "1e-5"
      if config.steps is not None:
        defaults["steps"] = str(config.steps)
      results[job] = run_example(config, ["examples/tiny/tiny_rl.py"], defaults, watch=watch, prefix=f"[{job}] ")
    except BaseException as exc:
      results[job] = exc

  threads = [threading.Thread(target=train, args=(job,)) for job in models]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  for job, result in sorted(results.items()):
    if isinstance(result, BaseException):
      raise RuntimeError(f"{config.scenario} {job} ({models[job]}) failed") from result

  for job, model in models.items():
    rows = [row for row in read_jsonl(log_dirs[job] / "metrics.jsonl") if row.get("phase") == "train"]
    if not rows:
      raise RuntimeError(f"{job} ({model}) logged no training steps in {log_dirs[job]}")
    best = max(require_finite_metric(row, "mean_reward") for row in rows)
    if best <= 0:
      raise RuntimeError(
        f"{job} ({model}) never earned a reward in {len(rows)} steps; its samples are most likely "
        "token soup from the other model's tokenizer (check the API server's per-model metadata)"
      )
    print(f"[training-e2e] {job} {model}: best mean_reward={best:.2f} over {len(rows)} steps")


def read_jsonl(path: Path) -> list[dict]:
  if not path.exists() or path.stat().st_size == 0:
    raise RuntimeError(f"Expected {path} to exist and be non-empty")
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_finite_metric(row: dict, key: str) -> float:
  value = row.get(key)
  if not isinstance(value, int | float) or not math.isfinite(float(value)):
    raise RuntimeError(f"Expected finite metric {key!r}, got {value!r} in {row}")
  return float(value)


def run_textsql(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  log_dir = Path(config.log_dir) / config.scenario.replace("-", "_")
  defaults = {
    "phase": "rl_only",
    "base_url": base_url,
    "log_dir": str(log_dir),
    "model.base_model": config.base_model,
    "model.tokenizer_name": config.base_model,
    "model.rank": "16",
    "dataset.train_limit": "64",
    "dataset.rl_train_limit": "64",
    # Eval runs through the torch sampler (no vLLM during training), so keep it
    # small here; override with extra='dataset.eval_limit=N'.
    "dataset.eval_limit": "8",
    "dataset.eval_max_tokens": "64",
    "sft.steps": "0",
    "rl.steps": str(config.steps if config.steps is not None else 2),
    "rl.prompts_per_step": "4",
    "rl.samples_per_prompt": "4",
    "rl.max_tokens": "64",
    "rl.eval_every": "1",
    "fine_tuning_type": "full" if "fft" in config.scenario else "lora",
  }
  run_example(config, ["examples/text-to-sql/texttosql_sft_grpo.py", "gemma4_e2b_rl_recipe"], defaults, watch=watch)

  rows = read_jsonl(log_dir / "metrics.jsonl")
  train_rows = [row for row in rows if row.get("phase") == "rl_train"]
  eval_rows = [row for row in rows if row.get("phase") == "rl_eval"]
  if not train_rows or not eval_rows:
    raise RuntimeError(f"Text-to-SQL RL must log rl_train and rl_eval metrics in {log_dir / 'metrics.jsonl'}")
  rollouts = sum(int(require_finite_metric(row, "num_rollouts")) for row in train_rows)
  if rollouts <= 0:
    raise RuntimeError("Text-to-SQL RL did not produce any trainable rollouts")
  execution_match = require_finite_metric(eval_rows[-1], "execution_match")
  print(f"[training-e2e] textsql rollouts={rollouts} final_execution_match={execution_match:.1%}")


def run_textsql_rl_x2(config: RunConfig, base_url: str, watch: list[ManagedProcess]) -> None:
  """Two concurrent Text-to-SQL FFT RL jobs against the same backend: each create_model spawns
  its own trainer and dedicated sampler worker, and the accel timeslicer time-slices them.

  `extra_a` / `extra_b` override job-a / job-b separately (on top of `extra`), so
  one run can compare two learning rates, or two base models when a job's
  overrides set model.base_model; that job's tokenizer follows its base model."""
  results: dict[str, str | BaseException] = {}

  def train(job: str) -> None:
    try:
      log_dir = Path(config.log_dir) / f"{config.scenario.replace('-', '_')}_{job}"
      if log_dir.exists():
        shutil.rmtree(log_dir)
      overrides = job_overrides(config, job)
      base_model = overrides.get("model.base_model", config.base_model)
      print(f"[training-e2e] {job}: base_model={base_model} overrides={overrides}")
      defaults = {
        "phase": "rl_only",
        "base_url": base_url,
        "log_dir": str(log_dir),
        "model.base_model": base_model,
        "model.tokenizer_name": base_model,
        "model.rank": "16",
        "dataset.train_limit": "64",
        "dataset.rl_train_limit": "64",
        "dataset.eval_limit": "8",
        "dataset.eval_max_tokens": "64",
        "sft.steps": "0",
        "rl.steps": str(config.steps if config.steps is not None else 2),
        "rl.prompts_per_step": "4",
        "rl.samples_per_prompt": "4",
        "rl.max_tokens": "64",
        "rl.eval_every": "1",
      }
      results[job] = run_example(
        config,
        ["examples/text-to-sql/texttosql_sft_grpo.py", "gemma4_e2b_rl_recipe"],
        defaults,
        watch=watch,
        prefix=f"[{job}] ",
        overrides=overrides,
      )
    except BaseException as exc:
      results[job] = exc

  threads = [threading.Thread(target=train, args=(job,)) for job in ("job-a", "job-b")]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  for job, result in sorted(results.items()):
    if isinstance(result, BaseException):
      raise RuntimeError(f"fft-textsql-rl-x2 {job} failed") from result

    log_dir = Path(config.log_dir) / f"{config.scenario.replace('-', '_')}_{job}"
    rows = read_jsonl(log_dir / "metrics.jsonl")
    train_rows = [row for row in rows if row.get("phase") == "rl_train"]
    eval_rows = [row for row in rows if row.get("phase") == "rl_eval"]
    if not train_rows or not eval_rows:
      raise RuntimeError(f"Text-to-SQL RL {job} must log rl_train and rl_eval metrics in {log_dir / 'metrics.jsonl'}")
    rollouts = sum(int(require_finite_metric(row, "num_rollouts")) for row in train_rows)
    if rollouts <= 0:
      raise RuntimeError(f"Text-to-SQL RL {job} did not produce any trainable rollouts")
    execution_match = require_finite_metric(eval_rows[-1], "execution_match")
    print(f"[training-e2e] textsql {job} rollouts={rollouts} final_execution_match={execution_match:.1%}")

  check_snapshot_interleaving(config)


def main() -> None:
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  Path(config.log_dir).mkdir(parents=True, exist_ok=True)
  processes: list[ManagedProcess] = []
  try:
    base_url = start_backend(config, processes)
    if config.scenario == "fft-gsm8k":
      run_gsm8k(config, base_url, processes)
    elif config.scenario == "fft-gsm8k-x2":
      run_gsm8k_x2(config, base_url, processes)
    elif config.scenario in {"fft-gsm8k-rl", "lora-gsm8k-rl"}:
      run_gsm8k_rl(config, base_url, processes)
    elif config.scenario in {"fft-gsm8k-rl-x2", "lora-gsm8k-rl-x2"}:
      run_gsm8k_rl_x2(config, base_url, processes)
    elif config.scenario == "lora-fft-gsm8k-rl-x4":
      run_gsm8k_rl_x4_mixed(config, base_url, processes)
    elif config.scenario == "fft-gsm8k-rl-x2-compare":
      run_gsm8k_rl_x2_compare(config, base_url, processes)
    elif config.scenario == "fft-gsm8k-rl-x3":
      run_gsm8k_rl_x3(config, base_url, processes)
    elif config.scenario == "fft-gsm8k-rl-x3-hetero-8b-0.6b":
      run_gsm8k_rl_x3_hetero_8b_0_6b(config, base_url, processes)
    elif config.scenario == "fft-gsm8k-rl-hetero":
      run_gsm8k_rl_hetero(config, base_url, processes)
    elif config.scenario in {"lora-textsql", "fft-textsql-rl"}:
      run_textsql(config, base_url, processes)
    elif config.scenario == "fft-textsql-rl-x2":
      run_textsql_rl_x2(config, base_url, processes)
    elif config.scenario == "tiny-fft-rl-x2":
      run_tiny_fft_rl_x2(config, base_url, processes)
    elif config.scenario in {"tiny-rl-x2-families", "tiny-fft-rl-x2-families"}:
      run_tiny_rl_x2_families(config, base_url, processes)
    else:
      run_tiny(config, base_url, processes)
  finally:
    for managed in reversed(processes):
      stop_process(managed)


if __name__ == "__main__":
  main()
