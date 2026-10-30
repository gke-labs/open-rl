"""One configuration shared by training, evaluation, and LAB environments."""

from pathlib import Path

import chz
from tinker_cookbook import model_info


@chz.chz
class RunConfig:
  # Defaults are the Qwen3.5-9B recipe that raised held-out reward from 0.57
  # to 0.72-0.74 over 30 steps (run 5b, Automodel LoRA at a 196k window).
  model_name: str = "Qwen/Qwen3.5-9B"
  renderer_name: str | None = None
  base_url: str | None = None
  lab_root: Path = chz.field(default=Path(__file__).resolve().parent / "harvey-labs", munger=lambda _, path: Path(path).expanduser().resolve())
  log_path: str = "artifacts/harvey-labs"

  task: str | None = None  # A single training task; otherwise use the seeded split.
  train_tasks: int = 300
  eval_tasks: int = 50
  task_split_seed: int = 0
  train_split_seed: int | None = 242  # Reorders the train pool, as the reference runs did.
  batch_size: int = 8
  rollouts_per_example: int = 6
  # One attempt per eval task; four made a 50-task eval take ~100 min.
  eval_rollouts_per_task: int = 1

  max_steps: int = 35
  max_turns: int = 40
  max_tokens: int = 32 * 1024
  # The gateway's VLLM_MAX_MODEL_LEN and OPEN_RL_TRAIN_TOKEN_BUDGET must fit it.
  max_trajectory_tokens: int = 192 * 1024
  max_tool_result_tokens: int = 16 * 1024
  # When a turn stops on the per-turn max_tokens cap: False ends the episode
  # there, as the reference LAB harness does, and it is graded on what was
  # produced; True keeps the truncated turn in history and lets the agent
  # continue (the cookbook's LENGTH-continue), bounded by max_trajectory_tokens.
  continue_after_truncation: bool = False
  command_timeout: int = 60
  sandbox_warmpool: str | None = None  # SandboxWarmPool name; runs sandboxes on Agent Sandbox instead of Podman.
  sandbox_namespace: str = "lab-sandboxes"
  judge_model: str = "gpt-glm-5.2"  # OpenAI-compatible GLM endpoint via OPENAI_BASE_URL / OPENAI_API_KEY.
  judge_parallel: int = 0  # Auto: 16 for GLM, 1 otherwise.

  learning_rate: float = 2e-4
  lora_rank: int = 32
  save_every: int = 5
  eval_every: int = 5
  final_eval: bool = True
  stream_minibatches: bool = True
  num_substeps: int = 1
  kl_penalty_coef: float = 0.0
  kl_discount_factor: float = 0.0
  # Warm-start weights with a fresh optimizer and batch counter; not a resume.
  load_checkpoint_path: str | None = None
  log_groups: int = 0  # Trajectory groups printed to the console per step; 0 keeps the console to steps and metrics.

  @property
  def resolved_renderer(self) -> str:
    if self.renderer_name:
      return self.renderer_name
    if "gemma" in self.model_name.lower():
      return "gemma4"
    return model_info.get_recommended_renderer_name(self.model_name)
