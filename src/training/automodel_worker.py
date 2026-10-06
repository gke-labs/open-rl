"""A LoRA trainer worker built on NVIDIA NeMo Automodel.

The model loads through Automodel's FSDP2 path. Under torchrun the mesh is
DP x CP, with DP whatever the world has left after OPEN_RL_AUTOMODEL_CP. Each
DP rank runs a round-robin share of the datums and FSDP2 reduces the grads.
Alone it is a one-rank mesh. Batching and the loss are the base class's.

Under CP the ranks of a CP group run one datum per pass together. Automodel
round-robin shards the sequence, the ring-attention context stays open through
the backward, and the local logprobs are gathered back into position order. The
gather's backward scales by CP to undo FSDP2's mean over the CP ranks. This path
matched a single-GPU reference at CP2 and CP4 on Qwen3.5-9B (adapter grad cosine
>= 0.9987); compare again after touching it. Per-token logprobs come out of the final hidden states
in chunks, so full [seq, vocab] logits never exist, and decoder layers are
checkpointed in groups so long sequences fit.

One process serves every adapter on one base model. Automodel's LoRA modules
hold a single adapter, so the one being used sits in them and every other
adapter's weights, grads and optimizer wait off to the side. Each call copies
its adapter in first.

The modules are built once, at the sampler's max rank on every target. A
smaller adapter keeps its extra A rows at zero, and a module it does not train
keeps A and B at zero. B starts at zero too, so none of those ever get a
gradient. Scale and dropout are set per adapter on each swap.
"""

import contextlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from transformers import AutoConfig, AutoTokenizer

from training.distributed import is_primary
from training.trainer_worker import BaseTrainerWorker, Datum
from training.types import FFTConfig, LoraConfig

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
AUTOMODEL_SEED = int(os.getenv("OPEN_RL_AUTOMODEL_SEED", "1234"))
# Ranks that share one sequence. The rest of the torchrun world is DP.
AUTOMODEL_CP = int(os.getenv("OPEN_RL_AUTOMODEL_CP", "1"))
# The rank the LoRA modules are built at. Saved adapters carry it, so it must
# fit the sampler's limit, and it is the same knob.
MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", "64"))

# LoRA targets per LoraConfig flag, matched as model.*.layers.*.<name> so a
# vision tower or an MTP head the samplers cannot place is never wrapped.
# train_attn covers Qwen3.5's linear-attention projections too. The kaiming A
# init is PEFT's, so an lr means the same thing here as on the LoRA worker.
LORA_TARGETS = {
  "train_attn": ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
  "train_mlp": ("gate_proj", "up_proj", "down_proj"),
}

# Decoder layers per activation checkpoint. Automodel's own checkpointing wraps
# attention and MLP separately and stashes four tensors per layer; a group of
# whole layers stashes one. 0 turns it off.
RECOMPUTE_NUM_LAYERS = int(os.getenv("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", "4"))
# Rows of hidden states projected through the vocab at a time. 1024 rows of a
# 248k vocab is a 1 GiB fp32 chunk.
LOGPROB_CHUNK = int(os.getenv("OPEN_RL_LOGPROB_CHUNK", "1024"))


def require_automodel():
  try:
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=automodel needs nemo-automodel, which is not a project dependency. "
      f"Run the trainer from the automodel image or scripts/setup_automodel_env.sh. Import failed with: {exc}"
    ) from exc
  return NeMoAutoModelForCausalLM


def is_dtensor(tensor: Any) -> bool:
  from torch.distributed.tensor import DTensor

  return isinstance(tensor, DTensor)


def zero_rows_from(weight: torch.Tensor, start: int) -> None:
  """Zero rows start: of a weight that FSDP2 may have sharded by row."""
  with torch.no_grad():
    if not is_dtensor(weight):
      weight[start:].zero_()
      return
    from torch.distributed.tensor import distribute_tensor

    full = weight.full_tensor()
    full[start:].zero_()
    weight.copy_(distribute_tensor(full, weight.device_mesh, weight.placements))


def barrier() -> None:
  if dist.is_initialized() and dist.get_world_size() > 1:
    dist.barrier()


def initialize_single_process_group(device: torch.device) -> None:
  """Automodel's FSDP2 load needs a process group; one process makes a group of one."""
  if not dist.is_initialized():
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", store=dist.HashStore(), rank=0, world_size=1)


def chunk_target_logprob(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, softcap: float | None) -> torch.Tensor:
  """logit[target] - logsumexp for one chunk of hidden states."""
  logits = torch.nn.functional.linear(hidden, weight).float()
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  return logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def round_robin_permutation(cp_size: int, padded_seq_len: int, device: torch.device) -> torch.Tensor:
  """Global position of every element of the rank-major all-gather of CP shards."""
  chunks = torch.arange(padded_seq_len, device=device).chunk(2 * cp_size)
  return torch.cat([part for rank in range(cp_size) for part in (chunks[rank], chunks[2 * cp_size - 1 - rank])])


class GatherSequenceShards(torch.autograd.Function):
  """All-gather [batch, local_seq] shards along the sequence. The backward
  keeps this rank's slice, scaled by CP."""

  @staticmethod
  def forward(ctx, local: torch.Tensor, group: dist.ProcessGroup, cp_size: int, cp_rank: int) -> torch.Tensor:
    ctx.cp_size = cp_size
    ctx.cp_rank = cp_rank
    ctx.local_len = local.shape[1]
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=1)

  @staticmethod
  def backward(ctx, grad: torch.Tensor):
    start = ctx.cp_rank * ctx.local_len
    return grad[:, start : start + ctx.local_len] * ctx.cp_size, None, None, None


def lora_target_names(config: LoraConfig) -> set[str]:
  return {name for flag, names in LORA_TARGETS.items() if getattr(config, flag) for name in names}


def lora_target_patterns(config: LoraConfig) -> list[str]:
  """Automodel PEFT patterns for the modules a LoraConfig asks to train."""
  patterns = [f"model.*.layers.*.{name}" for flag, names in LORA_TARGETS.items() if getattr(config, flag) for name in names]
  if config.train_unembed:
    # The tinker LoraConfig asks for this by default, but vLLM rejects the whole
    # adapter when it carries lm_head weights for models such as Qwen3.5 and Gemma.
    print("[LoRA] Ignoring train_unembed=True: vLLM cannot load an lm_head adapter for this model.")
  if not patterns:
    raise ValueError("At least one LoRA training target must be enabled.")
  return patterns


class LayerGroup:
  """Runs a run of decoder layers under one non-reentrant checkpoint."""

  def __init__(self, layers: list[torch.nn.Module]):
    self.layers = layers

  def run(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x, **kwargs)
    return x

  def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    if not torch.is_grad_enabled():
      return self.run(x, **kwargs)
    return torch.utils.checkpoint.checkpoint(self.run, x, use_reentrant=False, **kwargs)


class GroupCheckpointedLayers(torch.nn.ModuleDict):
  """Automodel's native backbones run `for layer in self.layers.values()`.
  Swapping this class in after sharding changes only that iteration, so
  parameter names and FSDP2 units are untouched."""

  group_size = 1

  def values(self):
    layers = list(self._modules.values())
    if not self.training:
      return iter(layers)
    return iter([LayerGroup(layers[start : start + self.group_size]) for start in range(0, len(layers), self.group_size)])


def install_group_checkpointing(model: torch.nn.Module, group_size: int) -> None:
  """Native backbones keep a ModuleDict of layers and get grouped checkpoints.
  A stock HF backbone keeps a ModuleList and gets HF's per-layer checkpointing."""
  backbone = model.model.language_model if hasattr(model.model, "language_model") else model.model
  layers = getattr(backbone, "layers", None)
  if isinstance(layers, torch.nn.ModuleDict):
    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = group_size
  elif isinstance(layers, torch.nn.ModuleList) and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  else:
    raise RuntimeError(f"No decoder layers to checkpoint on {type(backbone).__name__}; set OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS=0.")


@dataclass
class AdapterState:
  """One adapter's config, and its tensors while another adapter holds the LoRA modules."""

  config: LoraConfig
  weights: list[torch.Tensor] = field(default_factory=list)
  grads: list[torch.Tensor | None] = field(default_factory=list)
  optimizer: torch.optim.Optimizer | None = None


class AutomodelTrainingWorker(BaseTrainerWorker):
  # FSDP2 reduces grads inside backward, so every rank needs the same number of passes.
  backward_runs_collectives = True

  def __init__(self):
    super().__init__()
    self.model: torch.nn.Module | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.adapters: dict[str, AdapterState] = {}
    # The adapter whose weights are in the LoRA modules.
    self.active: str | None = None
    self.checkpointer: Any = None
    # The LoRA modules as built, at MAX_LORA_RANK on every target.
    self.peft_config: Any = None
    self.device_mesh: Any = None
    self.cp_size = AUTOMODEL_CP
    # Open from a CP forward until the next pass, so its backward runs under it too.
    self.cp_context: contextlib.ExitStack | None = None

  def build_distributed_setup(self) -> Any:
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    # torchrun's processor has already made the group.
    initialize_single_process_group(self.device)
    world = dist.get_world_size()
    if world % self.cp_size:
      raise RuntimeError(f"{world} trainer GPUs do not split into CP groups of {self.cp_size}.")
    strategy = FSDP2Config(activation_checkpointing=False)
    mesh_context = MeshContext.build(strategy, ParallelismSizes(cp_size=self.cp_size), world_size=world)
    self.device_mesh = mesh_context.device_mesh
    print(f"Automodel device mesh: DP={world // self.cp_size} CP={self.cp_size}")
    return DistributedSetup(mesh_context=mesh_context, strategy_config=strategy, activation_checkpointing=False)

  # Datums shard over the dp_shard axis. The ranks of a CP group see the same datums.

  def dp_group(self):
    return self.device_mesh["dp_shard"].get_group()

  def shard_rank(self) -> int:
    return self.device_mesh["dp_shard"].get_local_rank() if self.device_mesh is not None else 0

  def shard_count(self) -> int:
    return self.device_mesh["dp_shard"].size() if self.device_mesh is not None else 1

  def shard_all_reduce_max(self, value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.dp_group())
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group())
    return float(tensor.item())

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    gathered: list[Any] = [None] * self.shard_count()
    dist.all_gather_object(gathered, value, group=self.dp_group())
    return gathered

  def build_peft_config(self, config: LoraConfig) -> Any:
    from nemo_automodel.components._peft.lora import PeftConfig

    return PeftConfig(
      target_modules=lora_target_patterns(config),
      dim=config.rank,
      alpha=config.lora_alpha,
      dropout=config.lora_dropout,
      lora_A_init="kaiming",
      use_triton=False,
    )

  def load_base_model(self, base_model_name: str) -> None:
    """The processor preloads BASE_MODEL before any create_model arrives. Automodel
    applies LoRA at load, so the load itself waits for the client's LoraConfig;
    only the tokenizer is fetched here."""
    if self.tokenizer is None or self.base_model_name != base_model_name:
      self.base_model_name = base_model_name
      self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

  def load_model(self, base_model_name: str) -> None:
    NeMoAutoModelForCausalLM = require_automodel()
    if self.device.type == "cuda":
      torch.cuda.set_device(self.device)
    self.load_base_model(base_model_name)
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "use_liger_kernel": False, "peft_config": self.peft_config}
    # Qwen3.5 ships a multi-token-prediction head that would run over the whole
    # sequence on every forward; the loss never reads it.
    if AutoConfig.from_pretrained(base_model_name).get_text_config().model_type.startswith("qwen3_5"):
      kwargs["num_nextn_predict_layers"] = 0
    # The CP context swaps SDPA for ring attention.
    if self.cp_size > 1:
      kwargs["attn_implementation"] = "sdpa"
    # from_pretrained applies LoRA, loads the base weights and freezes all but the adapters.
    self.model = NeMoAutoModelForCausalLM.from_pretrained(base_model_name, distributed_setup=self.build_distributed_setup(), **kwargs)
    if RECOMPUTE_NUM_LAYERS > 0:
      install_group_checkpointing(self.model, RECOMPUTE_NUM_LAYERS)
    print(f"Loaded Automodel {base_model_name} with LoRA rank {MAX_LORA_RANK}.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: LoraConfig | FFTConfig | None = None) -> None:
    """Add a fresh adapter. The first one loads the model with LoRA modules
    every later adapter fits in."""
    if not isinstance(config, LoraConfig):
      raise ValueError("The Automodel trainer supports LoRA only.")
    if config.rank > MAX_LORA_RANK:
      raise ValueError(f"LoRA rank {config.rank} is over this trainer's max of {MAX_LORA_RANK} (VLLM_MAX_LORA_RANK).")
    targets = lora_target_names(config)
    lora_target_patterns(config)  # refuses a config with no targets
    if self.model is None:
      self.peft_config = self.build_peft_config(LoraConfig(rank=MAX_LORA_RANK, lora_alpha=MAX_LORA_RANK, lora_dropout=0.0))
      self.load_model(base_model_name)
      self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
      if not self.trainable_params:
        raise ValueError("No trainable parameters found in the Automodel model")
    elif self.base_model_name != base_model_name:
      raise RuntimeError(f"This Automodel trainer holds {self.base_model_name}, not {base_model_name}.")
    self.stash()
    torch.manual_seed(config.seed if config.seed is not None else AUTOMODEL_SEED)
    for name, module in self.lora_modules():
      module.init_lora_weights("kaiming")
      zero_rows_from(module.lora_A.weight, config.rank if name.rsplit(".", 1)[-1] in targets else 0)
    self.adapters[model_id] = AdapterState(config)
    self.active = model_id
    self.apply_scale_and_dropout(config)

  def lora_modules(self) -> list[tuple[str, torch.nn.Module]]:
    return [(name, module) for name, module in self.model.named_modules() if hasattr(module, "init_lora_weights")]

  def apply_scale_and_dropout(self, config: LoraConfig) -> None:
    for _, module in self.lora_modules():
      module.scale = config.lora_alpha / config.rank
      module.dropout_p = config.lora_dropout

  def stash(self) -> None:
    """Move the active adapter's weights and grads out of the LoRA modules."""
    if self.active is None:
      return
    state = self.adapters[self.active]
    state.weights = [param.detach().clone() for param in self.trainable_params]
    state.grads = [param.grad for param in self.trainable_params]
    for param in self.trainable_params:
      param.grad = None
    self.active = None

  def activate(self, model_id: str) -> AdapterState:
    """Put this adapter's weights and grads into the LoRA modules. Copying into
    the existing parameters keeps FSDP's view of them intact."""
    if model_id not in self.adapters:
      raise ValueError(f"No adapter {model_id} on this Automodel trainer; create_model first.")
    state = self.adapters[model_id]
    if self.active == model_id:
      return state
    self.stash()
    with torch.no_grad():
      for param, weight, grad in zip(self.trainable_params, state.weights, state.grads, strict=True):
        param.copy_(weight)
        param.grad = grad
    state.weights, state.grads = [], []
    self.active = model_id
    self.apply_scale_and_dropout(state.config)
    return state

  def delete_model(self, model_id: str) -> None:
    """Drop the adapter's tensors and optimizer. If it is the active one, the
    LoRA modules keep its weights until the next adapter is copied in."""
    if self.adapters.pop(model_id, None) is None:
      return
    if self.active == model_id:
      for param in self.trainable_params:
        param.grad = None
      self.active = None
    print(f"Automodel adapter '{model_id}' deleted.")

  # -- the step --------------------------------------------------------------------

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    self.activate(model_id)
    try:
      return super().forward_backward(self.model, data, loss_fn, loss_config, forward_only=forward_only)
    finally:
      self.close_cp_context()

  def close_cp_context(self) -> None:
    if self.cp_context is not None:
      self.cp_context.close()
      self.cp_context = None

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    # The CP shard takes one unpadded sequence per pass.
    if self.cp_size > 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]
    return super().make_training_batches(data)

  def compute_target_logprobs(
    self, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_token_ids: torch.Tensor
  ) -> torch.Tensor:
    """Per-position logprob of each target, projected from the final hidden states in chunks."""
    seq_len = target_token_ids.shape[1]
    if self.cp_size > 1:
      return self.compute_target_logprobs_cp(model, input_ids[:, :seq_len], target_token_ids)
    # An all-ones mask is plain causal attention; dropping it lets SDPA use flash.
    mask = None if bool(attention_mask.all()) else attention_mask
    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=False, logits_to_keep=1, output_hidden_states=True)
    return self.project_target_logprobs(model, outputs.hidden_states[-1][:, :seq_len], target_token_ids)

  def compute_target_logprobs_cp(self, model: torch.nn.Module, input_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """The logprobs of one sequence sharded over the CP group."""
    from nemo_automodel.components.distributed.context_parallel.sharder import shard_batch_aux_only

    seq_len = target_token_ids.shape[1]
    cp_mesh = self.device_mesh["cp"]
    context, batch, layout = shard_batch_aux_only(cp_mesh, None, {"input_ids": input_ids, "labels": target_token_ids.clone()})
    # Closing the context before the backward runs the attention grads and the
    # checkpoint recompute as local attention, and the grads come out wrong.
    self.close_cp_context()
    self.cp_context = contextlib.ExitStack()
    self.cp_context.enter_context(context())
    aux = {key: batch[key] for key in ("padding_mask", "_packed_seq_ids") if key in batch}
    outputs = model(
      input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False, logits_to_keep=1, output_hidden_states=True, **aux
    )
    # Padding slots carry an ignore index and are cut off after the gather.
    local = self.project_target_logprobs(model, outputs.hidden_states[-1], batch["labels"].clamp_min(0))
    gathered = GatherSequenceShards.apply(local, cp_mesh.get_group(), self.cp_size, cp_mesh.get_local_rank())
    order = round_robin_permutation(self.cp_size, layout.padded_seq_len, gathered.device)
    return torch.zeros_like(gathered).index_copy(1, order, gathered)[:, :seq_len]

  def project_target_logprobs(self, model: torch.nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logit[target] - logsumexp over the vocab, in checkpointed chunks."""
    seq_len = targets.shape[1]
    if is_dtensor(hidden):
      hidden = hidden.full_tensor()
    head = model.get_output_embeddings()
    weight = head.weight.full_tensor() if is_dtensor(head.weight) else head.weight
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    batch = hidden.shape[0]
    flat_hidden = hidden.reshape(batch * seq_len, -1)
    flat_targets = targets.reshape(batch * seq_len)
    chunks = []
    for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK):
      args = (flat_hidden[start : start + LOGPROB_CHUNK], weight, flat_targets[start : start + LOGPROB_CHUNK], softcap)
      chunks.append(
        torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
        if torch.is_grad_enabled()
        else chunk_target_logprob(*args)
      )
    return torch.cat(chunks).reshape(batch, seq_len)

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    state = self.activate(model_id)
    # Each adapter's optimizer keys its moments by the shared LoRA parameters,
    # so its state only ever sees that adapter's weights.
    if state.optimizer is None:
      state.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=adam_params.get("learning_rate", 1e-4),
        betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
        eps=adam_params.get("eps", 1e-12),
        weight_decay=adam_params.get("weight_decay", 0.0),
      )
    if adam_params.get("learning_rate") is not None:
      for param_group in state.optimizer.param_groups:
        param_group["lr"] = adam_params["learning_rate"]

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    total_norm = self.clip_gradients(max_grad_norm if max_grad_norm > 0 else math.inf)
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    return {"metrics": {"grad_norm:mean": self.sanitize_float(total_norm)}}

  def clip_gradients(self, max_grad_norm: float) -> float:
    """Global grad norm over the trainable parameters, DTensors or not."""
    norms = []
    for param in self.trainable_params:
      if param.grad is None:
        continue
      norm = torch.linalg.vector_norm(param.grad.detach().float())
      norms.append(norm.full_tensor() if is_dtensor(norm) else norm)
    if not norms:
      return 0.0
    total_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
      for param in self.trainable_params:
        if param.grad is not None:
          param.grad.mul_(clip_coef)
    return total_norm

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Sampling from the Automodel trainer is unsupported; use the vLLM sampler.")

  # -- checkpoints -----------------------------------------------------------------

  def get_checkpointer(self) -> Any:
    """Automodel's Checkpointer gathers DTensor shards, maps native keys back to
    the hub layout, and writes a PEFT adapter that vLLM loads."""
    if self.checkpointer is None:
      from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

      config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.join(TMP_DIR, "automodel"),
        model_save_format="safetensors",
        save_consolidated=False,
        is_peft=True,
        model_repo_id=self.base_model_name,
      )
      dp_rank = self.shard_rank()
      self.checkpointer = Checkpointer(config, dp_rank=dp_rank, tp_rank=0, pp_rank=0)
    return self.checkpointer

  def saved_peft_config(self, config: LoraConfig) -> Any:
    """The adapter is saved at the modules' rank, with alpha raised to match so
    vLLM's alpha / r is still the adapter's own scale."""
    return replace(self.peft_config, alpha=config.lora_alpha * MAX_LORA_RANK / config.rank, dropout=config.lora_dropout)

  def write_staged(self, path: str, peft_config: Any, metadata: dict[str, Any] | None = None) -> None:
    """Write the adapter into a staging dir and rename it over path, so a reader
    never sees a half-written directory. The checkpointer nests its output
    under model/, which is lifted into the staging dir. Every rank joins the
    save's gather and rank 0 writes and moves the files."""
    staging, previous = f"{path}.staging", f"{path}.previous"
    if is_primary():
      shutil.rmtree(staging, ignore_errors=True)
      os.makedirs(staging)
    barrier()
    self.get_checkpointer().save_model(self.model, weights_path=staging, peft_config=peft_config, tokenizer=self.tokenizer)
    if is_primary():
      self.publish_staged(staging, path, previous, metadata)
    barrier()

  def publish_staged(self, staging: str, path: str, previous: str, metadata: dict[str, Any] | None) -> None:
    model_dir = os.path.join(staging, "model")
    for entry in os.listdir(model_dir):
      os.replace(os.path.join(model_dir, entry), os.path.join(staging, entry))
    shutil.rmtree(model_dir, ignore_errors=True)
    if metadata is not None:
      with open(os.path.join(staging, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    if os.path.exists(path):
      os.rename(path, previous)
    os.rename(staging, path)
    shutil.rmtree(previous, ignore_errors=True)

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    state = self.activate(model_id)
    # Weights only for now; the optimizer state is not saved.
    self.write_staged(state_path, self.saved_peft_config(state.config), self.metadata(model_id, kind))
    return {"path": state_path}

  def load_from_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("The Automodel trainer does not load checkpoints yet.")

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str | None:
    """Write the adapter where the LoRA sampler hot-loads it, peft/<id>/<id>,
    so there is no checkpoint to announce."""
    assert self.model is not None, "Model must be loaded first."
    state = self.activate(model_id)
    self.write_staged(os.path.join(TMP_DIR, "peft", model_id, model_id), self.saved_peft_config(state.config))
    return None

  def metadata(self, model_id: str, kind: str) -> dict[str, Any]:
    return {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": False,
      "model_id": model_id,
      "timestamp": time.time(),
    }
