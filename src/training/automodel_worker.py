# NeMo Automodel training-backend worker.

"""A training-backend worker built on NVIDIA NeMo Automodel.

Why this exists. Automodel is HF-native (no format conversion, no bridge) and
its context-parallel path shards the residual stream, which is what takes a
27B or 31B model to a 180k-260k token window on four GPUs. Its GatedDeltaNet CP
runs FLA's context-parallel gated delta rule, which passes conv and recurrent
state between ranks, so it is also the home for long-context Qwen3.5 training.

Layout. Automodel builds a (pp, dp_replicate, dp_shard, cp, tp) mesh from the
sizes below; DP is what is left of the torchrun world after CP and TP. FSDP2
shards parameters over dp_shard x cp, so under CP the weights are sharded too.
Only the DP axis shards datums: CP and TP ranks cooperate on one sequence and
must see identical data, which is what data_parallel_group scopes to.

Logprobs never materialise [seq, vocab] logits. The forward returns only the
last logit row (logits_to_keep=1), the final normed hidden states are taken
from a hook on the backbone's last norm, the lm_head weight is gathered out
of its DTensor, and hidden states are projected in checkpointed chunks.

Under CP each pass is one datum: the sequence is padded to a multiple of 2*CP,
round-robin sharded (head and tail chunk per rank, torch's load-balanced
context_parallel layout), forwarded under the ring-attention context, projected
locally, and the local logprobs are all-gathered back into position order. The
backward of that gather scales by CP to cancel FSDP2's mean over the
dp_shard x cp mesh, matching data_parallel_loss_scale on the DP axis.

A wrong CP trains on a corrupted gradient without crashing, so this one was
checked before use on Qwen3.5-9B (LoRA r32, 4096 tokens, 2026-09-10) against a
single-GPU reference: per-position logprobs within bf16 noise on every chunk
and adapter gradients at cosine >= 0.9987 for CP2, CP4 and TP2xCP2, the same
agreement TP2 alone shows. Compare against a single-GPU reference again after
touching this path.

Two modes, chosen by the model's fine-tuning type. LoRA (the default) freezes
the base, trains PEFT adapters with the rank and alpha the client sent, and
publishes the adapter where the LoRA sampler workers load it. Full-parameter
trains everything and publishes whole checkpoints through the FFT route.
"""

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer

from training.commands import SaveWeightsForSampler
from training.distributed import barrier, group_rank, group_size, is_primary
from training.fft_trainer_worker import trainable_model_parameters
from training.trainer_worker import ENABLE_GRADIENT_CHECKPOINTING, BaseTrainerWorker, Datum, chunk_target_logprob, tmp_dir
from training.types import FFTConfig, LoraConfig, SamplerWeights

# Parallel layout. DP is inferred as world / (CP * TP).
AUTOMODEL_TP = int(os.getenv("OPEN_RL_AUTOMODEL_TP", "1"))
AUTOMODEL_CP = int(os.getenv("OPEN_RL_AUTOMODEL_CP", "1"))
DEFAULT_SEED = 1234
# Global grad-norm clip used when the client sends no grad_clip_norm. The
# cookbook builds AdamParams without one, and Gemma's steps carry rare 20-130x
# gradient spikes that an unclipped Adam turns into one bad step and a long
# second-moment hangover. 0 = off.
GRAD_CLIP_NORM = float(os.getenv("OPEN_RL_GRAD_CLIP_NORM", "0"))

# LoRA leaf module names, matched as model.*.layers.*.<name>. That covers the
# attention, GDN and MLP projections of every decoder layer and nothing else:
# the vision tower has blocks, not layers, and the multi-token-prediction head
# lives under mtp, not model. Adapters on either would be tensors the
# language-model-only samplers cannot place.
ATTENTION_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
MLP_LORA_TARGETS = ("gate_proj", "up_proj", "down_proj")

# Activation checkpointing, done here rather than by Automodel. For its native
# Qwen3.5 model Automodel wraps self_attn, linear_attn and mlp separately and
# leaves the norms outside, so every layer stashes four sequence-length tensors
# instead of one: measured 2.47 GiB per 1k tokens on the 27B at TP4 and a
# 47k-token ceiling. Checkpointing a group of whole layers keeps one stash per
# group; recompute cost is one extra forward regardless of the group size. 4
# is where that trade bottomed out for this model. 0 turns it off.
RECOMPUTE_NUM_LAYERS = int(os.getenv("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", "4")) if ENABLE_GRADIENT_CHECKPOINTING else 0

# Rows of hidden states projected through the vocab per chunk: a 1 GiB fp32
# logits chunk on a 248k vocab, which an H200 does not notice.
LOGPROB_CHUNK = 1024


def attention_kwargs(model_type: str, cp_size: int) -> dict[str, Any]:
  """from_pretrained kwargs selecting the attention kernels for this model.

  Gemma 4's global layers have 512-wide heads, past what SDPA's fused kernels
  take, so they go through Automodel's FFPA route (FFPA for the 512 heads,
  FlexAttention for the sliding-window layers). Under CP the ring swaps SDPA
  itself, so the model is built with sdpa and FFPA is requested for the
  full-attention ring chunks via the text config, the way Automodel's own
  Gemma 4 CP recipes do. Everything else is plain SDPA, which is what the CP
  ring-attention context patches.
  """
  if not model_type.startswith("gemma4"):
    return {"attn_implementation": "sdpa"}
  if cp_size > 1:
    return {
      "attn_implementation": "sdpa",
      "use_sdpa_patching": False,
      "text_config": {"use_cache": False, "cp_full_attn_backend": "ffpa"},
    }
  return {"attn_implementation": "ffpa", "use_sdpa_patching": False}


def widest_head_dim(text_config: Any) -> int:
  """The largest attention head width in the model (checks per-layer views first for heterogeneous models like Gemma 4)."""
  with contextlib.suppress(Exception):
    if layers := getattr(text_config, "per_layer_config", None):
      return max(int(getattr(layer, "head_dim", 0) or 0) for layer in layers)
    return int(getattr(text_config, "head_dim", 0) or 0)
  return 0


def flex_kernel_options(text_config: Any) -> dict[str, Any]:
  """Smaller FlexAttention tiles for models with heads 256 wide or wider."""
  if widest_head_dim(text_config) < 256:
    return {}
  return {
    "kernel_options": {
      "fwd_BLOCK_M": 64,
      "fwd_BLOCK_N": 64,
      "fwd_num_stages": 1,
      "bwd_BLOCK_M1": 32,
      "bwd_BLOCK_N1": 32,
      "bwd_BLOCK_M2": 32,
      "bwd_BLOCK_N2": 32,
      "bwd_num_stages": 1,
    }
  }


def require_automodel():
  """Import NeMo Automodel, or explain how to get it."""
  try:
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=automodel needs nemo-automodel, which is not a dependency of this "
      "project (no extra in pyproject.toml installs it). Build the trainer interpreter with "
      f"scripts/setup_automodel_env.sh and point AUTOMODEL_PYTHON at it. Import failed with: {exc}"
    ) from exc
  return NeMoAutoModelForCausalLM


def text_backbone(model: torch.nn.Module) -> torch.nn.Module:
  """The decoder stack: nested under language_model on multimodal checkpoints."""
  return model.model.language_model if hasattr(model.model, "language_model") else model.model


def round_robin_permutation(cp_size: int, padded_seq_len: int, device: torch.device) -> torch.Tensor:
  """Global position of every element of the rank-major all-gather of CP shards."""
  chunks = torch.arange(padded_seq_len, device=device).chunk(2 * cp_size)
  return torch.cat([part for r in range(cp_size) for part in (chunks[r], chunks[2 * cp_size - 1 - r])])


class GatherSequenceShards(torch.autograd.Function):
  """All-gather [batch, local_seq] shards along the sequence and scale backward by CP."""

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


class GroupCheckpointedLayers(torch.nn.ModuleDict):
  """Iterates decoder layers as checkpointed groups during training."""

  group_size = 1

  def values(self):
    layers = list(self._modules.values())
    if not self.training:
      return iter(layers)

    def make_group(group: list[torch.nn.Module]):
      def run(x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        for layer in group:
          x = layer(x, **kwargs)
        return x

      return lambda x, **kwargs: torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False, **kwargs)

    return (make_group(layers[i : i + self.group_size]) for i in range(0, len(layers), self.group_size))


def install_group_checkpointing(model: torch.nn.Module, group_size: int) -> None:
  """Checkpoint groups of layers on Automodel's native backbones.

  Automodel's native models keep their layers in a ModuleDict and iterate its
  values, which the class swap below hooks. A stock HF backbone (what Automodel
  builds for architectures it has no native class for) keeps a ModuleList and
  computes per-layer arguments inside its own loop, so it gets HF's per-layer
  gradient checkpointing instead: one stash per layer rather than per group.
  """
  layers = getattr(text_backbone(model), "layers", None)
  if isinstance(layers, torch.nn.ModuleDict):
    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = group_size
    print(f"[Automodel Worker] activation checkpointing in groups of {group_size} layers.")
    return
  if isinstance(layers, torch.nn.ModuleList) and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[Automodel Worker] HF backbone: per-layer gradient checkpointing (group size not configurable here).")
    return
  raise RuntimeError(
    f"Activation checkpointing found neither a ModuleDict nor a ModuleList of layers ({type(layers).__name__}); "
    "set OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS=0 to run without checkpointing."
  )


def lora_target_modules(config: LoraConfig) -> list[str]:
  if config.train_unembed:
    raise ValueError("The Automodel backend does not train lm_head adapters (train_unembed).")
  names = (ATTENTION_LORA_TARGETS if config.train_attn else ()) + (MLP_LORA_TARGETS if config.train_mlp else ())
  if not names:
    raise ValueError("LoRA config trains neither attention nor MLP modules.")
  return [f"model.*.layers.*.{name}" for name in names]


class AutomodelTrainingWorker(BaseTrainerWorker):
  def __init__(self, full_parameter: bool = False):
    super().__init__()
    self.full_parameter = full_parameter
    self.model: torch.nn.Module | None = None
    self.distributed_setup: Any = None
    self.device_mesh: Any = None
    self.peft_config: Any = None
    self.checkpointer: Any = None
    self.forward_kwargs: dict[str, Any] = {}
    self.cp_context: contextlib.ExitStack | None = None
    self.final_norm_output: torch.Tensor | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.tp_size = AUTOMODEL_TP
    self.cp_size = AUTOMODEL_CP

  def save_needs_gpu(self) -> bool:
    # Every save gathers DTensor shards over the GPUs.
    return True

  # -- distributed layout ---------------------------------------------------

  def build_distributed_setup(self, config: Any) -> Any:
    """Build the FSDP2 mesh and policy once for the life of the process."""
    if not dist.is_initialized():
      raise RuntimeError("The Automodel backend runs under torchrun; torch.distributed is not initialized.")
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    world = dist.get_world_size()
    if world % (self.cp_size * self.tp_size):
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_AUTOMODEL_CP={self.cp_size} * OPEN_RL_AUTOMODEL_TP={self.tp_size}")
    strategy = FSDP2Config(
      activation_checkpointing=False,
      tp_plan=self.tensor_parallel_plan(config) if self.tp_size > 1 else None,
    )
    mesh_context = MeshContext.build(strategy, ParallelismSizes(tp_size=self.tp_size, cp_size=self.cp_size), world_size=world)
    print(f"Automodel device mesh: DP={world // (self.cp_size * self.tp_size)} CP={self.cp_size} TP={self.tp_size}")
    return DistributedSetup(
      mesh_context=mesh_context,
      strategy_config=strategy,
      activation_checkpointing=strategy.activation_checkpointing,
    )

  def tensor_parallel_plan(self, config: Any) -> dict[str, Any]:
    """The HF text-model TP plan, addressed from the top of the model.

    Automodel's automatic plan reads _tp_plan off the language model, which its
    native Qwen3.5 backbone does not carry, so on that model TP would shard the
    embeddings and nothing else. The plan lives on the text config either way;
    prefix it the way the module tree is actually laid out. GDN layers are not
    in it: they are not TP-shardable and stay replicated.
    """
    from nemo_automodel.components.distributed.parallelizer import translate_to_torch_parallel_style
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    text_config = config.get_text_config()
    architectures = config.architectures or []
    prefix = "model.language_model" if architectures and architectures[0].endswith("ForConditionalGeneration") else "model"
    plan: dict[str, Any] = {f"{prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate())}
    for name, style_name in (getattr(text_config, "base_model_tp_plan", None) or {}).items():
      try:
        style = translate_to_torch_parallel_style(style_name)
      except ValueError:
        # HF ships one plan per family; the MoE-only entries (packed experts)
        # match no module on a dense checkpoint and Automodel has no style for
        # them. Dropping them leaves those modules replicated, never wrong.
        print(f"[Automodel Worker] TP plan: skipping {name} ({style_name}) with no torch parallel style.")
        continue
      if style is not None:
        plan[f"{prefix}.{name}"] = style
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
    return plan

  def data_parallel_group(self) -> dist.ProcessGroup | None:
    if self.device_mesh is None or self.device_mesh["dp_shard"].size() == 1:
      return None
    return self.device_mesh["dp_shard"].get_group()

  def data_parallel_loss_scale(self) -> float:
    # FSDP2 averages gradients over the data-parallel mesh in every backward.
    return float(group_size(self.data_parallel_group()))

  # -- model construction ---------------------------------------------------

  def build_peft_config(self, config: LoraConfig) -> Any:
    from nemo_automodel.components._peft.lora import PeftConfig

    # Alpha and the kaiming A init are PEFT's defaults, which is what the
    # LoRA worker trains with, so an lr means the same thing on both backends.
    return PeftConfig(
      target_modules=lora_target_modules(config),
      dim=config.rank,
      alpha=config.lora_alpha,
      dropout=config.lora_dropout,
      lora_A_init="kaiming",
      use_triton=False,
    )

  def load_base_model(self, base_model_name: str) -> None:
    """Record the base model. The model itself is built in build_model, once
    the LoRA settings are known: Automodel applies adapters inside
    from_pretrained, before FSDP2 shards anything."""
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

  def build_model(self, base_model_name: str) -> None:
    NeMoAutoModelForCausalLM = require_automodel()
    torch.cuda.set_device(self.device)
    self.load_base_model(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)
    if self.distributed_setup is None:
      self.distributed_setup = self.build_distributed_setup(config)
      self.device_mesh = self.distributed_setup.mesh_context.device_mesh
    print(f"Loading Automodel {base_model_name} (rank {os.getenv('RANK', '0')}/{os.getenv('WORLD_SIZE', '1')})...")

    # from_pretrained applies LoRA to the matched linears before FSDP2 shards
    # anything, loads the base weights, freezes everything but the adapters,
    # and wraps with the strategy above. Liger stays off so the forward is the
    # plain HF graph. Qwen3.5 checkpoints ship a multi-token-prediction head
    # that the native model would build and run over the whole sequence on
    # every training forward; the loss never reads it, so it is not built.
    text_config = config.get_text_config()
    model_type = text_config.model_type
    extra = attention_kwargs(model_type, self.cp_size)
    self.forward_kwargs = flex_kernel_options(text_config) if model_type.startswith("gemma4") else {}
    if model_type.startswith("qwen3_5"):
      extra["num_nextn_predict_layers"] = 0
    print(f"[Automodel Worker] {model_type}: {extra}")
    self.model = NeMoAutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype=torch.bfloat16,
      use_liger_kernel=False,
      distributed_setup=self.distributed_setup,
      peft_config=self.peft_config,
      **extra,
    )
    if RECOMPUTE_NUM_LAYERS > 0:
      install_group_checkpointing(self.model, RECOMPUTE_NUM_LAYERS)
    text_backbone(self.model).norm.register_forward_hook(self.keep_final_norm_output)
    if self.is_lora:
      trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
      total = sum(param.numel() for param in self.model.parameters())
      print(
        f"[Automodel Worker] LoRA rank={self.peft_config.dim} alpha={self.peft_config.alpha} on "
        f"{self.peft_config.target_modules}: {trainable:,} trainable of {total:,} ({100 * trainable / max(1, total):.3f}%)."
      )
    print("Successfully loaded Automodel.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: LoraConfig | FFTConfig | None = None) -> None:
    if self.is_lora:
      if not isinstance(config, LoraConfig):
        raise ValueError("A LoRA Automodel worker needs the model's LoraConfig to build its adapters.")
      self.peft_config = self.build_peft_config(config)
    if self.model is not None and self.base_model_name == base_model_name:
      print(f"Automodel model {base_model_name} already loaded.")
    else:
      self.build_model(base_model_name)
    torch.manual_seed(config.seed if config is not None and config.seed is not None else DEFAULT_SEED)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call create_model first."
    if not self.is_lora:
      for param in self.model.parameters():
        param.requires_grad_(True)
    self.trainable_params = trainable_model_parameters(self.model)
    self.model.train()

  # -- forward / backward ---------------------------------------------------

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    # The CP path shards one unpadded sequence; packing several datums into a
    # padded batch would need the padding mask sharded alongside.
    if self.cp_size > 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]
    return super().make_training_batches(data)

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    try:
      res = super().forward_backward(self.model, data, loss_fn, loss_config)
    finally:
      self.close_cp_context()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def close_cp_context(self) -> None:
    if self.cp_context is not None:
      self.cp_context.close()
      self.cp_context = None

  def compute_target_logprobs(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return [batch, seq] target logprobs.

    Targets are already position-aligned by the Tinker client (input_ids[t]
    predicts target_token_ids[t]) and no shifting happens here.
    """
    seq_len = target_token_ids.shape[1]
    input_ids = input_ids[:, :seq_len]
    if self.cp_size > 1:
      return self.compute_target_logprobs_cp(model, input_ids, target_token_ids)

    # A dense all-ones mask only describes ordinary causal attention; omitting
    # it lets SDPA pick flash attention instead of building an additive mask.
    if attention_mask is not None and bool(attention_mask.all()):
      attention_mask = None
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1, **self.forward_kwargs)
    return self.project_target_logprobs(model, self.final_hidden_states()[:, :seq_len], target_token_ids)

  def keep_final_norm_output(self, module: torch.nn.Module, args: Any, output: torch.Tensor) -> None:
    self.final_norm_output = output

  def final_hidden_states(self) -> torch.Tensor:
    """The last forward's final normed hidden states, taken once.

    Model-owned CP forwards (Gemma4) drop output_hidden_states on the way to
    the text model, so the norm hook is the one source that works everywhere.
    Clearing it here keeps the previous pass's graph from outliving the pass.
    """
    hidden, self.final_norm_output = self.final_norm_output, None
    if hidden is None:
      raise RuntimeError("The backbone norm hook saw no output; the logprob path needs the final hidden states.")
    return hidden.full_tensor() if isinstance(hidden, DTensor) else hidden

  def head_weight(self, weight: torch.Tensor) -> torch.Tensor:
    """The lm_head weight as one plain [vocab, hidden] tensor on this rank."""
    if not isinstance(weight, DTensor):
      return weight
    if weight.requires_grad:
      # A trained head needs gradient-correct reduction semantics for the
      # gather; Automodel's helper has them for the DP/CP case and refuses TP.
      from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

      return FusedLinearCrossEntropy.materialize_lm_weight(weight, grad_reduce_group=self.device_mesh["dp_shard_cp"].get_group())
    return weight.full_tensor()

  def project_target_logprobs(self, model: torch.nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logit[target] - logsumexp over the vocab, in checkpointed chunks."""
    head = model.get_output_embeddings()
    weight = self.head_weight(head.weight)
    bias = getattr(head, "bias", None)
    if bias is not None:
      bias = self.head_weight(bias)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    batch, seq_len, _ = hidden.shape
    flat_hidden = hidden.reshape(batch * seq_len, -1)
    flat_targets = targets.reshape(batch * seq_len)
    needs_grad = flat_hidden.requires_grad or weight.requires_grad

    def project(start: int) -> torch.Tensor:
      args = (flat_hidden[start : start + LOGPROB_CHUNK], weight, bias, flat_targets[start : start + LOGPROB_CHUNK], softcap)
      if needs_grad:
        return torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
      return chunk_target_logprob(*args)

    return torch.cat([project(start) for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK)]).reshape(batch, seq_len)

  def compute_target_logprobs_cp(self, model: torch.nn.Module, input_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Return [batch, seq] target logprobs with the sequence sharded over CP."""
    if input_ids.shape[0] != 1:
      raise RuntimeError(f"The Automodel CP path takes one sequence per pass, got a batch of {input_ids.shape[0]}.")
    from nemo_automodel.components.distributed.context_parallel.sharder import shard_batch_aux_only

    seq_len = target_token_ids.shape[1]
    cp_mesh = self.device_mesh["cp"]
    cp_group = cp_mesh.get_group()
    cp_rank = cp_mesh.get_local_rank()
    batch = {"input_ids": input_ids, "labels": target_token_ids.clone()}
    # Two CP contracts. Models that own their CP (Gemma4) shard the aux streams
    # into contiguous slices, embed the full input_ids and slice their own hidden
    # states, and run a p2p ring inside every attention layer, so no context is
    # needed and the gather comes back in order. Everything else goes through
    # Automodel's round-robin shard plus torch's ring-attention context.
    owns_cp = bool(getattr(model, "_owns_cp_attention", False))
    if owns_cp:
      sharder = model.prepare_model_inputs_for_cp(batch)["cp_sharder"]
      context, batch, layout = sharder.shard_batch(cp_mesh, self.device_mesh["tp"], batch)
    else:
      context, batch, layout = shard_batch_aux_only(cp_mesh, self.device_mesh["tp"], batch)
    aux = {key: batch[key] for key in ("padding_mask", "_packed_seq_ids") if key in batch}
    # The context must outlive this call. It swaps SDPA for ring attention by
    # dispatch, and the backward dispatches SDPA again, both for the attention
    # gradient and for the activation-checkpoint recompute of the forward.
    # Closed after the forward, both run plain local attention over this
    # rank's shard and every gradient upstream of the last attention layer is
    # silently wrong (measured: cosine 0.24 on o_proj adapters, logprobs fine).
    # forward_backward closes it once this pass's backward is done.
    self.close_cp_context()
    self.cp_context = contextlib.ExitStack()
    self.cp_context.enter_context(context())
    model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False, logits_to_keep=1, **aux, **self.forward_kwargs)
    # Padding slots carry an ignore index; they are cut off after the gather.
    local_logprobs = self.project_target_logprobs(model, self.final_hidden_states(), batch["labels"].clamp_min(0))

    gathered = GatherSequenceShards.apply(local_logprobs, cp_group, self.cp_size, cp_rank)
    if owns_cp:
      return gathered[:, :seq_len]
    perm = round_robin_permutation(self.cp_size, layout.padded_seq_len, gathered.device)
    ordered = torch.zeros_like(gathered).index_copy(1, perm, gathered)
    return ordered[:, :seq_len]

  # -- optimizer ------------------------------------------------------------

  def optim_step(self, adam_params: dict[str, Any]) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    if self.optimizer is None:
      self.optimizer = self.build_optimizer(adam_params)

    total_norm, clip_coef, *_ = self.step_optimizer(
      self.optimizer,
      self.trainable_params,
      adam_params,
      default_clip=GRAD_CLIP_NORM,
    )
    if self.is_lora:
      self.save_adapter(self.model_id)
    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm),
        "grad_clip_coef:mean": clip_coef,
        **self.ratio_metrics(),
      }
    }

  def build_optimizer(self, adam_params: dict[str, Any], *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
    return super().build_optimizer(self.trainable_params, adam_params, label="Automodel", foreach=False)

  def clip_gradients(self, params: list[torch.nn.Parameter] | float, max_grad_norm: float | None = None) -> tuple[float, float]:
    """Clip to a global grad norm over DTensors across potentially heterogeneous meshes."""
    if max_grad_norm is None:
      params, max_grad_norm = self.trainable_params, float(params)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
      return 0.0, 1.0
    norms = [n.full_tensor() if isinstance(n := torch.linalg.vector_norm(g.detach().float()), DTensor) else n for g in grads]
    total_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
    if clip_coef < 1.0:
      for g in grads:
        g.mul_(clip_coef)
    return total_norm, clip_coef

  # -- checkpointing --------------------------------------------------------

  def get_checkpointer(self) -> Any:
    """One Automodel Checkpointer for the process."""
    if self.checkpointer is None:
      from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

      config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.join(tmp_dir(), "checkpoints"),
        model_save_format="safetensors",
        save_consolidated=not self.is_lora,
        is_peft=self.is_lora,
        model_repo_id=self.base_model_name,
      )
      self.checkpointer = Checkpointer(
        config,
        dp_rank=group_rank(self.data_parallel_group()),
        tp_rank=self.device_mesh["tp"].get_local_rank(),
        pp_rank=0,
      )
    return self.checkpointer

  def write_weights(self, save_path: str) -> None:
    """Write the adapter (LoRA) or the consolidated HF model (full) into save_path."""
    checkpointer = self.get_checkpointer()
    checkpointer.save_model(self.model, weights_path=save_path, peft_config=self.peft_config, tokenizer=self.tokenizer)
    if is_primary():
      model_dir = os.path.join(save_path, "model")
      source = model_dir if self.is_lora else os.path.join(model_dir, "consolidated")
      for entry in os.listdir(source):
        os.replace(os.path.join(source, entry), os.path.join(save_path, entry))
      shutil.rmtree(model_dir, ignore_errors=True)
    barrier()

  def stage_and_swap(self, final_dir: str, write_extra: Callable[[str], None] | None = None) -> None:
    """Write weights into a sibling staging directory and swap it into final_dir."""
    staging_dir = f"{final_dir}.staging"
    previous_dir = f"{final_dir}.previous"
    if is_primary():
      shutil.rmtree(staging_dir, ignore_errors=True)
      os.makedirs(staging_dir, exist_ok=True)
    barrier()
    self.write_weights(staging_dir)
    if is_primary():
      if write_extra is not None:
        write_extra(staging_dir)
      shutil.rmtree(previous_dir, ignore_errors=True)
      if os.path.exists(final_dir):
        os.rename(final_dir, previous_dir)
      os.rename(staging_dir, final_dir)
      shutil.rmtree(previous_dir, ignore_errors=True)
    barrier()

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    include_optimizer = include_optimizer and self.optimizer is not None
    if include_optimizer:
      from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

      optimizer_state = get_optimizer_state_dict(self.model, self.optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    metadata = self.checkpoint_metadata(self.model_id, kind=kind, has_optimizer=include_optimizer, lora=self.is_lora)

    def write_extra(staged: str) -> None:
      if include_optimizer:
        torch.save(optimizer_state, os.path.join(staged, "optimizer.pt"))
      with open(os.path.join(staged, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    self.stage_and_swap(state_path, write_extra)
    print(f"Saved Automodel state to {state_path}")
    return {"path": state_path}

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    name = alias or "automodel-model"
    save_path = name if os.path.isabs(name) else os.path.join(tmp_dir(), "automodel", name)
    return self.save_state(save_path, kind="weights")

  def load_from_state(self, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    meta_file = next((p for f in ("metadata.json", "adapter_config.json") if os.path.exists(p := os.path.join(state_path, f))), None)
    if not meta_file:
      raise FileNotFoundError(f"{state_path} has neither metadata.json nor adapter_config.json")
    with open(meta_file) as f:
      metadata = json.load(f)
    base_model = metadata.get("base_model") or metadata.get("base_model_name_or_path")
    if not base_model:
      raise ValueError(f"{state_path} does not name its base model")

    if self.is_lora:
      if self.model is None:
        if self.peft_config is None:
          raise RuntimeError("A LoRA Automodel worker restores into adapters built by create_model; create the model first.")
        self.build_model(base_model)
      self.get_checkpointer().load_model(self.model, model_path=state_path)
    else:
      self.model = None
      self.build_model(state_path)
    self.base_model_name = base_model
    self.prepare_model_for_training()

    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and metadata.get("has_optimizer") and os.path.exists(optimizer_path):
      from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

      self.optimizer = self.build_optimizer({})
      full_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
      set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=full_state, options=StateDictOptions(full_state_dict=True))
      print(f"Restored Automodel optimizer state from {optimizer_path}")
    print(f"Loaded Automodel state from {state_path}")
    return {"model_id": self.model_id, "base_model": base_model}

  # -- sampler weights ------------------------------------------------------

  def publish_sampler_weights(self, command: SaveWeightsForSampler) -> SamplerWeights:
    if self.full_parameter:
      return super().publish_sampler_weights(command)
    return self.save_adapter(self.model_id, command.alias)

  def save_weights(self, alias: str | None = None) -> dict[str, Any]:
    if self.full_parameter:
      return super().save_weights(alias)
    return {"path": self.save_adapter(self.model_id, alias).path}

  def save_adapter(self, adapter_id: str, alias: str | None = None) -> SamplerWeights:
    """Publish the LoRA adapter where the sampler workers load it."""
    if not self.is_lora:
      raise RuntimeError("A full-parameter Automodel worker has no adapter to publish; it saves whole checkpoints.")
    adapter_root = os.path.join(tmp_dir(), "peft", adapter_id)
    final_dir = os.path.join(adapter_root, adapter_id)
    self.stage_and_swap(final_dir)
    if is_primary():
      metadata = {
        "model_id": adapter_id,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time(),
        **({"alias": alias} if alias is not None else {}),
      }
      with open(os.path.join(adapter_root, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      print(f"[Automodel Worker] Saved LoRA adapter to {final_dir}.")
    return SamplerWeights(kind="adapter", path=final_dir)

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Sampling from the Automodel trainer is unsupported; use the vLLM sampler worker.")
