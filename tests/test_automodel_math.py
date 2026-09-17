"""Numerical checks for the Automodel trainer path.

Everything here runs on CPU without nemo-automodel. The distributed cases use
real FSDP2 (fully_shard on a gloo mesh), so the claims about cancelling FSDP2's
gradient averaging are checked against FSDP2 itself, not a stand-in.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from tests.test_automodel_worker import rank_shards
from training import automodel_worker
from training.automodel_worker import AutomodelTrainingWorker, GatherSequenceShards, round_robin_permutation
from training.distributed import group_size
from training.trainer_worker import BaseTrainerWorker, Datum

HIDDEN, VOCAB = 8, 11


class HeadOnlyModel(torch.nn.Module):
  def __init__(self, softcap: float | None):
    super().__init__()
    self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    text_config = SimpleNamespace(final_logit_softcapping=softcap)
    self.config = SimpleNamespace(get_text_config=lambda: text_config)

  def get_output_embeddings(self):
    return self.lm_head


def reference_logprobs(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, softcap: float | None) -> torch.Tensor:
  logits = hidden @ weight.T
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class ProjectTargetLogprobsTest(unittest.TestCase):
  """Chunked, checkpointed projection equals the full log_softmax, in value and gradient."""

  def check(self, softcap: float | None) -> None:
    torch.manual_seed(0)
    worker = AutomodelTrainingWorker()
    model = HeadOnlyModel(softcap)
    hidden = torch.randn(2, 7, HIDDEN, requires_grad=True)
    targets = torch.randint(0, VOCAB, (2, 7))

    with patch.object(automodel_worker, "LOGPROB_CHUNK", 4):
      actual = worker.project_target_logprobs(model, hidden, targets)
    expected = reference_logprobs(hidden, model.lm_head.weight, targets, softcap)
    torch.testing.assert_close(actual, expected)

    weights = torch.rand(2, 7)
    actual_grads = torch.autograd.grad((actual * weights).sum(), [hidden, model.lm_head.weight])
    expected_grads = torch.autograd.grad((expected * weights).sum(), [hidden, model.lm_head.weight])
    for a, e in zip(actual_grads, expected_grads, strict=True):
      torch.testing.assert_close(a, e)

  def test_matches_full_logits(self) -> None:
    self.check(softcap=None)

  def test_matches_full_logits_with_softcap(self) -> None:
    self.check(softcap=30.0)

  def test_low_precision_logits_are_reduced_in_fp32(self) -> None:
    worker = AutomodelTrainingWorker()
    model = HeadOnlyModel(None).to(torch.bfloat16)
    hidden = torch.randn(1, 5, HIDDEN).to(torch.bfloat16)
    targets = torch.randint(0, VOCAB, (1, 5))
    with torch.no_grad():
      actual = worker.project_target_logprobs(model, hidden, targets)
    self.assertEqual(actual.dtype, torch.float32)
    logits = (hidden @ model.lm_head.weight.T).float()
    expected = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)


class ClipGradientsTest(unittest.TestCase):
  """The per-tensor norm stack equals torch's clip_grad_norm_ on plain tensors."""

  def make_worker(self):
    torch.manual_seed(1)
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    for p in params:
      p.grad = torch.randn_like(p)
    worker = AutomodelTrainingWorker()
    worker.trainable_params = params
    return worker, params

  def test_clips_like_torch(self) -> None:
    worker, params = self.make_worker()
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    for r, p in zip(reference, params, strict=True):
      r.grad = p.grad.clone()

    total, clip_coef = worker.clip_gradients(0.5)
    expected_total = torch.nn.utils.clip_grad_norm_(reference, 0.5)

    self.assertAlmostEqual(total, expected_total.item(), places=5)
    self.assertAlmostEqual(clip_coef, 0.5 / (total + 1e-6), places=6)
    for r, p in zip(reference, params, strict=True):
      torch.testing.assert_close(p.grad, r.grad)

  def test_no_clipping_under_the_threshold(self) -> None:
    worker, params = self.make_worker()
    before = [p.grad.clone() for p in params]
    worker.clip_gradients(float("inf"))
    for b, p in zip(before, params, strict=True):
      torch.testing.assert_close(p.grad, b)


class RatioStatsTest(unittest.TestCase):
  def test_tail_counts_only_positions_with_an_advantage(self) -> None:
    worker = BaseTrainerWorker()
    target = torch.tensor([[-1.0, -2.0, -0.5, -9.0]])
    old = torch.tensor([[0.0, -2.0, -2.0, -1.0]])
    weights = torch.ones(1, 4)
    # Position 0 is a prompt position: weight 1, no advantage, logprob 0.
    advantages = torch.tensor([[0.0, 1.0, 1.0, -1.0]])

    worker.record_ratio_stats(target, old, weights, advantages)
    metrics = worker.ratio_metrics()

    self.assertEqual(metrics["ratio/max_abs_log:max"], 8.0)
    self.assertEqual(metrics["ratio/tokens_abs_log_gt1:sum"], 2.0)
    self.assertEqual(metrics["ratio/tokens_abs_log_gt5:sum"], 1.0)
    self.assertAlmostEqual(metrics["ratio/frac_abs_log_gt1:mean"], 2 / 3)
    # Reported once, then reset.
    self.assertEqual(worker.ratio_metrics()["ratio/max_abs_log:max"], 0.0)


# -- FSDP2 on a CPU mesh ------------------------------------------------------


class TableModel(torch.nn.Module):
  """logprob of a token is -table[token]; FSDP2 shards the table over the mesh."""

  def __init__(self):
    super().__init__()
    self.table = torch.nn.Parameter(torch.linspace(0.1, 1.2, VOCAB).unsqueeze(-1))

  def forward(self, targets: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.embedding(targets, self.table).squeeze(-1)


class TableWorker(BaseTrainerWorker):
  def __init__(self, group):
    super().__init__()
    self.device = torch.device("cpu")
    self.group = group

  def data_parallel_group(self):
    return self.group

  def data_parallel_loss_scale(self):
    return float(group_size(self.group))

  def compute_target_logprobs(self, model, input_ids, attention_mask, target_token_ids):
    return model(target_token_ids)


def table_data() -> list[Datum]:
  return [
    Datum(model_input=[1, 2, 3], loss_fn_inputs={"target_tokens": {"data": [2, 3, 4]}, "weights": {"data": [1.0, 0.5, 0.25]}}),
    Datum(model_input=[5, 6], loss_fn_inputs={"target_tokens": {"data": [6, 7]}, "weights": {"data": [2.0, 0.75]}}),
    Datum(model_input=[8], loss_fn_inputs={"target_tokens": {"data": [9]}, "weights": {"data": [1.5]}}),
  ]


def init_group(rank: int, world_size: int, port: int) -> None:
  os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": str(world_size)})
  dist.init_process_group("gloo", rank=rank, world_size=world_size)


def run_fsdp_data_parallel(rank: int, world_size: int, port: int) -> None:
  init_group(rank, world_size, port)
  try:
    reference_model = TableModel()
    expected = TableWorker(None).forward_backward(reference_model, table_data(), "cross_entropy")

    mesh = init_device_mesh("cpu", (world_size,))
    model = fully_shard(TableModel(), mesh=mesh)
    actual = TableWorker(mesh.get_group()).forward_backward(model, table_data(), "cross_entropy")

    # FSDP2 averaged the gradient over the mesh; the loss scaling in
    # forward_backward must turn that average back into the full-batch sum.
    torch.testing.assert_close(model.table.grad.full_tensor(), reference_model.table.grad)
    assert actual["metrics"] == expected["metrics"], (actual["metrics"], expected["metrics"])
    assert actual["loss_fn_outputs"] == expected["loss_fn_outputs"]
  finally:
    dist.destroy_process_group()


def run_fsdp_context_parallel_gather(rank: int, world_size: int, port: int) -> None:
  init_group(rank, world_size, port)
  try:
    torch.manual_seed(0)
    padded = 4 * world_size
    tokens = torch.randint(0, VOCAB, (1, padded))
    weights = torch.rand(1, padded)

    reference_model = TableModel()
    (reference_model(tokens) * weights).sum().backward()

    mesh = init_device_mesh("cpu", (world_size,))
    model = fully_shard(TableModel(), mesh=mesh)
    local_tokens = tokens[:, rank_shards(world_size, padded)[rank]]
    local_logprobs = model(local_tokens)
    gathered = GatherSequenceShards.apply(local_logprobs, mesh.get_group(), world_size, rank)
    perm = round_robin_permutation(world_size, padded, gathered.device)
    ordered = torch.zeros_like(gathered).index_copy(1, perm, gathered)
    (ordered * weights).sum().backward()

    # Every rank computed the same full-sequence loss; its gradient covers only
    # its own shard, and FSDP2 averaged those partials over the mesh. The
    # gather's backward scales by the CP size so the average is the true sum.
    torch.testing.assert_close(ordered, reference_model(tokens))
    torch.testing.assert_close(model.table.grad.full_tensor(), reference_model.table.grad)
  finally:
    dist.destroy_process_group()


class Fsdp2ScalingTest(unittest.TestCase):
  def test_data_parallel_loss_scaling_cancels_fsdp2_averaging(self) -> None:
    mp.spawn(run_fsdp_data_parallel, args=(2, 29541), nprocs=2, join=True)

  def test_context_parallel_gather_scaling_cancels_fsdp2_averaging(self) -> None:
    mp.spawn(run_fsdp_context_parallel_gather, args=(2, 29542), nprocs=2, join=True)


if __name__ == "__main__":
  unittest.main()
