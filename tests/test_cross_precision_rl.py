"""Unit tests for cross-precision RL off-policy correction toggles:
1. Top-p Replay (loss_fn_config['top_p'], loss_fn_config['temperature'])
2. Masked Importance Sampling (loss_fn_config['mis_rho_min'], loss_fn_config['mis_rho_max'])
3. Score Centering (loss_fn_config['score_centering'] + topk_sample_logprobs)
"""

import struct
import unittest
from types import SimpleNamespace

import torch

from server import proto_codec
from server.proto import tinker_public_pb2 as pb
from training import commands, losses
from training.trainer_worker import BaseTrainerWorker, Datum

VOCAB_SIZE = 6


class _TokenizerStub:
  pad_token_id = 0
  eos_token_id = 1


class _LinearLogitModel(torch.nn.Module):
  """Deterministic causal LM stub whose logits are a trainable parameter table."""

  def __init__(self, seq_len: int = 4, vocab_size: int = VOCAB_SIZE):
    super().__init__()
    torch.manual_seed(42)
    self.logit_table = torch.nn.Parameter(torch.randn(seq_len, vocab_size, dtype=torch.float32))

  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
    batch_size, seq_len = input_ids.shape
    logits = self.logit_table[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1).contiguous()
    return SimpleNamespace(logits=logits)


class TopPReplayTest(unittest.TestCase):
  def test_apply_top_p_truncates_tail_and_renormalizes_nucleus(self) -> None:
    # Probabilities before truncation: [0.6, 0.3, 0.07, 0.03]
    probs = torch.tensor([[0.60, 0.30, 0.07, 0.03]], dtype=torch.float32)
    logits = probs.log().requires_grad_(True)

    # With top_p=0.90, tokens 0 and 1 have cumulative mass 0.90, so tokens 2 and 3 are masked to -inf.
    truncated = losses.apply_top_p(logits, top_p=0.90)
    log_probs = torch.nn.functional.log_softmax(truncated, dim=-1)

    self.assertTrue(torch.isfinite(log_probs[0, 0]))
    self.assertTrue(torch.isfinite(log_probs[0, 1]))
    self.assertEqual(log_probs[0, 2].item(), -float("inf"))
    self.assertEqual(log_probs[0, 3].item(), -float("inf"))

    # Inside the nucleus, probabilities renormalize to [0.6 / 0.9, 0.3 / 0.9] = [2/3, 1/3].
    torch.testing.assert_close(log_probs[0, :2].exp(), torch.tensor([2.0 / 3.0, 1.0 / 3.0]), atol=1e-5, rtol=1e-5)

    # Backward through a kept token only touches logits inside the nucleus.
    (-log_probs[0, 0]).backward()
    assert logits.grad is not None
    self.assertAlmostEqual(logits.grad[0, 2].item(), 0.0)
    self.assertAlmostEqual(logits.grad[0, 3].item(), 0.0)
    self.assertAlmostEqual((logits.grad[0, 0] + logits.grad[0, 1]).item(), 0.0, places=6)

  def test_apply_top_p_keeps_at_least_top_one_token(self) -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0]], dtype=torch.float32)
    # Even when top_p is smaller than the top-1 token's probability, the argmax token is kept.
    truncated = losses.apply_top_p(logits, top_p=0.01)
    log_probs = torch.nn.functional.log_softmax(truncated, dim=-1)
    self.assertAlmostEqual(log_probs[0, 0].item(), 0.0, places=6)
    self.assertEqual(log_probs[0, 1].item(), -float("inf"))
    self.assertEqual(log_probs[0, 2].item(), -float("inf"))

  def test_invalid_top_p_or_temperature_is_rejected(self) -> None:
    logits = torch.zeros(1, 4)
    for bad_top_p in (0.0, -0.5, 1.05):
      with self.subTest(top_p=bad_top_p), self.assertRaisesRegex(ValueError, "top_p"):
        losses.apply_top_p(logits, top_p=bad_top_p)
    for bad_temp in (0.0, -1.0):
      with self.subTest(temperature=bad_temp), self.assertRaisesRegex(ValueError, "temperature"):
        losses.apply_top_p(logits, top_p=0.9, temperature=bad_temp)


class MaskedImportanceSamplingTest(unittest.TestCase):
  def test_mis_zeros_loss_and_gradient_outside_bounds(self) -> None:
    # Three tokens with ratios [0.4, 1.0, 2.5]:
    # Token 0 (0.4 < 0.5) and Token 2 (2.5 > 2.0) should be rejected with zero gradient;
    # Token 1 (1.0 in [0.5, 2.0]) should keep its exact importance-sampling gradient.
    old_logprobs = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
    target_ratios = torch.tensor([[0.4, 1.0, 2.5]], dtype=torch.float32)
    target_logprobs = (old_logprobs + target_ratios.log()).detach().requires_grad_(True)
    weights = torch.ones_like(old_logprobs)
    advantages = torch.tensor([[1.5, 2.0, -1.0]], dtype=torch.float32)

    loss_config = {"mis_rho_min": 0.5, "mis_rho_max": 2.0}
    elementwise = losses.importance_sampling_loss(
      target_logprobs,
      weights,
      old_logprobs,
      advantages,
      loss_config=loss_config,
    )
    self.assertAlmostEqual(elementwise[0, 0].item(), 0.0, places=6)
    self.assertAlmostEqual(elementwise[0, 1].item(), -2.0, places=6)
    self.assertAlmostEqual(elementwise[0, 2].item(), 0.0, places=6)

    elementwise.sum().backward()
    assert target_logprobs.grad is not None
    torch.testing.assert_close(target_logprobs.grad, torch.tensor([[0.0, -2.0, 0.0]]))

  def test_mis_composes_with_ppo_loss(self) -> None:
    old_logprobs = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    target_ratios = torch.tensor([[0.9, 3.0]], dtype=torch.float32)
    target_logprobs = (old_logprobs + target_ratios.log()).detach().requires_grad_(True)
    weights = torch.ones_like(old_logprobs)
    advantages = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    # Without MIS, PPO clamps token 1's objective to (1 + 0.2) * 1.0 = 1.2 (non-zero loss).
    # With MIS (mis_rho_max=2.0), token 1 is masked to 0.0.
    elementwise = losses.ppo_loss(
      target_logprobs,
      weights,
      old_logprobs,
      advantages,
      loss_config={"clip_range": 0.2, "mis_rho_min": 0.5, "mis_rho_max": 2.0},
    )
    self.assertAlmostEqual(elementwise[0, 0].item(), -0.9, places=5)
    self.assertAlmostEqual(elementwise[0, 1].item(), 0.0, places=6)

  def test_invalid_mis_bounds_are_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "mis_rho_min.*<=.*mis_rho_max"):
      losses.parse_off_policy_config({"mis_rho_min": 2.0, "mis_rho_max": 0.5})
    with self.assertRaisesRegex(ValueError, "mis_rho_min"):
      losses.parse_off_policy_config({"mis_rho_min": -0.1})
    with self.assertRaisesRegex(ValueError, "mis_rho_max"):
      losses.parse_off_policy_config({"mis_rho_max": 0.0})


class ScoreCenteringTest(unittest.TestCase):
  def test_score_centering_restores_zero_expected_score_under_cross_precision(self) -> None:
    """When K covers the vocabulary, the expected score under pi_sampler is identically zero
    even when pi_trainer (fp32) and pi_sampler (bf16) differ due to precision mismatch."""
    torch.manual_seed(7)
    logits_fp32 = torch.randn(1, 1, VOCAB_SIZE, dtype=torch.float32)
    # Simulate a significant precision perturbation between sampler (bf16) and trainer (fp32)
    logits_sampler = (logits_fp32.to(torch.bfloat16).float() + torch.tensor([[[0.15, -0.1, 0.05, -0.2, 0.08, 0.02]]])).detach()

    sampler_lp_dist = torch.nn.functional.log_softmax(logits_sampler, dim=-1)
    sampler_probs = sampler_lp_dist.exp()

    # Compute the expected score gradient E_{y ~ pi_sampler}[grad_{logits} (log pi_trainer(y) - S)]
    # across all possible sampled tokens y in {0, ..., VOCAB_SIZE - 1}.
    topk_ids = torch.arange(VOCAB_SIZE, dtype=torch.long).view(1, 1, VOCAB_SIZE)
    expected_uncentered_grad = torch.zeros_like(logits_fp32)
    expected_centered_grad = torch.zeros_like(logits_fp32)

    for y_token in range(VOCAB_SIZE):
      p_y = sampler_probs[0, 0, y_token].item()
      target_ids = torch.tensor([[y_token]], dtype=torch.long)
      old_lp = sampler_lp_dist[:, :, y_token]
      weights = torch.ones((1, 1), dtype=torch.float32)
      # Use advantage=1 / ratio so we directly test the score function E_{pi_sampler}[grad log pi_trainer(y)]
      with torch.no_grad():
        trainer_lp_val = torch.nn.functional.log_softmax(logits_fp32, dim=-1)[:, :, y_token]
        inv_ratio = torch.exp(old_lp - trainer_lp_val)
      advantages = inv_ratio

      # Uncentered gradient
      logits_uncentered = logits_fp32.clone().detach().requires_grad_(True)
      lp_uncentered = torch.nn.functional.log_softmax(logits_uncentered, dim=-1)
      target_lp_u = lp_uncentered.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
      loss_u = losses.importance_sampling_loss(target_lp_u, weights, old_lp, advantages)
      loss_u.sum().backward()
      assert logits_uncentered.grad is not None
      expected_uncentered_grad += p_y * logits_uncentered.grad

      # Score-centered gradient
      logits_centered = logits_fp32.clone().detach().requires_grad_(True)
      lp_centered = torch.nn.functional.log_softmax(logits_centered, dim=-1)
      target_lp_c = lp_centered.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
      trainer_topk_lp_c = lp_centered.gather(-1, topk_ids)
      loss_c = losses.importance_sampling_loss(
        target_lp_c,
        weights,
        old_lp,
        advantages,
        loss_config={"score_centering": 1},
        trainer_topk_logprobs=trainer_topk_lp_c,
        sampler_topk_logprobs=sampler_lp_dist,
      )
      # Forward loss value must be unchanged by score centering
      torch.testing.assert_close(loss_c, loss_u)
      loss_c.sum().backward()
      assert logits_centered.grad is not None
      expected_centered_grad += p_y * logits_centered.grad

    # Without score centering, E_{y ~ pi_sampler}[grad log pi_trainer(y)] = pi_trainer - pi_sampler != 0.
    self.assertGreater(expected_uncentered_grad.abs().max().item(), 1e-2)
    # With score centering, the expected gradient under pi_sampler is identically zero!
    torch.testing.assert_close(expected_centered_grad, torch.zeros_like(expected_centered_grad), atol=1e-6, rtol=1e-6)

  def test_score_centering_handles_sentinel_and_top_p_minus_inf_without_nan(self) -> None:
    logits = torch.tensor([[[2.0, 1.0, -5.0, -6.0]]], dtype=torch.float32, requires_grad=True)
    masked_logits = losses.apply_top_p(logits, top_p=0.95)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)

    target_lp = log_probs[:, :, 0]
    old_lp = torch.tensor([[-0.3]], dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)
    advantages = torch.tensor([[1.0]], dtype=torch.float32)

    # Include token 2 (masked to -inf by top_p) and token 0 (sentinel -99999.0 in sampler_topk_lp)
    topk_ids = torch.tensor([[[0, 2, 0]]], dtype=torch.long)
    trainer_topk_lp = log_probs.gather(-1, topk_ids)
    sampler_topk_lp = torch.tensor([[[-0.3, -4.0, -99999.0]]], dtype=torch.float32)

    loss = losses.importance_sampling_loss(
      target_lp,
      weights,
      old_lp,
      advantages,
      loss_config={"score_centering": "true"},
      trainer_topk_logprobs=trainer_topk_lp,
      sampler_topk_logprobs=sampler_topk_lp,
    )
    self.assertTrue(torch.isfinite(loss).all())
    loss.sum().backward()
    assert logits.grad is not None
    self.assertTrue(torch.isfinite(logits.grad).all())

  def test_missing_topk_inputs_raises_clear_error(self) -> None:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    model = _LinearLogitModel(seq_len=3)
    datum = Datum(
      model_input=[1, 2, 3],
      loss_fn_inputs={
        "target_tokens": [2, 3, 4],
        "weights": [0.0, 1.0, 1.0],
        "logprobs": [0.0, -0.5, -0.5],
        "advantages": [0.0, 1.0, -1.0],
      },
    )
    with self.assertRaisesRegex(ValueError, "topk_token_ids"):
      worker.forward_backward(model, [datum], "importance_sampling", loss_config={"score_centering": 1})


class TrainerEndToEndTogglesTest(unittest.TestCase):
  def test_forward_backward_combines_all_three_toggles_and_aligns_completion_topk(self) -> None:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    model = _LinearLogitModel(seq_len=4)

    # Datum 0 supplies completion-only top-K (2 rows for a 3-token sequence where position 0 has weight=0).
    # Datum 1 supplies full-length top-K (4 rows for a 4-token sequence).
    datums = [
      Datum(
        model_input=[1, 2, 3],
        loss_fn_inputs={
          "target_tokens": [2, 3, 4],
          "weights": [0.0, 1.0, 1.0],
          "logprobs": [0.0, -1.2, -1.5],
          "advantages": [0.0, 1.0, -0.5],
          "topk_token_ids": {"data": [0, 1, 2, 3], "shape": [2, 2]},
          "topk_logprobs": {"data": [-0.5, -1.2, -0.6, -1.1], "shape": [2, 2]},
        },
      ),
      Datum(
        model_input=[1, 2, 3, 4],
        loss_fn_inputs={
          "target_tokens": [2, 3, 4, 5],
          "weights": [0.0, 1.0, 1.0, 1.0],
          "logprobs": [0.0, -1.1, -1.3, -1.4],
          "advantages": [0.0, 0.8, 0.4, -0.6],
          "topk_token_ids": [[0, 1], [1, 2], [2, 3], [3, 4]],
          "topk_logprobs": [[-99999.0, -99999.0], [-0.4, -1.0], [-0.5, -1.1], [-0.6, -1.2]],
        },
      ),
    ]

    loss_config = {
      "top_p": 0.95,
      "temperature": 1.0,
      "mis_rho_min": 0.1,
      "mis_rho_max": 5.0,
      "score_centering": 1.0,
    }
    for loss_fn in ("importance_sampling", "ppo"):
      model.zero_grad()
      result = worker.forward_backward(model, datums, loss_fn, loss_config=loss_config)
      self.assertIn("loss:mean", result["metrics"])
      self.assertIn("loss:sum", result["metrics"])
      self.assertEqual(len(result["loss_fn_outputs"]), 2)
      self.assertEqual(result["loss_fn_outputs"][0]["logprobs"]["shape"], [3])
      self.assertEqual(result["loss_fn_outputs"][1]["logprobs"]["shape"], [4])
      assert model.logit_table.grad is not None
      self.assertTrue(torch.isfinite(model.logit_table.grad).all())
      self.assertGreater(model.logit_table.grad.abs().sum().item(), 0.0)


class TopkWireCodecTest(unittest.TestCase):
  def test_encode_sample_response_packs_topk_sampled_logprobs_with_sentinel_padding(self) -> None:
    raw = {
      "sequences": [
        {
          "tokens": [10, 20],
          "logprobs": [-0.25, -0.5],
          "stop_reason": "stop",
          "topk_logprobs": [
            [(10, -0.25), (11, -1.5), (12, -2.5)],
            [(20, -0.5)],  # shorter row: padded to k=3 with (0, -99999.0)
          ],
          "topk_k": 3,
        }
      ]
    }
    encoded = proto_codec.encode_sample_response(raw)
    msg = pb.SampleResponse()
    msg.ParseFromString(encoded)
    self.assertEqual(len(msg.sequences), 1)
    topk = msg.sequences[0].topk_sampled_logprobs
    self.assertEqual(topk.length, 2)
    self.assertEqual(topk.k, 3)

    token_ids = list(struct.unpack("<6i", topk.token_ids))
    logprobs = list(struct.unpack("<6f", topk.logprobs))
    self.assertEqual(token_ids, [10, 11, 12, 20, 0, 0])
    self.assertAlmostEqual(logprobs[0], -0.25, places=5)
    self.assertAlmostEqual(logprobs[3], -0.5, places=5)
    self.assertAlmostEqual(logprobs[4], -99999.0, places=1)
    self.assertAlmostEqual(logprobs[5], -99999.0, places=1)

  def test_sample_command_carries_topk_and_top_p(self) -> None:
    cmd = commands.Sample(
      request_id="req-1",
      model_id="m",
      prompt_tokens=[1, 2],
      max_tokens=8,
      temperature=0.8,
      top_p=0.95,
      top_k=50,
      topk_sample_logprobs=20,
    )
    restored = commands.parse_command(commands.wire(cmd))
    self.assertEqual(restored, cmd)


if __name__ == "__main__":
  unittest.main()
