import unittest

import torch

from training.automodel_worker import GroupCheckpointedLayers, attention_kwargs, lora_target_modules, round_robin_permutation
from training.lora_trainer_worker import LoraConfig


def rank_shards(cp_size: int, padded: int) -> list[torch.Tensor]:
  """Positions each rank holds under torch's load-balanced CP layout."""
  chunk = padded // (2 * cp_size)
  chunks = torch.arange(padded).split(chunk)
  return [torch.cat((chunks[r], chunks[2 * cp_size - 1 - r])) for r in range(cp_size)]


class Layer(torch.nn.Module):
  def __init__(self, scale: float):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(scale))

  def forward(self, x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x * self.weight + shift


class AutomodelWorkerHelpersTest(unittest.TestCase):
  def test_round_robin_permutation_restores_position_order(self):
    for cp_size, padded in ((2, 16), (4, 64), (8, 4096)):
      positions = torch.arange(padded, dtype=torch.float32)
      gathered = torch.cat([positions[idx] for idx in rank_shards(cp_size, padded)]).unsqueeze(0)
      perm = round_robin_permutation(cp_size, padded, torch.device("cpu"))
      restored = torch.zeros_like(gathered).index_copy(1, perm, gathered)
      self.assertTrue(torch.equal(restored[0], positions))
      self.assertEqual(perm.unique().numel(), padded)

  def test_group_checkpointing_matches_plain_forward_and_backward(self):
    layers = torch.nn.ModuleDict({str(i): Layer(1.0 + i / 10) for i in range(6)})
    x = torch.randn(4, requires_grad=True)
    shift = torch.ones(4)

    def run(module_dict):
      h = x
      for layer in module_dict.values():
        h = layer(x=h, shift=shift)
      return h.sum()

    expected = run(layers)
    expected_grads = torch.autograd.grad(expected, [x, *layers.parameters()])

    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = 4
    self.assertEqual(len(list(layers.values())), 2)
    actual = run(layers)
    actual_grads = torch.autograd.grad(actual, [x, *layers.parameters()])
    self.assertTrue(torch.allclose(actual, expected))
    for a, e in zip(actual_grads, expected_grads):
      self.assertTrue(torch.allclose(a, e))

    layers.eval()
    self.assertEqual(len(list(layers.values())), 6)

  def test_attention_kwargs_by_model_family(self):
    self.assertEqual(attention_kwargs("qwen3_5_text", cp_size=4), {"attn_implementation": "sdpa"})
    self.assertEqual(attention_kwargs("gemma4_text", cp_size=1), {"attn_implementation": "ffpa", "use_sdpa_patching": False})
    under_cp = attention_kwargs("gemma4_text", cp_size=4)
    self.assertEqual(under_cp["attn_implementation"], "sdpa")
    self.assertEqual(under_cp["text_config"]["cp_full_attn_backend"], "ffpa")

  def test_lora_target_modules_follow_the_client_config(self):
    self.assertIn("model.*.layers.*.q_proj", lora_target_modules(LoraConfig()))
    self.assertIn("model.*.layers.*.down_proj", lora_target_modules(LoraConfig()))
    self.assertNotIn("model.*.layers.*.down_proj", lora_target_modules(LoraConfig(train_mlp=False)))
    with self.assertRaises(ValueError):
      lora_target_modules(LoraConfig(train_unembed=True))

  def test_stage_and_swap_uses_rank_invariant_staging_path(self):
    import os
    import tempfile
    from unittest.mock import patch

    from training.automodel_worker import AutomodelTrainingWorker

    with tempfile.TemporaryDirectory() as tmpdir:
      target = os.path.join(tmpdir, "step-1")
      worker = AutomodelTrainingWorker(full_parameter=False)
      worker.model = Layer(1.0)
      seen: list[str] = []
      worker.write_weights = lambda path: seen.append(path) or (open(f"{path}/w", "w").close() if os.path.isdir(path) else None)  # type: ignore[method-assign]
      with patch("training.automodel_worker.is_primary", side_effect=[True, True, False, False]):
        worker.model_id = "job"
        worker.save_state(target)
        worker.save_state(target)
      self.assertEqual(seen, [f"{target}.staging", f"{target}.staging"])
      self.assertTrue(os.path.isfile(f"{target}/metadata.json"))


if __name__ == "__main__":
  unittest.main()
