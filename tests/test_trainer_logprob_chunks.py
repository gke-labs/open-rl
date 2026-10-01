import os
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from training.trainer_worker import BaseTrainerWorker

VOCAB = 96
TINY = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
# Small enough that tanh is far from linear on these logits, so a missing softcap changes the result.
SOFTCAP = 0.25


def qwen3() -> torch.nn.Module:
  torch.manual_seed(0)
  return Qwen3ForCausalLM(Qwen3Config(**TINY)).eval()


def gemma4() -> torch.nn.Module:
  torch.manual_seed(0)
  text = dict(TINY, final_logit_softcapping=SOFTCAP, vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=8)
  return Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=None, audio_config=None)).eval()


def with_lora(model: torch.nn.Module) -> torch.nn.Module:
  peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
  # LoRA B starts at zero; perturb it so the adapter changes the logits and gets nonzero gradients.
  with torch.no_grad():
    for name, param in peft_model.named_parameters():
      if "lora_B" in name:
        param.normal_(std=0.5)
  return peft_model


class LogitsOnlyModel(torch.nn.Module):
  """A model that returns logits but exposes no decoder body or output head."""

  def __init__(self):
    super().__init__()
    self.embed = torch.nn.Embedding(VOCAB, VOCAB)

  def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
    return type("Output", (), {"logits": self.embed(input_ids)})()


def batch():
  torch.manual_seed(1)
  input_ids = torch.randint(0, VOCAB, (2, 9))
  attention_mask = torch.ones_like(input_ids)
  attention_mask[1, 6:] = 0
  target_token_ids = torch.randint(0, VOCAB, (2, 7))
  return input_ids, attention_mask, target_token_ids


def full_logits_logprobs(model, input_ids, attention_mask, target_token_ids) -> torch.Tensor:
  logits = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).logits[:, : target_token_ids.shape[1], :]
  return torch.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)


def record_head_positions(model) -> list[int]:
  """Record the number of positions in each call to the model's output head."""
  positions = []
  model.get_output_embeddings().register_forward_hook(lambda _module, args, _output: positions.append(args[0].shape[1]))
  return positions


def logprobs_and_grads(fn, model, *inputs):
  model.zero_grad(set_to_none=True)
  logprobs = fn(model, *inputs)
  weights = torch.linspace(0.5, 1.5, logprobs.numel()).reshape(logprobs.shape)
  (logprobs * weights).sum().backward()
  grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
  return logprobs.detach(), grads


class TestChunkedTargetLogprobs(unittest.TestCase):
  def worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    return worker

  def assert_matches_full_logits(self, model, chunk_tokens: int):
    inputs = batch()
    expected, expected_grads = logprobs_and_grads(full_logits_logprobs, model, *inputs)
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_LOGPROB_CHUNK_TOKENS": str(chunk_tokens)}):
      actual, actual_grads = logprobs_and_grads(self.worker().compute_target_logprobs, model, *inputs)
    torch.testing.assert_close(actual, expected)
    self.assertTrue(expected_grads)
    self.assertEqual(actual_grads.keys(), expected_grads.keys())
    # Chunking changes the fp32 summation order, so gradients agree to rounding
    # relative to each tensor's scale, not bit for bit.
    for name, grad in expected_grads.items():
      torch.testing.assert_close(actual_grads[name], grad, rtol=0, atol=1e-4 * grad.abs().max().item(), msg=name)

  def test_matches_full_logits_path(self) -> None:
    # 2 rows: 1 token gives one position per chunk; 6 gives chunks of 3, 3, 1; 1024 is one chunk.
    models = [("qwen3", qwen3), ("gemma4", gemma4), ("qwen3+lora", lambda: with_lora(qwen3())), ("gemma4+lora", lambda: with_lora(gemma4()))]
    for name, build in models:
      for chunk_tokens in (1, 6, 1024):
        with self.subTest(model=name, chunk_tokens=chunk_tokens):
          self.assert_matches_full_logits(build(), chunk_tokens)

  def test_output_head_runs_per_chunk(self) -> None:
    for name, build in [("qwen3", qwen3), ("gemma4+lora", lambda: with_lora(gemma4()))]:
      with self.subTest(model=name):
        model = build()
        head_positions = record_head_positions(model)
        with patch.dict(os.environ, {"OPEN_RL_TRAIN_LOGPROB_CHUNK_TOKENS": "6"}):
          self.worker().compute_target_logprobs(model, *batch())
        self.assertEqual(head_positions, [3, 3, 1])

  def test_default_chunk_size(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(self.worker().logprob_chunk_tokens(), 1024)

  def test_split_finds_body_head_and_softcap(self) -> None:
    for name, model, softcap in [("qwen3", qwen3(), None), ("gemma4", gemma4(), SOFTCAP), ("gemma4+lora", with_lora(gemma4()), SOFTCAP)]:
      with self.subTest(model=name):
        body, head, found_softcap = self.worker().split_causal_lm(model)
        self.assertIs(head, model.get_output_embeddings())
        self.assertNotIsInstance(body, type(model))
        self.assertEqual(found_softcap, softcap)

  def test_model_without_body_and_head_uses_full_logits(self) -> None:
    model = LogitsOnlyModel()
    self.assertIsNone(self.worker().split_causal_lm(model))
    inputs = batch()
    torch.testing.assert_close(self.worker().compute_target_logprobs(model, *inputs), full_logits_logprobs(model, *inputs))


if __name__ == "__main__":
  unittest.main()
