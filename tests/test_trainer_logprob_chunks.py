import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
  CohereConfig,
  CohereForCausalLM,
  Gemma4Config,
  Gemma4ForConditionalGeneration,
  GraniteConfig,
  GraniteForCausalLM,
  Qwen3Config,
  Qwen3ForCausalLM,
)

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


def cohere() -> torch.nn.Module:
  torch.manual_seed(0)
  return CohereForCausalLM(CohereConfig(**TINY, logit_scale=0.5)).eval()


def granite() -> torch.nn.Module:
  torch.manual_seed(0)
  return GraniteForCausalLM(GraniteConfig(**TINY, logits_scaling=4.0)).eval()


def with_lora(model: torch.nn.Module) -> torch.nn.Module:
  peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
  # LoRA B starts at zero; perturb it so the adapter changes the logits and gets nonzero gradients.
  with torch.no_grad():
    for name, param in peft_model.named_parameters():
      if "lora_B" in name:
        param.normal_(std=0.5)
  return peft_model


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


def logprobs_and_grads(fn, model, *inputs):
  model.zero_grad(set_to_none=True)
  logprobs = fn(model, *inputs)
  weights = torch.linspace(0.5, 1.5, logprobs.numel()).reshape(logprobs.shape)
  (logprobs * weights).sum().backward()
  grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
  return logprobs.detach(), grads


def record_head_rows(model) -> list[int]:
  """Record the number of rows in each call to the model's output head."""
  rows = []
  model.get_output_embeddings().register_forward_hook(lambda _module, args, _output: rows.append(args[0].shape[0]))
  return rows


def chunk_size(rows: int):
  return patch.object(BaseTrainerWorker, "LOGPROB_CHUNK_TOKENS", rows)


class TestChunkedTargetLogprobs(unittest.TestCase):
  def test_matches_full_logits_path(self) -> None:
    # Cohere and Granite rescale logits after the head, so they take the full-logits path.
    models = {
      "qwen3": qwen3,
      "gemma4": gemma4,
      "qwen3+lora": lambda: with_lora(qwen3()),
      "gemma4+lora": lambda: with_lora(gemma4()),
      "cohere": cohere,
      "granite": granite,
    }
    # 2 x 7 = 14 rows: 1 gives one row per chunk; 6 gives chunks of 6, 6, 2; 1024 is one chunk.
    for name, build in models.items():
      for rows in (1, 6, 1024):
        with self.subTest(model=name, rows=rows):
          model = build()
          expected, expected_grads = logprobs_and_grads(full_logits_logprobs, model, *batch())
          with chunk_size(rows):
            actual, actual_grads = logprobs_and_grads(BaseTrainerWorker().compute_target_logprobs, model, *batch())
          torch.testing.assert_close(actual, expected)
          self.assertTrue(expected_grads)
          self.assertEqual(actual_grads.keys(), expected_grads.keys())
          # Chunking changes the fp32 summation order, so gradients agree to rounding
          # relative to each tensor's scale, not bit for bit.
          for param, grad in expected_grads.items():
            torch.testing.assert_close(actual_grads[param], grad, rtol=0, atol=1e-4 * grad.abs().max().item(), msg=param)

  def test_output_head_runs_per_chunk(self) -> None:
    for name, build, rows, expected in [("qwen3", qwen3, 1, [1] * 14), ("gemma4+lora", lambda: with_lora(gemma4()), 6, [6, 6, 2])]:
      with self.subTest(model=name, rows=rows):
        model = build()
        head_rows = record_head_rows(model)
        with chunk_size(rows):
          BaseTrainerWorker().compute_target_logprobs(model, *batch())
        self.assertEqual(head_rows, expected)


if __name__ == "__main__":
  unittest.main()
