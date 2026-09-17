"""A real, tiny Llama checkpoint built offline for tests.

Two layers, hidden size 32, a 64-token word-level tokenizer. Small enough to
train on CPU in a second, real enough that transformers and peft treat it as
any other checkpoint, so the workers under test run their production code.
"""

import os

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

VOCAB_SIZE = 64


def build_tiny_llama(directory: str, seed: int = 0) -> str:
  torch.manual_seed(seed)
  config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    tie_word_embeddings=False,
  )
  LlamaForCausalLM(config).save_pretrained(directory)

  vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, **{f"w{i}": i for i in range(3, VOCAB_SIZE)}}
  tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
  tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
  PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="</s>").save_pretrained(directory)
  return os.path.abspath(directory)
