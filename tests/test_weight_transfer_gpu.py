"""Opt-in GPU smoke test: local tiny Qwen2, no model download or cluster required."""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_delta_weight_transfer_engine import write_delta


def runtime_weights(worker):
  return {name: value.detach().cpu().clone() for name, value in worker.get_model().named_parameters()}


@unittest.skipUnless(os.getenv("OPEN_RL_GPU_TESTS") == "1", "Set OPEN_RL_GPU_TESTS=1 with make test-weight-transfer")
class WeightTransferGPUTest(unittest.TestCase):
  def test_sparse_and_dense_updates_match_real_qwen_loader_and_generation(self):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from vllm import LLM, SamplingParams
    from vllm.config import WeightTransferConfig

    # collective_rpc(runtime_weights) ships a function to the worker, which vLLM only pickles when this is set.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      config = Qwen2Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
      )
      hf_model = Qwen2ForCausalLM(config).to(torch.bfloat16)
      hf_model.save_pretrained(path / "base")
      llm = LLM(
        model=str(path / "base"),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=32,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        kv_cache_memory_bytes=16 << 20,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        enable_prefix_caching=False,
        weight_transfer_config=WeightTransferConfig(backend="delta_snapshot"),
      )
      try:
        sampling = SamplingParams(temperature=0, max_tokens=2, logprobs=5, detokenize=False, ignore_eos=True)
        llm.generate([{"prompt_token_ids": [1, 2, 3]}], sampling, use_tqdm=False)
        before = llm.collective_rpc(runtime_weights)[0]
        names = [
          "model.layers.0.self_attn.q_proj.weight",
          "model.layers.0.self_attn.k_proj.bias",
          "model.layers.1.mlp.up_proj.weight",
          "model.embed_tokens.weight",
          "model.norm.weight",
        ]
        tensors, shapes = {}, []
        for i, name in enumerate(names):
          param = hf_model.get_parameter(name)
          shapes.append(list(param.shape))
          indices = (
            torch.arange(param.numel(), dtype=torch.int32) if name == "model.norm.weight" else torch.tensor([0, param.numel() - 1], dtype=torch.int32)
          )
          values = (
            torch.zeros(indices.numel(), dtype=param.dtype) if name == "model.norm.weight" else torch.full((indices.numel(),), 0.5, dtype=param.dtype)
          )
          param.data.view(-1)[indices.long()] = values
          tensors[f"{i}.indices"] = indices
          tensors[f"{i}.values"] = values
        write_delta(path / "delta", names, shapes, tensors)
        llm.start_weight_update()
        llm.update_weights({"update_info": {"target_weights_path": str(path / "delta")}})
        llm.finish_weight_update(weight_version="sparse")
        sparse = llm.collective_rpc(runtime_weights)[0]
        sparse_output = llm.generate([{"prompt_token_ids": [1, 2, 3]}], sampling, use_tqdm=False)[0].outputs[0]
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in sparse.items()))
        hf_model.save_pretrained(path / "full")
        llm.start_weight_update()
        llm.update_weights({"update_info": {"target_weights_path": str(path / "full")}})
        llm.finish_weight_update(weight_version="dense")
        dense = llm.collective_rpc(runtime_weights)[0]
        for name in sparse:
          self.assertTrue(torch.equal(sparse[name], dense[name]), name)
        dense_output = llm.generate([{"prompt_token_ids": [1, 2, 3]}], sampling, use_tqdm=False)[0].outputs[0]
        self.assertEqual(sparse_output.token_ids, dense_output.token_ids)
        self.assertEqual(sparse_output.logprobs, dense_output.logprobs)
      finally:
        llm.llm_engine.engine_core.shutdown()
