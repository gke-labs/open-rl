"""Zero-GPU CPU Unit Test for vLLM Sampler Parameter Fusion & Weight Loading."""

import os
import sys
import unittest

import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


try:
  import vllm  # noqa: F401

  HAS_VLLM = True
except ImportError:
  HAS_VLLM = False


@unittest.skipUnless(HAS_VLLM, "vLLM not installed in current environment")
class TestSamplerWeightFusionCPU(unittest.TestCase):
  """Tests parameter loading and unfused-to-fused parameter packing on CPU."""

  def test_load_weights_unfused_to_fused_on_cpu(self):
    """Test native load_weights packs unfused HF parameters into fused qkv_proj and gate_up_proj on CPU."""
    from unittest.mock import MagicMock

    from transformers import Qwen3Config
    from vllm.config import VllmConfig
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    hf_config = Qwen3Config(
      hidden_size=256,
      intermediate_size=512,
      num_attention_heads=4,
      num_key_value_heads=4,
      head_dim=64,
      num_hidden_layers=2,
      vocab_size=1000,
    )
    import os

    os.environ["VLLM_USE_BYTECODE_HOOK"] = "0"

    from vllm.config.compilation import CompilationConfig, CompilationMode

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_config = hf_config
    vllm_config.quant_config = None
    comp_config = CompilationConfig(mode=CompilationMode.NONE)
    comp_config.custom_ops = ["all"]
    vllm_config.compilation_config = comp_config
    vllm_config.cache_config.cache_dtype = "auto"

    import vllm.distributed.parallel_state as ps
    from vllm.config.vllm import set_current_vllm_config

    mock_group = MagicMock()
    mock_group.is_first_rank = True
    mock_group.is_last_rank = True
    mock_group.rank_in_group = 0
    mock_group.world_size = 1

    ps._TP = mock_group
    ps._PP = mock_group

    from unittest.mock import patch

    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    with (
      patch("vllm.v1.attention.selector._cached_get_attn_backend", return_value=CPUAttentionBackend),
      set_current_vllm_config(vllm_config),
    ):
      model = Qwen3ForCausalLM(vllm_config=vllm_config)

    unfused_weights = [
      ("model.layers.0.self_attn.q_proj.weight", torch.ones(256, 256)),
      ("model.layers.0.self_attn.k_proj.weight", torch.ones(256, 256) * 2),
      ("model.layers.0.self_attn.v_proj.weight", torch.ones(256, 256) * 3),
      ("model.layers.0.mlp.gate_proj.weight", torch.ones(512, 256) * 4),
      ("model.layers.0.mlp.up_proj.weight", torch.ones(512, 256) * 5),
    ]

    # Invoke native vLLM load_weights on CPU
    model.load_weights(unfused_weights)

    # Verify that fused qkv_proj and gate_up_proj received non-zero weight tensors on CPU
    qkv_tensor = model.model.layers[0].self_attn.qkv_proj.weight
    gate_up_tensor = model.model.layers[0].mlp.gate_up_proj.weight

    self.assertTrue(torch.all(qkv_tensor[0:256] == 1.0), "q_proj slice was not loaded correctly!")
    self.assertTrue(torch.all(qkv_tensor[256:512] == 2.0), "k_proj slice was not loaded correctly!")
    self.assertTrue(torch.all(qkv_tensor[512:768] == 3.0), "v_proj slice was not loaded correctly!")
    self.assertTrue(torch.all(gate_up_tensor[0:512] == 4.0), "gate_proj slice was not loaded correctly!")
    self.assertTrue(torch.all(gate_up_tensor[512:1024] == 5.0), "up_proj slice was not loaded correctly!")


if __name__ == "__main__":
  unittest.main()
