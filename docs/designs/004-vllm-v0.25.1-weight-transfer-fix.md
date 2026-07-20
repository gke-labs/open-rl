# 004 Design Document: vLLM v0.25.1 Delta Weight Transfer & Sampler Sync Fix

**Status:** Proposed  
**Author:** Open-RL Engineering  
**Date:** 2026-07-20  
**Target Branch:** `feat/vllm-v0.25.1-upgrade`  
**Related Issues / Design Docs:** `003-vllm-v0.25.1-upgrade-and-cuda-devel-migration.md`

---

## 1. Executive Summary

During the upgrade of vLLM from `v0.20.0` to `v0.25.1`, End-to-End Reinforcement Learning (RL) benchmark runs (`fft-gsm8k-rl`) executed cleanly without runtime exceptions, but exhibited severe convergence degradation under **Delta Weight Sync** (`OPEN_RL_WEIGHT_SYNC_STRATEGY=delta`). Full weight reloads via disk checkpointing converged as expected, identifying the weight transfer and sampler synchronization subsystem as the primary source of divergence.

Critical investigation revealed two core failure modes:
1. **Silent Parameter Drop on GPU (Fused Parameter Mismatch)**: The patched `load_weights_fn` in `patch_vllm_weight_transfer.py` performs direct key matching against `dict(model.named_parameters())`. Because vLLM fuses attention (`q_proj`, `k_proj`, `v_proj` $\rightarrow$ `qkv_proj`) and MLP (`gate_proj`, `up_proj` $\rightarrow$ `gate_up_proj`) layers, **over 95% of model parameter tensors were silently skipped during GPU VRAM weight loading**.
2. **Omitted KV Cache Invalidation**: The KV cache invalidation step (`await engine.wake_up(tags=["kv_cache"])`) was omitted during `vllm_sampler.py` refactoring, leaving stale KV states in GPU memory across rollout batches.

This design document outlines the root cause analysis, proposed architectural fixes, clean in-memory runtime extension pattern (replacing brittle file patching), fail-fast refactoring guidelines, shift-left testing strategy (including realistic end-to-end CPU delta sync testing), and verification strategy to restore delta weight transfer convergence in vLLM `v0.25.1`.

---

## 2. Problem Statement & Root Cause Analysis

### 2.1 Root Cause 1: Fused Parameter Name Mismatch

In HuggingFace checkpoints and Open-RL's CPU snapshot (`_cpu_snapshot`), model parameters are stored under standard HuggingFace naming:
- `model.layers.N.self_attn.q_proj.weight`
- `model.layers.N.self_attn.k_proj.weight`
- `model.layers.N.self_attn.v_proj.weight`
- `model.layers.N.mlp.gate_proj.weight`
- `model.layers.N.mlp.up_proj.weight`

In vLLM `v0.25.1`, vLLM's internal model executor fuses these projection matrices into combined tensors:
- `model.layers.N.self_attn.qkv_proj.weight`
- `model.layers.N.mlp.gate_up_proj.weight`

In [`src/server/scripts/patch_vllm_weight_transfer.py`](file:///usr/local/google/home/sunilarora/open-rl/src/server/scripts/patch_vllm_weight_transfer.py#L56-L65), `load_weights_fn` was patched as follows:

```python
# CURRENT BROKEN IMPLEMENTATION in patch_vllm_weight_transfer.py
def load_weights_fn(weights):
    if model is not None:
        named_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in weights:
                if name in named_params:
                    named_params[name].copy_(tensor)
```

Because `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`, and `up_proj.weight` do **not** exist in `dict(model.named_parameters())`, `name in named_params` evaluates to `False`. The loop skips every attention and feed-forward weight tensor without throwing an exception. Only unfused parameters (such as `input_layernorm.weight` and `post_attention_layernorm.weight`) are copied to GPU VRAM.

Consequently, during RL sampling rollouts, the GPU sampler operates on **frozen step-0 base model attention and MLP weights**, causing training policy rollouts to diverge.

---

### 2.2 Root Cause 2: Omitted KV Cache Invalidation

In `upstream/fft` (v0.20.0), weight updates in `vllm_sampler.py` executed a strict sequence:
```python
# WORKING SEQUENCE in upstream/fft (v0.20.0)
await engine.sleep(level=1)
await engine.wake_up(tags=["weights"])
await engine.collective_rpc("update_weights", ...)
await engine.wake_up(tags=["kv_cache"])  # Invalidate & reset KV cache
```

In `feat/vllm-v0.25.1-upgrade`, the KV cache wake-up call was removed:
```python
# BROKEN SEQUENCE in feat/vllm-v0.25.1-upgrade
await engine.collective_rpc("update_weights", ...)
await engine.wake_up(tags=["weights"])
# Missing: await engine.wake_up(tags=["kv_cache"])
```

Without invalidating `kv_cache`, generation requests reuse cached KV attention blocks computed from previous rollout steps or prior weight states, corrupting autoregressive token sampling.

---

### 2.3 Deep-Dive: Fused vs. Unfused Parameters in vLLM

To assist developers implementing and maintaining this fix, this section details how weight layouts differ between HuggingFace (PyTorch training) and vLLM (high-performance inference serving):

#### 1. Unfused Format (HuggingFace Checkpoints & PyTorch Trainers)
Standard PyTorch models (e.g. `Qwen3ForCausalLM`, `Qwen2ForCausalLM`, `LlamaForCausalLM`) maintain distinct `nn.Linear` layers for each projection:
- **Attention Projections**: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
- **MLP Projections**: `gate_proj.weight`, `up_proj.weight`

These weights are stored as separate tensors in `.safetensors` files and in Open-RL's host CPU snapshot (`_cpu_snapshot`).

#### 2. Fused Format (vLLM Engine)
To maximize GPU memory bandwidth efficiency and reduce CUDA kernel launch overhead, vLLM concatenates (fuses) related parameter matrices into single combined tensors on GPU VRAM:
- **`qkv_proj.weight`**: Concatenates `q_proj`, `k_proj`, and `v_proj` along dimension 0.
- **`gate_up_proj.weight`**: Concatenates `gate_proj` and `up_proj` along dimension 0.

#### 3. Why Naive Dictionary Matching Fails
When `patch_vllm_weight_transfer.py` attempted to load weights using direct dictionary key matching:
```python
if name in dict(model.named_parameters()):
    named_params[name].copy_(tensor)
```
- `name` from HuggingFace checkpoint: `"model.layers.0.self_attn.q_proj.weight"`
- `dict(model.named_parameters())` keys in vLLM: `"model.layers.0.self_attn.qkv_proj.weight"`

Because `"q_proj.weight"` is **not** a key in vLLM's `named_parameters()`, the expression `if name in named_params` evaluated to `False` for all Q, K, V, Gate, and Up projections (representing >95% of total model parameters). The loop silently skipped these weights without throwing an exception.

#### 4. The Fix Mechanism (`model.load_weights`)
vLLM's internal model classes (`Qwen3ForCausalLM.load_weights`, `Qwen2ForCausalLM.load_weights`, `LlamaForCausalLM.load_weights`) implement a specialized `load_weights` method. This method accepts an iterator of HuggingFace `(name, tensor)` tuples, recognizes unfused names (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`), and automatically performs the necessary concatenation into vLLM's fused GPU `qkv_proj` and `gate_up_proj` parameters.

Calling `model.load_weights(weights)` guarantees that unfused HuggingFace tensors are correctly packed and transferred to GPU VRAM.

---

### 2.4 Diagnostic Signature: Parameter Change Density Trajectory (10-12% → 2.5% vs. Flatlined ~0.25%)

A key diagnostic indicator distinguishes a functioning weight-sync pipeline from a broken one:

#### Healthy Delta Sync Trajectory (`upstream/fft` Baseline & Full Weight Sync)
When sampler weights update successfully on every step:
- **Steps 1–5 (Initial Exploration)**: As the policy begins shifting, gradient updates occur across a broad set of layers, resulting in **10%–12% changed parameter density**.
- **Steps 5–30 (Convergence)**: As policy rollouts improve and converge toward optimal rewards, policy KL-divergence shrinks, and parameter change density smoothly decreases to **~2.5%**.

#### Buggy Delta Sync Trajectory (`feat/vllm-v0.25.1-upgrade`)
When sampler GPU weights are skipped due to fused name mismatch:
- The GPU sampler remains frozen on the step-0 base model parameters.
- Every rollout step is generated from the exact same initial step-0 distribution.
- The trainer computes gradient steps on static rollouts, causing parameter change density to **flatline at ~0.25% across all steps**.

---

## 3. Proposed Fix Architecture

```
                                  +---------------------------------------+
                                  | DeltaSnapshotWeightTransferEngine     |
                                  | (Reconstructs CPU Snapshot in RAM)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Native vLLM Model load_weights()      |
                                  | (Handles HF -> vLLM fused qkv/gate_up)|
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | GPU Worker Model VRAM                 |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | engine.wake_up(tags=["kv_cache"])     |
                                  | (Resets KV Cache for clean sampling)  |
                                  +---------------------------------------+
```

---

### 3.1 Component 1: Native vLLM `load_weights` Dispatch in `patch_vllm_weight_transfer.py`

Replace the naive dictionary matching in `patch_vllm_weight_transfer.py` with a call to vLLM's native model loader (`model.load_weights`), which understands HuggingFace parameter mapping and tensor fusion:

```python
def load_weights_fn(weights):
    if model is None:
        raise RuntimeError("GPUWorker model instance is None during update_weights call.")

    # 1. Prefer native vLLM model.load_weights() if available
    if hasattr(model, "load_weights"):
        model.load_weights(weights)
        return

    # 2. Fallback for runner / model_runner wrappers
    runner = getattr(self, "model_runner", None)
    if runner is not None:
        runner_model = getattr(runner, "model", None)
        if runner_model is not None and hasattr(runner_model, "load_weights"):
            runner_model.load_weights(weights)
            return

    raise RuntimeError(
        f"Target vLLM model '{type(model).__name__}' does not implement native load_weights(). "
        "Refusing to execute naive parameter copy to prevent silent parameter drops."
    )
```

---

### 3.2 Component 2: Parameter Mapper for Delta Snapshot Engine

Enhance `DeltaSnapshotWeightTransferEngine.receive_weights` in [`src/server/delta_weight_transfer_engine.py`](file:///usr/local/google/home/sunilarora/open-rl/src/server/delta_weight_transfer_engine.py) to explicitly obtain the underlying model instance and invoke `model.load_weights()` without catching and swallowing exceptions:

```python
# In DeltaSnapshotWeightTransferEngine.receive_weights:
model = getattr(load_weights_fn, "__self__", None) or getattr(self, "model", None)
if model is not None and hasattr(model, "load_weights"):
    model.load_weights(weights_to_load)
else:
    raise RuntimeError("No valid vLLM model instance with load_weights() found for delta weight transfer.")
```

---

### 3.3 Component 3: Restore KV Cache Invalidation in `vllm_sampler.py`

Update `vllm_sampler.py` to enforce KV cache reset after delta weight remapping:

```python
# In process_sampling_request in src/server/vllm_sampler.py:
if os.getenv("OPEN_RL_WEIGHT_SYNC_STRATEGY", "delta").lower() == "delta":
    print(f"[vLLM Worker] Receiving incremental delta weights from {weights_path}...")
    await engine.collective_rpc(
        "update_weights",
        kwargs={
            "update_info": {
                "kind": "delta_snapshot",
                "target_weights_path": weights_path,
                "base_model_path": base_model_path,
            }
        },
    )
    print("[vLLM Worker] Remapping updated host CPU weights to GPU VRAM...")
    await engine.wake_up(tags=["weights"])
    print("[vLLM Worker] Resetting KV cache for updated weights...")
    await engine.wake_up(tags=["kv_cache"])  # RESTORED KV CACHE RESET
```

---

### 3.4 Component 4: Clean In-Memory Runtime Extension (Replacing Disk-Level File Patching)

To eliminate brittle disk-level file modifications (`patch_vllm_weight_transfer.py` doing string search-and-replace inside `.venv/site-packages/vllm/...`), we replace file patching with **Clean In-Memory Runtime Extension** executed dynamically on import:

#### 1. In-Memory Engine Registration
vLLM's `WeightTransferEngineFactory` provides a public `@classmethod register_engine(name, cls)`. We register Open-RL's engine in Python memory at sampler startup without touching disk files:

```python
# In server/vllm_sampler.py or plugin init module (Zero disk modifications)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from server.delta_weight_transfer_engine import DeltaSnapshotWeightTransferEngine

# Register Open-RL engine dynamically in vLLM's in-memory factory registry
WeightTransferEngineFactory.register_engine("delta_snapshot", DeltaSnapshotWeightTransferEngine)
```

#### 2. Dynamic In-Memory Method Binding for `GPUWorker`
Instead of string-editing `vllm/v1/worker/gpu_worker.py` on disk during container builds, we wrap `GPUWorker.update_weights` dynamically in memory:

```python
# In server/scripts/vllm_patches.py (Executed dynamically in memory)
from vllm.v1.worker.gpu_worker import GPUWorker

_original_update_weights = getattr(GPUWorker, "update_weights", None)

def _open_rl_update_weights(self, update_info: dict = None, **kwargs):
    if isinstance(update_info, dict) and update_info.get("kind") == "delta_snapshot":
        model = self.get_model() if hasattr(self, "get_model") else self.model_runner.get_model()
        if not hasattr(self, "_delta_transfer_engine"):
            self._delta_transfer_engine = DeltaSnapshotWeightTransferEngine(model=model)
        
        self._delta_transfer_engine.receive_weights(
            update_info=update_info,
            load_weights=model.load_weights,  # REUSES NATIVE vLLM LOAD_WEIGHTS
        )
        return

    if _original_update_weights is not None:
        return _original_update_weights(self, update_info, **kwargs)

# Bind method in memory cleanly
GPUWorker.update_weights = _open_rl_update_weights
```

---

## 4. Elimination of Defensive Fallbacks & Fail-Fast Refactoring Guidance

To adhere to Open-RL architectural standards (explicit initialization, fail-fast error handling, and avoiding defensive fallbacks that mask bugs), the implementor **must remove all silent fallbacks** in the delta weight transfer path.

### 4.1 Identified Defensive Patterns & Refactoring Directives

| Code Location | Existing Defensive Fallback (Anti-Pattern) | Why It Masks Bugs | Required Fail-Fast Refactoring |
| :--- | :--- | :--- | :--- |
| `patch_vllm_weight_transfer.py` | Silent loop over `named_parameters()` when `load_weights` is missing. | Silently skips unmatched parameters (95%+ of model) without raising an error. | Raise an explicit `RuntimeError("Target vLLM model does not implement native load_weights()")` if `load_weights` cannot be resolved. |
| `delta_weight_transfer_engine.py` | `try...except Exception:` block around `load_weights_fn`, catching errors and doing a partial dict copy. | Catches and swallows genuine weight-loading exceptions, leaving GPU in a corrupt state while printing a warning. | Remove `try...except` fallback entirely. Allow exceptions from `model.load_weights()` to fail fast with a full stack trace. |
| `vllm_sampler.py` | Falling back from `update_weights` (delta sync) to `reload_weights` (full disk sync) on exception. | Silently degrades to slow disk reloads without alerting developers that delta sync is broken. | Remove silent disk-reload fallback. Fail immediately if RPC `update_weights` fails under delta strategy. |
| `delta_weight_transfer_engine.py` | Catching base model safetensors loading errors and falling back to model parameter iteration. | Obscures invalid HF base model paths or corrupt safetensors files. | Raise `FileNotFoundError` or `ValueError` immediately if base model safetensors cannot be loaded. |
| `_resolve_target_key` | Dynamic guessing (`model.` prefix stripping/prepending) during delta patching. | Obscures key format mismatches between trainer state dicts and CPU snapshots. | Normalize state dict parameter keys strictly at CPU snapshot initialization time. Fail fast with `KeyError` showing exact missing keys if a delta key is unrecognized. |

### 4.2 Explicit Invariants for Implementor

1. **No Silent Error Swallowing**: Never catch broad `Exception` only to print a warning and continue execution in a degraded or un-updated state.
2. **Explicit Attribute Initialization**: All engine state variables (`self._cpu_snapshot`, `self.current_weights_path`, `self._base_model`) must be explicitly declared and initialized in `__init__`. Avoid dynamic `hasattr(self, ...)` or `getattr(self, ..., None)` defensive checks where possible.
3. **Fail-Fast Error Reporting**: If a delta tensor shape, dtype, or key name does not match expected model parameters, fail immediately during `receive_weights` with clear diagnostic context.

---

## 5. Verification & Testing Plan

### 5.1 Unit Testing (`make test`)

1. **Zero-GPU CPU Unit Test (Non-Mocked Real vLLM Load Test)** (`tests/test_sampler_patch.py`):
   - Instantiate a tiny 2-layer `Qwen3Config` (`hidden_size=64`, `num_attention_heads=2`) directly on CPU (`torch.device("cpu")`).
   - Construct real vLLM model instance `Qwen3ForCausalLM(config=config)` on CPU without any GPU requirements.
   - Pass synthetic unfused HuggingFace weight tuples (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`).
   - Call `model.load_weights(unfused_weights)` on CPU.
   - Assert that fused `qkv_proj.weight` and `gate_up_proj.weight` tensors on the CPU model instance are updated with non-zero values.

2. **Realistic End-to-End CPU Delta Sync Unit Test (Compute & Reconstruct)** (`tests/test_delta_weight_transfer_engine.py`):
   - Run a 100% real CPU end-to-end test pipeline: Base Model Setup $\rightarrow$ Trainer Mutation $\rightarrow$ `save_state_delta()` $\rightarrow$ `receive_weights()` $\rightarrow$ vLLM CPU Model parameter verification.

### 5.2 End-to-End Cluster Benchmark Verification (`make cluster-e2e`)

1. **Overfit Benchmark (`tiny-rl` / `fft-gsm8k-rl`)**:
   - Run 10–30 steps of `Qwen3-8B` or `Qwen2.5-0.5B-Instruct` RL training under `OPEN_RL_WEIGHT_SYNC_STRATEGY=delta`.
   - Verify live metrics progression (`Step | Accuracy | Reward | Sampling | Train Step`):
     - Assert that `env/all/reward/total` increases monotonically over iterations.
     - Assert that `env/all/correct` matches baseline rates achieved in `upstream/fft`.
   - **Parameter Delta Density Curve Assertion**:
     - Verify that per-step parameter change density starts high (~10%–12%) in early steps and gradually settles to ~2.5% as training converges, rather than remaining flatlined at ~0.25%.

---

### 5.3 Shift-Left Testing Strategy for Early Local Bug Detection

To prevent bugs from only being discovered late during multi-minute Kubernetes GPU cluster runs, the engineering workflow adopts a **4-tier local fast-feedback testing strategy**:

#### Tier 1: Zero-GPU CPU Unit Test for Parameter Fusion (`tests/test_sampler_patch.py`)
- **Execution Time**: $< 0.3$ seconds (`python3 -m unittest tests/test_sampler_patch.py`).
- **GPU Requirement**: **Zero (Runs 100% on CPU)**.
- **Implementation**: Instantiates a tiny 2-layer Qwen3 architecture on CPU (`device="cpu"`).
- **What it Catches**: Tests that passing unfused HuggingFace weight names (`q_proj`, `k_proj`, `v_proj`) to `model.load_weights()` correctly updates vLLM fused layer parameters (`qkv_proj`). If a developer re-introduces string matching against `named_parameters()`, this unit test fails locally on any laptop/Cloudtop in under 300 milliseconds.

```python
# Clean, Explicit Zero-GPU CPU Unit Test Pattern in tests/test_sampler_patch.py
import unittest
import torch
from transformers import Qwen3Config
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

class TestSamplerWeightFusionCPU(unittest.TestCase):
    def test_load_weights_unfused_to_fused_on_cpu(self):
        config = Qwen3Config(
            hidden_size=64, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=2,
            num_hidden_layers=2, vocab_size=1000,
        )
        model = Qwen3ForCausalLM(config=config)
        unfused_weights = [
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(64, 64)),
            ("model.layers.0.self_attn.k_proj.weight", torch.ones(64, 64)),
            ("model.layers.0.self_attn.v_proj.weight", torch.ones(64, 64)),
            ("model.layers.0.mlp.gate_proj.weight", torch.ones(128, 64)),
            ("model.layers.0.mlp.up_proj.weight", torch.ones(128, 64)),
        ]
        model.load_weights(unfused_weights)
        qkv_tensor = model.model.layers[0].self_attn.qkv_proj.weight
        gate_up_tensor = model.model.layers[0].mlp.gate_up_proj.weight
        self.assertFalse(torch.all(qkv_tensor == 0), "qkv_proj was not updated by load_weights!")
        self.assertFalse(torch.all(gate_up_tensor == 0), "gate_up_proj was not updated by load_weights!")
```

#### Tier 2: Realistic End-to-End CPU Delta Sync Test (`tests/test_delta_weight_transfer_engine.py`)
- **Execution Time**: $< 1.0$ seconds (`python3 -m unittest tests/test_delta_weight_transfer_engine.py`).
- **GPU Requirement**: **Zero (Runs 100% on CPU)**.
- **What it Catches**: Tests the full pipeline from trainer diff calculation, saving `delta.safetensors`, CPU snapshot patching, to vLLM model `load_weights` on CPU.

```python
# Realistic End-to-End CPU Delta Sync Unit Test Pattern in tests/test_delta_weight_transfer_engine.py
import tempfile
import unittest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from server.delta_weight_transfer_engine import DeltaSnapshotWeightTransferEngine
from training.fft_trainer_worker import FFTTrainingWorker

class TestEndToEndDeltaSyncCPU(unittest.TestCase):
    def test_compute_and_reconstruct_delta_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, vocab_size=1000)
            base_model = AutoModelForCausalLM.from_config(config)
            base_path = f"{tmp_dir}/base_model"
            base_model.save_pretrained(base_path)

            vllm_sampler_model = Qwen3ForCausalLM(config=config)
            engine = DeltaSnapshotWeightTransferEngine()
            engine._ensure_cpu_snapshot(base_model=base_path, model=None)

            trainer = FFTTrainingWorker()
            trainer.model = base_model
            trainer.set_weight_sync_strategy("delta")
            trainer.total_model_elements = sum(p.numel() for p in base_model.parameters())
            trainer.model_layer_names = [n for n, _ in base_model.named_parameters()]

            with torch.no_grad():
                base_model.model.layers[0].self_attn.q_proj.weight.add_(0.5)

            delta_dir = f"{tmp_dir}/delta_step_1"
            trainer.save_state_delta(model_id="test", state_path=delta_dir)

            engine.receive_weights(
                update_info={"target_weights_path": delta_dir},
                load_weights=vllm_sampler_model.load_weights,
            )

            qkv_weight = vllm_sampler_model.model.layers[0].self_attn.qkv_proj.weight
            self.assertFalse(torch.all(qkv_weight == 0), "Fused qkv_proj was not updated on CPU!")
```

#### Tier 3: Fast Syntax & Compilation Checks (`make lint`)
- **Execution Time**: $< 2$ seconds (`python3 -m py_compile ... && make lint`).
- **GPU Requirement**: Zero.
- **What it Catches**: Catches syntax errors, indentation bugs, unused imports, and uninitialized attribute type errors prior to Docker container image builds (`make build-images`).

#### Tier 4: Local Overfit Smoke Harness (`make test`)
- **Execution Time**: $< 15$ seconds (`export PATH=$PATH:$HOME/.local/bin && make test`).
- **GPU Requirement**: Zero (Runs CPU discovery tests in `examples` environment).
- **What it Catches**: Executes unit tests in the client/examples environment (`uv --project examples ...`) prior to pushing container images to remote container registries or Kubernetes clusters.

---

## 6. Upstream Contribution Strategy (vLLM Plugin Entrypoints)

To eliminate monkey-patching entirely in future vLLM versions, Open-RL will contribute two clean enhancements to `vllm-project/vllm`:

1. **Upstream PR for Dynamic Engine Discovery**: Add standard Python `importlib.metadata.entry_points(group="vllm.weight_transfer_engines")` discovery to `WeightTransferEngineFactory`.
2. **Native Entry Point Registration**: Configure Open-RL's `pyproject.toml` to expose `delta_snapshot` natively:
   ```toml
   [project.entry-points."vllm.weight_transfer_engines"]
   delta_snapshot = "server.delta_weight_transfer_engine:DeltaSnapshotWeightTransferEngine"
   ```

---

## 7. Timeline & Migration Risk

- **Risk Assessment**: Low. The fix restores standard vLLM weight loading patterns without modifying GPU memory layouts or altering vLLM's public API interfaces.
- **Backwards Compatibility**: Fully compatible with vLLM `v0.25.1` and `v0.20.0`.
