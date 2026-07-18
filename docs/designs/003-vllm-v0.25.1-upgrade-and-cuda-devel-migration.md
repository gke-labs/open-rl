# Design Doc 003: vLLM v0.25.1 Upgrade & CUDA Devel Base Image Migration

**Status**: Accepted  
**Author**: Open-RL Engineering  
**Date**: 2026-07-17  
**Target Branch**: `fft`  

---

## 1. Executive Summary

This design document specifies the architecture, dependency upgrades, container base image migration, and validation strategy for upgrading vLLM from `v0.20.0` to `v0.25.1` in the Open-RL framework.

### Core Objectives
1. **vLLM Upgrade**: Upgrade vLLM dependency to `v0.25.1` across `pyproject.toml` and lockfiles.
2. **CUDA Devel Base Image Migration**: Transition the server container image from `nvidia/cuda:12.1.0-base-ubuntu22.04` to `nvidia/cuda:12.4.1-devel-ubuntu22.04` to support compiling custom runtime Triton kernels during engine startup.
3. **Eliminating Weight Transfer File Patching & Using Native Hooks**:
   - Retain `patch_vllm_lora_dedup.py` for Gemma4 LoRA module deduplication during activation.
   - Eliminate file-level monkey-patching (`patch_vllm_weight_transfer.py` removed from build pipeline). Use vLLM's native extension hook `WeightTransferEngineFactory.register_engine("delta_snapshot", DeltaSnapshotWeightTransferEngine)` dynamically in `delta_weight_transfer_engine.py` at runtime.
4. **Comprehensive Validation & End-to-End Cluster Benchmark**:
   - Perform local syntax checks (`py_compile`), formatting/linting (`make lint` / `make fmt`), and unit testing (`make test`).
   - Rebuild and push container images (`make build-images push-images`).
   - Deploy to the Kubernetes GPU cluster and run functional E2E verification using a 3-step delta weight sync run with `Qwen3-0.6B` / `Qwen2.5-0.5B-Instruct`.

---

## 2. Scope & Target Files

- **Dependency & Build Configuration**:
  - `pyproject.toml` (bump `vllm==0.25.1` and update wheel index URL)
  - `uv.lock`
  - `src/server/Dockerfile` (update base image to `nvidia/cuda:12.4.1-devel-ubuntu22.04`)
- **Patch & Engine Files**:
  - `src/server/scripts/patch_vllm_weight_transfer.py`
  - `src/server/scripts/patch_vllm_lora_dedup.py`
  - `src/server/delta_weight_transfer_engine.py`
  - `src/server/vllm_sampler.py`
- **Design & Test Files**:
  - `docs/designs/003-vllm-v0.25.1-upgrade-and-cuda-devel-migration.md`
  - `tests/test_vllm_sampler.py` / `tests/test_diffing_backends.py`

---

## 3. Detailed Architectural Specification

### 3.1 Base Image & Environment Changes
Update `src/server/Dockerfile`:
```dockerfile
# Use NVIDIA CUDA 12.4.1 devel base image for PyTorch/vLLM runtime compilation
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
```

Update `pyproject.toml`:
```toml
vllm = [
    "torch; sys_platform == 'linux'",
    "torchvision; sys_platform == 'linux'",
    "torch-c-dlpack-ext; sys_platform == 'linux'",
    "vllm==0.25.1; sys_platform == 'linux'",
]
```
And update `[tool.uv.index]` for `vllm-cu129` or CUDA 12 wheel sources accordingly.

### 3.2 Delta Weight Transfer & Patch Retaining Strategy
1. **LoRA Activation Patch (`patch_vllm_lora_dedup.py`)**:
   Keep module identity deduplication (`seen_modules: set[int] = set()`) intact to prevent double reset/setting of Gemma4 decoder shared aliases.
2. **Weight Transfer Config & Factory Patch (`patch_vllm_weight_transfer.py`)**:
   Inspect vLLM 0.25.1's `WeightTransferEngineFactory` and `WeightTransferConfig` to determine if `Literal["nccl", "ipc", "delta_snapshot"]` registration requires string replacement or uses a dynamic `register_engine` API.

---

## 4. Verification & Testing Strategy

### Phase 1: Local Code Integrity & Compilation
- `python3 -m py_compile src/server/delta_weight_transfer_engine.py src/server/vllm_sampler.py src/server/scripts/patch_vllm_*.py`
- `export PATH=$PATH:$HOME/.local/bin && make lint && make fmt`
- `make test`

### Phase 2: Container Image Build & Push
- `make build-images push-images IMAGE_TAG=$(cat VERSION 2>/dev/null || echo latest)`

### Phase 3: Kubernetes Cluster E2E Functional Verification
Deploy to Kubernetes cluster and run 3-step delta weight sync verification:
```bash
make cluster-e2e IMAGE_TAG=$(cat VERSION 2>/dev/null || echo latest) \
  E2E_SCENARIO=fft-gsm8k-rl \
  E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=3"
```
Verify step-by-step metric logging and successful delta weight reloads in gateway/sampler pod output.
