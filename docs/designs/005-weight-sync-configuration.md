# Design Doc 005: Unified Weight Synchronization Configuration & Header Extraction Architecture

**Author:** Open-RL Engineering Team  
**Status:** Proposed Design (`v0.5.0`)  
**Target Component:** Gateway Server, Worker Managers (Local & K8s), Trainer Engine, Sampler Engine, Client HTTP API  

---

## 1. Executive Summary

In distributed Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) workflows, policy weights are continuously synchronized between the PyTorch Training Worker and the vLLM Sampling Worker. Over recent iterations, various weight sync flags (`OPEN_RL_WEIGHT_SYNC_STRATEGY`, `OPEN_RL_IN_PLACE_DELTA`, `OPEN_RL_EMIT_VLLM_FUSED_DELTAS`, etc.) were introduced across different modules, leading to configuration overlap and implicit defaults.

This document specifies a **unified, hierarchical Weight Synchronization Configuration Architecture** (`WeightSyncConfig`). 

Key highlights of this design:
1. **Hierarchical Schema**: A clean separation between the top-level sync strategy (`strategy`) and delta-specific execution sub-knobs (`format`, `apply_method`, `enable_prefetching`).
2. **Client-Side HTTP Header Control**: Full control exposed to clients via `x-open-rl-weight-sync-*` HTTP headers.
3. **Gateway Extraction & Defaulting**: Robust HTTP header parsing and automatic defaulting in the Gateway server, storing `weight_sync_cfg` in canonical Redis model metadata.
4. **Explicit Worker Pod Template Injection**: Worker Managers (`worker_manager.py` and `k8s_worker_manager.py`) dynamically read model metadata and set explicit environment variables on Trainer and Sampler pod templates.
5. **Comprehensive Unit Test Suite**: Dedicated test coverage verifying HTTP header extraction, case-insensitivity, invalid fallbacks, and Redis metadata persistence.

---

## 2. Motivation & Problem Statement

### 2.1 Current Friction
Currently, configuring weight synchronization suffers from three friction points:
1. **Scattered Environment Variables**: Flags like `OPEN_RL_IN_PLACE_DELTA` and `OPEN_RL_EMIT_VLLM_FUSED_DELTAS` are parsed in separate files without a centralized configuration dataclass.
2. **Dual-Location Defaulting (Anti-Pattern)**: Default values (such as `"vllm_fused"` or `"patch_in_place"`) were evaluated in multiple locations (both Gateway and downstream Worker processes), risking default drift bugs.
3. **Overloaded Terminology**: Terms like `sync_cfg` are overloaded across multiple system boundaries. Replacing them with `weight_sync_cfg` establishes explicit domain boundaries.

### 2.2 Objective
Establish a single, canonical hierarchy that enforces **Single-Location Defaulting** at the Gateway entry point. The Gateway extracts HTTP headers, applies default values ONCE, and persists a fully-resolved `weight_sync_cfg` into Redis model metadata. Downstream Worker Managers and engines consume these fully-resolved values directly without duplicating default fallbacks.

---

## 3. Hierarchical Configuration Schema (`WeightSyncConfig`)

The configuration is structured as a hierarchical tree:

```
                            [ WEIGHT SYNC STRATEGY ]
                                       |
                   +-------------------+-------------------+
                   |                                       |
               1. FULL                                 2. DELTA
      (Full Model Checkpoint Reload)              (Sparse Delta Sync)
                                                           |
               +-------------------------------------------+-------------------------------------------+
               |                                           |                                           |
       A. Layer Format                            B. Apply Method                             C. Prefetching
   (`vllm_fused` | `native`)                  (`patch_in_place` | `full_replace`)              (`True` | `False`)
```

### 3.1 Python Dataclass Definition

```python
from dataclasses import dataclass


@dataclass
class WeightSyncConfig:
  strategy: str = "delta"  # Options: "delta" | "full"
  delta_format: str = "vllm_fused"  # Options: "vllm_fused" | "native"
  delta_apply_method: str = "patch_in_place"  # Options: "patch_in_place" | "full_replace"
  enable_prefetching: bool = True  # Options: True | False
```

### 3.2 Field Specification & Defaulting Rules

| Field | Type | Allowed Values | Default (Gateway Entry Point Only) | Description / Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **`strategy`** | `str` | `"delta"`, `"full"` | `"delta"` | Top-level weight transfer strategy. |
| **`delta_format`** | `str` | `"vllm_fused"`, `"native"` | `"vllm_fused"` | Delta layer encoding format. `"vllm_fused"` remaps split HF attention/MLP layers (`q/k/v_proj`) to vLLM fused representations (`qkv_proj`). Active when `strategy == "delta"`. |
| **`delta_apply_method`** | `str` | `"patch_in_place"`, `"full_replace"` | `"patch_in_place"` | Delta VRAM mutation technique. `"patch_in_place"` mutates GPU memory directly via PCIe DMA (`index_copy_`). `"full_replace"` merges into CPU state_dict. Active when `strategy == "delta"`. |
| **`enable_prefetching`** | `bool` | `True`, `False` | `True` | Background DRAM pre-staging. Preloads pinned DRAM buffers via Redis Pub/Sub signals while Sampler is in sleep mode. |

---

## 4. Client HTTP Header Interface

Clients (SDKs, CLI tools, or curl scripts) pass optional HTTP headers using the `x-open-rl-weight-sync-` prefix during fine-tuning job creation:

```http
POST /v1/fine_tuning/jobs HTTP/1.1
Host: open-rl-gateway:8000
Content-Type: application/json
x-open-rl-weight-sync-strategy: delta
x-open-rl-weight-sync-delta-format: vllm_fused
x-open-rl-weight-sync-delta-apply-method: patch_in_place
x-open-rl-weight-sync-enable-prefetching: true
```

---

## 5. Gateway Header Extraction & Metadata Store

Inside `src/server/gateway.py`, the Gateway server parses incoming HTTP headers, applies defaults for missing fields **once**, and persists `weight_sync_cfg` to Redis.

```python
def extract_weight_sync_config(headers: Any = None) -> WeightSyncConfig:
  """Extract and normalize WeightSyncConfig from HTTP headers with single-location defaults."""
  if not headers:
    return WeightSyncConfig()

  get_header = headers.get if hasattr(headers, "get") else (lambda k, default=None: default)

  strategy = (get_header("x-open-rl-weight-sync-strategy") or "delta").lower()
  if strategy not in ("delta", "full"):
    strategy = "delta"

  delta_fmt = (get_header("x-open-rl-weight-sync-delta-format") or get_header("x-open-rl-weight-sync-format") or "vllm_fused").lower()
  if delta_fmt not in ("vllm_fused", "native"):
    delta_fmt = "vllm_fused"

  delta_apply_method = (
    get_header("x-open-rl-weight-sync-delta-apply-method") or get_header("x-open-rl-weight-sync-apply-method") or "patch_in_place"
  ).lower()
  if delta_apply_method not in ("patch_in_place", "full_replace"):
    delta_apply_method = "patch_in_place"

  raw_prefetch = get_header("x-open-rl-weight-sync-enable-prefetching")
  if raw_prefetch is not None:
    enable_prefetching = str(raw_prefetch).lower() in ("true", "1", "yes")
  else:
    enable_prefetching = True

  return WeightSyncConfig(
    strategy=strategy,
    delta_format=delta_fmt,
    delta_apply_method=delta_apply_method,
    enable_prefetching=enable_prefetching,
  )
```

### 5.1 Canonical Redis Model Metadata Storage

The extracted configuration is stored inside `TrainingModelMetadata` under Redis key `open_rl:model_meta:<model_id>`:

```json
{
  "base_model": "Qwen/Qwen3-8B",
  "created_at": 1753401200.0,
  "training_kind": "fft",
  "weight_sync_strategy": "delta",
  "weight_sync_config": {
    "strategy": "delta",
    "delta_format": "vllm_fused",
    "delta_apply_method": "patch_in_place",
    "enable_prefetching": true
  }
}
```

---

## 6. Worker Managers & Pod Template Environment Injection

Worker Managers (`src/server/worker_manager.py` for local dev and `src/server/k8s_worker_manager.py` for Kubernetes) retrieve `weight_sync_cfg` from the store and set explicit environment variables on worker process / Pod templates.

**Single Source of Truth Principle**: Worker Managers do not repeat default logic. They read the fully-resolved `weight_sync_config` dictionary from Redis and pass the exact values down:

```python
# Centralized Model Metadata Dataclasses in src/server/model_metadata.py:
@dataclass
class TrainingModelMetadata:
  base_model: str | None
  created_at: float
  training_kind: str
  weight_sync_strategy: str | None = None
  weight_sync_config: dict[str, Any] | None = None
  full_config: dict[str, Any] | None = None
  lora_config: dict[str, Any] | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "TrainingModelMetadata":
    return cls(...)


# Metadata retrieval in worker_manager.py:
def _fetch_metadata_from_store(model_id: str) -> TrainingModelMetadata | None:
  val = get_store().get_value_sync(f"open_rl:model_meta:{model_id}")
  if val:
    meta_dict = json.loads(val) if isinstance(val, str) else val
    if isinstance(meta_dict, dict):
      return TrainingModelMetadata.from_dict(meta_dict)
  return None


# Pod Template Environment Setup in k8s_worker_manager.py:
meta = _fetch_metadata_from_store(model_id)

# Fully-resolved by Gateway — no duplicate fallback defaults needed:
if meta and meta.weight_sync_config:
  cfg = meta.weight_sync_config
  set_env(container, "OPEN_RL_WEIGHT_SYNC_STRATEGY", cfg["strategy"])
  set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT", cfg["delta_format"])
  set_env(container, "OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD", cfg["delta_apply_method"])
  set_env(container, "OPEN_RL_WEIGHT_SYNC_ENABLE_PREFETCHING", str(cfg["enable_prefetching"]).lower())
```

---

## 7. Engine Environment Consumption & Cluster Execution Integration

Inside worker processes, engines reconstruct the `WeightSyncConfig` dataclass instance directly from environment variables via `WeightSyncConfig.from_env()`:

1. **Trainer Engine (`src/training/fft_trainer_worker.py`)**:
   - Reconstructs `self.weight_sync_cfg = WeightSyncConfig.from_env()`.
   - Accesses `self.weight_sync_cfg.strategy` and `self.weight_sync_cfg.delta_format`.
   - Executes fused layer remapping (`qkv_proj`, `gate_up_proj`) when `delta_format == "vllm_fused"`.

2. **Sampler Engine (`src/server/delta_weight_transfer_engine.py` & `vllm_sampler.py`)**:
   - Reconstructs `weight_sync_cfg = WeightSyncConfig.from_env()`.
   - Accesses `weight_sync_cfg.delta_apply_method` and `weight_sync_cfg.enable_prefetching`.
   - Executes direct PCIe DMA `index_copy_` into GPU VRAM when `delta_apply_method == "patch_in_place"`.
   - Listens to Redis Pub/Sub signals for background DRAM pre-staging when `enable_prefetching == True`.

### 7.1 Makefile & `cluster-e2e` Command Line Integration
Clients launching in-cluster benchmarks via `make cluster-e2e` can configure all four weight sync parameters on the command line:

```bash
make cluster-e2e E2E_SCENARIO=fft-gsm8k-rl \
  WEIGHT_SYNC_STRATEGY=delta \
  WEIGHT_SYNC_DELTA_FORMAT=vllm_fused \
  WEIGHT_SYNC_DELTA_APPLY_METHOD=patch_in_place \
  WEIGHT_SYNC_ENABLE_PREFETCHING=true
```

`scripts/run_cluster_e2e.py` parses these overrides via canonical CLI flags (`--weight-sync-strategy`, `--weight-sync-delta-format`, `--weight-sync-delta-apply-method`, `--weight-sync-enable-prefetching`) and dynamically injects `OPEN_RL_WEIGHT_SYNC_*` environment variables onto the `e2e-client` job container, where `patch_tinker_default_headers()` (`examples/common/tinker_utils.py`) automatically propagates them as HTTP headers to the Gateway.

### 7.2 Kubernetes Manifest Standardisation
Base Kubernetes pod templates (`05-worker-pod-template.yaml` and `09-sampler-pod-template.yaml`) are updated to specify canonical `OPEN_RL_WEIGHT_SYNC_*` environment keys:

```yaml
        - name: OPEN_RL_WEIGHT_SYNC_STRATEGY
          value: "delta"
        - name: OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT
          value: "vllm_fused"
        - name: OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD
          value: "patch_in_place"
        - name: OPEN_RL_WEIGHT_SYNC_ENABLE_PREFETCHING
          value: "true"
```

---

## 8. Unit Testing & Verification Plan

A dedicated test module `tests/test_weight_sync_config.py` verifies:

1. **Default Header Parsing**: Verifies that calling `extract_weight_sync_config({})` returns `WeightSyncConfig` with default fields (`"delta"`, `"vllm_fused"`, `"patch_in_place"`, `True`).
2. **Explicit Header Overrides**: Tests passing custom headers (e.g. `x-open-rl-weight-sync-delta-format: native`, `x-open-rl-weight-sync-delta-apply-method: full_replace`) and asserts correct dataclass populating.
3. **Case-Insensitivity & Fallback Resilience**: Tests uppercase strings (`"DELTA"`, `"PATCH_IN_PLACE"`) and invalid enum values, ensuring fallbacks revert to default settings safely.
4. **Metadata Serialization & Pod Environment Propagation**: Tests saving `TrainingModelMetadata` to `In-Memory Store` / `Redis`, calling `k8s_worker_manager`, and verifying environment variables on the resulting pod container.
