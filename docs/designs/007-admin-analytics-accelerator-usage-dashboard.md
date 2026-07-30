# Design Doc 007: Phase 1 Admin Analytics & Accelerator Usage Dashboard

**Author:** Open-RL Engineering Team  
**Status:** Approved Design (`v1.2.0`)  
**Target Component:** Multi-Tenant Gateway, Accelerator Time-Slicer (`accel_timeslicer`), Admin Dashboard  
**Target Endpoints:** `GET /api/v1/admin/accel_usage`, `GET /admin/dashboard/`  

---

## 1. Executive Summary

As OpenRL scales to support multi-tenant Reinforcement Learning (RL) workloads, maximizing GPU utilization and providing transparent hardware telemetry is essential for both **Platform Engineers** and **AI Researchers**.

This design document specifies **Phase 1 of the Admin Analytics & Dashboard initiative**, focusing strictly on **Accelerator Usage**. The goal is to provide real-time, low-overhead visibility into physical GPU time-slicing, workload volume (token throughput), and hardware allocation without impacting training performance or duplicating large request/response payloads in storage.

### Key Highlights
1. **DRA ResourceClaim Hardware Anchoring**: Anchoring hardware telemetry directly to Dynamic Resource Allocation (DRA) `ResourceClaim` objects (`resource_claim_id`), representing the true Kubernetes hardware allocation handle shared across co-located trainer and sampler pods.
2. **Work Volume Timeline Graph & Custom Time Range Filter**: Rendering a 2D timeline bar chart where the **X-Axis is Time**, the **Y-Axis is Work Volume** (tokens generated or trained per slice), and the **Bar Width is Slice Duration**, with support for quick presets (`1m`, `5m`, `15m`, `1h`, `All History`) and **Custom Time Ranges (`start_ts` to `end_ts`)**.
3. **Lean Telemetry (Zero Payload Duplication)**: Storing lightweight telemetry metadata (~120 bytes per slice event) in Redis while strictly excluding heavy prompt token arrays, generated text samples, or logprobs.
4. **Phase 1 Administrative Routes**: Exposing the internal infrastructure analytics via `GET /api/v1/admin/accel_usage` and serving the web interface at `http://localhost:8000/admin/dashboard/`.

---

## 2. Motivation & Problem Statement

### 2.1 Current Telemetry Gaps
1. **Lack of GPU Time-Slicing Visibility**: The Accelerator Time-Slicer (`accel_timeslicer`) dynamically coordinates process access to CUDA devices across co-located sampler and trainer workers. However, there is no centralized dashboard showing when a GPU slice was acquired or released by a specific tenant job.
2. **Obscured Workload Throughput**: Standard GPU metrics (e.g. raw `nvidia-smi` utilization) show *that* a GPU is busy, but fail to show *how much useful RL compute or token generation* was accomplished during a specific time-slice.
3. **Payload Bloat Risk**: Naively logging complete request payloads (prompts, completion text, logprob arrays) to monitoring stores duplicates megabytes of JSON per step, risking Redis memory exhaustion.

### 2.2 Phase 1 Objectives
- **Build Phase 1 First**: Focus exclusively on **Accelerator Usage** to deliver immediate, high-impact hardware visualization for demoing multi-tenant GPU sharing before expanding to request queue state in Phase 2.
- **Hardware-Anchored Telemetry**: Group telemetry by DRA `resource_claim_id` and K8s `node_name`.
- **Zero Performance Impact**: Ensure telemetry recording adds $< 1 \text{ ms}$ overhead to the `time_slicer.acquire()` lifecycle.

---

## 3. Technical Architecture & Data Schemas

### 3.1 DRA ResourceClaim Hardware Anchoring
In Kubernetes clusters utilizing DRA (e.g. `k8s/deploy/distributed-fft-timeslice/`), GPUs are allocated via K8s `ResourceClaim` objects. Multiple pods (Tenant A's sampler, Tenant B's trainer, etc.) reference the same shared claim.

Telemetry events are anchored by `resource_claim_id`, ensuring all time-slicer acquire/release events map to the exact physical GPU allocation regardless of pod restart or worker process ID.

### 3.2 Model-Size Aware Volume Metrics (Tokens vs Compute FLOPs)
Measuring work volume purely in raw tokens creates a mismatch when comparing workloads across different model parameter scales (e.g. `Qwen2.5-0.5B` vs `Qwen3-8B` vs `Llama-3.1-70B`). Processing 1,000 tokens on a 70B model requires ~140x more floating-point operations than on a 0.5B model.

To solve this, telemetry captures both **Raw Tokens** and **Model-Adjusted Compute FLOPs**:
- **Sampler Pass Compute (vLLM)**: $\text{FLOPs} \approx 2 \times N_{\text{params}} \times N_{\text{tokens}}$
- **Trainer Pass Compute (PyTorch)**: $\text{FLOPs} \approx 6 \times N_{\text{params}} \times N_{\text{tokens}}$
- **Compute Density (TFLOPS/sec)**: $\text{TFLOPS/s} = \frac{\text{FLOPs} / 10^{12}}{\text{duration\_sec}}$

### 3.3 Redis Telemetry Schema (`accel_usage_history`)
* **Redis Key**: `open_rl:accel_usage_history:<resource_claim_id>`
* **Data Structure**: Capped Redis List (`LPUSH` + `LTRIM` to keep the last 5000 slice events per claim).

**Event JSON Payload Schema:**
```json
{
  "event_id": "accel-evt-991",
  "resource_claim_id": "open-rl-shared-gpu-claim-01",
  "node_name": "gke-gpu-node-01",
  "gpu_index": 0,
  "job_id": "math-rl-job-01",
  "tenant_id": "tenant-math",
  "model_name": "Qwen/Qwen3-8B",
  "num_params_billions": 8.0,
  "worker_role": "trainer",
  "acquire_time": 1774417700.150,
  "release_time": 1774417701.950,
  "duration_ms": 1800,
  "tokens_processed": 1000,
  "work_volume_tflops": 48.0,
  "tflops_per_sec": 26.66
}
```

#### Field Definitions:
- `resource_claim_id` *(string, required)*: The K8s DRA ResourceClaim name or local GPU identifier (`"open-rl-shared-gpu-claim-01"` or `"gpu-0"`).
- `node_name` *(string, required)*: The physical K8s Node name (`spec.nodeName` via Downward API) or `"localhost"`.
- `gpu_index` *(int, required)*: Device index on host (e.g. `0`).
- `job_id` *(string, required)*: Identifier of the active model/job holding the slice lock.
- `tenant_id` *(string, required)*: Tenant namespace (e.g. `"tenant-math"`).
- `model_name` *(string, required)*: Model checkpoint string (e.g. `"Qwen/Qwen3-8B"`).
- `num_params_billions` *(float, required)*: Model parameter count in billions (e.g. `8.0`).
- `worker_role` *(string, required)*: `"trainer"` (PyTorch compute) or `"sampler"` (vLLM inference).
- `acquire_time` *(float, required)*: Epoch timestamp (seconds) when `time_slicer.acquire()` succeeded.
- `release_time` *(float, required)*: Epoch timestamp (seconds) when `time_slicer.acquire()` block exited.
- `duration_ms` *(int, required)*: Total slice duration in milliseconds (`(release_time - acquire_time) * 1000`).
- `tokens_processed` *(int, required)*: Raw tokens generated (for samplers) or batch tokens trained (for trainers).
- `work_volume_tflops` *(float, required)*: Total TFLOPs computed during slice (`6 * N * T / 1e12` for trainer, `2 * N * T / 1e12` for sampler).
- `tflops_per_sec` *(float, required)*: Hardware compute density achieved (`work_volume_tflops / (duration_ms / 1000)`).

---

## 4. Administrative Endpoints & UI Specification

### 4.1 API Endpoint: `GET /api/v1/admin/accel_usage`
Returns a JSON snapshot of active DRA claims and their recent slice history.

**Query Parameters:**
- `resource_claim_id` *(string, optional)*: Filter history to a specific DRA claim.
- `window_sec` *(float, optional, default: 300.0)*: Rolling duty cycle calculation window in seconds (`60` for 1m, `300` for 5m, `900` for 15m, `3600` for 1h, `0` for All History).

**Response Schema (`200 OK`):**
```json
{
  "timestamp": 1774417702.500,
  "claims": {
    "open-rl-shared-gpu-claim-01": {
      "resource_claim_id": "open-rl-shared-gpu-claim-01",
      "node_name": "gke-gpu-node-01",
      "gpu_index": 0,
      "hardware_name": "NVIDIA L4 (24GB)",
      "active_slice": {
        "job_id": "math-rl-job-01",
        "tenant_id": "tenant-math",
        "worker_role": "trainer",
        "acquire_time": 1774417700.150
      },
      "history": [
        {
          "event_id": "accel-evt-991",
          "job_id": "math-rl-job-01",
          "tenant_id": "tenant-math",
          "worker_role": "trainer",
          "acquire_time": 1774417700.150,
          "release_time": 1774417701.950,
          "duration_ms": 1800,
          "work_volume": 3800,
          "throughput_tps": 2111.1
        }
      ]
    }
  }
}
```

### 4.2 Per-Accelerator Hardware Metrics & Consumption Breakdown (Phase 1 Scope)
For Phase 1 (demo scope), hardware metrics and tenant consumption pie charts are calculated and rendered **per individual accelerator** (DRA ResourceClaim card). Multi-node cluster-wide aggregation is deferred to a future phase.

Each **DRA ResourceClaim Card** contains:
1. **Header & Badges**:
   - `resource_claim_id`, `node_name`, and hardware device model.
   - **Idle Time %**: $\text{Idle \%} = \frac{\text{Total Idle ms in Window}}{\text{Total Window ms}} \times 100\%$
   - **Duty Cycle %**: $\text{Duty Cycle \%} = 100\% - \text{Idle \%}$
2. **Accelerator Consumption Pie / Donut Chart**:
   - Displays percentage breakdown of GPU time by Tenant/Job for that specific accelerator, with **Idle time explicitly represented as a slice**:
     - `Tenant A (Math RL)`: 45.2%
     - `Tenant B (SQL RL)`: 40.6%
     - `Idle Time`: 14.2%
3. **Work Volume Timeline Chart**:
   - 2D Gantt chart (X-Axis: Time, Y-Axis: Tokens or Compute TFLOPs).

### 4.3 Web Admin Dashboard Layout (`GET /admin/dashboard/`)

```text
====================================================================================================
 🎛️ OPEN-RL ADMIN DASHBOARD | ACCELERATOR USAGE                     [ Window: Last 1 Hour ]
====================================================================================================

 📟 DRA RESOURCECLAIM CARD #1: open-rl-shared-gpu-claim-01 (Node: gke-gpu-node-01 | NVIDIA L4)
 ───────────────────────────────────────────────────────────────────────────────────────────────────
  📊 Accelerator Stats:   🔥 85.8% Duty Cycle   │   🟩 14.2% Idle   │   ⚡ 142.8 TFLOPs Delivered

  🍰 Tenant Consumption Breakdown:
   [████████████████ 45.2% Tenant A (Math RL) | ██████████████ 40.6% Tenant B (SQL RL) | ░░░ 14.2% IDLE]

  📈 Work Volume Timeline:   Display Metric: (●) Compute FLOPs (TFLOPs)   ( ) Raw Tokens
  Y-Axis: Compute Density (TFLOPs)
   50 TFLOPs ┼                                               ┌─────────────────────┐
             │                                               │ 🟧 Qwen-8B Trainer  │
   10 TFLOPs ┼                     ┌──────────────────┐      │ (48.0 TFLOPs)       │
             │ ┌───────────────────┤ 🟦 Gemma-2B      │      │                     │
    1 TFLOP  ┼ │ 🟦 Qwen-0.5B Samp  │ Sampler          │      │                     │
             │ │ (1.5 TFLOPs)      │ (8.0 TFLOPs)     │      │                     │
           0 ┴─┴───────────────────┴──────────────────┴──────┴─────────────────────┴─────────────► X-Axis (Time)
               10:00:01            10:00:04.2                10:00:07.5

====================================================================================================
### 4.4 Light Theme UI & Aesthetic Specifications
To ensure the dashboard is easy on the eyes during prolonged monitoring without harsh contrast or glaring neon colors, the interface uses a warm alabaster light palette:

- **Page Background**: Warm Alabaster (`#F8FAFC`)
- **Card Containers**: Clean White (`#FFFFFF`) with soft borders (`#E2E8F0`) and subtle shadows (`0 1px 3px rgba(0,0,0,0.04)`).
- **Primary Text**: Soft Slate (`#334155`)
- **Secondary Labels**: Muted Gray (`#64748B`)
- **Soft Pastel Workload Roles**:
  - 🟦 **Sampler (vLLM)**: **Soft Sky Blue** (Background: `#E0F2FE`, Border: `#38BDF8`, Text: `#0369A1`)
  - 🟧 **Trainer (PyTorch)**: **Soft Peach / Coral** (Background: `#FFEDD5`, Border: `#FB923C`, Text: `#C2410C`)
  - ░ **Idle Time**: **Subtle Slate Haze** (Background: `#F1F5F9`, Border: `#CBD5E1`)

### 4.5 Real-Time Streaming & Client Technology Stack
- **Real-Time Updates**: Supported via Server-Sent Events (SSE) at `GET /api/v1/admin/accel_usage/stream` with a 500ms REST polling fallback (`GET /api/v1/admin/accel_usage`). As slice lock events occur on worker nodes, new timeline bars animate smoothly into the chart.
- **Client Architecture**: Zero-dependency, single-file native HTML5 + ES6 Modern Vanilla JS served directly by FastAPI via `HTMLResponse`. Includes ApexCharts (via CDN) for 60fps timeline and donut chart rendering without NPM or build steps.

---

## 5. Implementation Strategy & Code Touchpoints

### 5.1 Worker Telemetry Hook
Inside `training_requests_processor.py` (and `vllm_sampler.py`), wrap `time_slicer.acquire()` blocks to capture timing and token counts:

```python
t_start = time.time()
async with self.time_slicer.acquire(self.workload):
    # Execute batch compute...
    token_count = sum(len(req.get("tokens", [])) for req in gpu_reqs)
    # Execute handle_request...

t_end = time.time()
dur_ms = int((t_end - t_start) * 1000)

event = {
    "event_id": f"evt-{uuid.uuid4().hex[:8]}",
    "resource_claim_id": os.getenv("OPEN_RL_DRA_CLAIM_ID", "gpu-claim-0"),
    "node_name": os.getenv("NODE_NAME", "localhost"),
    "gpu_index": int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0] if os.getenv("CUDA_VISIBLE_DEVICES") else 0),
    "job_id": self.model_id,
    "tenant_id": getattr(self, "tenant_id", "default"),
    "worker_role": "trainer",
    "acquire_time": t_start,
    "release_time": t_end,
    "duration_ms": dur_ms,
    "work_volume": token_count,
    "throughput_tps": round(token_count / (dur_ms / 1000.0), 1) if dur_ms > 0 else 0.0,
}

await self.store.record_accel_usage_event(event["resource_claim_id"], event)
```

### 5.2 Store Layer Updates (`src/server/store.py`)
Add methods `record_accel_usage_event` and `get_accel_usage_history` to `RequestStore`, `InMemoryStore`, and `RedisStore`.

### 5.3 Gateway Route Additions (`src/server/gateway.py`)
- Endpoint `@app.get("/api/v1/admin/accel_usage")`
- Route `@app.get("/admin/dashboard/", response_class=HTMLResponse)` serving the web view.

---

## 6. Verification & Testability

1. **Unit Test Suite**: Add `tests/test_admin_accel_usage.py` testing telemetry recording, event list capping, and API response formatting.
2. **Validation**: Execute `make test` to guarantee zero regressions.
3. **Demo Execution**: Run `make cluster-e2e E2E_SCENARIO=fft-gsm8k-rl-x2` and open `http://localhost:8000/admin/dashboard/` to observe live multi-tenant GPU time-slicing.
