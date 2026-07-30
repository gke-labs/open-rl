# Design Doc 009: Phase 2 Admin Analytics — Job Lifecycle, Request Queues & Step Feed Dashboard

**Author:** Open-RL Engineering Team & User  
**Status:** Approved Design (`v1.2.0`)  
**Target Component:** Multi-Tenant Gateway, RequestStore, Admin Dashboard (`tab-jobs`, `tab-job-details`)  
**Target Endpoints:** `GET /api/v1/admin/jobs`, `GET /api/v1/admin/jobs/{job_id}/requests`  

---

## 1. Executive Summary

Phase 2 expands the Open-RL Admin Dashboard beyond hardware-level GPU usage to provide **Job Orchestration, Request Lifecycle, and Step-Level Workflow Visibility**.

While Phase 1 (Design Doc 007) provided physical GPU time-slicing metrics, Phase 2 provides full operational transparency into software workloads:
1. **Macro Level:** Directory of all active, queued, and recently completed training/sampling jobs across the cluster.
2. **Micro Level (Per-Job Inspection):** Deep inspection of a job structured as a **3-Tier Step-Centric Reverse Chronological Timeline** (Step $N \rightarrow$ Step $0$), grouping raw lower-level API requests into **Rollout** and **Training** phase containers per step with color-coded request status blocks.

---

## 2. Hierarchical Domain Object Model

Every Open-RL workload is structured into a 3-tier operational hierarchy:

```text
Level 1: JOB (e.g. `fft-gsm8k-rl`, Tenant: `tenant-math`, Model: `Qwen/Qwen2.5-0.5B-Instruct`)
  │  • Status: Active / Completed / Failed / Terminated
  │  • Global Metrics: Steps Completed, Elapsed Runtime, Total Tokens, Total TFLOPs
  │
  └── Level 2: STEP (Step #1, Step #2, ..., Step #N)
        │  • Status: IN_PROGRESS | COMPLETED | FAILED
        │  • Step Metrics: Total Step Duration (s), Total Tokens, Accuracy, Loss
        │
        ├── Level 3A: ROLLOUT PHASE (Sampling Stage)
        │     │  • Phase Status: PENDING | PROCESSING | COMPLETED
        │     │  • Worker Role: "sampler" (vLLM)
        │     │  • Requests:
        │     │      1. sample (Rollout Answer Generation)
        │     │      2. sample (Reference Policy Logprobs Evaluation)
        │     │
        └── Level 3B: TRAINING PHASE (Training Stage)
              │  • Phase Status: PENDING | PROCESSING | COMPLETED
              │  • Worker Role: "trainer" (PyTorch)
              │  • Requests:
              │      1. forward_backward (PyTorch Loss & Backprop)
              │      2. optim_step (AdamW Optimizer Update & Delta Sync)
              │      3. save_weights_for_sampler (Sparse Weight Export)
              │      4. save_weights_and_get_sampling_client_async (SamplingClient Creation)
```

---

## 3. Redis Data Schemas & Telemetry Extensions

To prevent performance degradation on primary execution queues (`open_rl:queue:<model_id>`), Phase 2 uses dedicated Redis structures:

### 3.1 Extended Job Metadata (`open_rl:model_meta:<model_id>`)
We extend the existing `TrainingModelMetadata` class (`src/server/model_metadata.py`) stored in Redis:

```json
{
  "model_id": "fft-gsm8k-rl",
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "training_kind": "fft",
  "created_at": 1785104000.120,
  "updated_at": 1785104610.500,
  "completed_at": null,
  "status": "active",
  "total_steps_completed": 18,
  "max_steps": 30,
  "tenant_id": "tenant-math",
  "weight_sync_config": {
    "strategy": "delta",
    "delta_format": "vllm_fused"
  },
  "full_config": {
    "learning_rate": 1e-4,
    "batch_size": 24,
    "seed": 42
  }
}
```

### 3.2 Per-Request Lifecycle Hash (`open_rl:job_requests:<model_id>`)
A dedicated Redis Hash (`HSET`) per job tracking request lifecycle states:

```json
{
  "request_id": "req_9921_fwd_bwd",
  "model_id": "fft-gsm8k-rl",
  "op": "forward_backward",
  "role": "trainer",
  "status": "processing",
  "session_id": "sess_8820_eval",
  "worker_pod": "open-rl-trainer-0.default.pod",
  "created_at": 1785104610.150,
  "started_at": 1785104610.165,
  "completed_at": null,
  "duration_ms": null,
  "token_count": 12288
}
```

#### Field Definitions:
- `request_id` *(string)*: Unique HTTP request identifier.
- `model_id` *(string)*: Job / Model identifier.
- `op` *(string)*: Operation type (`forward_backward`, `optim_step`, `sample`, `save_weights_for_sampler`).
- `role` *(string)*: Worker role executing the request (`"trainer"` or `"sampler"`).
- `status` *(string)*: Lifecycle state (`"pending"`, `"processing"`, `"done"`, `"failed"`).
- `session_id` *(string, optional)*: Sampling session ID or client trace ID.
- `worker_pod` *(string, optional)*: Pod name or worker ID executing the request.
- `created_at` *(float)*: Epoch timestamp when Gateway received the HTTP request.
- `started_at` *(float, optional)*: Epoch timestamp when Worker acquired GPU and started execution.
- `completed_at` *(float, optional)*: Epoch timestamp when request finished.
- `duration_ms` *(int, optional)*: Total execution time in milliseconds (`(completed_at - started_at) * 1000`).
- `token_count` *(int, optional)*: Tokens processed or generated.

---

## 4. Administrative Endpoints

### 4.1 `GET /api/v1/admin/jobs`
Returns all active and recently completed jobs.

**Response Schema (`200 OK`):**
```json
{
  "active_jobs": [
    {
      "model_id": "fft-gsm8k-rl",
      "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
      "tenant_id": "tenant-math",
      "status": "active",
      "active_phase": "training",
      "current_step": 18,
      "max_steps": 30,
      "pending_trainer_reqs": 1,
      "pending_sampler_reqs": 0,
      "created_at": 1785104000.120,
      "updated_at": 1785104610.500
    }
  ],
  "completed_jobs": []
}
```

### 4.2 `GET /api/v1/admin/jobs/{job_id}/requests`
Returns detailed request status and step feed for `job_id`.

**Response Schema (`200 OK`):**
```json
{
  "job_id": "fft-gsm8k-rl",
  "currently_executing": {
    "request_id": "req_9921_fwd_bwd",
    "op": "forward_backward",
    "role": "trainer",
    "worker_pod": "open-rl-trainer-0",
    "started_at": 1785104610.165,
    "elapsed_sec": 34.2
  },
  "pending_queues": {
    "trainer": [
      { "request_id": "req_9922_optim", "op": "optim_step", "waiting_sec": 34.2 }
    ],
    "sampler": []
  },
  "step_feed": [
    {
      "step_index": 18,
      "duration_sec": 52.8,
      "sampling_stage": {
        "duration_sec": 5.97,
        "requests": [
          { "op": "sample", "duration_ms": 4820, "description": "Rollout Answer Generation" },
          { "op": "sample", "duration_ms": 1150, "description": "Reference Policy Logprobs" }
        ]
      },
      "training_stage": {
        "duration_sec": 46.83,
        "requests": [
          { "op": "forward_backward", "duration_ms": 40270, "description": "Loss & Backprop" },
          { "op": "optim_step", "duration_ms": 5410, "description": "Optimizer Step & Weight Delta Sync" },
          { "op": "save_weights_for_sampler", "duration_ms": 1150, "description": "Export Sparse Delta Weights" }
        ]
      }
    }
  ]
}
```

---

## 5. User Interface Specification (`GET /admin/dashboard/`)

### 5.1 Active & Completed Job Directory Table (`tab-jobs` / `#/jobs`)
- Clean directory table listing all active and completed jobs with Job / Model ID, Base Model, Tenant, Status (`🟢 ACTIVE`, `🏁 COMPLETED`), Current Step Progress (`Step 18 / 30`), Pending Queue Counts (`0 Trainer / 56 Sampler`), and an **Inspect Requests** button.

### 5.2 Dedicated Job Visualization View (`tab-job-details` / `#/job/<job_id>`)
Clicking any active job opens a **Dedicated Full-Page Job Details View** featuring:
1. **Header & Navigation Bar**:
   - `← Back to Jobs` button to return to the cluster directory seamlessly without full reloads.
   - Dynamic Job Title and metadata summary badges (Base Model, Training Kind, Tenant ID, Step Progress).
2. **Currently Executing Request Card**:
   - Pulsing card highlighting the operation currently holding the GPU lock, including request ID, role (`[TRAINER]` vs `[SAMPLER]`), worker pod ID, and live running time (`elapsed_sec`).
3. **Compact Visual Block Status Legend Bar**:
   - Sleek horizontal status legend displaying live counters:
     `[🟢 Executing: X]` `[🟧 Trainer Pending: Y]` `[🟦 Sampler Pending: Z]` `[🩵 Sampler Done: A]` `[🟧 Trainer Done: B]`
4. **Option 2: Step-Based Visual Block Matrix Grid**:
   - Every request is represented as an interactive $14\text{px} \times 14\text{px}$ square block (`border-radius: 3px`).
   - Blocks feature rich state-aware color coding:
     - 🟢 **Executing (On GPU)**: `#22c55e` (Pulsing bright green with glowing `@keyframes pulse-block` animation).
     - 🟧 **Pending Trainer**: `#f97316` (Amber/Orange block waiting for PyTorch trainer slice).
     - 🟦 **Pending Sampler**: `#0284c7` (Sky Blue block waiting for vLLM sampler batch).
     - 🩵 **Completed Sampler**: `#7dd3fc` (Light Sky Blue).
     - 🟧 **Completed Trainer**: `#ffedd5` (Light Amber).
     - 🟥 **Failed**: `#ef4444` (Solid Red).
   - **Interactive Hover Tooltips**: Hovering over any block reveals `Req ID`, `Operation`, `State`, and execution duration.
   - **Phase Containers**:
     - **`🎯 Rollout Phase (vLLM Sampler)`**: Visual grid of all prompt generation/sampling requests (e.g. `56 Prompts / Rollouts`).
     - **`🏋️ Training Phase (PyTorch Trainer)`**: Visual grid of training operations (`forward_backward`, `optim_step`, `save_weights_for_sampler`) labeled cleanly as **`X Training Requests`**.

---

## 6. Implementation Plan & Status (`v1.1.0 Completed & Verified`)

1. **Model Metadata Extension (`src/server/model_metadata.py`)**:
   - Extended `TrainingModelMetadata` with `status`, `updated_at`, `completed_at`, `total_steps_completed`, `max_steps`, `tenant_id`.
2. **Store Layer Methods (`src/server/store.py`)**:
   - Implemented `record_job_request_event`, `get_job_requests`, `list_jobs_metadata` in `RequestStore`, `InMemoryStore`, and `RedisStore`.
3. **Gateway Telemetry Hooks (`src/server/gateway.py`)**:
   - Recorded `status: "pending"` in `open_rl:job_requests:<model_id>` upon request arrival in `enqueue()` and `asample()`.
4. **Worker Telemetry Hooks (`src/server/training_requests_processor.py` & `vllm_sampler.py`)**:
   - Updated `status: "processing"` upon GPU lock acquire, and `status: "done"` / `"failed"` upon completion.
5. **Gateway REST Endpoints (`src/server/gateway.py`)**:
   - Implemented `GET /api/v1/admin/jobs` and `GET /api/v1/admin/jobs/{job_id}/requests`.
6. **Dedicated Dashboard UI (`src/server/admin_dashboard_template.py`)**:
   - Implemented Dedicated Job Page, Option 2 Visual Block Matrix Grid, Visual Block Legend Header, and live JS polling.

