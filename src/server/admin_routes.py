import json
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from server.admin_dashboard_template import ADMIN_DASHBOARD_HTML
from server.store import get_store

admin_router = APIRouter()


@admin_router.get("/admin", response_class=HTMLResponse)
@admin_router.get("/admin/dashboard/", response_class=HTMLResponse)
async def get_admin_dashboard():
  """Serve the interactive OpenRL Admin Infrastructure Dashboard."""
  return HTMLResponse(content=ADMIN_DASHBOARD_HTML)


@admin_router.get("/api/v1/admin/accel_usage")
async def get_admin_accel_usage(
  resource_claim_id: str | None = None,
  window_sec: float = 300.0,
  start_ts: float | None = None,
  end_ts: float | None = None,
):
  """Admin Infrastructure API — Accelerator time-slicing telemetry & duty cycle stats."""
  store = get_store()
  raw_history = await store.get_accel_usage_history(resource_claim_id)
  now = time.time()
  claims_data: dict[str, Any] = {}

  for c_id, events in raw_history.items():
    if not events:
      continue

    # Only filter out stale claims when using short rolling windows (window_sec > 0 and no custom range)
    most_recent_time = max(ev.get("release_time", 0) for ev in events)
    if window_sec > 0 and start_ts is None and len(raw_history) > 1 and (now - most_recent_time > max(window_sec, 86400.0)):
      continue

    node_name = events[0].get("node_name", "localhost")
    gpu_index = events[0].get("gpu_index", 0)

    if start_ts is not None and end_ts is not None and end_ts > start_ts:
      filtered_events = [ev for ev in events if ev.get("release_time", now) >= start_ts and ev.get("acquire_time", 0) <= end_ts]
      effective_window_sec = max(end_ts - start_ts, 1.0)
    elif window_sec > 0:
      window_start = now - window_sec
      filtered_events = [ev for ev in events if ev.get("release_time", now) >= window_start]
      effective_window_sec = window_sec
    else:
      filtered_events = events
      if events:
        min_acquire = min(ev.get("acquire_time", now) for ev in events)
        effective_window_sec = max(now - min_acquire, 1.0)
      else:
        effective_window_sec = 300.0

    total_active_ms = 0
    tenant_active_ms: dict[str, float] = {}

    for ev in filtered_events:
      dur_ms = ev.get("duration_ms", 0)
      total_active_ms += dur_ms
      tenant = ev.get("tenant_id", "default")
      tenant_active_ms[tenant] = tenant_active_ms.get(tenant, 0.0) + dur_ms

    total_window_ms = max(effective_window_sec * 1000, 1.0)
    duty_cycle_pct = round(min((total_active_ms / total_window_ms) * 100.0, 100.0), 1)
    idle_pct = round(max(100.0 - duty_cycle_pct, 0.0), 1)

    tenant_breakdown = []
    for tenant, t_ms in tenant_active_ms.items():
      pct = round((t_ms / total_window_ms) * 100.0, 1)
      tenant_breakdown.append(
        {
          "tenant_id": tenant,
          "active_ms": t_ms,
          "percentage": pct,
        }
      )

    if idle_pct > 0:
      idle_ms = max(total_window_ms - total_active_ms, 0.0)
      tenant_breakdown.append(
        {
          "tenant_id": "Idle",
          "active_ms": idle_ms,
          "percentage": idle_pct,
        }
      )

    claims_data[c_id] = {
      "resource_claim_id": c_id,
      "node_name": node_name,
      "gpu_index": gpu_index,
      "duty_cycle_pct": duty_cycle_pct,
      "idle_pct": idle_pct,
      "tenant_breakdown": tenant_breakdown,
      "history": filtered_events,
    }

  return {
    "timestamp": now,
    "claims": claims_data,
  }


@admin_router.get("/api/v1/admin/jobs")
async def admin_list_jobs():
  """Admin API — List all registered models/jobs with active queue counts & execution status."""
  store = get_store()
  raw_jobs = await store.list_jobs_metadata()
  now = time.time()
  active_jobs = []
  completed_jobs = []

  for job in raw_jobs:
    m_id = job.get("model_id")
    if not m_id:
      continue

    reqs_map = await store.get_job_requests(m_id)
    pending_trainer = sum(
      1 for r in reqs_map.values() if r.get("status") == "pending" and r.get("role") == "trainer" and (now - r.get("created_at", now)) <= 300.0
    )
    pending_sampler = sum(
      1 for r in reqs_map.values() if r.get("status") == "pending" and r.get("role") == "sampler" and (now - r.get("created_at", now)) <= 300.0
    )
    completed_steps = sum(1 for r in reqs_map.values() if r.get("op") == "optim_step" and r.get("status") == "done")
    current_step = max(job.get("total_steps_completed", 0), completed_steps)

    last_activity = max(
      [job.get("updated_at", 0.0), job.get("created_at", 0.0)] + [r.get("completed_at", 0.0) for r in reqs_map.values() if r.get("completed_at")]
    )
    status = job.get("status", "active")
    if status == "active" and pending_trainer == 0 and pending_sampler == 0 and (now - last_activity) > 900.0:
      status = "completed"

    weight_sync_cfg = job.get("weight_sync_config") or {}
    weight_sync_strategy = weight_sync_cfg.get("strategy") if isinstance(weight_sync_cfg, dict) else getattr(weight_sync_cfg, "strategy", "full")

    item = {
      "model_id": m_id,
      "base_model": job.get("base_model", "Unknown"),
      "training_kind": job.get("training_kind", "fft"),
      "tenant_id": job.get("tenant_id", "default"),
      "status": status,
      "current_step": current_step,
      "max_steps": job.get("max_steps"),
      "weight_sync_strategy": weight_sync_strategy,
      "latest_mutation_pct": job.get("latest_mutation_pct"),
      "latest_changed_elements": job.get("latest_changed_elements"),
      "latest_total_elements": job.get("latest_total_elements"),
      "pending_trainer_reqs": pending_trainer,
      "pending_sampler_reqs": pending_sampler,
      "created_at": job.get("created_at", now),
      "updated_at": job.get("updated_at", now),
    }

    if status == "active":
      active_jobs.append(item)
    else:
      completed_jobs.append(item)

  return {
    "timestamp": now,
    "active_jobs": active_jobs,
    "completed_jobs": completed_jobs,
  }


@admin_router.get("/api/v1/admin/jobs/{job_id}/requests")
@admin_router.get("/api/v1/admin/job_requests")
async def admin_get_job_requests(job_id: str):
  """Admin API — Retrieve request queue details (pending, processing, done) for a specific job."""
  store = get_store()
  reqs_map = await store.get_job_requests(job_id)
  now = time.time()

  pending_trainer = []
  pending_sampler = []
  currently_executing = None
  recent_completed = []

  for r_id, req in reqs_map.items():
    st = req.get("status", "unknown")
    op = req.get("op", "unknown")
    role = req.get("role", "unknown")
    worker_pod = req.get("worker_pod", "")

    if role == "unknown":
      if "sampler" in worker_pod or op in ("sample", "sample_completed"):
        role = "sampler"
      elif "trainer" in worker_pod or op in ("forward_backward", "optim_step", "save_weights_for_sampler", "create_model"):
        role = "trainer"
    if op == "unknown" and role == "sampler":
      op = "sample"

    if st == "processing":
      t_start = req.get("started_at", now)
      currently_executing = {
        "request_id": r_id,
        "op": op,
        "role": role,
        "worker_pod": req.get("worker_pod", "worker-0"),
        "started_at": t_start,
        "elapsed_sec": round(now - t_start, 1) if t_start else 0.0,
      }
    elif st == "pending":
      t_created = req.get("created_at", now)
      if (now - t_created) <= 300.0:
        item = {
          "request_id": r_id,
          "op": op,
          "role": role,
          "created_at": t_created,
          "waiting_sec": round(now - t_created, 1) if t_created else 0.0,
        }
        if role == "sampler":
          pending_sampler.append(item)
        else:
          pending_trainer.append(item)
    elif st in ("done", "failed"):
      recent_completed.append(
        {
          "request_id": r_id,
          "op": op,
          "role": role,
          "status": st,
          "session_id": req.get("session_id"),
          "created_at": req.get("created_at"),
          "started_at": req.get("started_at"),
          "completed_at": req.get("completed_at", now),
        }
      )

  raw_meta = await store.get_value(f"open_rl:model_meta:{job_id}")
  meta_dict = json.loads(raw_meta) if raw_meta else {}
  mutation_history = meta_dict.get("mutation_history", [])
  weight_sync_cfg = meta_dict.get("weight_sync_config") or {}
  weight_sync_strategy = weight_sync_cfg.get("strategy") if isinstance(weight_sync_cfg, dict) else getattr(weight_sync_cfg, "strategy", "full")

  return {
    "job_id": job_id,
    "timestamp": now,
    "weight_sync_strategy": weight_sync_strategy,
    "currently_executing": currently_executing,
    "pending_queues": {
      "trainer": pending_trainer,
      "sampler": pending_sampler,
    },
    "recent_completed": recent_completed,
    "mutation_history": mutation_history,
  }
