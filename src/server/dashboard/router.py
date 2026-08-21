# HTTP surface of the operational dashboard. The JSON endpoints are the same primitives the
# ops CLI uses (health, problems, inspect, logs, stop); the static files are the human UI.

import asyncio
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.dashboard import data, demo
from server.store import get_store

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter(prefix="/api/v1/dashboard")


@router.get("/cluster")
async def dashboard_cluster():
  if data.demo_mode_enabled():
    return demo.demo_cluster()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  return await data.cluster_snapshot(get_store(), k8s)


@router.get("/runs")
async def dashboard_runs(request: Request):
  if data.demo_mode_enabled():
    return demo.demo_runs()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  return await data.runs_snapshot(get_store(), request.app.state.fft_worker_manager, k8s["pods"])


@router.get("/runs/{run_id}")
async def dashboard_run_detail(run_id: str, request: Request, logs: int = 0):
  log_tail = min(max(logs, 0), 2000)
  if data.demo_mode_enabled():
    detail = demo.demo_run_detail(run_id, log_tail)
  else:
    k8s = await asyncio.to_thread(data.k8s_snapshot)
    detail = await data.run_detail(get_store(), request.app.state.fft_worker_manager, run_id, k8s, log_tail)
  if detail is None:
    return JSONResponse(status_code=404, content={"error": f"unknown run: {run_id}"})
  return detail


@router.post("/runs/{run_id}/stop")
async def dashboard_stop_run(run_id: str, request: Request):
  if data.demo_mode_enabled():
    return {"demo": True, "notice": demo.DEMO_NOTICE, "run_id": run_id, "stopped": True, "actions": ["demo mode — nothing was actually stopped"]}
  result = await data.stop_run(get_store(), request.app.state.fft_worker_manager, run_id)
  if not result["stopped"]:
    return JSONResponse(status_code=409, content={**result, "error": "nothing to stop for this run"})
  return result


@router.get("/health")
async def dashboard_health(request: Request):
  if data.demo_mode_enabled():
    return demo.demo_health()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  checks = await data.health_checks(get_store(), k8s)
  stats, queues = await data.operational_stats(get_store(), k8s, request.app.state.fft_worker_manager)
  return {"demo": False, "checks": checks, "stats": stats, "queues": queues}


@router.get("/problems")
async def dashboard_problems():
  if data.demo_mode_enabled():
    return demo.demo_problems()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  checks = await data.health_checks(get_store(), k8s)
  return {"demo": False, "problems": data.derive_problems(checks, k8s)}


@router.get("/pods/{pod}/logs")
async def dashboard_pod_logs(pod: str, container: str | None = None, tail: int = 500):
  if data.demo_mode_enabled():
    return demo.demo_pod_logs(pod)
  try:
    return await asyncio.to_thread(data.k8s_pod_logs, pod, container, min(max(tail, 1), 5000))
  except Exception as exc:
    return JSONResponse(status_code=503, content={"error": str(exc)})


def mount_dashboard(app: FastAPI) -> None:
  # The gateway lifespan replaces this with the live manager once FFT workers exist.
  app.state.fft_worker_manager = None
  app.include_router(router)
  app.mount("/dashboard/static", StaticFiles(directory=STATIC_DIR), name="dashboard-static")

  @app.get("/dashboard", include_in_schema=False)
  async def dashboard_index():
    return FileResponse(STATIC_DIR / "index.html")
