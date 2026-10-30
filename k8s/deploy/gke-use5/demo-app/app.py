"""OpenRL demo app: start and stop a sweep of RL client Jobs from a web page, and show the fleet from the
dashboard's inspection API. Runs in the cluster on the API server image; the code comes from a ConfigMap.

Routes
  GET  /                         the page
  GET  /api/inspect/{path}       proxy to the dashboard's /api/v1/inspect/{path}
  GET  /api/sweep/presets        built-in job mixes
  GET  /api/sweep/jobs           the demo Jobs with their state and the run each created
  POST /api/sweep/start          {"run": "...", "jobs": [{"kind","model","count","steps","seed"}]}
  POST /api/sweep/stop           {"name": "<job>"}: release the run's workers, then delete the Job
  POST /api/sweep/stop_all       the same for every demo Job
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from kubernetes import client, config

NAMESPACE = os.getenv("POD_NAMESPACE", "openrl-system")
DASHBOARD = os.getenv("DASHBOARD_URL", "http://open-rl-dashboard:8000").rstrip("/")
API_SERVER = os.getenv("API_SERVER_URL", "http://open-rl-api-server-service:8000").rstrip("/")
CLIENT_IMAGE = os.getenv("CLIENT_IMAGE", "ghcr.io/gke-labs/open-rl/client:latest")
PVC = os.getenv("SHARED_PVC", "open-rl-shared-pvc")
APP_LABEL = "openrl-demo"
RUN_LABEL = "openrl.io/demo-run"
MODEL_ID = re.compile(r"(?:TrainingClient|ServiceClient) initialized for (?:model|session) ([a-f0-9-]{36})")
HERE = Path(__file__).parent

PRESETS = {
  "keynote-20": {
    "steps": 4,
    "jobs": [
      {"kind": "lora", "model": "google/gemma-4-e2b", "count": 5, "steps": 8},
      {"kind": "lora", "model": "Qwen/Qwen3-0.6B", "count": 5, "steps": 8},
      {"kind": "fft", "model": "google/gemma-4-e2b", "count": 5, "steps": 4},
      {"kind": "fft", "model": "Qwen/Qwen3-0.6B", "count": 5, "steps": 4},
    ],
  },
  "qwen-8": {
    "steps": 5,
    "jobs": [
      {"kind": "lora", "model": "Qwen/Qwen3-0.6B", "count": 4, "steps": 5},
      {"kind": "fft", "model": "Qwen/Qwen3-0.6B", "count": 4, "steps": 5},
    ],
  },
  "smoke-2": {"steps": 2, "jobs": [{"kind": "fft", "model": "Qwen/Qwen3-0.6B", "count": 2, "steps": 2}]},
}

config.load_incluster_config()
batch = client.BatchV1Api()
core = client.CoreV1Api()
custom = client.CustomObjectsApi()
app = FastAPI(title="OpenRL demo")
run_ids: dict[str, str] = {}  # job name -> run id, once seen in the client's log


def slug(model: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")


def job_manifest(run: str, job: str, kind: str, model: str, steps: int, seed: int) -> dict:
  scenario = f"{kind}-gsm8k-rl"
  labels = {"app": APP_LABEL, RUN_LABEL: run, "openrl.io/demo-kind": kind, "openrl.io/demo-model": slug(model), "openrl.io/demo-seed": str(seed)}
  return {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
      "name": job,
      "namespace": NAMESPACE,
      "labels": labels,
      "annotations": {"openrl.io/demo-steps": str(steps), "openrl.io/demo-base-model": model},
    },
    "spec": {
      "backoffLimit": 0,
      "ttlSecondsAfterFinished": 86400,
      "template": {
        "metadata": {"labels": labels},
        "spec": {
          "restartPolicy": "Never",
          "serviceAccountName": "open-rl-sa",
          "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
          "containers": [
            {
              "name": "client",
              "image": CLIENT_IMAGE,
              "imagePullPolicy": "IfNotPresent",
              "args": [
                f"scenario={scenario}",
                "uv_extra=cpu",
                f"base_url={API_SERVER}",
                # one directory per job directly under runs/, so the dashboard's client-directory search
                # (three levels below its root, set to /mnt/shared/open-rl/runs here) finds the recipe's metrics
                f"log_dir=/mnt/shared/open-rl/runs/{job}",
                f"base_model={model}",
                f"steps={steps}",
                "group_size=8",
                "groups_per_batch=8",
                "max_tokens=512",
                f"extra=seed={seed}",
              ],
              # openrl.recipe=rl makes the API server start the sampler with the trainer instead of on the first sample
              # request; the e2e script merges its own openrl. settings into this variable.
              "env": [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "HF_HOME", "value": "/mnt/shared/open-rl/huggingface"},
                {"name": "TINKER_TAGS", "value": "openrl.recipe=rl"},
              ],
              "resources": {"requests": {"cpu": "1", "memory": "4Gi"}, "limits": {"memory": "12Gi"}},
              "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
            }
          ],
          "volumes": [{"name": "shared-storage", "persistentVolumeClaim": {"claimName": PVC}}],
        },
      },
    },
  }


def job_state(job) -> str:
  st = job.status
  if st.succeeded:
    return "succeeded"
  if st.failed:
    return "failed"
  if st.active:
    return "running"
  return "pending"


def pods_by_job() -> dict[str, object]:
  pods = core.list_namespaced_pod(NAMESPACE, label_selector=f"app={APP_LABEL}").items
  return {p.metadata.labels.get("job-name"): p for p in pods}


RUN_ID_ANNOTATION = "openrl.io/demo-run-id"


def find_run_id(job: str, pod, annotations: dict | None = None) -> str | None:
  """The run a Job's client created: from memory, else the Job's annotation, else the client log (whole log, once),
  which is then written back as an annotation so a restart of this app does not lose it."""
  if job in run_ids:
    return run_ids[job]
  if annotations and (found := annotations.get(RUN_ID_ANNOTATION)):
    run_ids[job] = found
    return found
  if pod is None or pod.status.phase == "Pending":
    return None
  try:
    text = core.read_namespaced_pod_log(pod.metadata.name, NAMESPACE)
  except Exception:
    return None
  if m := MODEL_ID.search(text):
    run_ids[job] = m.group(1)
    try:
      batch.patch_namespaced_job(job, NAMESPACE, {"metadata": {"annotations": {RUN_ID_ANNOTATION: m.group(1)}}})
    except Exception:
      pass
    return m.group(1)
  return None


def list_jobs() -> list[dict]:
  jobs = batch.list_namespaced_job(NAMESPACE, label_selector=f"app={APP_LABEL}").items
  pods = pods_by_job()
  out = []
  for j in sorted(jobs, key=lambda j: j.metadata.creation_timestamp, reverse=True):  # newest first
    labels, ann = j.metadata.labels or {}, j.metadata.annotations or {}
    pod = pods.get(j.metadata.name)
    out.append(
      {
        "name": j.metadata.name,
        "run": labels.get(RUN_LABEL),
        "kind": labels.get("openrl.io/demo-kind"),
        "model": ann.get("openrl.io/demo-base-model"),
        "seed": labels.get("openrl.io/demo-seed"),
        "steps": int(ann.get("openrl.io/demo-steps", "0") or 0),
        "created_at": j.metadata.creation_timestamp.timestamp(),
        "started_at": j.status.start_time.timestamp() if j.status.start_time else None,
        "completed_at": j.status.completion_time.timestamp() if j.status.completion_time else None,
        "state": job_state(j),
        "pod": pod.metadata.name if pod else None,
        "pod_phase": pod.status.phase if pod else None,
        "run_id": find_run_id(j.metadata.name, pod, ann),
      }
    )
  return out


async def release(run_id: str) -> bool:
  async with httpx.AsyncClient(timeout=20) as http:
    try:
      r = await http.post(f"{API_SERVER}/api/v1/delete_model", json={"model_id": run_id})
      return r.status_code < 300
    except httpx.HTTPError:
      return False


async def stop_job(name: str) -> dict:
  run_id = run_ids.get(name)
  if run_id is None:
    try:
      job = batch.read_namespaced_job(name, NAMESPACE)
      run_id = (job.metadata.annotations or {}).get(RUN_ID_ANNOTATION)
    except Exception:
      run_id = None
  if run_id is None:
    try:
      pod = pods_by_job().get(name)
      run_id = await asyncio.to_thread(find_run_id, name, pod)
    except Exception:
      run_id = None
  released = await release(run_id) if run_id else False
  try:
    batch.delete_namespaced_job(name, NAMESPACE, propagation_policy="Background")
    deleted = True
  except client.ApiException as exc:
    deleted = exc.status == 404
  print(json.dumps({"event": "stop", "job": name, "run_id": run_id, "released": released, "deleted": deleted}), flush=True)
  return {"name": name, "run_id": run_id, "released": released, "deleted": deleted}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
  return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/inspect/{path:path}")
async def inspect(path: str, request: Request):
  async with httpx.AsyncClient(timeout=120) as http:
    try:
      r = await http.get(f"{DASHBOARD}/api/v1/inspect/{path}", params=dict(request.query_params))
    except httpx.HTTPError as exc:
      raise HTTPException(502, f"dashboard unreachable: {exc}") from exc
  is_json = r.headers.get("content-type", "").startswith("application/json")
  return JSONResponse(status_code=r.status_code, content=r.json() if is_json else {"text": r.text})


def list_placements() -> list[dict]:
  """Live placements from Workload status, which the scheduler stamps within seconds; the dashboard's own
  placement records trail by up to 30 s because the API server sweeps Workloads on its reaper interval."""
  items = custom.list_namespaced_custom_object("openrl.io", "v1alpha1", NAMESPACE, "workloads").get("items", [])
  out = []
  for w in items:
    spec, st = w.get("spec", {}), w.get("status", {})
    if not st.get("nodeName") or st.get("releasedAt"):
      continue
    out.append(
      {
        "name": w["metadata"]["name"],
        "role": spec.get("role"),
        "owner": spec.get("modelID") or spec.get("ownerID"),
        "phase": st.get("phase"),
        "kind": spec.get("trainingKind"),
        "exclusive": spec.get("exclusive"),
        "node": st.get("nodeName"),
        "pod": st.get("podName"),
        "devices": st.get("gpuUUIDs") or [],
        "placed_at": st.get("placedAt"),
        "shared": w["metadata"]["name"].startswith("lora-"),
      }
    )
  return out


@app.get("/api/sweep/placements")
async def placements():
  return {"placements": await asyncio.to_thread(list_placements), "now": time.time()}


DELTA_LINE = re.compile(r"Saved sparse delta \(([0-9.]+)% changed elements, (\d+)/(\d+)\) to (\S+)")
ADAPTER_LINE = re.compile(r"Auto-saved adapter '([0-9a-f-]+)' to (\S+)")


@app.get("/api/sweep/deltas")
async def deltas(run_id: str, since: str = "7d", until: str | None = None, kind: str = "fft"):
  """Each sparse delta the run's trainer published, from its log lines through the dashboard's logs route:
  when, which sampler version, the share of weights that changed, and the counts. LoRA runs report adapter saves.
  Stopgap: the trainer returns density_pct from the save but the op record keeps only result["metrics"], so the
  figure is not in /runs/{id}/steps yet (see the findings report)."""
  out, cursor, pages = [], None, 0
  async with httpx.AsyncClient(timeout=120) as http:
    while pages < 20:
      params = {"run_id": run_id, "since": since, "q": "Saved sparse delta" if kind == "fft" else "Auto-saved adapter"}
      if kind == "fft":
        params["pod"] = f"orw-fft-{run_id}-trainer"
      if until:
        params["until"] = until
      if cursor:
        params["cursor"] = cursor
      try:
        r = await http.get(f"{DASHBOARD}/api/v1/inspect/logs", params=params)
        body = r.json()
      except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"dashboard logs unreachable: {exc}") from exc
      if not body.get("available", True):
        raise HTTPException(502, body.get("error") or "logs unavailable")
      for rec in body.get("records") or []:
        msg = rec.get("message") or ""
        if m := DELTA_LINE.search(msg):
          path = m.group(4)
          out.append(
            {
              "time": rec.get("timestamp"),
              "kind": "delta",
              "version": path.rsplit("/", 1)[-1],
              "density_pct": float(m.group(1)),
              "changed": int(m.group(2)),
              "total": int(m.group(3)),
              "final": path.endswith("/final"),
            }
          )
        elif m := ADAPTER_LINE.search(msg):
          if run_id.startswith(m.group(1)[:8]):
            out.append({"time": rec.get("timestamp"), "kind": "adapter", "version": m.group(2).rsplit("/", 1)[-1]})
      cursor, pages = body.get("next_cursor"), pages + 1
      if not cursor:
        break
  out.sort(key=lambda x: x["time"] or "")
  return {"run_id": run_id, "saves": out, "pages": pages}


@app.get("/api/sweep/presets")
async def presets():
  return PRESETS


@app.get("/api/sweep/jobs")
async def jobs():
  return {"jobs": await asyncio.to_thread(list_jobs), "now": time.time()}


@app.post("/api/sweep/start")
async def start(body: dict):
  run = re.sub(r"[^a-z0-9-]", "-", str(body.get("run") or time.strftime("demo-%H%M%S")).lower())[:30].strip("-")
  specs = body.get("jobs") or []
  if not specs:
    raise HTTPException(400, "no jobs")
  existing = {j.metadata.name for j in batch.list_namespaced_job(NAMESPACE, label_selector=f"app={APP_LABEL}").items}
  created, n = [], 0
  for spec in specs:
    kind, model = spec.get("kind", "fft"), spec.get("model", "Qwen/Qwen3-0.6B")
    if kind not in ("lora", "fft"):
      raise HTTPException(400, f"kind must be lora or fft, not {kind!r}")
    steps, count, seed0 = int(spec.get("steps") or body.get("steps") or 3), int(spec.get("count") or 1), int(spec.get("seed") or 1)
    for i in range(count):
      n += 1
      name = f"{run}-{n:02d}-{kind}-{slug(model)}"[:63].rstrip("-")
      if name in existing:
        continue
      manifest = job_manifest(run, name, kind, model, steps, seed0 + i)
      await asyncio.to_thread(batch.create_namespaced_job, NAMESPACE, manifest)
      created.append({"name": name, "kind": kind, "model": model, "steps": steps, "seed": seed0 + i})
      await asyncio.sleep(0.3)  # one every third of a second, so the submit log reads as a stream
  return {"run": run, "created": created}


@app.post("/api/sweep/stop")
async def stop(body: dict):
  name = body.get("name")
  if not name:
    raise HTTPException(400, "name required")
  return await stop_job(name)


@app.post("/api/sweep/stop_all")
async def stop_all():
  names = [j.metadata.name for j in batch.list_namespaced_job(NAMESPACE, label_selector=f"app={APP_LABEL}").items]
  return {"stopped": [await stop_job(n) for n in names]}


if __name__ == "__main__":
  import uvicorn

  # Access log on, so every start/stop/stop_all call is auditable in the pod log (the polling GETs are noisy but cheap).
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
