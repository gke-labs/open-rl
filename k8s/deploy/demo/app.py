"""OpenRL demo app: start and stop a sweep of RL client Jobs from a web page, and show the fleet from the
API server's inspection API. Runs in the cluster on the API server image; the code comes from a ConfigMap.

Routes
  GET  /                         the page
  GET  /api/inspect/{path}       proxy to the API server's /api/v1/inspect/{path}
  GET  /api/sweep/presets        built-in job mixes, and the namespace and cluster name for the page
  GET  /api/sweep/jobs           the demo Jobs with their state and the run each created
  GET  /api/sweep/metrics?jobs=  each Job's client metrics per step, from the metrics.jsonl it wrote to its own log_dir
  POST /api/sweep/start          {"run": "...", "jobs": [{"kind","model","count","steps","seed"}]}
  POST /api/sweep/stop           {"name": "<job>"}: release the run's workers, then delete the Job
  POST /api/sweep/stop_all       the same for every demo Job still going
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
API_SERVER = os.getenv("API_SERVER_URL", "http://open-rl-api-server-service:8000").rstrip("/")
CLIENT_IMAGE = os.getenv("CLIENT_IMAGE", "ghcr.io/openrl-dev/openrl/client:latest")
CLUSTER_NAME = os.getenv("CLUSTER_NAME", "")  # shown on the Fleet tab; the page falls back to the namespace
PVC = os.getenv("SHARED_PVC", "open-rl-shared-pvc")
# Each Job writes its client logs to RUNS_ROOT/<job>. The app mounts the same volume read-only and reads only those
# directories: the client's own output, which the control plane neither holds nor serves.
RUNS_ROOT = os.getenv("RUNS_ROOT", "/mnt/shared/open-rl/runs")
JOB_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
# The cookbook's names for what the Metrics tab charts.
CLIENT_METRICS = {
  "reward": "env/all/reward/total",
  "correct": "env/all/correct",
  "format": "env/all/format",
  "output_tokens": "env/all/ac_tokens_per_turn",
  "entropy": "optim/entropy",
  "kl": "optim/kl_sample_train_v1",
}
APP_LABEL = "openrl-demo"
RUN_LABEL = "openrl.io/demo-run"
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
app = FastAPI(title="OpenRL demo")
run_ids: dict[str, str] = {}  # job name -> run id, once the inventory lists the run the Job's client created


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
                # one directory per job under RUNS_ROOT; the Metrics tab reads the job's metrics.jsonl from it
                f"log_dir={RUNS_ROOT}/{job}",
                f"base_model={model}",
                f"steps={steps}",
                "group_size=8",
                "groups_per_batch=8",
                "max_tokens=512",
                f"extra=seed={seed}",
              ],
              # openrl.recipe=rl makes the API server start the sampler with the trainer instead of on the first sample
              # request; openrl.client_log_id names the Job on the run's record, which is how this app finds the run
              # its Job created. The e2e script merges its own openrl. settings into this variable.
              "env": [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "HF_HOME", "value": "/mnt/shared/open-rl/huggingface"},
                {"name": "TINKER_TAGS", "value": f"openrl.recipe=rl,openrl.client_log_id={job}"},
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


session_tags: dict[str, dict] = {}  # session id -> the tags it was opened with; a session's metadata never changes


def runs_by_log_id(since: float) -> dict[str, str]:
  """client_log_id -> run id, from the inspection API alone: the inventory lists each run's sessions, and a session's
  summary carries the tags the client opened it with, among them the openrl.client_log_id this app set to the Job's name.
  Only runs created since `since` are read, so a cluster's whole history is not walked for one new Job; a session
  whose summary fails is skipped and asked again next time."""
  found: dict[str, str] = {}
  try:
    with httpx.Client(base_url=f"{API_SERVER}/api/v1/inspect", timeout=10) as http:
      runs = http.get("/inventory").raise_for_status().json()["runs"]
      for run in runs:
        if (run.get("created_at") or 0) < since:
          continue
        for session_id in run.get("session_ids") or []:
          if session_id not in session_tags:
            try:
              tags = http.get(f"/sessions/{session_id}/summary").raise_for_status().json().get("metadata") or {}
            except (httpx.HTTPError, ValueError):
              continue
            if not tags:
              continue  # the session's record is not written yet; ask again next time rather than remember nothing
            session_tags[session_id] = tags
          if log_id := session_tags[session_id].get("openrl.client_log_id"):
            found.setdefault(log_id, run["run_id"])
  except (httpx.HTTPError, ValueError, KeyError):
    pass
  return found


def find_run_id(job: str, annotations: dict | None = None, by_log_id: dict[str, str] | None = None) -> str | None:
  """The run a Job's client created: from memory, else the Job's annotation, else the inspection API (by the
  client_log_id the Job set), which is then written back as an annotation so a restart of this app does not lose it."""
  if job in run_ids:
    return run_ids[job]
  if annotations and (found := annotations.get(RUN_ID_ANNOTATION)):
    run_ids[job] = found
    return found
  found = (by_log_id if by_log_id is not None else runs_by_log_id(0.0)).get(job)
  if found:
    run_ids[job] = found
    try:
      batch.patch_namespaced_job(job, NAMESPACE, {"metadata": {"annotations": {RUN_ID_ANNOTATION: found}}})
    except Exception:
      pass
  return found


def list_jobs() -> list[dict]:
  jobs = batch.list_namespaced_job(NAMESPACE, label_selector=f"app={APP_LABEL}").items
  pods, by_log_id = pods_by_job(), None
  # the API is asked only while some Job still lacks its run id, and only for runs created since the oldest such Job
  unmatched = [j for j in jobs if j.metadata.name not in run_ids and not (j.metadata.annotations or {}).get(RUN_ID_ANNOTATION)]
  since = min((j.metadata.creation_timestamp.timestamp() for j in unmatched), default=0.0) - 60
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
        "run_id": find_run_id(j.metadata.name, ann, by_log_id)
        if j.metadata.name in run_ids or ann.get(RUN_ID_ANNOTATION)
        else find_run_id(j.metadata.name, ann, by_log_id if by_log_id is not None else (by_log_id := runs_by_log_id(since))),
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
      job = batch.read_namespaced_job(name, NAMESPACE)
      run_id = await asyncio.to_thread(find_run_id, name, None, runs_by_log_id(job.metadata.creation_timestamp.timestamp() - 60))
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
      r = await http.get(f"{API_SERVER}/api/v1/inspect/{path}", params=dict(request.query_params))
    except httpx.HTTPError as exc:
      raise HTTPException(502, f"API server unreachable: {exc}") from exc
  is_json = r.headers.get("content-type", "").startswith("application/json")
  return JSONResponse(status_code=r.status_code, content=r.json() if is_json else {"text": r.text})


def read_metrics(job: str) -> dict:
  """The Job's metrics.jsonl, found under its own log_dir (the recipe writes it a level or two down), as one row per
  step with the charted metrics. A Job that has not written one yet has no rows."""
  root = Path(RUNS_ROOT, job)
  files = sorted(root.glob("**/metrics.jsonl"), key=lambda f: len(f.parts))[:1] if root.is_dir() else []
  rows = []
  for line in files[0].read_text(encoding="utf-8", errors="replace").splitlines() if files else []:
    try:
      raw = json.loads(line)
    except ValueError:
      continue  # a line the client is still writing
    if not isinstance(raw.get("step"), int):
      continue
    row = {"step": raw["step"]}
    for name, key in CLIENT_METRICS.items():
      value = raw.get(key)
      if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
        row[name] = value
    rows.append(row)
  return {"rows": rows}


@app.get("/api/sweep/metrics")
async def metrics(jobs: str):
  """Client metrics per step for each named demo Job, read from the log_dir this app gave it."""
  names = [n for n in jobs.split(",") if n][:64]
  bad = [n for n in names if not JOB_NAME.match(n)]
  if bad:
    raise HTTPException(400, f"not a Job name: {bad[0]}")
  found = await asyncio.gather(*(asyncio.to_thread(read_metrics, n) for n in names))
  return {"jobs": dict(zip(names, found)), "metrics": CLIENT_METRICS}


@app.get("/api/sweep/presets")
async def presets():
  return {"presets": PRESETS, "namespace": NAMESPACE, "cluster": CLUSTER_NAME}


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
  created, skipped, n = [], [], 0
  for spec in specs:
    kind, model = spec.get("kind", "fft"), spec.get("model", "Qwen/Qwen3-0.6B")
    if kind not in ("lora", "fft"):
      raise HTTPException(400, f"kind must be lora or fft, not {kind!r}")
    steps, count, seed0 = int(spec.get("steps") or body.get("steps") or 3), int(spec.get("count") or 1), int(spec.get("seed") or 1)
    for i in range(count):
      n += 1
      name = f"{run}-{n:02d}-{kind}-{slug(model)}"[:63].rstrip("-")
      if name in existing:
        skipped.append(name)  # a Job by this name already exists: the page names them so a reused run is not silent
        continue
      manifest = job_manifest(run, name, kind, model, steps, seed0 + i)
      await asyncio.to_thread(batch.create_namespaced_job, NAMESPACE, manifest)
      created.append({"name": name, "kind": kind, "model": model, "steps": steps, "seed": seed0 + i})
      await asyncio.sleep(0.3)  # one every third of a second, so the submit log reads as a stream
  return {"run": run, "created": created, "skipped": skipped}


@app.post("/api/sweep/stop")
async def stop(body: dict):
  name = body.get("name")
  if not name:
    raise HTTPException(400, "name required")
  return await stop_job(name)


@app.post("/api/sweep/stop_all")
async def stop_all():
  """Stop every demo Job still going; finished ones stay, so the Sweep list keeps its history."""
  jobs = batch.list_namespaced_job(NAMESPACE, label_selector=f"app={APP_LABEL}").items
  names = [j.metadata.name for j in jobs if job_state(j) not in ("succeeded", "failed")]
  return {"stopped": [await stop_job(n) for n in names]}


if __name__ == "__main__":
  import uvicorn

  # Access log on, so every start/stop/stop_all call is auditable in the pod log (the polling GETs are noisy but cheap).
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
