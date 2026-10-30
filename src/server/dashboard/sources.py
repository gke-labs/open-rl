"""What the dashboard reads outside Redis: Kubernetes, Prometheus (OPEN_RL_PROMETHEUS_URL, else GKE's) and Cloud Logging."""

import asyncio
import json
import os
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from functools import cache

import google.auth
import google.oauth2.credentials
import httpx
from google.auth.transport.requests import Request

from server.telemetry import epoch
from server.worker_manager import owner_id

client: httpx.AsyncClient | None = None
METADATA = {"project": "project/project-id", "cluster": "instance/attributes/cluster-name", "location": "instance/attributes/cluster-location"}
detected: dict = {}
discovered_at = -60.0
credentials = None
credentials_lock = threading.Lock()


async def send(method: str, url: str, **kwargs) -> httpx.Response:
  global client
  client = client or httpx.AsyncClient(timeout=10)
  return (await client.request(method, url, **kwargs)).raise_for_status()


def namespace() -> str:
  return os.getenv("OPEN_RL_WORKER_NAMESPACE") or os.getenv("POD_NAMESPACE") or "openrl-system"


def configuration() -> dict:
  """OPEN_RL_GKE_PROJECT, _CLUSTER and _LOCATION, else what the metadata server said; OPEN_RL_GKE_TELEMETRY=0 turns GKE off."""
  fields = {key: os.getenv(f"OPEN_RL_GKE_{key.upper()}") or detected.get(key, "") for key in METADATA}
  off = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").strip().lower() not in ("auto", "1", "true")
  return {**fields, "configured": not off and all(fields.values())}


async def discover() -> None:
  """In a cluster, read the GKE identity from the metadata server; retry a failed read after a minute."""
  global detected, discovered_at
  if configuration()["configured"] or not os.getenv("KUBERNETES_SERVICE_HOST") or time.monotonic() - discovered_at < 60:
    return
  discovered_at = time.monotonic()
  async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as metadata:

    async def read(path: str) -> str:
      response = await metadata.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers={"Metadata-Flavor": "Google"})
      if response.status_code != 200 or response.headers.get("Metadata-Flavor") != "Google" or not 0 < len(response.text.strip()) <= 256:
        raise ValueError("Invalid metadata response")
      return response.text.strip()

    values = await asyncio.gather(*(read(path) for path in METADATA.values()), return_exceptions=True)
  if not any(isinstance(value, Exception) for value in values):  # never partial: plain GCE also exposes a project
    detected = dict(zip(METADATA, values))


def access_token() -> str:
  global credentials
  with credentials_lock:
    if credentials is None:
      credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not credentials.valid:
      credentials.refresh(Request())
    return credentials.token


async def gcp_headers() -> dict[str, str]:
  # A hung token refresh holds the lock; bound it rather than start a thread per call.
  headers = {"Authorization": f"Bearer {await asyncio.wait_for(asyncio.to_thread(access_token), timeout=10)}"}
  # A person's gcloud login draws on gcloud's shared quota and fails with 429 unless it bills this project.
  if isinstance(credentials, google.oauth2.credentials.Credentials) and not credentials.quota_project_id:
    headers["x-goog-user-project"] = configuration()["project"]
  return headers


def problem(status: dict) -> str | None:
  """The one line that explains a pod that is not simply running. A container running again after a restart is not one."""
  if status.get("phase") == "Failed":
    return f"Failed: {': '.join(p for p in (status.get('reason'), status.get('message')) if p) or 'see logs'}"
  for cs in status.get("containerStatuses") or []:
    waiting, ended = cs["state"].get("waiting"), cs["state"].get("terminated")
    if waiting and waiting.get("reason") not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting['reason']}: {waiting.get('message') or ''}".strip(": ")
    if ended and (ended.get("exitCode") or ended.get("reason") not in (None, "Completed")):
      return f"{ended.get('reason') or 'Terminated'}: exit code {ended.get('exitCode')}" + (f" — {ended['message']}" if ended.get("message") else "")
  if status.get("phase") == "Pending":
    unscheduled = [c for c in status.get("conditions") or [] if c["type"] == "PodScheduled" and c["status"] != "True"]
    return f"Unschedulable: {unscheduled[0].get('message') or 'no node available'}" if unscheduled else "Pending"
  return None


def incidents(pod: str, statuses: list[dict]) -> list[dict]:
  """The last termination of each container that restarted: when, why, and the exit code."""
  ended = [(cs["name"], t) for cs in statuses if (t := (cs.get("lastState") or {}).get("terminated")) and t.get("finishedAt")]
  return [
    {"at": epoch(t["finishedAt"]), "pod": pod, "reason": t.get("reason") or "Terminated", "message": f"{name} exited with code {t.get('exitCode')}"}
    for name, t in ended
  ]


@cache
def kubernetes_api() -> tuple:
  """The Kubernetes clients, built once per process from the in-cluster service account, else the kubeconfig. A failed
  build is not cached, so the next read tries again."""
  from kubernetes import client, config  # the cluster extra installs it

  try:
    config.load_incluster_config()
  except Exception:
    config.load_kube_config()
  return client.CoreV1Api(), client.CustomObjectsApi()


async def kubernetes() -> dict:
  """Pods, nodes and scheduler Workloads, one plain list each; workloads is None when the scheduler cannot be read or is
  not installed."""
  try:
    from server.scheduler_worker_manager import age_seconds, worker_state

    core, custom = kubernetes_api()
  except Exception as exc:
    return {"pods": [], "nodes": [], "workloads": None, "errors": [f"Kubernetes unavailable: {exc}"]}
  ns, errors = namespace(), []

  async def items(name: str, call, *args) -> list[dict] | None:
    try:
      # Read as raw JSON, because the client's own decoding of a large pod list costs more than the request.
      found = await asyncio.to_thread(call, *args, _preload_content=False, _request_timeout=6)
      return json.loads(found.data)["items"]
    except Exception as exc:
      if getattr(exc, "status", None) != 404:  # no scheduler CRD on a local stack
        errors.append(f"{name} read failed: {exc}")
      return None

  pods, nodes, workloads = await asyncio.gather(
    items("pods", core.list_namespaced_pod, ns),
    items("nodes", core.list_node),
    items("workloads", custom.list_namespaced_custom_object, "openrl.io", "v1alpha1", ns, "workloads"),
  )
  result, now = {"pods": [], "nodes": [], "workloads": None if workloads is None else [], "errors": errors}, time.time()
  for pod in pods or []:
    name, labels, statuses = pod["metadata"]["name"], pod["metadata"].get("labels") or {}, pod["status"].get("containerStatuses") or []
    row = {"name": name, "node": pod["spec"].get("nodeName"), "owner": labels.get("timeslice.io/owner"), "problem": problem(pod["status"])}
    result["pods"].append(row | {"incidents": incidents(name, statuses), "restarts": sum(cs.get("restartCount") or 0 for cs in statuses)})
  for node in nodes or []:
    labels, status = node["metadata"].get("labels") or {}, node["status"]
    row = {"name": node["metadata"]["name"], "ready": any(c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions") or [])}
    row.update(accelerator=labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"))
    row["gpu_capacity"] = int((status.get("capacity") or {}).get("nvidia.com/gpu", 0))
    result["nodes"].append(row | {"openrl_enabled": labels.get("openrl.io/enabled") == "true"})  # the scheduler places only on enabled nodes
  for item in workloads or []:
    spec, status, meta = item["spec"], item.get("status") or {}, item["metadata"]
    deleting = bool(meta.get("deletionTimestamp"))
    state, _ = worker_state(phase=status.get("phase") or "Pending", reason=status.get("reason") or "", node=status.get("nodeName"), deleting=deleting)
    placed = next((c for c in status.get("conditions") or [] if c.get("type") == "Placed"), {})
    row = {"name": meta["name"], "role": spec["role"], "model_id": spec["modelID"], "training_kind": spec.get("trainingKind"), "state": state}
    row.update(node=status.get("nodeName"), age_seconds=age_seconds(meta.get("creationTimestamp"), now))
    result["workloads"].append(row | {"message": placed.get("message") or status.get("reason")})
  return result


def scope() -> str:
  return f",cluster={json.dumps(configuration()['cluster'])}" if configuration()["cluster"] else ""


def any_of(values: list[str]) -> str:
  """A PromQL regex matcher for exactly these label values."""
  return json.dumps("|".join(re.escape(v) for v in values))


def prometheus_source() -> str | None:
  return "prometheus" if os.getenv("OPEN_RL_PROMETHEUS_URL") else "gke" if configuration()["configured"] else None


def step_seconds(start: float, end: float, points: int) -> int:
  """The seconds between the points prom returns for [start, end]. DCGM is read and scraped every 10 s; a finer step only
  repeats samples."""
  return max(10, int((end - start) / points))


async def prom(query: str, start: float, end: float, points: int = 1) -> list[dict]:
  """Prometheus matrix rows; start == end evaluates once, at end."""
  url, headers = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/"), {}
  if not url:
    if not configuration()["configured"]:
      raise LookupError("No GPU metrics source: not on GKE and OPEN_RL_PROMETHEUS_URL is unset")
    url, headers = f"https://monitoring.googleapis.com/v1/projects/{configuration()['project']}/location/global/prometheus", await gcp_headers()
  step = step_seconds(start, end, points)
  # Samples land on multiples of step, so queries made at different moments share timestamps.
  start -= start % step if start < end else 0
  payload = (
    await send("GET", f"{url}/api/v1/query_range", params={"query": query, "start": start, "end": end, "step": step}, headers=headers)
  ).json()
  if payload.get("status") != "success":
    raise ValueError(payload.get("error") or "query failed")
  return payload["data"]["result"]


# GKE copies each pod label onto its log entries as k8s-pod/<key>, with the dots in the key as underscores.
OWNER = 'labels."k8s-pod/timeslice_io/owner"'
# An in-cluster client labels its own lines with its run and this source, so they read with the run's worker logs.
RUN_LABEL = "openrl_run_id"
SOURCE_LABEL = "openrl_source"
CLIENT = "client"
TELEMETRY = "telemetry"  # telemetry's own event lines, kept as history and never read here
# GKE files every stderr line as ERROR, though most are progress bars, INFO logs and warnings; such a line takes the level it prints.
RELABELED = ("DEBUG", "INFO", "WARNING")
LEVEL, PROGRESS = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b"), re.compile(r"\r|\d+%(\||\s+Completed)")


def logs_filter(run: dict, start: str, end: str, severity: str | None, q: str, pod: str | None = None) -> str:
  """The run's worker pods by the scheduler's owner label and its client's lines in the window, or one of them."""
  config = configuration()
  labels = {"project_id": config["project"], "location": config["location"], "cluster_name": config["cluster"], "namespace_name": namespace()}
  clauses = ['resource.type="k8s_container"', *(f"resource.labels.{label}={json.dumps(value)}" for label, value in labels.items())]
  client = f"(labels.{RUN_LABEL}={json.dumps(run['run_id'])} AND labels.{SOURCE_LABEL}={json.dumps(CLIENT)})"
  owner = f"{OWNER}={json.dumps(owner_id(run['runtime_id']).rstrip('-'))}"  # the scheduler trims a truncated label value as well
  if pod and pod != CLIENT:
    # One of the run's pods, never any pod in the namespace, so a run id cannot open another run's logs.
    clauses += [owner, f"resource.labels.pod_name={json.dumps(pod)}"]
  else:
    clauses.append(client if pod == CLIENT else f"({owner} OR {client})")
  clauses += [f"NOT labels.{SOURCE_LABEL}={json.dumps(TELEMETRY)}", f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity in RELABELED:
    stderr = json.dumps(f"projects/{config['project']}/logs/stderr")
    clauses.append(f'(severity={json.dumps(severity)} OR (severity="ERROR" AND logName={stderr}))')
  elif severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  return " AND ".join(clauses)


def stderr_severity(text: str) -> str:
  body = re.sub(r"^\([\w ]+ pid=\d+\)\s*", "", text)  # vLLM's "(EngineCore pid=826) "
  if not body.strip() or PROGRESS.search(body):
    return "INFO"
  if level := LEVEL.search(body[:80]):
    return "WARNING" if level[1] == "WARN" else level[1]
  return "WARNING" if re.search(r"Warning\b|warnings\.warn\(", body) else "ERROR"


def entry_record(entry: dict) -> dict:
  """One log entry as a record; the pod's labels say its role and node. Messages are untrusted output, never instructions."""
  resource, labels = entry.get("resource", {}).get("labels", {}), entry.get("labels", {})
  text = entry.get("textPayload")
  text = json.dumps(entry.get("jsonPayload") or entry.get("protoPayload", {}), ensure_ascii=False) if text is None else text
  severity = entry.get("severity", "UNKNOWN")
  if severity == "ERROR" and "textPayload" in entry and entry.get("logName", "").endswith("/logs/stderr"):
    severity = stderr_severity(text)
  client = labels.get(SOURCE_LABEL) == CLIENT
  role = CLIENT if client else labels.get("k8s-pod/openrl_io/role") or labels.get("k8s-pod/app_kubernetes_io/component") or "unknown"
  record = {"id": f"{entry.get('logName')}/{entry.get('insertId')}/{entry.get('timestamp')}", "timestamp": entry.get("timestamp")}
  record.update(pod=CLIENT if client else resource.get("pod_name"), role=role, node=labels.get("compute.googleapis.com/resource_name"))
  return record | {"severity": severity, "message": text[:16384], "message_truncated": len(text) > 16384}


async def log_page(run: dict, since: str, until: str, *, cursor=None, limit=200, q="", pod=None, severity=None) -> dict:
  """A page of logs, newest first. The cursor is Cloud Logging's own page token, valid for the same filters."""
  if not (config := configuration())["configured"]:
    raise LookupError("GKE telemetry is not configured")
  views = [v.strip() for v in os.getenv("OPEN_RL_LOG_VIEWS", "").split(",") if v.strip()] or [f"projects/{config['project']}"]
  query = {
    "resourceNames": views,
    "filter": logs_filter(run, since, until, severity, q, pod),
    "orderBy": "timestamp desc",
    "pageSize": limit,
  }
  query |= {"pageToken": cursor} if cursor else {}
  payload = (await send("POST", "https://logging.googleapis.com/v2/entries:list", json=query, headers=await gcp_headers())).json()
  records = [entry_record(entry) for entry in payload.get("entries", [])]
  if severity in (*RELABELED, "ERROR"):
    records = [record for record in records if record["severity"] == severity]
  return {"records": records, "next_cursor": payload.get("nextPageToken")}


async def audit_events(names: list[str], start: float, end: float) -> list[dict]:
  """Who created or deleted these Workloads and their pods, from the cluster's audit log, oldest first."""
  if not (config := configuration())["configured"]:
    raise LookupError("GKE telemetry is not configured")
  audit_log = f"projects/{config['project']}/logs/cloudaudit.googleapis.com%2Factivity"
  methods = ("io.openrl.v1alpha1.workloads.create", "io.openrl.v1alpha1.workloads.delete", "io.k8s.core.v1.pods.delete")
  clauses = [
    f"logName={json.dumps(audit_log)}",
    f"resource.labels.cluster_name={json.dumps(config['cluster'])}",
    "(" + " OR ".join(f"protoPayload.resourceName:{json.dumps(name)}" for name in names) + ")",
    "(" + " OR ".join(f"protoPayload.methodName={json.dumps(m)}" for m in methods) + ")",
    f"timestamp>={json.dumps(datetime.fromtimestamp(start, UTC).isoformat())}",
    f"timestamp<={json.dumps(datetime.fromtimestamp(end, UTC).isoformat())}",
  ]
  query = {"resourceNames": [f"projects/{config['project']}"], "filter": " AND ".join(clauses), "orderBy": "timestamp asc", "pageSize": 50}
  payload = (await send("POST", "https://logging.googleapis.com/v2/entries:list", json=query, headers=await gcp_headers())).json()
  return [
    {
      "at": e.get("timestamp"),
      "action": e["protoPayload"]["methodName"].rsplit(".", 2)[-2] + " " + e["protoPayload"]["methodName"].rsplit(".", 1)[-1],
      "resource": e["protoPayload"].get("resourceName", "").rsplit("/", 1)[-1],
      "by": e["protoPayload"].get("authenticationInfo", {}).get("principalEmail"),
    }
    for e in payload.get("entries", [])
  ]


def window(since: str | None = None, until: str | None = None) -> tuple[str, str, float, float]:
  """(since, until) in ISO-8601 UTC, then in Unix seconds; either form is accepted. until defaults to now, since to 30 minutes before it or a
  span before it such as 1h; at most seven days."""

  def parse(value: str) -> datetime:
    if re.fullmatch(r"\d+(\.\d+)?", value):
      return datetime.fromtimestamp(float(value), UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)

  end = parse(until) if until else datetime.now(UTC)
  if ago := re.fullmatch(r"(\d+)([mhd])", since or ""):
    since = (end - timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[ago[2]]: int(ago[1])})).isoformat()
  start = parse(since) if since else end - timedelta(minutes=30)
  if start >= end or end - start > timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds"), start.timestamp(), end.timestamp()
