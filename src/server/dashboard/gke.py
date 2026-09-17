"""Cloud Logging as the run-log source on GKE.

The gateway learns its project, cluster and location from the metadata
server, authenticates with Application Default Credentials (Workload Identity
in the cluster), and queries entries for the pods a run has owned. Paging is
stateless: the cursor is Cloud Logging's own page token, valid for the same
filters and window.
"""

import asyncio
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta

import google.auth
import httpx
from google.auth.transport.requests import Request

from server.dashboard import cluster

MAX_MESSAGE = 16 * 1024
MAX_SOURCES = 64
REFUSED = "Cloud Logging refused this gateway's credentials (missing logging read scope or IAM role)"
UNAVAILABLE = "Cloud Logging unavailable; check credentials, IAM and telemetry configuration"

_METADATA = {"project": "project/project-id", "cluster": "instance/attributes/cluster-name", "location": "instance/attributes/cluster-location"}
_detected: dict | None = None
_discovery_at = 0.0
_discovery_lock = asyncio.Lock()
_credentials = None
_LOCK = threading.Lock()
_refused_until = 0.0


# ---- identity -------------------------------------------------------------------


def configuration() -> dict:
  mode = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").strip().lower()
  fields = {
    "project": os.getenv("OPEN_RL_GKE_PROJECT") or (_detected or {}).get("project", ""),
    "cluster": os.getenv("OPEN_RL_GKE_CLUSTER") or (_detected or {}).get("cluster", ""),
    "location": os.getenv("OPEN_RL_GKE_LOCATION") or (_detected or {}).get("location", ""),
    "mode": mode,
  }
  complete = all(fields[key] for key in ("project", "cluster", "location"))
  fields["enabled"] = mode in {"1", "true"} or (mode == "auto" and complete)
  fields["configured"] = fields["enabled"] and complete
  return fields


async def discover() -> dict:
  """Resolve GKE identity once; retry unavailable metadata after a minute."""
  global _detected, _discovery_at
  config = configuration()
  if config["mode"] not in {"auto", "1", "true"} or config["configured"] or not os.getenv("KUBERNETES_SERVICE_HOST"):
    return config
  async with _discovery_lock:
    if time.monotonic() - _discovery_at < 60:
      return configuration()
    async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:

      async def read(key, path):
        response = await client.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers={"Metadata-Flavor": "Google"})
        response.raise_for_status()
        value = response.text.strip()
        if response.headers.get("Metadata-Flavor") != "Google" or not value or len(value) > 256:
          raise ValueError("Invalid metadata response")
        return key, value

      values = await asyncio.gather(*(read(key, path) for key, path in _METADATA.items()), return_exceptions=True)
    # Never enable from partial metadata: generic GCE also exposes project ID.
    if all(isinstance(value, tuple) for value in values):
      _detected = dict(values)
    _discovery_at = time.monotonic()
    return configuration()


def access_token() -> str:
  global _credentials
  with _LOCK:
    if _credentials is None:
      _credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not _credentials.valid:
      _credentials.refresh(Request())
    return _credentials.token


async def request(method: str, url: str, **kwargs) -> dict:
  token = await asyncio.wait_for(asyncio.to_thread(access_token), timeout=10)
  async with httpx.AsyncClient(timeout=15) as client:
    response = await client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs)
    response.raise_for_status()
    return response.json()


def refused() -> bool:
  return time.monotonic() < _refused_until


def note_refusal() -> None:
  """A scope or IAM refusal does not clear itself; stop asking for a while."""
  global _refused_until
  _refused_until = time.monotonic() + 600


# ---- time and sources -----------------------------------------------------------


def timestamp(value) -> str:
  """ISO-8601 in UTC with microseconds, the form Cloud Logging filters compare."""
  if isinstance(value, (int, float)):
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="microseconds")
  parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=UTC)
  return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def time_range(since=None, until=None) -> tuple[str, str]:
  end = datetime.fromisoformat(timestamp(until)) if until else datetime.now(UTC)
  start = datetime.fromisoformat(timestamp(since)) if since else end - timedelta(minutes=30)
  if start >= end or end - start > timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")


def pod_sources(run: dict | None, past: list[dict], observed_at: str) -> list[dict]:
  """The pods a run's logs can come from: live ones until now, and any recorded
  placement of the run until it was last seen. Names bound to lifetimes, since
  Cloud Logging labels carry pod names and not UIDs."""
  sources: dict[str, dict] = {}
  run_id = run["run_id"] if run else None
  for entry in past:
    if run_id in (entry.get("run_ids") or []) and entry.get("pod"):
      sources[entry["pod"]] = {
        "pod": entry["pod"],
        "node": entry.get("node"),
        "role": entry.get("role") or "unknown",
        "created_at": timestamp(entry["first_seen"]),
        "until": timestamp(entry["last_seen"] + 60),
      }
  for pod in (run or {}).get("pods", []):
    if pod.get("created_at"):
      sources[pod["name"]] = {
        "pod": pod["name"],
        "node": pod.get("node"),
        "role": pod.get("role", "unknown"),
        "created_at": timestamp(pod["created_at"]),
        "until": timestamp(observed_at),
        "shared_runtime": pod.get("shared_runtime", False),
      }
  return list(sources.values())


# ---- query -------------------------------------------------------------------------


def scope_filter() -> list[str]:
  config = configuration()
  labels = (
    ("project_id", config["project"]),
    ("location", config["location"]),
    ("cluster_name", config["cluster"]),
    ("namespace_name", cluster.namespace()),
  )
  return ['resource.type="k8s_container"', *[f"resource.labels.{label}={json.dumps(value)}" for label, value in labels]]


def source_filter(sources: list[dict]) -> str:
  def clause(s: dict) -> str:
    return f"(resource.labels.pod_name={json.dumps(s['pod'])} AND timestamp>={json.dumps(s['created_at'])} AND timestamp<={json.dumps(s['until'])})"

  return "(" + " OR ".join(clause(s) for s in sources) + ")"


def logs_filter(sources: list[dict], start: str, end: str, severity: str | None, q: str) -> str:
  clauses = scope_filter() + [source_filter(sources), f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  return " AND ".join(clauses)


def entry_text(entry: dict) -> str:
  text = entry.get("textPayload")
  if text is not None:
    return text
  fields = entry.get("jsonPayload") or {}
  return json.dumps(fields, ensure_ascii=False) if fields else json.dumps(entry.get("protoPayload", {}), ensure_ascii=False)


def entry_record(entry: dict, sources: list[dict], run_id: str) -> dict | None:
  """One log entry as a dashboard record, or None when it belongs to no source lifetime."""
  labels = entry.get("resource", {}).get("labels", {})
  at = timestamp(entry["timestamp"])
  match = next((s for s in sources if s["pod"] == labels.get("pod_name") and s["created_at"] <= at <= s["until"]), None)
  if match is None:
    return None
  fields = entry.get("jsonPayload") or {}
  text = entry_text(entry)
  identity = hashlib.sha256(json.dumps([entry.get("logName"), entry.get("insertId"), entry.get("timestamp"), text]).encode()).hexdigest()
  return {
    "id": identity,
    "run_id": run_id,
    "timestamp": entry.get("timestamp"),
    "pod": match["pod"],
    "container": labels.get("container_name"),
    "node": match.get("node"),
    "role": match.get("role", "unknown"),
    "severity": entry.get("severity", "UNKNOWN"),
    "rank": fields.get("rank"),
    "request_id": fields.get("request_id"),
    "message": text[:MAX_MESSAGE],
    "message_truncated": len(text) > MAX_MESSAGE,
  }


async def run_logs(
  run_id: str, sources: list[dict], *, since=None, until=None, cursor=None, limit=200, q="", pod=None, node=None, severity=None
) -> dict:
  result = {"run_id": run_id, "source": "gke", "records": [], "sources": sources, "order": "newest_first", "next_cursor": None, "available": False}
  if not configuration()["configured"]:
    return {**result, "error": "GKE telemetry is not configured"}
  if refused():
    return {**result, "error": REFUSED}
  selected = [s for s in sources if (pod is None or s["pod"] == pod) and (node is None or s.get("node") == node)]
  if not selected:
    return {**result, "error": "No pod lifetimes match this run and filter"}
  if len(selected) > MAX_SOURCES:
    return {**result, "error": f"Select a pod to narrow this query to at most {MAX_SOURCES} pod lifetimes"}
  start, end = time_range(since, until)
  body = {
    "resourceNames": [f"projects/{configuration()['project']}"],
    "filter": logs_filter(selected, start, end, severity, q),
    "orderBy": "timestamp desc",
    "pageSize": limit,
  }
  if cursor:
    body["pageToken"] = cursor
  try:
    payload = await request("POST", "https://logging.googleapis.com/v2/entries:list", json=body)
    records = [record for entry in payload.get("entries", []) if (record := entry_record(entry, selected, run_id))]
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code in (401, 403):
      note_refusal()
      return {**result, "error": REFUSED}
    if exc.response.status_code == 400 and cursor:
      raise ValueError("Invalid or expired log cursor; repeat the query without it") from None
    return {**result, "error": UNAVAILABLE}
  except Exception:
    return {**result, "error": UNAVAILABLE}
  return {**result, "records": records, "available": True, "next_cursor": payload.get("nextPageToken")}
