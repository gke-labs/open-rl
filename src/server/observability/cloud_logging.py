"""A run's logs from Cloud Logging, read with the server's Google credentials."""

import asyncio
import json
import re
import threading
from datetime import UTC, datetime, timedelta

import httpx

from server.observability.log_selector import ClientLines, Selector

# GKE files every stderr line as ERROR, though most are progress bars, INFO logs and warnings; such a line takes the level it prints.
RELABELED = ("DEBUG", "INFO", "WARNING")
LEVEL, PROGRESS = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b"), re.compile(r"\r|\d+%(\||\s+Completed)")
MESSAGE_CHARS = 16384
client: httpx.AsyncClient | None = None
credentials = None
project: str | None = None  # the one the credentials name
credentials_lock = threading.Lock()


async def send(method: str, url: str, **kwargs) -> httpx.Response:
  global client
  client = client or httpx.AsyncClient(timeout=10)
  return (await client.request(method, url, **kwargs)).raise_for_status()


def access_token() -> str:
  import google.auth  # the cluster extra installs it
  from google.auth.transport.requests import Request

  global credentials, project
  with credentials_lock:
    if credentials is None:
      credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not credentials.valid:
      credentials.refresh(Request())
    return credentials.token


async def gcp_headers() -> dict[str, str]:
  import google.oauth2.credentials

  # A hung token refresh holds the lock; bound it rather than start a thread per call.
  headers = {"Authorization": f"Bearer {await asyncio.wait_for(asyncio.to_thread(access_token), timeout=10)}"}
  # A person's gcloud login draws on gcloud's shared quota and fails with 429 unless it bills its project.
  if isinstance(credentials, google.oauth2.credentials.Credentials) and not credentials.quota_project_id and project:
    headers["x-goog-user-project"] = project
  return headers


def window(since: str | None, until: str | None) -> tuple[str, str]:
  """ISO-8601 or Unix seconds; since also takes a span before until, such as 1h. Default: the 30 minutes ending now; at most seven days."""

  def parse(value: str) -> datetime:
    parsed = datetime.fromtimestamp(float(value), UTC) if re.fullmatch(r"\d+(\.\d+)?", value) else datetime.fromisoformat(value)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)

  end = parse(until) if until else datetime.now(UTC)
  if ago := re.fullmatch(r"(\d+)([mhd])", since or ""):
    start = end - timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[ago[2]]: int(ago[1])})
  else:
    start = parse(since) if since else end - timedelta(minutes=30)
  if not timedelta(0) < end - start <= timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")


def logs_filter(selector: Selector, project: str, start: str, end: str, severity: str | None, q: str) -> str:
  """The selected entries in the window. Worker pods are the namespace's pods whose names start with the run's prefix, so
  a run id cannot open another run's pods. GKE copies a client pod's openrl_client label to k8s-pod/openrl_client; a
  client outside Kubernetes sets openrl_client itself."""
  if isinstance(selector, ClientLines):
    client_id = json.dumps(selector.client_id)
    clauses = [f'(labels."k8s-pod/openrl_client"={client_id} OR labels."openrl_client"={client_id})']
  else:
    clauses = ['resource.type="k8s_container"', f"resource.labels.namespace_name={json.dumps(selector.namespace)}"]
    clauses.append(f"resource.labels.pod_name=~{json.dumps('^' + selector.prefix)}")
    if selector.pod:
      clauses.append(f"resource.labels.pod_name={json.dumps(selector.pod)}")
  clauses += [f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity in RELABELED:
    stderr = json.dumps(f"projects/{project}/logs/stderr")
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
  role = labels.get("k8s-pod/openrl_io/role") or labels.get("k8s-pod/app_kubernetes_io/component")
  role = role or ("client" if {"openrl_client", "k8s-pod/openrl_client"} & labels.keys() else "unknown")
  record = {"id": f"{entry.get('logName')}/{entry.get('insertId')}/{entry.get('timestamp')}", "timestamp": entry.get("timestamp")}
  record.update(pod=resource.get("pod_name"), role=role, node=labels.get("compute.googleapis.com/resource_name"))
  return record | {"severity": severity, "message": text[:MESSAGE_CHARS], "message_truncated": len(text) > MESSAGE_CHARS}


async def log_page(resource: str | None, selector: Selector, since: str, until: str, *, cursor=None, limit=200, q="", severity=None) -> dict:
  """A page of logs, newest first, from the Cloud Logging resource, by default the credentials' project. The cursor is
  Cloud Logging's own page token, valid for the same filters."""
  headers = await gcp_headers()  # loads the credentials and their project
  if not (resource := resource or (f"projects/{project}" if project else "")):
    raise LookupError("The Google credentials name no project; set OPEN_RL_LOG_STORE=cloud-logging:projects/<project>")
  owner = resource.split("/")[1] if resource.startswith("projects/") else project
  filters = logs_filter(selector, owner or "", since, until, severity, q)
  query = {"resourceNames": [resource], "filter": filters, "orderBy": "timestamp desc", "pageSize": limit}
  query |= {"pageToken": cursor} if cursor else {}
  payload = (await send("POST", "https://logging.googleapis.com/v2/entries:list", json=query, headers=headers)).json()
  records = [entry_record(entry) for entry in payload.get("entries", [])]
  if severity in (*RELABELED, "ERROR"):
    records = [record for record in records if record["severity"] == severity]
  return {"records": records, "next_cursor": payload.get("nextPageToken")}
