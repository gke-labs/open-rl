"""Serve the autoresearch browser UI."""
# ruff: noqa: E501

from __future__ import annotations

import http.server
import json
import socketserver
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

import chz

from harness.utils import AGENT_ARTIFACTS, ATTEMPT_ARTIFACTS, AgentPaths, agent_id, run_id

UI_DIR = Path(__file__).parents[1] / "ui"
STALE_GRACE_SECONDS = 60
TAIL_CHUNK_BYTES = 256 * 1024


@dataclass(frozen=True)
class FileTarget:
  path: Path
  start: int = 0
  end: int | None = None


def checked_name(value: str, field: str) -> str:
  if not value or value in {".", ".."} or "/" in value or "\\" in value:
    raise ValueError(f"invalid {field}")
  return value


def query_int(query: dict[str, list[str]], name: str, default: int = 0) -> int:
  try:
    return max(int(query.get(name, [str(default)])[0]), 0)
  except ValueError:
    return default


def agent_paths(log_root: Path, run: str, agent: str) -> AgentPaths:
  if agent_id(agent) != agent:
    raise ValueError("invalid agent")
  return AgentPaths.from_run(log_root, checked_name(run, "run"), agent)


def read_bytes(
  target: Path,
  offset: int = 0,
  tail_bytes: int = 0,
  max_bytes: int = 0,
  start_offset: int = 0,
  end_offset: int | None = None,
) -> tuple[int, int, bytes]:
  """Read `target` from `offset` (or its last `tail_bytes`) to EOF; returns (start, end, data)."""
  size = target.stat().st_size
  limit = min(size, end_offset) if end_offset is not None else size
  floor = min(max(0, start_offset), limit)
  start = max(floor, limit - tail_bytes) if tail_bytes else min(max(floor, offset), limit)
  with target.open("rb") as f:
    f.seek(start)
    length = limit - start
    if max_bytes:
      length = min(length, max_bytes)
    data = f.read(length)
  return start, start + len(data), data


def latest_mtime(paths: list[Path], fallback: float) -> float:
  latest = fallback
  for path in paths:
    if path.exists():
      latest = max(latest, path.stat().st_mtime)
  return latest


def stale_status(status: str, latest: float, timeout_minutes: float | None) -> str:
  if status != "running" or not timeout_minutes:
    return status
  limit = float(timeout_minutes) * 60 + STALE_GRACE_SECONDS
  return "stale" if time.time() - latest > limit else status


def ui_agent_id(run: str, agent: str) -> str:
  return f"{run}/{agent}" if run else agent


def agent_label(run: str, agent: str) -> str:
  if not run:
    return agent
  return run if run == agent or run.endswith(f"-{agent}") else f"{run}/{agent}"


def agent_record(path: Path, run: str = "") -> dict:
  data = json.loads(path.read_text(encoding="utf-8"))
  agent = path.parent.name
  started_at = float(data["started_at"])
  latest = latest_mtime([path.parent / name for name in AGENT_ARTIFACTS.values()], started_at)
  status = stale_status(data["status"], latest, data["agent_timeout_minutes"])
  return {
    "id": ui_agent_id(run, agent),
    "label": agent_label(run, agent),
    "run": run,
    "artifact_agent": agent,
    "status": status,
    "started_at": started_at,
    "finished_at": data["finished_at"],
    "spec_hash": data["spec_hash"],
    "agent_model": data["agent_model"],
    "agent_timeout_minutes": data["agent_timeout_minutes"],
    "exit_code": data["exit_code"],
  }


def attempt_record(path: Path, run: str = "") -> dict:
  data = json.loads(path.read_text(encoding="utf-8"))
  agent = path.parents[2].name
  attempt = path.parent.name
  started_at = float(data["started_at"])
  latest = latest_mtime([path.parent / name for name in ATTEMPT_ARTIFACTS.values()], started_at)
  status = stale_status(data["status"], latest, data["attempt_timeout_minutes"])
  return {
    "agent": ui_agent_id(run, agent),
    "run": run,
    "artifact_agent": agent,
    "id": attempt,
    "status": status,
    "baseline": data["baseline"],
    "started_at": started_at,
    "finished_at": data["finished_at"],
    "attempt_timeout_minutes": data["attempt_timeout_minutes"],
    "agent_log_start": int(data.get("agent_log_start", 0) or 0),
    "agent_log_end": data.get("agent_log_end"),
    "spec_hash": data["spec_hash"],
    "branch": data["branch"],
    "commit": data["commit"],
    "parent": data["parent"],
    "description": data["description"],
    "exit_code": data["exit_code"],
    "error": data["error"],
    "metric": data["metric"],
  }


def run_roots(log_root: Path, run: str = "") -> list[tuple[str, Path]]:
  if run:
    return [(run, log_root / checked_name(run, "run"))]
  if not log_root.exists():
    return []
  return [(path.name, path) for path in sorted(log_root.iterdir()) if (path / "researchers").is_dir()]


def metric_title(value: str = "score") -> str:
  return str(value or "score").split("/")[-1].replace("_", " ").replace("-", " ").title()


def metric_axis(value: str = "score") -> str:
  return str(value or "score").split("/")[-1].replace("_", " ").lower()


def spec_meta(spec_hashes: list[str]) -> str:
  unique = sorted(set(value for value in spec_hashes if value))
  if len(unique) > 1:
    return "mixed spec"
  if not unique:
    return "missing spec"
  return "spec " + unique[0].removeprefix("sha256:")[:8]


def artifact_url(route: str, run: str, agent: str, artifact: str, attempt: str = "") -> str:
  query = {"run": run, "agent": agent, "artifact": artifact}
  if attempt:
    query["attempt"] = attempt
  return f"{route}?{urlencode(query)}"


def artifact_ref(run: str, agent: str, attempt: str, artifact: str, live: bool = False) -> dict:
  ref = {
    "label": {"agent": "Agent", "logs": "Logs", "diff": "Diff"}[artifact],
    "format": artifact,
    "url": artifact_url("file", run, agent, artifact, attempt),
    "live": live,
  }
  if artifact in {"agent", "logs"}:
    ref["tail_url"] = artifact_url("tail", run, agent, artifact, attempt)
  return ref


def text_tail(path: Path, tail_bytes: int = TAIL_CHUNK_BYTES) -> tuple[str, int]:
  if not path.is_file():
    return "", 0
  _, end, data = read_bytes(path, tail_bytes=tail_bytes)
  return data.decode("utf-8", errors="replace"), end


def notes_text(path: Path) -> str:
  if not path.is_file():
    return ""
  return path.read_text(encoding="utf-8", errors="replace")


def snapshot(log_root: Path, run: str = "", include_text: bool = True) -> dict:
  log_root.mkdir(parents=True, exist_ok=True)
  roots = run_roots(log_root, run)
  entries = []
  for name, root in roots:
    for path in sorted(root.glob("researchers/*/metadata.json")):
      attempts = [attempt_record(item, name) for item in sorted(path.parent.glob("attempts/*/metadata.json"))]
      attempts.sort(key=lambda row: (row["started_at"], row["id"]))
      agent = agent_record(path, name)
      live = agent["status"] == "running" or any(attempt["status"] == "running" for attempt in attempts)
      attempts_word = "attempt" if len(attempts) == 1 else "attempts"
      count = "running" if live else f"{len(attempts)} {attempts_word}"
      spec_hashes = [agent.get("spec_hash", ""), *(attempt.get("spec_hash", "") for attempt in attempts)]
      entries.append(
        (
          {
            **agent,
            "live": live,
            "status": "running" if live else "complete" if agent["status"] == "completed" else agent["status"],
            "meta": f"{count} · {spec_meta(spec_hashes)}",
          },
          attempts,
        )
      )

  entries.sort(key=lambda row: (row[0]["started_at"], row[0]["id"]))
  agents = [agent for agent, _ in entries]
  rows = []

  for agent, attempts in entries:
    metric = next((attempt["metric"]["name"] for attempt in attempts if attempt.get("metric")), "score")
    paths = AgentPaths.from_run(log_root, agent["run"], agent["artifact_agent"])
    agent_tail, agent_offset = text_tail(paths.agent_log) if include_text else ("", paths.agent_log.stat().st_size if paths.agent_log.exists() else 0)
    activity_id = f"{agent['id']}:activity"
    rows.append(
      {
        "id": activity_id,
        "agent_id": agent["id"],
        "run": agent["run"],
        "artifact_agent": agent["artifact_agent"],
        "kind": "activity",
        "label": "Activity",
        "number": "",
        "status": agent["status"],
        "live": agent["status"] == "running",
        "score": None,
        "description": f"agent {agent['status']}",
        "metric_label": metric_title(metric),
        "axis_label": metric_axis(metric),
        "tab_order": ["agent", "notes"],
        "meta": f"Activity · {agent['label']} · agent {agent['status']}",
        "agent": agent_tail,
        "agent_offset": agent_offset,
        "notes": notes_text(paths.notes) if include_text else "",
      }
    )

    for index, attempt in enumerate(attempts, start=1):
      metric_data = attempt.get("metric") or {}
      score = None if attempt["status"] == "running" else metric_data.get("value")
      metric_name = metric_data.get("name") or metric
      metric_label = metric_title(metric_name)
      commit = f" · {attempt['commit']}" if attempt.get("commit") else ""
      score_text = f" · {metric_label} {score:.4g}" if isinstance(score, int | float) else ""
      tab_order = ["agent", "logs"] if attempt["baseline"] else ["agent", "logs", "diff"]
      next_attempt = attempts[index] if index < len(attempts) else None
      agent_log_end = attempt.get("agent_log_end")
      if agent_log_end is None and next_attempt and next_attempt.get("agent_log_start") is not None:
        agent_log_end = next_attempt["agent_log_start"]
      agent_live = agent["status"] == "running" and index == len(attempts) and agent_log_end is None
      artifacts = {
        "agent": artifact_ref(attempt["run"], attempt["artifact_agent"], attempt["id"], "agent", agent_live),
        "logs": artifact_ref(attempt["run"], attempt["artifact_agent"], attempt["id"], "logs", attempt["status"] == "running"),
      }
      if not attempt["baseline"]:
        artifacts["diff"] = artifact_ref(attempt["run"], attempt["artifact_agent"], attempt["id"], "diff")
      rows.append(
        {
          "id": f"{agent['id']}:{attempt['id']}",
          "agent_id": agent["id"],
          "attempt_id": attempt["id"],
          "kind": "attempt",
          "label": f"Attempt {index}",
          "number": index,
          "status": attempt["status"],
          "live": attempt["status"] == "running",
          "score": score,
          "score_mode": metric_data.get("mode") or "max",
          "description": "Default config" if attempt["baseline"] else attempt.get("description") or attempt["id"],
          "metric_label": metric_label,
          "axis_label": metric_axis(metric_name),
          "tab_order": tab_order,
          "artifacts": artifacts,
          "agent_log_start": attempt.get("agent_log_start"),
          "agent_log_end": agent_log_end,
          "meta": f"Attempt {index} · {agent['label']} · {attempt['status']}{score_text}{commit}",
          "started_at": attempt["started_at"],
        }
      )

  return {"task": run or "all runs", "runs": [name for name, _ in roots], "agents": agents, "rows": rows}


def snapshot_fingerprint(data: dict) -> str:
  compact = {
    **data,
    "rows": [{key: value for key, value in row.items() if key not in {"agent", "agent_offset"}} for row in data["rows"]],
  }
  return json.dumps(compact, sort_keys=True, allow_nan=False, separators=(",", ":"))


@chz.chz
class UiConfig:
  log_root: Path = Path("artifacts/autoresearch")
  run_name: str = ""
  interval_seconds: float = 15
  host: str = "0.0.0.0"
  port: int = 8080
  serve: bool = False


class Handler(http.server.SimpleHTTPRequestHandler):
  log_root: Path
  run_name: str
  interval_seconds: float

  def log_message(self, format: str, *args) -> None:
    return

  def send_data(self, data: bytes, content_type: str = "application/json", status_code: int = 200, headers: dict[str, str] | None = None) -> None:
    self.send_response(status_code)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    extra = headers or {}
    if "Cache-Control" not in extra:
      self.send_header("Cache-Control", "no-store, max-age=0")
    for key, value in extra.items():
      self.send_header(key, value)
    self.end_headers()
    self.wfile.write(data)

  def stream_sse(self, source) -> None:
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.send_header("X-Accel-Buffering", "no")
      self.end_headers()
      for item in source:
        if item is None:
          payload = b": keepalive\n\n"
        elif isinstance(item, tuple):
          event, data = item
          payload = f"event: {event}\ndata: {data}\n\n".encode()
        else:
          payload = f"data: {item}\n\n".encode()
        self.wfile.write(payload)
        self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
      return

  def do_GET(self) -> None:
    path = urlsplit(self.path).path
    if path == "/events":
      return self.events()
    if path == "/tail":
      return self.tail()
    if path == "/file":
      return self.send_file()
    return self.send_static(path)

  def send_static(self, path: str) -> None:
    filename = "index.html" if path in {"/", "/experiments.html"} else unquote(path).lstrip("/")
    target = (UI_DIR / filename).resolve()
    try:
      target.relative_to(UI_DIR.resolve())
    except ValueError:
      return self.send_error(404, "file not found")
    if not target.is_file():
      return self.send_error(404, "file not found")
    return self.send_data(target.read_bytes(), self.guess_type(str(target)))

  def attempt_agent_target(self, paths: AgentPaths, attempt: str) -> FileTarget:
    attempt_dir = paths.attempt_dir(checked_name(attempt, "attempt"))
    manifest_path = attempt_dir / "metadata.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = int(data.get("agent_log_start", 0) or 0)
    end = data.get("agent_log_end")
    if end is None:
      current = (float(data.get("started_at", 0) or 0), attempt)
      for path in sorted(paths.attempts.glob("*/metadata.json")):
        if path.parent.name == attempt:
          continue
        row = json.loads(path.read_text(encoding="utf-8"))
        next_start = row.get("agent_log_start")
        if next_start is None:
          continue
        if (float(row.get("started_at", 0) or 0), path.parent.name) > current:
          end = int(next_start)
          break
    return FileTarget(paths.agent_log, start, int(end) if end is not None else None)

  def target(self) -> FileTarget | None:
    query = parse_qs(urlsplit(self.path).query)
    run = self.run_name or query.get("run", [""])[0]
    agent = query.get("agent", query.get("researcher", [""]))[0]
    artifact = query.get("artifact", [""])[0]
    attempt = query.get("attempt", [""])[0]
    if not run or not agent or not artifact:
      self.send_error(400, "missing run, agent, or artifact")
      return None
    try:
      paths = agent_paths(self.log_root, run, agent)
      if attempt:
        if artifact == "agent":
          return self.attempt_agent_target(paths, attempt)
        attempt_dir = paths.attempt_dir(checked_name(attempt, "attempt"))
        return FileTarget(attempt_dir / ATTEMPT_ARTIFACTS[artifact])
      return FileTarget(paths.root / AGENT_ARTIFACTS[artifact])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
      self.send_error(404, "unknown artifact")
      return None

  def send_file(self) -> None:
    target = self.target()
    if target is None:
      return
    if not target.path.is_file():
      return self.send_error(404, "file not found")
    tail_bytes = query_int(parse_qs(urlsplit(self.path).query), "tail_bytes")
    start, end, data = read_bytes(target.path, tail_bytes=tail_bytes, start_offset=target.start, end_offset=target.end)
    self.send_data(data, "text/plain; charset=utf-8", headers={"X-OpenRL-Start": str(start), "X-OpenRL-End": str(end)})

  def tail(self) -> None:
    target = self.target()
    if target is None:
      return
    offset = query_int(parse_qs(urlsplit(self.path).query), "offset", target.start)

    def source():
      nonlocal offset
      while True:
        payload = None
        if target.path.is_file():
          chunk_start, offset, data = read_bytes(
            target.path, offset=offset, max_bytes=TAIL_CHUNK_BYTES, start_offset=target.start, end_offset=target.end
          )
          if data:
            payload = json.dumps({"start": chunk_start, "end": offset, "text": data.decode("utf-8", errors="replace")}, separators=(",", ":"))
        yield payload
        time.sleep(1)

    self.stream_sse(source())

  def events(self) -> None:
    def source():
      state = snapshot(self.log_root, self.run_name)
      last = snapshot_fingerprint(state)
      offsets = {row["id"]: row.get("agent_offset", 0) for row in state["rows"] if row["kind"] == "activity"}
      yield "snapshot", json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":"))

      while True:
        emitted = False
        for row in state["rows"]:
          if row["kind"] != "activity":
            continue
          paths = AgentPaths.from_run(self.log_root, row["run"], row["artifact_agent"])
          offset = offsets.get(row["id"], row.get("agent_offset", 0))
          if paths.agent_log.is_file():
            _, end, data = read_bytes(paths.agent_log, offset=offset, max_bytes=TAIL_CHUNK_BYTES)
            if data:
              offsets[row["id"]] = end
              patch = {"id": row["id"], "agent": data.decode("utf-8", errors="replace"), "agent_offset": end}
              yield "row", json.dumps(patch, sort_keys=True, allow_nan=False, separators=(",", ":"))
              emitted = True

        latest = snapshot(self.log_root, self.run_name)
        text = snapshot_fingerprint(latest)
        if text != last:
          state = latest
          last = text
          offsets = {row["id"]: row.get("agent_offset", 0) for row in state["rows"] if row["kind"] == "activity"}
          yield "snapshot", json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":"))
        elif not emitted:
          yield None
        time.sleep(self.interval_seconds)

    self.stream_sse(source())


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
  allow_reuse_address = True
  daemon_threads = True


def serve(log_root: Path, run_name: str, host: str, port: int, interval: float) -> None:
  Handler.log_root = log_root
  Handler.run_name = run_name
  Handler.interval_seconds = interval
  print(f"serving UI at http://{host}:{port}/", flush=True)
  ThreadingTCPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
  args = chz.entrypoint(UiConfig, argv=argv, allow_hyphens=True)
  root = args.log_root.resolve()
  run_name = run_id(args.run_name) if args.run_name else ""
  if args.serve:
    serve(root, run_name, args.host, args.port, args.interval_seconds)
  else:
    print(json.dumps(snapshot(root, run_name, include_text=False), indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
