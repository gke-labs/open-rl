"""Serve the dashboard UI over a recorded fixture instead of a cluster.

Static assets come from the working tree, so edits show on reload. API
requests are answered from dev/fixtures/dashboard, captured from a real
gateway by dev/capture_dashboard_fixture.py. Time and log filters operate
only on recorded rows; pagination never requests another Cloud Logging page.
The captured files remain unchanged, and missing responses are explicit.

  python dev/dashboard_fixture.py --port 9017
"""

import argparse
import hashlib
import http.server
import json
import pathlib
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "server" / "dashboard" / "static"
API = "/api/v1/dashboard"
TYPES = {".css": "text/css", ".js": "text/javascript", ".html": "text/html", ".json": "application/json"}


def epoch(value: str) -> float:
  parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
  return parsed.replace(tzinfo=parsed.tzinfo or UTC).timestamp()


def replay(data: dict, path: str, query: dict, captured_at: str) -> dict:
  data = {**data, "recorded_at": captured_at}
  start = epoch(query["since"]) if query.get("since") else float("-inf")
  end = epoch(query["until"]) if query.get("until") else float("inf")
  if start >= end:
    raise ValueError("Time range must end after it starts")
  if path.endswith("/logs"):
    limit = int(query.get("limit", 200))
    if not 1 <= limit <= 1000:
      raise ValueError("Log limit must be between 1 and 1000")
    original = data.get("records", [])
    times = [epoch(row["timestamp"]) for row in original]
    before = epoch(query["before"]) if query.get("before") else float("inf")
    rows = [
      row
      for row, at in zip(original, times)
      if start <= at <= end
      and at < before
      and query.get("q", "").casefold() in row.get("message", "").casefold()
      and all(not query.get(key) or row.get(key) == query[key] for key in ("pod", "node", "container", "severity"))
    ]
    rows.sort(key=lambda row: epoch(row["timestamp"]), reverse=True)
    scope = json.dumps([captured_at, path, {key: value for key, value in query.items() if key != "cursor"}], sort_keys=True)
    prefix = "recording-" + hashlib.sha256(scope.encode()).hexdigest()[:16] + ":"
    cursor = query.get("cursor", prefix + "0")
    if not cursor.startswith(prefix) or not cursor[len(prefix) :].isdigit():
      raise ValueError("Log cursor is not from this recording and query; repeat the query without it")
    offset = int(cursor[len(prefix) :])
    data.update(
      records=rows[offset : offset + limit],
      next_cursor=prefix + str(offset + limit) if offset + limit < len(rows) else None,
      order="newest_first",
      coverage={
        "recorded_records": len(original),
        "matching_records": len(rows),
        "since": original[times.index(min(times))]["timestamp"] if times else None,
        "until": original[times.index(max(times))]["timestamp"] if times else None,
        "older_not_recorded": bool(data.get("next_cursor")),
      },
    )
  elif path.endswith("/turns"):
    # Keep whole turns overlapping the window, including those that finish
    # after its end; the timeline clips their visible geometry locally.
    data["samples"] = [sample for sample in data.get("samples", []) if sample["at"] >= start and sample["started_at"] <= end]
    data["available"] = bool(data["samples"])
  elif path.endswith("/metrics"):
    if "samples" in data:
      data["samples"] = [sample for sample in data["samples"] if start <= sample["at"] <= end]
      data["available"] = bool(data["samples"])
    if "devices" in data:
      data["devices"] = [
        {**device, **{key: [sample for sample in device.get(key, []) if start <= sample[0] <= end] for key in ("utilization", "memory_mib")}}
        for device in data["devices"]
      ]
      data["available"] = any(device["utilization"] for device in data["devices"])
  return data


def make_handler(fixture: pathlib.Path):
  manifest = json.loads((fixture / "manifest.json").read_text())
  files = manifest["files"]

  class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
      pass

    def send(self, status: int, body: bytes, content_type: str) -> None:
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      request = urlparse(self.path)
      path = request.path
      if path in ("/", "/dashboard", "/dashboard/"):
        return self.send(200, (STATIC / "index.html").read_bytes(), "text/html")
      if path.startswith("/dashboard/assets/"):
        asset = (STATIC / path[len("/dashboard/assets/") :]).resolve()
        if asset.parent == STATIC.resolve() and asset.is_file():
          return self.send(200, asset.read_bytes(), TYPES.get(asset.suffix, "application/octet-stream"))
        return self.send(404, b"not found", "text/plain")
      if path == "/api/v1/healthz":
        return self.send(200, b'{"status":"ok"}', "application/json")
      if path in files:
        try:
          data = replay(json.loads((fixture / files[path]).read_text()), path, dict(parse_qsl(request.query)), manifest["captured_at"])
        except ValueError as exc:
          return self.send(400, json.dumps({"detail": str(exc)}).encode(), "application/json")
        return self.send(200, json.dumps(data).encode(), "application/json")
      resource = path.removeprefix(API + "/").split("/")
      label = "Run" if resource[0] == "runs" else "Allocation" if resource[0] == "allocations" else "Response"
      detail = f"{label} {resource[2] if len(resource) > 2 else 'details'} were not recorded"
      if len(resource) > 1:
        detail += f" for {resource[1]}"
      missing = {"detail": detail, "error": detail, "recorded_at": manifest["captured_at"]}
      return self.send(404, json.dumps(missing).encode(), "application/json")

  return Handler


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--fixture", default=str(ROOT / "dev" / "fixtures" / "dashboard"))
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=9017)
  args = parser.parse_args()
  fixture = pathlib.Path(args.fixture)
  manifest = json.loads((fixture / "manifest.json").read_text())
  print(f"fixture captured {manifest['captured_at']} from {manifest['base']}; {len(manifest['files'])} responses")
  print(f"http://{args.host}:{args.port}/dashboard")
  http.server.ThreadingHTTPServer((args.host, args.port), make_handler(fixture)).serve_forever()


if __name__ == "__main__":
  main()
