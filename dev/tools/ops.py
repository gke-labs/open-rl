#!/usr/bin/env python3
# JSON CLI for operating the cluster. Exposes the same primitives as the /dashboard UI —
# health, problems, inspect, runs, logs, launch, stop — and always prints JSON so agents
# and scripts can consume the output directly. Stdlib only; no extra dependencies.

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def base_url() -> str:
  return os.environ.get("BASE_URL", "http://localhost:9003").rstrip("/")


def request(method: str, path: str, body: dict | None = None) -> dict:
  url = f"{base_url()}{path}"
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=30) as resp:
      return json.load(resp)
  except urllib.error.HTTPError as exc:
    try:
      return {"error": True, "status": exc.code, **json.load(exc)}
    except Exception:
      return {"error": True, "status": exc.code, "message": exc.reason}
  except urllib.error.URLError as exc:
    return {"error": True, "message": f"gateway unreachable at {base_url()}: {exc.reason}"}


def emit(payload: dict) -> None:
  print(json.dumps(payload, indent=2))
  if payload.get("error"):
    sys.exit(1)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL cluster operations (JSON output). Set BASE_URL to target a gateway.")
  sub = parser.add_subparsers(dest="command", required=True)

  sub.add_parser("health", help="Gateway, storage, Kubernetes, and visibility checks")
  sub.add_parser("problems", help="Everything currently wrong, most severe first")
  sub.add_parser("inspect", help="Cluster snapshot: pools, nodes, pods, gateway, services")
  sub.add_parser("runs", help="List runs with lifecycle state")

  run = sub.add_parser("run", help="Everything about one run: state, pods, queue depth, GPU claims, optional logs")
  run.add_argument("run_id")
  run.add_argument("--logs", type=int, default=0, metavar="N", help="Include the last N log lines per pod")

  logs = sub.add_parser("logs", help="Fetch logs for a pod")
  logs.add_argument("pod")
  logs.add_argument("--container")
  logs.add_argument("--tail", type=int, default=500)

  launch = sub.add_parser("launch", help="Launch a run (create_model)")
  launch.add_argument("--base-model", required=True)

  stop = sub.add_parser("stop", help="Stop a run: its worker, queued work, and pods")
  stop.add_argument("run_id")

  args = parser.parse_args()

  if args.command == "health":
    emit(request("GET", "/api/v1/dashboard/health"))
  elif args.command == "problems":
    emit(request("GET", "/api/v1/dashboard/problems"))
  elif args.command == "inspect":
    emit(request("GET", "/api/v1/dashboard/cluster"))
  elif args.command == "runs":
    emit(request("GET", "/api/v1/dashboard/runs"))
  elif args.command == "run":
    path = f"/api/v1/dashboard/runs/{urllib.parse.quote(args.run_id)}"
    if args.logs:
      path += f"?logs={args.logs}"
    emit(request("GET", path))
  elif args.command == "logs":
    params = {"tail": str(args.tail)}
    if args.container:
      params["container"] = args.container
    emit(request("GET", f"/api/v1/dashboard/pods/{urllib.parse.quote(args.pod)}/logs?{urllib.parse.urlencode(params)}"))
  elif args.command == "launch":
    emit(request("POST", "/api/v1/create_model", {"base_model": args.base_model}))
  elif args.command == "stop":
    emit(request("POST", f"/api/v1/dashboard/runs/{urllib.parse.quote(args.run_id)}/stop"))


if __name__ == "__main__":
  main()
