"""Record the dashboard API from a live gateway into a fixture directory.

The fixture is real data: whatever the cluster showed at capture time. The
fixture server in dev/dashboard_fixture.py replays it for UI work without a
cluster.

  python dev/capture_dashboard_fixture.py --base http://127.0.0.1:18000 --out dev/fixtures/dashboard
"""

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

API = "/api/v1/dashboard"


def fetch(base: str, path: str) -> dict:
  with urllib.request.urlopen(base + path, timeout=60) as response:
    return json.load(response)


def name_for(path: str) -> str:
  """The file a request path is stored under: the path without its query, slashes as dashes."""
  return urllib.parse.urlparse(path).path[len(API) :].strip("/").replace("/", "--") or "index"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--base", default="http://127.0.0.1:18000")
  parser.add_argument("--out", default="dev/fixtures/dashboard")
  parser.add_argument("--runs", type=int, default=3, help="how many runs to record in detail")
  args = parser.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  paths = [API, f"{API}/snapshot", f"{API}/experiments"]
  snapshot = fetch(args.base, f"{API}/snapshot")
  runs = [r for r in snapshot["runs"] if r["display_status"] == "Running"][: args.runs] or snapshot["runs"][: args.runs]
  for run in runs:
    rid = run["run_id"]
    paths += [f"{API}/runs/{rid}", f"{API}/runs/{rid}/metrics", f"{API}/runs/{rid}/logs?limit=200"]
  for placement in snapshot["placements"][: args.runs * 2]:
    paths += [f"{API}/allocations/{placement['id']}/{resource}" for resource in ("metrics", "turns")]

  recorded = {}
  for path in paths:
    try:
      body = fetch(args.base, path)
    except Exception as exc:
      print(f"skip {path}: {exc}")
      continue
    name = name_for(path)
    (out / f"{name}.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    recorded[urllib.parse.urlparse(path).path] = f"{name}.json"
    print(f"{path} -> {name}.json ({len(json.dumps(body)) // 1024} KiB)")
  (out / "manifest.json").write_text(
    json.dumps({"base": args.base, "captured_at": snapshot["observed_at"], "files": recorded}, indent=1, sort_keys=True)
  )


if __name__ == "__main__":
  main()
