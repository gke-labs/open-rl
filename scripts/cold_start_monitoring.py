"""Filestore read throughput and IOPS from Cloud Monitoring over a time window.

  python3 scripts/cold_start_monitoring.py --start 2026-10-15T03:00:00Z --end 2026-10-15T03:10:00Z --out runs/<dir>/<case>

Reads file.googleapis.com/nfs/server/read_bytes_count and read_ops_count for
every Filestore instance in the project, aligned to 60 s rates (the finest
Filestore publishes), and writes monitoring.json plus a per-minute table.
Uses application-default credentials (`gcloud auth application-default
print-access-token`); the plain user token is refused by this API.
"""

import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path

PROJECT = "cdrollouts-sunilarora"
METRICS = {
  "read_bytes_count": "file.googleapis.com/nfs/server/read_bytes_count",
  "read_ops_count": "file.googleapis.com/nfs/server/read_ops_count",
  "write_bytes_count": "file.googleapis.com/nfs/server/write_bytes_count",
}


def token() -> str:
  return subprocess.run(["gcloud", "auth", "application-default", "print-access-token"], check=True, capture_output=True, text=True).stdout.strip()


def fetch(metric: str, start: str, end: str, tok: str) -> dict:
  # curl rather than urllib: the system Python here has no CA bundle.
  params = {
    "filter": f'metric.type="{metric}"',
    "interval.startTime": start,
    "interval.endTime": end,
    "aggregation.alignmentPeriod": "60s",
    "aggregation.perSeriesAligner": "ALIGN_RATE",
  }
  url = f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries?" + urllib.parse.urlencode(params)
  out = subprocess.run(["curl", "-sS", "-H", f"Authorization: Bearer {tok}", url], check=True, capture_output=True, text=True).stdout
  return json.loads(out)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--start", required=True)
  parser.add_argument("--end", required=True)
  parser.add_argument("--out", required=True, type=Path)
  args = parser.parse_args()
  tok = token()
  result = {name: fetch(metric, args.start, args.end, tok) for name, metric in METRICS.items()}
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "monitoring.json").write_text(json.dumps(result, indent=1))
  per_minute: dict[str, dict[str, float]] = {}
  for name, body in result.items():
    for series in body.get("timeSeries", []):
      for point in series["points"]:
        row = per_minute.setdefault(point["interval"]["endTime"], {})
        row[name] = float(point["value"].get("doubleValue", 0.0))
  lines = ["| minute ending (UTC) | read MiB/s | read ops/s | write MiB/s |", "|---|---|---|---|"]
  for end_time in sorted(per_minute):
    row = per_minute[end_time]
    lines.append(
      f"| {end_time} | {row.get('read_bytes_count', 0) / 2**20:.1f} | {row.get('read_ops_count', 0):.0f} "
      f"| {row.get('write_bytes_count', 0) / 2**20:.1f} |"
    )
  table = "\n".join(lines) + "\n"
  (args.out / "monitoring.md").write_text(table)
  print(table)


if __name__ == "__main__":
  main()
