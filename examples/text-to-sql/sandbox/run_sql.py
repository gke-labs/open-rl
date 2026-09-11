#!/usr/bin/env python3
"""Run one SQLite query inside the sandbox and print the rows as JSON.

Usage: python /opt/run_sql.py <job>

`<job>` is either a directory holding `schema.sql` and `query.sql`, or a JSON
file `{"schema": ..., "query": ...}` (one upload instead of two; the file is
deleted after it is read so the sandbox does not accumulate jobs). The schema
is executed, then the query, in an in-memory database, and one JSON object is
printed:

    {"rows": [[...], ...], "error": null}   on success
    {"rows": null, "error": "<sqlite message>"} on any sqlite error

The 250 ms progress-handler deadline and the 8-digit float rounding match
`examples/text-to-sql/utils/rewards.py:run_sql`, so a result computed here is
byte-comparable with one computed locally. Only untrusted, model-written SQL
should ever reach this script; the schema is trusted dataset content.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
import time
from pathlib import Path

DEADLINE_SECONDS = 0.25


def encode_value(value: object) -> object:
  if isinstance(value, float):
    return round(value, 8)
  if isinstance(value, bytes):
    return {"__bytes__": base64.b64encode(value).decode("ascii")}
  return value


def load_job(job: Path) -> tuple[str, str]:
  if job.is_dir():
    return (job / "schema.sql").read_text(), (job / "query.sql").read_text()
  payload = json.loads(job.read_text())
  job.unlink(missing_ok=True)
  return str(payload["schema"]), str(payload["query"])


def run(job: Path) -> dict[str, object]:
  try:
    schema, query = load_job(job)
  except (OSError, ValueError, KeyError) as exc:
    return {"rows": None, "error": f"bad job: {exc}"}
  connection = sqlite3.connect(":memory:")
  try:
    deadline = time.monotonic() + DEADLINE_SECONDS
    connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    connection.executescript(schema)
    rows = connection.execute(query).fetchall()
    return {"rows": [[encode_value(v) for v in row] for row in rows], "error": None}
  except sqlite3.Error as exc:
    return {"rows": None, "error": str(exc)}
  finally:
    connection.close()


def main() -> int:
  if len(sys.argv) != 2:
    print(json.dumps({"rows": None, "error": "usage: run_sql.py <job-dir-or-json>"}))
    return 2
  print(json.dumps(run(Path(sys.argv[1]))))
  return 0


if __name__ == "__main__":
  sys.exit(main())
