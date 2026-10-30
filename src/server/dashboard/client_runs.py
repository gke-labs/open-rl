"""A run's client directory: the tinker-cookbook's log_path on a volume the server also mounts, where the client writes
config.json, metrics.jsonl, checkpoints.jsonl and logs.log. The client does nothing for the server. Every read is
blocking, so callers run it in a thread."""

import json
import os
import re

from server.telemetry import finite

DEPTH = 3  # the root's run directories, such as /mnt/shared/<project>/<run> or one level deeper
MATCH_SECONDS = 60  # config.json is written just before the client creates its model
SCAN_BYTES = 4 << 20  # of logs.log per page
MESSAGE_CHARS = 16384
TINKER_PATH = re.compile(r'tinker://([^/"]+)/')
# The cookbook's file format, "%(name)s:%(lineno)d [%(levelname)s] %(message)s"; other lines continue the record above.
HEADER = re.compile(r"\S+:\d+ \[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\] ")
listings: dict[str, tuple[int, list[str], bool]] = {}  # directory -> (mtime, subdirectories, is a run directory)
parsed: dict[str, tuple[tuple[int, int], object]] = {}  # file -> ((mtime, size), parsed contents)
progress: dict[str, tuple[int, int, dict[int, dict[str, float]]]] = {}  # metrics.jsonl -> (inode, bytes read, metrics by step)


def root() -> str:
  return os.getenv("OPEN_RL_CLIENT_RUNS_ROOT", "/mnt/shared")


def directories(path: str | None = None, depth: int = DEPTH) -> list[str]:
  """Every directory under the root, to DEPTH, that holds config.json or checkpoints.jsonl. A run directory is not
  descended into, and a directory is listed again only when its mtime changes."""
  path = path or root()
  try:
    mtime = os.stat(path).st_mtime_ns
    if (listing := listings.get(path)) is None or listing[0] != mtime:
      with os.scandir(path) as scan:
        entries = list(scan)
      run = any(e.name in ("config.json", "checkpoints.jsonl") for e in entries)
      below = [] if run else [e.path for e in entries if not e.name.startswith(".") and e.name != "lost+found" and e.is_dir(follow_symlinks=False)]
      listing = listings[path] = (mtime, below, run)
  except OSError:  # no root here, or a directory removed while it was scanned
    return []
  if listing[2]:
    return [path]
  return [found for below in listing[1] for found in directories(below, depth - 1)] if depth > 0 else []


def read(path: str, parse) -> tuple[float, object] | None:
  """The file's mtime and parsed contents, read again only when its mtime or size changes. None when it is absent, as
  checkpoints.jsonl is until the first save, or is not what the cookbook writes."""
  try:
    stat = os.stat(path)
    if (entry := parsed.get(path)) is None or entry[0] != (stat.st_mtime_ns, stat.st_size):
      with open(path, encoding="utf-8", errors="replace") as f:
        entry = parsed[path] = ((stat.st_mtime_ns, stat.st_size), parse(f.read()))
  except (OSError, ValueError):
    return None
  return entry[0][0] / 1e9, entry[1]


def started(directory: str, run: dict) -> bool:
  """Whether config.json was written within MATCH_SECONDS of the run's creation, for its base model."""
  if not (found := read(os.path.join(directory, "config.json"), json.loads)) or not isinstance(config := found[1], dict):
    return False
  return abs(found[0] - run["created_at"]) <= MATCH_SECONDS and config.get("model_name") == run["model"]


def checkpoint_runs(directory: str) -> set[str]:
  found = read(os.path.join(directory, "checkpoints.jsonl"), lambda text: set(TINKER_PATH.findall(text)))
  return found[1] if found else set()


def match(run: dict, candidates: list[str]) -> str | None:
  """The run's directory: the one whose checkpoints.jsonl names tinker://<run_id>/, else the one started() for it. Of
  several, only one that was also started() for it is taken; otherwise none, never a guess."""
  exact = [d for d in candidates if run["run_id"] in checkpoint_runs(d)]
  if len(exact) == 1:
    return exact[0]
  timed = [d for d in exact or candidates if started(d, run)]
  return timed[0] if len(timed) == 1 else None


def metrics(directory: str) -> dict[int, dict[str, float]]:
  """metrics.jsonl's finite numbers by each line's "step", reading only what was appended since the last call."""
  path = os.path.join(directory, "metrics.jsonl")
  try:
    stat = os.stat(path)
  except FileNotFoundError:
    return {}  # no step has finished yet
  inode, done, rows = progress.get(path, (stat.st_ino, 0, {}))
  if inode != stat.st_ino or stat.st_size < done:  # replaced or truncated
    done, rows = 0, {}
  with open(path, "rb") as f:
    f.seek(done)
    data = f.read(stat.st_size - done)
  data = data[: data.rfind(b"\n") + 1]  # a line still being written waits for the next read
  rows = dict(rows)
  for row in map(json.loads, data.splitlines()):
    if isinstance(step := row.pop("step", None), int):  # a call to log_metrics without a step has none
      rows[step] = rows.get(step, {}) | finite(row)
  progress[path] = (stat.st_ino, done + len(data), rows)
  return rows


def log_page(directory: str, before: int | None, limit: int, q: str = "", severity: str | None = None) -> dict:
  """logs.log, newest first, in the shape of a Cloud Logging page. Its lines carry no time, so a record is a header line
  and the lines under it, such as a traceback, and next_cursor is the byte offset the next older page ends at."""
  path = os.path.join(directory, "logs.log")
  with open(path, "rb") as f:
    size = f.seek(0, os.SEEK_END)
    end = size if before is None else min(before, size)
    start = max(0, end - SCAN_BYTES)
    f.seek(start)
    block = f.read(end - start)
  if start and (cut := block.find(b"\n") + 1):  # the first line began before the block
    block, start = block[cut:], start + cut
  lines, at = [], start
  for raw in block.splitlines(keepends=True):
    lines.append((at, raw.decode("utf-8", "replace").rstrip("\r\n")))
    at += len(raw)
  records, under, cursor = [], [], None
  for at, text in reversed(lines):
    under.append(text)
    # The file's first line starts a record, and so does a block's first line when the block has no header at all.
    if not (header := HEADER.match(text)) and at and not (at == start and cursor is None):
      continue
    message, under, cursor = "\n".join(reversed(under)), [], at
    level = header[1] if header else "UNKNOWN"
    if (not severity or level == severity) and q.lower() in message.lower():
      record = {"id": f"logs.log:{at}", "timestamp": None, "pod": "client", "role": "client", "node": None, "severity": level}
      records.append(record | {"message": message[:MESSAGE_CHARS], "message_truncated": len(message) > MESSAGE_CHARS})
      if len(records) == limit:
        break
  return {"records": records, "next_cursor": str(cursor) if cursor else None}
