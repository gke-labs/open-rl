"""glob and grep for LAB's tools, run inside the sandbox so they see its filesystem.

Same ordering, limits, and symlink rules as LAB's host-side versions. Takes one
JSON argument and prints the results as a JSON list:
  {"kind": "glob", "root": ..., "pattern": ...}
  {"kind": "grep", "root": ..., "pattern": ..., "file_glob": ..., "mode": "files_with_matches" | "count" | "content"}
Standard library only: the guest is LAB's sandbox image, not this project.
"""

import json
import re
import sys
from pathlib import Path


def main() -> None:
  args = json.loads(sys.argv[1])
  root = Path(args["root"])
  real = root.resolve()

  def under(p):
    try:
      p.resolve().relative_to(real)
      return True
    except ValueError:
      return False

  if args["kind"] == "glob":
    found = sorted((m for m in root.glob(args["pattern"]) if m.is_file() and under(m)), key=lambda p: p.stat().st_mtime, reverse=True)
    print(json.dumps([str(m.relative_to(root)) for m in found[:100]]))
    return
  regex = re.compile(args["pattern"])
  out = []
  for f in root.glob(args["file_glob"] or "**/*"):
    if not f.is_file() or not under(f):
      continue
    try:
      text = f.read_text(encoding="utf-8", errors="replace")
    except Exception:
      continue
    hits = list(regex.finditer(text))
    if not hits:
      continue
    rel = str(f.relative_to(root))
    if args["mode"] == "files_with_matches":
      out.append(rel)
    elif args["mode"] == "count":
      out.append(f"{rel}: {len(hits)}")
    elif args["mode"] == "content":
      out.extend(f"{rel}:{i + 1}: {line}" for i, line in enumerate(text.split("\n")) if regex.search(line))
  print(json.dumps(out[:250]))


if __name__ == "__main__":
  main()
