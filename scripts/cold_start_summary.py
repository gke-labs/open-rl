"""Summarize cold-start timelines across cases: one row per run, medians per group.

  python3 scripts/cold_start_summary.py runs/2026-10-15_cold_start

Reads every <case>/timeline.json the timeline tool produced and prints a
markdown table of the phases that matter for the demo, then medians and
ranges for groups of runs whose directory names share a stem after the
`after-` prefix and before a trailing run number (B1/B2 -> B).
"""

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

PHASES = [
  ("total", "client.create_model.start", "client.sample.done"),
  ("create_model", "client.create_model.start", "client.create_model.done"),
  ("trainer pull", "trainer.pod_PodScheduled", "trainer.event_Pulled"),
  ("trainer load", "trainer.load_start", "trainer.load_done"),
  ("first sample", "client.sample.start", "client.sample.done"),
  ("sampler pull", "sampler.pod_PodScheduled", "sampler.event_Pulled"),
  ("vLLM load", "sampler.load_start", "sampler.weights_loaded"),
  ("compile", "sampler.weights_loaded", "sampler.compile_done"),
  ("capture", "sampler.compile_done", "sampler.engine_init_done"),
]


def ts(value):
  if not value:
    return None
  return datetime.fromisoformat(value.replace("Z", "+00:00"))


def phases_of(report: dict) -> dict[str, float | None]:
  marks = dict(report["client"])
  for w in report["workers"].values():
    marks.update(w["marks"])
  out = {}
  for name, a, b in PHASES:
    ta, tb = ts(marks.get(a, {}).get("ts")), ts(marks.get(b, {}).get("ts"))
    out[name] = round((tb - ta).total_seconds(), 1) if ta and tb else None
  return out


def main() -> None:
  root = Path(sys.argv[1])
  rows = []
  for path in sorted(root.glob("*/timeline.json")):
    report = json.loads(path.read_text())
    rows.append((path.parent.name, phases_of(report)))
  names = [p[0] for p in PHASES]
  print("| run | " + " | ".join(names) + " |")
  print("|---|" + "---|" * len(names))
  for case, ph in rows:
    print(f"| {case} | " + " | ".join("unmeasured" if ph[n] is None else f"{ph[n]:.1f}" for n in names) + " |")
  groups: dict[str, list[dict]] = {}
  for case, ph in rows:
    stem = re.sub(r"^(after-)?([A-Z])\d*-.*$", lambda m: ("after " if m.group(1) else "before ") + m.group(2), case)
    groups.setdefault(stem, []).append(ph)
  print("\n| group | n | " + " | ".join(f"{n} median (min-max)" for n in names) + " |")
  print("|---|---|" + "---|" * len(names))
  for stem, phs in sorted(groups.items()):
    cells = []
    for n in names:
      vals = [p[n] for p in phs if p[n] is not None]
      cells.append("unmeasured" if not vals else f"{statistics.median(vals):.1f} ({min(vals):.1f}-{max(vals):.1f})")
    print(f"| {stem} | {len(phs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
  main()
