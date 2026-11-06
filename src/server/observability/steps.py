TRAINING = {"forward_backward", "optim_step"}


def steps(ops: list[dict]) -> list[dict]:
  """Trainer op records grouped by the step they belong to: the step's window, whether its optim_step succeeded, and
  per op its count, failures, seconds and metric spread. Ops after the last optim_step that do no training, such as
  the final save or an eval, belong to the last step rather than to one of their own."""
  numbered, completed = [], 0
  for s in ops:
    # A record kept before ops carried their step is placed by the optim_steps before it.
    numbered.append((s["step"] if isinstance(s.get("step"), int) else completed, s))
    completed += s["operation"] == "optim_step" and s["status"] == "succeeded"
  numbers = sorted({number for number, _ in numbered})
  if len(numbers) > 1 and not any(s["operation"] in TRAINING for number, s in numbered if number == numbers[-1]):
    numbered = [(numbers[-2] if number == numbers[-1] else number, s) for number, s in numbered]
  found: dict[int, dict] = {}
  for number, s in numbered:
    step = found.setdefault(number, {"step": number, "started_at": s["started_at"], "ended_at": s["at"], "ops": {}, "nonfinite": set()})
    step["started_at"], step["ended_at"] = min(step["started_at"], s["started_at"]), max(step["ended_at"], s["at"])
    op = step["ops"].setdefault(s["operation"], {"count": 0, "failed": 0, "seconds": 0.0, "metrics": {}})
    op["count"] += 1
    op["failed"] += s["status"] != "succeeded"
    op["seconds"] += s["elapsed_seconds"]
    for name, value in s["metrics"].items():
      op["metrics"].setdefault(name, []).append(value)
    step["nonfinite"].update(s.get("nonfinite", ()))
  for step in found.values():
    optim = step["ops"].get("optim_step", {"count": 0, "failed": 0})
    step.update(nonfinite=sorted(step["nonfinite"]), complete=optim["count"] > optim["failed"])
    for op in step["ops"].values():
      op["metrics"] = {name: {"min": min(v), "max": max(v), "mean": sum(v) / len(v)} for name, v in op["metrics"].items()}
  return [found[k] for k in sorted(found)]
