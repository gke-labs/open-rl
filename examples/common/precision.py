"""The openrl.* precision settings every model must name.

Kept free of third-party imports so any example can use it.
"""

from __future__ import annotations

import os

DTYPES = ("float32", "bfloat16")
KEYS = ("openrl.trainer_dtype", "openrl.sampler_dtype")


def ensure_precision_tags(dtype: str) -> None:
  """Name the trainer and sampler dtype in TINKER_TAGS, which the SDK sends
  when it opens a session, so it covers every client a script (or the
  cookbook) creates. A dtype already in TINKER_TAGS is kept, so a launcher's
  choice wins over the script's default."""
  if dtype not in DTYPES:
    raise ValueError(f"dtype must be one of {', '.join(DTYPES)}, got {dtype!r}")
  tags = [tag for tag in os.getenv("TINKER_TAGS", "").split(",") if tag]
  present = dict(tag.partition("=")[::2] for tag in tags)
  # One dtype set by the launcher names the other too: they must be equal.
  dtype = next((present[key] for key in KEYS if key in present), dtype)
  tags += [f"{key}={dtype}" for key in KEYS if key not in present]
  os.environ["TINKER_TAGS"] = ",".join(tags)
