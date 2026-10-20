"""The openrl.* precision settings every model must name.

Kept free of third-party imports so any example can use it.
"""

from __future__ import annotations

import os

DTYPES = ("float32", "bfloat16")
KEYS = ("openrl.trainer_dtype", "openrl.sampler_dtype")


def ensure_precision_tags(
  dtype: str = "bfloat16",
  *,
  trainer_dtype: str | None = None,
  sampler_dtype: str | None = None,
) -> None:
  """Name the trainer and sampler dtype in TINKER_TAGS, which the SDK sends
  when it opens a session, so it covers every client a script (or the
  cookbook) creates. A dtype already in TINKER_TAGS is kept, so a launcher's
  choice wins over the script's default."""
  resolved_trainer = trainer_dtype or dtype
  resolved_sampler = sampler_dtype or dtype
  for label, value in (("dtype", dtype), ("trainer_dtype", resolved_trainer), ("sampler_dtype", resolved_sampler)):
    if value not in DTYPES:
      raise ValueError(f"{label} must be one of {', '.join(DTYPES)}, got {value!r}")
  tags = [tag for tag in os.getenv("TINKER_TAGS", "").split(",") if tag]
  present = dict(tag.partition("=")[::2] for tag in tags)
  target = {
    "openrl.trainer_dtype": resolved_trainer,
    "openrl.sampler_dtype": resolved_sampler,
  }
  tags += [f"{key}={target[key]}" for key in KEYS if key not in present]
  os.environ["TINKER_TAGS"] = ",".join(tags)
