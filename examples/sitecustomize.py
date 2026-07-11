"""sitecustomize for examples project: runs automatically when Python starts to inject custom Open-RL strategy headers."""

import os

try:
  import tinker.lib.public_interfaces.service_client as _sc

  _orig_get_default_headers = _sc._get_default_headers

  def _patched_get_default_headers() -> dict[str, str]:
    headers = _orig_get_default_headers()
    if strategy := os.getenv("OPEN_RL_WEIGHT_SYNC_STRATEGY"):
      headers["X-Open-RL-Weight-Sync-Strategy"] = strategy
    if ft_type := os.getenv("OPEN_RL_FINE_TUNING_TYPE"):
      headers["X-Open-RL-Fine-Tuning-Type"] = ft_type
    return headers

  _sc._get_default_headers = _patched_get_default_headers
except Exception:
  pass
