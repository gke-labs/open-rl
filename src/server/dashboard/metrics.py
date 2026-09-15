"""GPU utilization and memory for one allocation, from DCGM through Prometheus.

Devices are matched by UUID, which both the DRA ResourceSlice and the DCGM
exporter publish, so nothing is inferred from a GPU index.
"""

import json
import math
import os

import httpx

from server.dashboard import gke
from server.dashboard.snapshot import snapshot
from server.store import get_store

SERIES = (("utilization", "DCGM_FI_DEV_GPU_UTIL"), ("memory_mib", "DCGM_FI_DEV_FB_USED"))
STEP_SECONDS = 15


def device_uuids(state: dict) -> dict[str, str | None]:
  return {device["id"]: device.get("uuid") for node in state["cluster"]["nodes"] for device in node["devices"]}


async def query_range(client: httpx.AsyncClient, url: str, metric: str, uuid: str, start: str, end: str) -> list[list[float]]:
  response = await client.get(
    f"{url}/api/v1/query_range", params={"query": f"{metric}{{UUID={json.dumps(uuid)}}}", "start": start, "end": end, "step": STEP_SECONDS}
  )
  response.raise_for_status()
  payload = response.json()
  if payload.get("status") != "success":
    raise ValueError("Query failed")
  # Duplicate scrape targets must not inflate utilization or memory.
  values: dict[float, float] = {}
  for result in payload["data"]["result"]:
    if result["metric"].get("UUID") != uuid:
      continue
    for at, value in result.get("values", []):
      number = float(value)
      if math.isfinite(number):
        values[float(at)] = max(number, values.get(float(at), number))
  return [list(item) for item in sorted(values.items())]


async def gpu_history(placement_id: str, since=None, until=None) -> dict:
  start, end = gke.time_range(since, until)
  state = await snapshot.current(get_store())
  placement = next((p for p in state["placements"] if p["id"] == placement_id), None) or next(
    (p for p in state["history"] if p["id"] == placement_id), None
  )
  if placement is None:
    return {"available": False, "reason": "Allocation not found", "devices": []}
  url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
  if not url:
    return {"available": False, "reason": "GPU metrics source is not configured (OPEN_RL_PROMETHEUS_URL)", "devices": []}
  uuids = device_uuids(state)
  devices = []
  async with httpx.AsyncClient(timeout=8) as client:
    for identity in placement["devices"]:
      uuid = uuids.get(identity)
      if not uuid:
        devices.append({"id": identity, "utilization": [], "memory_mib": [], "reason": "GPU UUID unavailable"})
        continue
      series = {"id": identity, "uuid": uuid}
      for name, metric in SERIES:
        try:
          series[name] = await query_range(client, url, metric, uuid, start, end)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
          series[name] = []
          series["reason"] = "GPU metrics query unavailable"
      devices.append(series)
  return {"available": any(d.get("utilization") for d in devices), "reason": None, "devices": devices, "since": start, "until": end}
