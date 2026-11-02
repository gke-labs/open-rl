"""Named PromQL queries over DCGM's GPU metrics, read from the Prometheus-compatible endpoint at OPEN_RL_PROMETHEUS_URL."""

import json
import math
import os
import re
from string import Template

import httpx

DCGM_SECONDS = 10  # how often k8s/deploy/dcgm collects; a finer step only repeats samples
# Per name, the metric each query reads and the query, tried in order until one has series. $gpus matches the GPU UUIDs,
# $step is the seconds between points.
QUERIES = {
  # The percent of each step the graphics engine was busy, so points=1 is the window's duty cycle. GPU_UTIL, where the
  # profiling metric is missing, samples only about the last second of each collection interval.
  "gpu_busy": {
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE": "max by (Hostname, UUID, gpu) (100 * avg_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID=~$gpus}[${step}s]))",
    "DCGM_FI_DEV_GPU_UTIL": "max by (Hostname, UUID, gpu) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus}[${step}s]))",
  },
  "gpu_util": {"DCGM_FI_DEV_GPU_UTIL": "max by (Hostname, UUID, gpu) (DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus})"},
  "gpu_memory_mib": {"DCGM_FI_DEV_FB_USED": "max by (Hostname, UUID, gpu) (DCGM_FI_DEV_FB_USED{UUID=~$gpus})"},
}


async def get(url: str, params: dict) -> dict:
  async with httpx.AsyncClient(timeout=10) as client:
    return (await client.get(url, params=params)).raise_for_status().json()


async def query(name: str, gpus: list[str], low: float, high: float, points: int) -> dict:
  """The named query over these GPUs in [low, high], from the first of its metrics that has series."""
  url, step = os.environ["OPEN_RL_PROMETHEUS_URL"].rstrip("/"), max(DCGM_SECONDS, int((high - low) / points))
  matcher = json.dumps("|".join(re.escape(uuid) for uuid in gpus))
  for metric, template in QUERIES[name].items():
    text = Template(template).substitute(gpus=matcher, step=step)
    params = {"query": text, "start": high if points == 1 else low, "end": high, "step": step}
    # Managed Service for Prometheus answers "result": null where plain Prometheus answers [].
    rows = (await get(f"{url}/api/v1/query_range", params))["data"].get("result") or []
    series = [{"labels": r["metric"], "points": [[float(t), float(v)] for t, v in r["values"] if math.isfinite(float(v))]} for r in rows]
    answer = {"metric": metric, "query": text, "step": step, "series": series}
    if series:
      return answer
  return answer
