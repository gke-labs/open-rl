import os
import uuid
from typing import Any


def extract_token_count(request: dict[str, Any]) -> int:
  payload = request.get("payload") or {}
  if not isinstance(payload, dict):
    return 0
  if "num_tokens" in payload and isinstance(payload["num_tokens"], int):
    return payload["num_tokens"]
  inputs = payload.get("inputs") or payload.get("input_ids") or []
  labels = payload.get("labels") or []
  if isinstance(inputs, list) and len(inputs) > 0:
    if isinstance(inputs[0], list):
      return sum(len(seq) for seq in inputs)
    return len(inputs)
  if isinstance(labels, list) and len(labels) > 0:
    if isinstance(labels[0], list):
      return sum(len(seq) for seq in labels)
    return len(labels)
  return 0


async def record_slice_telemetry(
  store: Any,
  model_id: str,
  worker_role: str,
  t_start: float,
  t_end: float,
  token_count: int,
  model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
  num_params_billions: float = 0.5,
) -> None:
  dur_sec = max(t_end - t_start, 0.001)
  dur_ms = int(dur_sec * 1000)

  claim_id = os.getenv("OPEN_RL_DRA_CLAIM_ID") or os.getenv("DRA_RESOURCE_CLAIM") or "open-rl-shared-gpu-claim-01"
  node_name = os.getenv("NODE_NAME") or os.getenv("HOSTNAME") or "localhost"
  gpu_index = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0] if os.getenv("CUDA_VISIBLE_DEVICES") else 0)

  event = {
    "event_id": f"accel-evt-{uuid.uuid4().hex[:8]}",
    "resource_claim_id": claim_id,
    "node_name": node_name,
    "gpu_index": gpu_index,
    "job_id": model_id,
    "tenant_id": model_id,
    "model_name": model_name,
    "num_params_billions": num_params_billions,
    "worker_role": worker_role,
    "acquire_time": t_start,
    "release_time": t_end,
    "duration_ms": dur_ms,
    "tokens_processed": token_count,
  }

  try:
    record_func = getattr(store, "record_accel_usage_event", None)
    if record_func is not None:
      await record_func(claim_id, event)
  except Exception as exc:
    print(f"[TELEMETRY] Failed to record slice event: {exc}")


async def record_job_request_event(store: Any, model_id: str, request_id: str, data: dict[str, Any]) -> None:
  record_func = getattr(store, "record_job_request_event", None)
  if record_func is not None:
    try:
      await record_func(model_id, request_id, data)
    except Exception as exc:
      print(f"[TELEMETRY] Failed to record job request event: {exc}")
