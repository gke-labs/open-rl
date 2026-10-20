"""Codec for the Tinker public protobuf wire format.

Tinker SDK 0.25.0 and later send ``POST /api/v1/forward_backward`` as
``application/x-protobuf`` and only accept protobuf bodies when polling
``/api/v1/retrieve_future`` for ``ForwardBackwardOutput`` and ``SampleResponse``.
Everything else on the API stays JSON.

This module converts between those messages and the JSON-shaped dicts the
API server already queues and the workers already produce, so nothing past the
API server edge changes. The schema is vendored in ``server.proto``; see
``scripts/sync_tinker_proto.sh``.

Only the stdlib ``array`` module is used for byte packing. The API server image
does not ship numpy.
"""

from __future__ import annotations

import array
import math
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from google.protobuf.message import DecodeError

from server.proto import tinker_public_pb2 as pb

PROTO_CONTENT_TYPE = "application/x-protobuf"

# Result types (as stored by the workers) that the SDK expects as protobuf.
PROTO_RESULT_TYPES = frozenset({"forward_backward_completed", "forward_backward", "sample_completed", "sample"})


class ProtoDecodeError(ValueError):
  """A protobuf request that parsed but asks for something the workers cannot run."""


_LITTLE_ENDIAN = sys.byteorder == "little"

# The wire is little-endian native layout: int32 tokens, int64/float32 tensors.
_INT32 = "i"
_INT64 = "q"
_FLOAT32 = "f"
for _code, _size in ((_INT32, 4), (_INT64, 8), (_FLOAT32, 4)):
  if array.array(_code).itemsize != _size:  # pragma: no cover - platform sanity check
    raise RuntimeError(f"array typecode {_code!r} is {array.array(_code).itemsize} bytes on this platform, expected {_size}")

# Proto dtype -> (array typecode, public dtype label). int32 widens to int64 to
# match the JSON path, where every integer tensor is labelled "int64".
_TENSOR_DTYPES: dict[int, tuple[str, str]] = {
  pb.DTYPE_FLOAT32: (_FLOAT32, "float32"),
  pb.DTYPE_INT64: (_INT64, "int64"),
  pb.DTYPE_INT32: (_INT32, "int64"),
}

_STOP_REASONS: dict[str, int] = {
  "stop": pb.STOP_REASON_STOP,
  "length": pb.STOP_REASON_LENGTH,
}


def _from_bytes(code: str, raw: bytes) -> array.array:
  arr = array.array(code)
  if len(raw) % arr.itemsize:
    raise ProtoDecodeError(f"tensor payload of {len(raw)} bytes is not a multiple of {arr.itemsize}")
  arr.frombytes(raw)
  if not _LITTLE_ENDIAN:
    arr.byteswap()
  return arr


def _to_bytes(code: str, values: Iterable[Any]) -> bytes:
  arr = array.array(code, values)
  if not _LITTLE_ENDIAN:
    arr.byteswap()
  return arr.tobytes()


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


def _decode_tensor(name: str, tensor: pb.Tensor) -> dict[str, Any]:
  if tensor.WhichOneof("encoding") == "sparse_csr":
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] is a sparse tensor; only dense tensors are supported")
  spec = _TENSOR_DTYPES.get(tensor.dtype)
  if spec is None:
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] has dtype {pb.DType.Name(tensor.dtype)}; only float32, int64, and int32 are supported")
  code, label = spec
  data = _from_bytes(code, tensor.dense).tolist()
  shape = list(tensor.shape) or [len(data)]
  return {"data": data, "dtype": label, "shape": shape}


def _decode_chunk(chunk: pb.Chunk) -> dict[str, Any]:
  kind = chunk.WhichOneof("chunk")
  if kind != "encoded_text":
    raise ProtoDecodeError(f"model_input chunk of kind {kind or 'unset'!r} is not supported; this server only runs text models")
  return {"type": "encoded_text", "tokens": _from_bytes(_INT32, chunk.encoded_text.tokens).tolist()}


def _decode_loss_config(msg: pb.ForwardBackwardRequest) -> dict[str, Any]:
  # v2 carries both numbers and strings and is preferred when present; the
  # SDK dual-writes floats into the legacy map for older servers.
  if msg.loss_fn_config_v2:
    out: dict[str, Any] = {}
    for key, value in msg.loss_fn_config_v2.items():
      out[key] = value.text if value.WhichOneof("value") == "text" else value.number
    return out
  return dict(msg.loss_fn_config)


def decode_forward_backward(body: bytes) -> dict[str, Any]:
  """Decode a protobuf ``ForwardBackwardRequest`` into the JSON request shape.

  The result matches what the SDK sends on the JSON path, so the API server
  handlers treat both encodings the same way.
  """
  msg = pb.ForwardBackwardRequest()
  try:
    msg.ParseFromString(body)
  except DecodeError as exc:
    raise ProtoDecodeError(f"malformed ForwardBackwardRequest protobuf: {exc}") from exc

  data = []
  for datum in msg.data:
    data.append(
      {
        "model_input": {"chunks": [_decode_chunk(chunk) for chunk in datum.model_input]},
        "loss_fn_inputs": {name: _decode_tensor(name, tensor) for name, tensor in datum.loss_fn_inputs.items()},
      }
    )
  return {
    "model_id": msg.model_id,
    "seq_id": msg.seq_id,
    "forward_only": msg.forward_only,
    "forward_backward_input": {
      "data": data,
      "loss_fn": msg.loss_fn,
      "loss_fn_config": _decode_loss_config(msg),
    },
  }


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


_MASK_LOGPROB = -99999.0


def _float_or_nan(value: Any) -> float:
  # Samplers emit None for positions without a logprob (the first prompt
  # token). The wire is a flat float32 buffer, so keep the slot as NaN.
  return math.nan if value is None else float(value)


def _encode_topk_logprobs(target: pb.TopkLogprobs, rows: Sequence[Any], k: int = 0) -> None:
  """Populate a ``TopkLogprobs`` message from per-position ``[(token_id, logprob), ...]`` rows."""
  length = len(rows)
  inferred_k = k or max((len(row) for row in rows if row), default=0)
  if length == 0 or inferred_k <= 0:
    return
  token_ids: list[int] = []
  logprobs: list[float] = []
  for row in rows:
    entries = list(row or [])[:inferred_k]
    for tid, lp in entries:
      token_ids.append(int(tid))
      logprobs.append(float(lp))
    pad = inferred_k - len(entries)
    if pad > 0:
      token_ids.extend([0] * pad)
      logprobs.extend([_MASK_LOGPROB] * pad)
  target.token_ids = _to_bytes(_INT32, token_ids)
  target.logprobs = _to_bytes(_FLOAT32, logprobs)
  target.k = inferred_k
  target.length = length


def encode_sample_response(result: dict[str, Any]) -> bytes:
  """Encode a sampler result dict as a protobuf ``SampleResponse``."""
  msg = pb.SampleResponse()
  for seq in result.get("sequences") or []:
    out = msg.sequences.add()
    out.stop_reason = _STOP_REASONS.get(str(seq.get("stop_reason")), pb.STOP_REASON_LENGTH)
    out.tokens = _to_bytes(_INT32, seq.get("tokens") or [])
    logprobs = seq.get("logprobs")
    if logprobs:
      out.logprobs = _to_bytes(_FLOAT32, (_float_or_nan(v) for v in logprobs))
    topk_logprobs = seq.get("topk_logprobs")
    if topk_logprobs is not None:
      _encode_topk_logprobs(out.topk_sampled_logprobs, topk_logprobs, int(seq.get("topk_k") or 0))
  prompt_logprobs = result.get("prompt_logprobs")
  if prompt_logprobs:
    msg.prompt_logprobs = _to_bytes(_FLOAT32, (_float_or_nan(v) for v in prompt_logprobs))
  topk_prompt_logprobs = result.get("topk_prompt_logprobs")
  if topk_prompt_logprobs is not None:
    _encode_topk_logprobs(msg.topk_prompt_logprobs, topk_prompt_logprobs, int(result.get("topk_prompt_k") or 0))
  msg.prompt_cache_hit_tokens = int(result.get("prompt_cache_hit_tokens") or 0)
  return msg.SerializeToString()


def _tensor_parts(value: Any) -> tuple[Sequence[Any], str, list[int] | None]:
  if isinstance(value, dict):
    data = value.get("data") or []
    return data, str(value.get("dtype") or "float32"), value.get("shape")
  return value or [], "float32", None


def encode_forward_backward_output(result: dict[str, Any]) -> bytes:
  """Encode a trainer result dict as a protobuf ``ForwardBackwardOutput``.

  Every per-datum field is packed into one ``ArrayRecord``: values
  concatenated into ``data`` with byte ``offsets`` (``num_datums + 1``
  entries) so the SDK can slice them back per datum.
  """
  msg = pb.ForwardBackwardOutput()
  msg.loss_fn_output_type = str(result.get("loss_fn_output_type") or "ArrayRecord")
  for key, value in (result.get("metrics") or {}).items():
    msg.metrics[key] = float(value)

  outputs: list[dict[str, Any]] = result.get("loss_fn_outputs") or []
  if not outputs:
    return msg.SerializeToString()

  record = msg.loss_fn_outputs.add()
  record.num_datums = len(outputs)
  names: list[str] = []
  for output in outputs:
    for name in output:
      if name not in names:
        names.append(name)

  for name in names:
    buf = bytearray()
    offsets = [0]
    label = "float32"
    trailing: list[int] = []
    for index, output in enumerate(outputs):
      data, dtype, shape = _tensor_parts(output.get(name))
      if index == 0:
        label = dtype
        if shape and len(shape) > 1:
          trailing = [int(s) for s in shape[1:]]
      code = _INT64 if label == "int64" else _FLOAT32
      buf += _to_bytes(code, data if label == "int64" else (_float_or_nan(v) for v in data))
      offsets.append(len(buf))
    field = record.fields[name]
    field.data = bytes(buf)
    field.offsets = _to_bytes(_INT64, offsets)
    field.dtype = pb.DTYPE_INT64 if label == "int64" else pb.DTYPE_FLOAT32
    field.trailing_shape.extend(trailing)
  return msg.SerializeToString()


def encode_future_result(result: dict[str, Any]) -> bytes | None:
  """Encode a completed future for a client that asked for protobuf.

  Returns ``None`` for result types the SDK still reads as JSON.
  """
  result_type = result.get("type")
  if result_type in ("forward_backward_completed", "forward_backward"):
    return encode_forward_backward_output(result)
  if result_type in ("sample_completed", "sample"):
    return encode_sample_response(result)
  return None
