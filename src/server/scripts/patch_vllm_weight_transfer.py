"""Deprecated vLLM Weight Transfer Patch Script.

In Open-RL (Design Doc #004 for vLLM v0.25.1+), disk-based file monkey patching is replaced
by clean in-memory engine registration via `WeightTransferEngineFactory.register_engine("delta_snapshot", DeltaSnapshotWeightTransferEngine)`.

This script is maintained as a no-op placeholder for backwards compatibility.
"""


def patch_weight_transfer(*args, **kwargs) -> int:
  print("INFO: vLLM weight transfer patching is obsolete in v0.25.1+. Clean in-memory WeightTransferEngineFactory registration is active.")
  return 0


if __name__ == "__main__":
  patch_weight_transfer()
