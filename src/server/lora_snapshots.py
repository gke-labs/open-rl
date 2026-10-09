"""Saved LoRA references name immutable directories on the shared volume."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def snapshot_path(ref: str) -> Path | None:
  if not ref.startswith("tinker://") or "/sampler_weights/" not in ref:
    return None
  return Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_lora", hashlib.sha256(ref.encode()).hexdigest())


def freeze_adapter(model_id: str, *refs: str | None) -> None:
  """Publish complete snapshots before acknowledging the save. Never overwrite
  a reference: vLLM caches adapters and prefixes by that reference's identity.
  Keep snapshots for as long as saved clients can use them, like checkpoints.
  """
  paths = {snapshot_path(ref) for ref in refs if ref}
  if None in paths:
    raise ValueError("LoRA sampler snapshots require tinker:// sampler_weights references")
  if any(path.exists() for path in paths):
    raise FileExistsError("Sampler weights reference already exists; save with a new name or sequence ID")
  source = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", model_id, model_id)
  if not (source / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Saved LoRA adapter is missing: {source}")
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=path.parent) as staging:
      staged = Path(staging, "adapter")
      shutil.copytree(source, staged)
      staged.rename(path)


def resolve_lora_path(lora_id: str, lora_path: str | None) -> str | None:
  if saved := snapshot_path(lora_id):
    if not (saved / "adapter_config.json").is_file():
      raise FileNotFoundError(f"Saved LoRA sampler weights are unavailable: {lora_id}")
    return str(saved)
  # Bare model IDs retain legacy base-model sampling before the first save.
  base_id = lora_id.split("://")[-1].split("/")[0]
  peft = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", base_id, base_id)
  candidates = [Path(lora_path), Path(lora_path, base_id)] if lora_path else []
  return next((str(path) for path in [*candidates, peft] if (path / "adapter_config.json").is_file()), None)
