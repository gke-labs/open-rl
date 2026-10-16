"""Each saved LoRA reference gets an immutable copy of the adapter in vLLM's
VLLM_LORA_RESOLVER_CACHE_DIR, so its directory name is the served model name."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def snapshot_root() -> Path:
  return Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_lora")


def snapshot_name(ref: str) -> str | None:
  # A hash, because a reference holds a user-chosen alias and slashes.
  if not ref.startswith("tinker://") or "/sampler_weights/" not in ref:
    return None
  return hashlib.sha256(ref.encode()).hexdigest()


def freeze_adapter(model_id: str, base_model: str, *refs: str | None) -> None:
  # Never overwrite a snapshot, because vLLM caches adapters and prefixes by name.
  paths = {snapshot_root() / snapshot_name(ref) for ref in refs if ref}
  if any(path.exists() for path in paths):
    raise FileExistsError("Sampler weights reference already exists; save with a new name or sequence ID")
  source = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", model_id, model_id)
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=path.parent) as staging:
      staged = Path(staging, "adapter")
      shutil.copytree(source, staged)
      # vLLM's filesystem resolver loads an adapter only when this names the
      # served model exactly, and a trainer may have loaded it from a local path.
      config_path = staged / "adapter_config.json"
      config = json.loads(config_path.read_text())
      config["base_model_name_or_path"] = base_model
      config_path.write_text(json.dumps(config))
      staged.rename(path)
