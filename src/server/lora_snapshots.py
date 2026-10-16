"""Each saved LoRA reference gets an immutable copy of the adapter in vLLM's
VLLM_LORA_RESOLVER_CACHE_DIR, so its directory name is the served model name."""

import filecmp
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


def same_files(a: Path, b: Path) -> bool:
  names = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
  return names == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file()) and all(filecmp.cmp(a / n, b / n, shallow=False) for n in names)


def freeze_adapter(model_id: str, base_model: str, *refs: str | None) -> None:
  paths = {snapshot_root() / snapshot_name(ref) for ref in refs if ref}
  source = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", model_id, model_id)
  snapshot_root().mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=".saving-", dir=snapshot_root()) as staging:
    staged = Path(staging, "adapter")
    shutil.copytree(source, staged)
    # vLLM's filesystem resolver loads an adapter only when this names the
    # served model exactly, and a trainer may have loaded it from a local path.
    config_path = staged / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = base_model
    config_path.write_text(json.dumps(config))
    # Never overwrite a snapshot, because vLLM caches adapters and prefixes by name.
    # Saving the same weights under a name again, as an eval before and after no training does, changes nothing.
    if any(path.exists() and not same_files(staged, path) for path in paths):
      raise FileExistsError("Sampler weights reference already exists; save with a new name or sequence ID")
    for path in paths:
      if not path.exists():
        copy = Path(staging, path.name)
        shutil.copytree(staged, copy)
        copy.rename(path)
