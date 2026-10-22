"""Read a checkpoint's files into the page cache before a loader maps them.

transformers' from_pretrained memory-maps safetensors and touches them tensor
by tensor. On NFS that is one synchronous read of a few hundred KiB at a time,
each paying the server's latency, so a large checkpoint on Filestore loads in
many minutes. vLLM avoids this by first streaming the files into the page cache
with several threads and large reads; this module does the same for the
trainers, so the mmap that follows hits memory.

Controlled by the environment:

  OPEN_RL_PREFETCH            auto (default: only on network filesystems), always, off
  OPEN_RL_PREFETCH_THREADS    8
  OPEN_RL_PREFETCH_BLOCK_MIB  16

The prefetch is skipped when the files would not fit in 90% of the memory
available to this process, because the page cache would evict them before
the loader arrives.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
NETWORK_FS = {"nfs", "nfs4", "lustre", "fuse", "fuse.gcsfuse", "cifs", "smb3"}
# One task per this many bytes, so a few large shards still spread over all threads.
CHUNK_BYTES = 256 * 1024 * 1024


def checkpoint_dir(name_or_path: str) -> Path | None:
  """The directory holding the checkpoint: the path itself, or the Hugging Face
  cache snapshot for a model id. No network: a model that is not cached yet is
  downloaded by the loader, not by us."""
  path = Path(name_or_path)
  if path.is_dir():
    return path
  try:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name_or_path, local_files_only=True, allow_patterns=list(PATTERNS)))
  except Exception:
    return None


def checkpoint_files(name_or_path: str) -> list[Path]:
  directory = checkpoint_dir(name_or_path)
  if directory is None:
    return []
  files = [p for pattern in PATTERNS for p in directory.glob(pattern) if p.is_file()]
  # The HF cache keeps symlinks into blobs/; stat follows them, which is what we want.
  return sorted(set(files), key=lambda p: p.stat().st_size, reverse=True)


def fs_type(path: Path) -> str:
  """The filesystem type of the mount that holds path, from /proc/mounts."""
  best, best_type = "", ""
  try:
    resolved = str(path.resolve())
    with open("/proc/mounts") as f:
      for line in f:
        parts = line.split()
        if len(parts) < 3:
          continue
        mount = parts[1]
        if (resolved == mount or resolved.startswith(mount.rstrip("/") + "/")) and len(mount) >= len(best):
          best, best_type = mount, parts[2]
  except OSError:
    return ""
  return best_type


def available_bytes() -> int:
  """What the page cache can hold for us: MemAvailable, capped by the cgroup limit."""
  avail = 0
  try:
    with open("/proc/meminfo") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          avail = int(line.split()[1]) * 1024
  except OSError:
    return 0
  for limit_file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
      raw = Path(limit_file).read_text().strip()
      if raw.isdigit():
        avail = min(avail, int(raw))
    except OSError:
      continue
  return avail


def read_range(path: Path, offset: int, length: int, block: int) -> int:
  """Read [offset, offset+length) with plain reads; the kernel keeps the pages."""
  done = 0
  fd = os.open(path, os.O_RDONLY)
  try:
    while done < length:
      chunk = os.pread(fd, min(block, length - done), offset + done)
      if not chunk:
        break
      done += len(chunk)
  finally:
    os.close(fd)
  return done


def prefetch(name_or_path: str, log=print) -> dict[str, Any] | None:
  """Warm the page cache for name_or_path's checkpoint files. Returns what it
  did, or None when it did nothing."""
  mode = os.getenv("OPEN_RL_PREFETCH", "auto").lower()
  if mode == "off":
    return None
  files = checkpoint_files(name_or_path)
  if not files:
    return None
  total = sum(p.stat().st_size for p in files)
  kind = fs_type(files[0])
  if mode == "auto" and kind not in NETWORK_FS:
    return None
  avail = available_bytes()
  if avail and total > 0.9 * avail and mode != "always":
    log(f"[prefetch] {total / 2**30:.1f} GiB of checkpoint files would not fit in {avail / 2**30:.1f} GiB of available memory; not prefetching")
    return None
  threads = int(os.getenv("OPEN_RL_PREFETCH_THREADS", "8"))
  block = int(os.getenv("OPEN_RL_PREFETCH_BLOCK_MIB", "16")) * 1024 * 1024
  tasks = []
  for path in files:
    size = path.stat().st_size
    for offset in range(0, size, CHUNK_BYTES):
      tasks.append((path, offset, min(CHUNK_BYTES, size - offset)))
  log(f"[prefetch] reading {len(files)} file(s), {total / 2**30:.2f} GiB, from {kind or 'unknown fs'} into the page cache with {threads} threads")
  started = time.time()
  with ThreadPoolExecutor(max_workers=threads) as pool:
    read = sum(pool.map(lambda t: read_range(t[0], t[1], t[2], block), tasks))
  seconds = time.time() - started
  log(f"[prefetch] done: {read / 2**30:.2f} GiB in {seconds:.1f} s ({read / 2**20 / max(seconds, 1e-6):.0f} MiB/s)")
  return {"files": len(files), "bytes": read, "seconds": round(seconds, 2), "fs": kind, "threads": threads, "started_at": started}
