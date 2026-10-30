"""Read a checkpoint's safetensors into the page cache before from_pretrained maps them.

from_pretrained memory-maps the files and faults them in a few pages at a time,
each read paying a round trip on a network filesystem. This uses vLLM's loader
prefetch policy (network filesystems only, 8 threads, 16 MiB reads, skipped when
the files exceed 90% of available memory) and splits files into chunks, so a
single-file checkpoint is read by every thread.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

NETWORK_FS = {"nfs", "nfs4", "lustre"}
THREADS = 8
BLOCK_BYTES = 16 * 1024 * 1024
# One task per chunk, so a single large file is read by every thread.
CHUNK_BYTES = 256 * 1024 * 1024
MEMINFO = Path("/proc/meminfo")
CGROUP = Path("/sys/fs/cgroup")


def checkpoint_files(name_or_path: str) -> list[Path]:
  """The safetensors of a checkpoint directory, or of a model id's Hugging Face
  cache snapshot. A model that is not cached yet is left to the loader."""
  directory = Path(name_or_path)
  if not directory.is_dir():
    try:
      directory = Path(snapshot_download(name_or_path, local_files_only=True, allow_patterns=["*.safetensors"]))
    except (LocalEntryNotFoundError, HFValidationError):
      return []
  # is_file follows the cache's symlinks into blobs/.
  return [p for p in directory.glob("*.safetensors") if p.is_file()]


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


def available_bytes() -> int | None:
  """MemAvailable, capped by the room left under the cgroup memory limit, or None
  when unknown. Cgroup usage is the working set (current minus inactive_file), as
  kubelet counts it, because the pod's own page cache can be reclaimed."""
  try:
    avail = next(int(line.split()[1]) * 1024 for line in MEMINFO.read_text().splitlines() if line.startswith("MemAvailable:"))
  except (OSError, StopIteration):
    return None
  try:
    limit = (CGROUP / "memory.max").read_text().strip()
    if limit != "max":
      stat = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
      working_set = int((CGROUP / "memory.current").read_text()) - int(stat["inactive_file"])
      avail = min(avail, max(0, int(limit) - working_set))
  except OSError:
    pass
  return avail


def read_range(path: Path, offset: int, length: int) -> int:
  """Read [offset, offset+length) with plain reads; the kernel keeps the pages."""
  done = 0
  fd = os.open(path, os.O_RDONLY)
  try:
    while done < length:
      chunk = os.pread(fd, min(BLOCK_BYTES, length - done), offset + done)
      if not chunk:
        break
      done += len(chunk)
  finally:
    os.close(fd)
  return done


def prefetch(name_or_path: str) -> int:
  """Warm the page cache for name_or_path's safetensors and return the bytes read.
  Errors are logged, not raised: without the prefetch the load is only slower."""
  try:
    return _prefetch(name_or_path)
  except Exception as e:
    print(f"[prefetch] failed, loading without it: {e!r}")
    return 0


def _prefetch(name_or_path: str) -> int:
  sizes = {path: path.stat().st_size for path in checkpoint_files(name_or_path)}
  if not sizes:
    return 0
  kind = fs_type(next(iter(sizes)))
  if kind not in NETWORK_FS:
    return 0
  total = sum(sizes.values())
  avail = available_bytes()
  if avail is not None and total > 0.9 * avail:
    print(f"[prefetch] {total / 2**30:.1f} GiB of checkpoint files would not fit in {avail / 2**30:.1f} GiB of available memory; not prefetching")
    return 0
  tasks = [(path, offset, min(CHUNK_BYTES, size - offset)) for path, size in sizes.items() for offset in range(0, size, CHUNK_BYTES)]
  print(f"[prefetch] reading {len(sizes)} file(s), {total / 2**30:.2f} GiB, from {kind} into the page cache with {THREADS} threads")
  started = time.monotonic()
  with ThreadPoolExecutor(max_workers=THREADS) as pool:
    read = sum(pool.map(read_range, *zip(*tasks)))
  seconds = time.monotonic() - started
  print(f"[prefetch] done: {read / 2**30:.2f} GiB in {seconds:.1f} s ({read / 2**20 / max(seconds, 1e-6):.0f} MiB/s)")
  return read
