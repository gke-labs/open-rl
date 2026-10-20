"""How much of each checkpoint file is resident in this node's page cache.

  python3 scripts/cold_start_resident.py '/mnt/shared/open-rl/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/*.safetensors'

Run inside a pod that mounts the shared volume (kubectl cp it in). mincore
answers per page without reading anything, so it does not disturb what it
measures. Used in runs/2026-10-15_cold_start to show that a pod's reads
survive in the page cache only while the volume stays mounted on the node.
"""

import ctypes
import glob
import mmap
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
PAGE = os.sysconf("SC_PAGE_SIZE")


def resident_pages(path: str) -> tuple[int, int]:
  size = os.path.getsize(path)
  fd = os.open(path, os.O_RDONLY)
  try:
    mm = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
      pages = (size + PAGE - 1) // PAGE
      vec = (ctypes.c_ubyte * pages)()
      addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
      if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
      resident = sum(1 for b in vec if b & 1)
      del vec
      return resident, pages
    finally:
      mm.close()
  finally:
    os.close(fd)


def main() -> None:
  total_resident = total_pages = 0
  for path in sorted(glob.glob(sys.argv[1])):
    resident, pages = resident_pages(path)
    total_resident += resident
    total_pages += pages
    print(f"{os.path.basename(path):40} {pages * PAGE / 2**30:6.2f} GiB  resident {100 * resident / max(pages, 1):5.1f}%")
  print(f"TOTAL resident {100 * total_resident / max(total_pages, 1):.1f}% of {total_pages * PAGE / 2**30:.1f} GiB")


if __name__ == "__main__":
  main()
