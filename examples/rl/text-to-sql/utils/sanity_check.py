#!/usr/bin/env python3
import os
import subprocess

# Requirements
MIN_VRAM_GB = 23
MIN_RAM_GB = 32
RECOMMENDED_GPUS = 2


def check_gpu():
  print("Checking GPUs...")
  try:
    # Run nvidia-smi to get GPU info
    result = subprocess.run(
      ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True
    )
    gpus = result.stdout.strip().split("\n")
    num_gpus = len(gpus)
    print(f"Found {num_gpus} GPU(s):")

    all_ok = True
    for i, gpu in enumerate(gpus):
      name, mem = gpu.split(",")
      mem = int(mem.strip())
      print(f"  GPU {i}: {name.strip()} with {mem} MB VRAM")
      if mem < MIN_VRAM_GB * 1000:
        print(f"    [WARNING] GPU {i} has less than {MIN_VRAM_GB}GB VRAM. Recipe might fail with OOM.")
        all_ok = False

    if num_gpus < RECOMMENDED_GPUS:
      print(f"[WARNING] Found less than {RECOMMENDED_GPUS} GPUs. The recipe recommends {RECOMMENDED_GPUS} GPUs (one for sampler, one for trainer).")
      all_ok = False

    return all_ok
  except FileNotFoundError:
    print("[ERROR] nvidia-smi command not found. Are NVIDIA drivers installed?")
    return False
  except subprocess.CalledProcessError as e:
    print(f"[ERROR] Failed to run nvidia-smi: {e}")
    return False


def check_ram():
  print("Checking System RAM...")
  try:
    with open("/proc/meminfo") as f:
      for line in f:
        if "MemTotal" in line:
          # MemTotal:       12345678 kB
          parts = line.split()
          mem_kb = int(parts[1])
          mem_gb = mem_kb / (1024 * 1024)
          print(f"Total System RAM: {mem_gb:.2f} GB")
          if mem_gb < MIN_RAM_GB:
            print(f"[WARNING] System RAM is less than {MIN_RAM_GB} GB. Loading large models might cause OOM hangs.")
            return False
          return True
  except FileNotFoundError:
    print("[WARNING] Could not read /proc/meminfo. Skipping RAM check.")
    return True
  return True


def check_packages():
  print("Checking for required packages...")
  missing = []

  # Check make
  if subprocess.run(["which", "make"], capture_output=True).returncode != 0:
    missing.append("make")

  # Check gcc
  if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
    missing.append("build-essential (gcc)")

  # Check uv
  if subprocess.run(["which", "uv"], capture_output=True).returncode != 0:
    missing.append("uv")

  # Check for Python headers
  # We can check if python3-dev is installed or if Python.h exists in standard paths

  try:
    result = subprocess.run(["dpkg", "-l", "python3-dev"], capture_output=True, text=True)
    if result.returncode != 0 or "ii" not in result.stdout:
      # Try to check if Python.h exists in typical include paths
      import sysconfig

      include_dir = sysconfig.get_path("include")
      if not os.path.exists(os.path.join(include_dir, "Python.h")):
        missing.append("python3-dev (Python headers)")
  except FileNotFoundError:
    # dpkg not found, likely not Debian/Ubuntu. Skip this check or use another method.
    pass

  if missing:
    print(f"[WARNING] Missing packages: {', '.join(missing)}")
    print("Please install them:")
    print("  For system packages: sudo apt update && sudo apt install -y build-essential python3-dev make")
    print("  For uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return False

  print("All required packages found.")
  return True


def main():
  print("=== Open-RL Text-to-SQL Recipe Sanity Check ===")
  gpu_ok = check_gpu()
  print("-" * 40)
  ram_ok = check_ram()
  print("-" * 40)
  pkg_ok = check_packages()
  print("-" * 40)

  if gpu_ok and ram_ok and pkg_ok:
    print("[SUCCESS] All checks passed! Your system seems ready for the recipe.")
  else:
    print("[ATTENTION] Some checks failed or produced warnings. Please review the messages above.")
    print(
      "You might still be able to run the recipe, but you may encounter issues. Please refer to the Troubleshooting section in the recipe README."
    )


if __name__ == "__main__":
  main()
