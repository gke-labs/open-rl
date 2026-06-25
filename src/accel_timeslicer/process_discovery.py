import os
import subprocess
from pathlib import Path

from .workload import DEFAULT_TIME_SLICE_GROUP, WorkloadRef

JOB_ID_ENV = "OPEN_RL_TIME_SLICE_JOB_ID"
GROUP_ENV = "OPEN_RL_TIME_SLICE_GROUP"


def discover_workload_gpu_pids(workload: WorkloadRef) -> list[int]:
  return sorted(pid for pid in nvidia_smi_compute_pids() if gpu_pid_matches_workload(pid, workload))


def workload_root_pids(workload: WorkloadRef) -> list[int]:
  roots = set()
  for pid in nvidia_smi_compute_pids():
    root = workload_root_pid(pid, workload)
    if root is not None:
      roots.add(root)
  return sorted(roots)


def gpu_pid_matches_workload(pid: int, workload: WorkloadRef) -> bool:
  return workload_root_pid(pid, workload) is not None


def workload_root_pid(pid: int, workload: WorkloadRef) -> int | None:
  if process_matches_workload(pid, workload):
    return pid
  pgid = process_group_id(pid)
  if pgid is not None and pgid != pid and process_matches_workload(pgid, workload):
    return pgid
  return None


def process_matches_workload(pid: int, workload: WorkloadRef) -> bool:
  env = process_environ(pid)
  return env.get(JOB_ID_ENV) == workload.job_id and env.get(GROUP_ENV, DEFAULT_TIME_SLICE_GROUP) == workload.group


def process_environ(pid: int) -> dict[str, str]:
  environ_path = Path("/proc") / str(pid) / "environ"
  try:
    raw = environ_path.read_bytes()
  except OSError:
    return {}

  env = {}
  for item in raw.split(b"\0"):
    if not item or b"=" not in item:
      continue
    key, value = item.split(b"=", 1)
    env[key.decode("utf-8", errors="ignore")] = value.decode("utf-8", errors="ignore")
  return env


def process_group_id(pid: int) -> int | None:
  try:
    return os.getpgid(pid)
  except OSError:
    return None


def nvidia_smi_compute_pids() -> list[int]:
  result = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    capture_output=True,
    check=False,
    text=True,
  )
  if result.returncode != 0:
    return []
  return sorted(int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit())
