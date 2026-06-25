from dataclasses import dataclass

DEFAULT_TIME_SLICE_GROUP = "shared-accelerator"
TRAINER_TIME_SLICE_GROUP = "trainers"
SAMPLER_TIME_SLICE_GROUP = "samplers"


def workload_job_id(role: str, model_id: str) -> str:
  return f"{role}-{model_id}"


@dataclass(frozen=True)
class WorkloadRef:
  job_id: str
  group: str = DEFAULT_TIME_SLICE_GROUP

  def __post_init__(self) -> None:
    if not self.job_id:
      raise ValueError("workload requires job_id")

  @property
  def key(self) -> str:
    return f"{self.group}:{self.job_id}"

  def as_payload(self) -> dict[str, str]:
    return {"job_id": self.job_id, "group": self.group}
