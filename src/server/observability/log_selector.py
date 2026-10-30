"""What a log store reads for a run: its worker pods, or its client's lines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerPods:
  """The namespace's pods whose names start with prefix, or the one of them named pod."""

  namespace: str
  prefix: str
  pod: str | None = None


@dataclass(frozen=True)
class ClientLines:
  """The lines labelled openrl_client=<client_id>, from wherever the client runs."""

  client_id: str


Selector = WorkerPods | ClientLines
